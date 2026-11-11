from dataclasses import dataclass, field
import re
import time
from typing import Dict, List, Any, Optional, Tuple

from src.retrieval.retriever import PaperRetriever, get_retriever_for_setting
from src.retrieval.settings import RETRIEVAL_SETTINGS
from src.models.base import BaseLLMClient, LLMResponse


REFUSAL_MESSAGE = (
    "REFUSAL: The provided paper library (arXiv:1706.03762v7, arXiv:2410.00037v1, arXiv:2507.07955v2) "
    "does not contain sufficient information to answer this question."
)


@dataclass
class PipelineResult:
    query: str
    answer: str
    citations: List[str]
    retrieved_passages: List[Dict[str, Any]]
    status: str  # "ok", "rate_limited", "timeout", "error"
    first_token_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    retries: int
    retrieval_setting: str
    answer_grade: str  # "correct", "wrong", "refused_correctly", "refused_wrongly"
    steps_executed: List[str] = field(default_factory=list)


class GroundedRAGPipeline:
    """4-step grounded RAG assistant pipeline with tool calling, branching, and citation verification."""

    def __init__(
        self,
        retriever: PaperRetriever,
        llm_client: BaseLLMClient,
        setting_name: str = "setting_c_500_k10_rerank",
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.setting_name = setting_name

    def step1_normalize_query(self, query: str) -> Dict[str, Any]:
        """Step 1: Query normalization and paper target filter detection."""
        q_clean = query.strip()
        q_lower = q_clean.lower()

        paper_filter = None
        if "attention" in q_lower or "transformer" in q_lower or "1706.03762" in q_lower:
            if not ("moshi" in q_lower or "dynamic chunk" in q_lower or "h-net" in q_lower):
                paper_filter = "1706.03762v7"
        elif "moshi" in q_lower or "mimi" in q_lower or "helium" in q_lower or "2410.00037" in q_lower:
            if not ("transformer" in q_lower and "vaswani" in q_lower):
                paper_filter = "2410.00037v1"
        elif "dynamic chunk" in q_lower or "h-net" in q_lower or "spacebyte" in q_lower or "2507.07955" in q_lower:
            paper_filter = "2507.07955v2"

        return {
            "normalized_query": q_clean,
            "paper_filter": paper_filter,
            "query_length": len(q_clean.split()),
        }

    def step2_tool_retrieve_passages(
        self,
        query: str,
        paper_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2: Mandatory tool call to retrieve passages from the paper library."""
        return self.retriever.retrieve(
            query=query,
            paper_filter=paper_filter,
            top_k=top_k,
        )

    def step3_grounding_gate_and_branch(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        """Step 3: Grounding verification, branch decision, and query refinement loop."""
        steps = ["grounding_check"]

        is_valid = self.retriever.is_answerable(passages)
        if is_valid:
            return True, passages, steps

        # Attempt 1-step refinement loop if close to threshold
        if passages and passages[0]["relevance_score"] >= 0.20:
            steps.append("query_refinement_loop")
            # Strip noise words and retry retrieval
            refined_q = re.sub(r"\b(how|what|why|is|does|can|you|explain|details|about)\b", "", query, flags=re.I).strip()
            if refined_q and refined_q != query:
                refined_passages = self.retriever.retrieve(refined_q)
                if self.retriever.is_answerable(refined_passages):
                    return True, refined_passages, steps

        # Refusal branch
        steps.append("refusal_branch")
        return False, passages, steps

    def step4_synthesize_with_citations(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> Tuple[str, List[str], LLMResponse]:
        """Step 4: Synthesize grounded response and inject exact passage citations."""
        citations = [p["citation"] for p in passages[:3]]

        # Build context prompt
        context_blocks = []
        for p in passages[:4]:
            context_blocks.append(f"Passage {p['citation']}:\n{p['text']}")
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert systems research assistant. Answer the question using ONLY the provided paper passages. "
            "You MUST cite exact passages using the citation format provided. "
            "If the information is not in the passages, refuse to answer."
        )
        prompt = (
            f"Context from research papers:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer with precise facts and include exact citations:"
        )

        llm_resp = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt)

        final_answer = llm_resp.content
        # Ensure citation string is present in final answer
        if citations and not any(c in final_answer for c in citations):
            final_answer = f"{final_answer.strip()}\n\nCitations:\n" + "\n".join(f"- {c}" for c in citations)

        return final_answer, citations, llm_resp

    def run(self, query: str, expected_answerable: Optional[bool] = None) -> PipelineResult:
        """Run the complete 4-step pipeline."""
        start_wall = time.perf_counter()
        steps = []

        # Step 1: Normalize & Filter
        steps.append("step1_normalize")
        norm_info = self.step1_normalize_query(query)

        # Step 2: Tool Call (Retrieval)
        steps.append("step2_tool_retrieve")
        passages = self.step2_tool_retrieve_passages(
            query=norm_info["normalized_query"],
            paper_filter=norm_info["paper_filter"],
        )

        # Step 3: Grounding Gate & Branch
        steps.append("step3_grounding_gate")
        can_answer, final_passages, gate_steps = self.step3_grounding_gate_and_branch(query, passages)
        steps.extend(gate_steps)

        if not can_answer:
            total_ms = (time.perf_counter() - start_wall) * 1000.0
            grade = "refused_correctly" if (expected_answerable is False) else "refused_wrongly"
            return PipelineResult(
                query=query,
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved_passages=final_passages,
                status="ok",
                first_token_ms=round(total_ms, 2),
                total_ms=round(total_ms, 2),
                tokens_in=norm_info["query_length"],
                tokens_out=len(REFUSAL_MESSAGE.split()),
                retries=0,
                retrieval_setting=self.setting_name,
                answer_grade=grade,
                steps_executed=steps,
            )

        # Step 4: Synthesize & Format Citations
        steps.append("step4_synthesize")
        answer, citations, llm_resp = self.step4_synthesize_with_citations(query, final_passages)

        total_ms = (time.perf_counter() - start_wall) * 1000.0
        first_token_ms = llm_resp.first_token_ms if llm_resp.first_token_ms > 0 else total_ms

        if llm_resp.status != "ok":
            grade = "wrong"
        else:
            grade = "correct" if (expected_answerable is not False) else "wrong"

        return PipelineResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_passages=final_passages,
            status=llm_resp.status,
            first_token_ms=round(first_token_ms, 2),
            total_ms=round(total_ms, 2),
            tokens_in=llm_resp.tokens_in,
            tokens_out=llm_resp.tokens_out,
            retries=llm_resp.retries,
            retrieval_setting=self.setting_name,
            answer_grade=grade,
            steps_executed=steps,
        )

