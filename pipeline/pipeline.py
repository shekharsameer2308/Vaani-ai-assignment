"""Core 4-step pipeline executing admit -> retrieve -> generate -> verify -> respond."""

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from retrieval.chunk import Chunk
from retrieval.search import search, SearchResult
from pipeline.verify import verify_citations, format_citations_block
from pipeline.fallback import get_fallback_answer
from pipeline.limits import RateLimiter, AdmissionResult
from llm.retry import evaluate_retry


REFUSAL_TEXT = "The papers do not say."


@dataclass
class StageTrace:
    stage: str
    timestamp_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDependencies:
    retriever_fn: Callable
    llm_client: Any
    limiter: RateLimiter
    cache: Dict[str, str] = field(default_factory=dict)
    clock_fn: Callable[[], float] = time.perf_counter
    piece_tokens: int = 200
    top_k: int = 4
    absence_threshold: Optional[float] = 12.0  # Calibrated minimum BM25 score for in-scope admission


PipelineDeps = PipelineDependencies


@dataclass
class Result:
    question: str
    answer: str
    citations: List[str]
    mode: str                    # "llm", "cache", "extractive", "daily_fallback", "refused", "given_up"
    status: str                  # "ok", "rate_limited", "timeout", "error"
    first_token_ms: float
    total_ms: float
    queue_wait_ms: float
    retry_sleep_ms: float
    tokens_in: int
    tokens_out: int
    retries: int
    trace: List[StageTrace] = field(default_factory=list)


async def answer(question: str, deps: PipelineDependencies) -> Result:
    """Async pipeline processing an engineer's question through 5 strict stages."""
    trace: List[StageTrace] = []
    t0 = deps.clock_fn()

    def record_stage(name: str, **kwargs):
        trace.append(StageTrace(stage=name, timestamp_ms=round((deps.clock_fn() - t0) * 1000.0, 2), details=kwargs))

    # 1. Admit stage (cache hit check)
    record_stage("admit_start")
    q_norm = question.strip().lower()

    if q_norm in deps.cache:
        cached_ans = deps.cache[q_norm]
        record_stage("admit_cache_hit")
        total_ms = (deps.clock_fn() - t0) * 1000.0
        return Result(
            question=question,
            answer=cached_ans,
            citations=[],
            mode="cache",
            status="ok",
            first_token_ms=round(total_ms, 2),
            total_ms=round(total_ms, 2),
            queue_wait_ms=0.0,
            retry_sleep_ms=0.0,
            tokens_in=len(question.split()),
            tokens_out=len(cached_ans.split()),
            retries=0,
            trace=trace,
        )

    # 2. Retrieve stage
    record_stage("retrieve_start")
    search_results = deps.retriever_fn(
        query=question,
        k=deps.top_k,
        piece_tokens=deps.piece_tokens,
        absence_threshold=deps.absence_threshold,
    )
    retrieved_chunks = [r.chunk for r in search_results]

    if not retrieved_chunks:
        # No passages passed absence threshold -> refuse with no model call
        record_stage("retrieve_empty_refusal")
        total_ms = (deps.clock_fn() - t0) * 1000.0
        return Result(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            mode="refused",
            status="ok",
            first_token_ms=round(total_ms, 2),
            total_ms=round(total_ms, 2),
            queue_wait_ms=0.0,
            retry_sleep_ms=0.0,
            tokens_in=len(question.split()),
            tokens_out=len(REFUSAL_TEXT.split()),
            retries=0,
            trace=trace,
        )

    record_stage("retrieve_passages_found", count=len(retrieved_chunks))

    # Check rate limiter / admission before calling LLM
    passages_text = format_citations_block(retrieved_chunks)
    estimated_prompt_tokens = len(question.split()) + len(passages_text.split()) + 60
    admission = await deps.limiter.acquire(estimated_tokens=estimated_prompt_tokens)
    record_stage("admission_evaluated", mode=admission.mode, wait_ms=admission.queue_wait_ms)

    if not admission.admitted:
        # Fallback path (extractive or daily_fallback)
        fallback_text = get_fallback_answer(
            top_chunk=retrieved_chunks[0] if retrieved_chunks else None,
            mode=admission.mode,
        )
        total_ms = (deps.clock_fn() - t0) * 1000.0
        return Result(
            question=question,
            answer=fallback_text,
            citations=[retrieved_chunks[0].citation] if retrieved_chunks else [],
            mode=admission.mode,
            status="ok",
            first_token_ms=round(total_ms, 2),
            total_ms=round(total_ms, 2),
            queue_wait_ms=admission.queue_wait_ms,
            retry_sleep_ms=0.0,
            tokens_in=len(question.split()),
            tokens_out=len(fallback_text.split()),
            retries=0,
            trace=trace,
        )

    # 3. Generate stage
    record_stage("generate_start")
    system_prompt = (
        "You are an assistant answering questions about research papers. "
        "Use ONLY the facts in the provided passages [P1]..[Pk]. "
        "Every claim sentence must cite the relevant passage tag (e.g. [P1]). "
        "If the passages do not contain the answer, say: 'The papers do not say.'"
    )
    user_prompt = f"Passages:\n{passages_text}\n\nQuestion: {question}\n\nAnswer:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    retries_executed = 0
    total_retry_sleep_ms = 0.0

    model_name = getattr(deps.llm_client, "default_model", None)
    if not model_name:
        if getattr(deps.llm_client, "base_url", "").startswith("https://api.x.ai"):
            model_name = "grok-beta"
        else:
            model_name = "llama-3.1-8b-instant"

    llm_resp = await deps.llm_client.stream_chat(
        messages=messages,
        model=model_name,
    )
    record_stage("generate_done", status=llm_resp.status, ttft_ms=llm_resp.first_token_ms)

    # Evaluate retry if generation failed
    if llm_resp.status != "ok":
        retry_decision = evaluate_retry(
            status_code=llm_resp.status_code,
            retry_count=retries_executed,
            first_token_received=(llm_resp.first_token_ms > 0),
            retry_after_header=llm_resp.retry_after_s,
        )
        if retry_decision.should_retry:
            retries_executed += 1
            total_retry_sleep_ms += retry_decision.sleep_s * 1000.0
            record_stage("generate_retry", attempt=retries_executed, reason=retry_decision.reason, sleep_s=retry_decision.sleep_s)
            await asyncio.sleep(retry_decision.sleep_s)
            llm_resp = await deps.llm_client.stream_chat(
                messages=messages,
                model="llama-3.1-8b-instant",
            )
            record_stage("generate_retry_done", status=llm_resp.status, ttft_ms=llm_resp.first_token_ms)

    if llm_resp.status != "ok":
        # Check if error message names daily limit
        if llm_resp.status_code == 429 and "daily" in (llm_resp.error_body or "").lower():
            deps.limiter.mark_daily_exhausted()
            fb = get_fallback_answer(retrieved_chunks[0], mode="daily_fallback")
            mode = "daily_fallback"
        else:
            fb = get_fallback_answer(retrieved_chunks[0], mode="extractive")
            mode = "extractive"

        total_ms = (deps.clock_fn() - t0) * 1000.0
        return Result(
            question=question,
            answer=fb,
            citations=[retrieved_chunks[0].citation],
            mode=mode,
            status=llm_resp.status,
            first_token_ms=llm_resp.first_token_ms,
            total_ms=round(total_ms, 2),
            queue_wait_ms=admission.queue_wait_ms,
            retry_sleep_ms=round(total_retry_sleep_ms, 2),
            tokens_in=llm_resp.usage.prompt_tokens,
            tokens_out=len(fb.split()),
            retries=retries_executed,
            trace=trace,
        )

    # 4. Verify stage
    record_stage("verify_start")
    raw_answer = llm_resp.content.strip()
    ver = verify_citations(raw_answer, retrieved_chunks)
    record_stage("verify_check", is_valid=ver.is_valid, reason=ver.reason)

    final_answer = raw_answer
    if not ver.is_valid:
        # One repair call on failure
        record_stage("verify_repair_start")
        repair_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_answer},
            {
                "role": "user",
                "content": f"Your response had citation errors ({ver.reason}). Rewrite your answer ensuring every claim cites a valid [P1]..[Pk] tag or say 'The papers do not say.'",
            },
        ]
        repair_resp = await deps.llm_client.stream_chat(messages=repair_messages, model=model_name)
        if repair_resp.status == "ok":
            repair_ver = verify_citations(repair_resp.content.strip(), retrieved_chunks)
            if repair_ver.is_valid:
                final_answer = repair_resp.content.strip()
                record_stage("verify_repair_success")
            else:
                final_answer = REFUSAL_TEXT
                record_stage("verify_repair_refusal")
        else:
            final_answer = REFUSAL_TEXT
            record_stage("verify_repair_failed")

    # 5. Respond stage
    record_stage("respond_done")
    total_ms = (deps.clock_fn() - t0) * 1000.0

    # Cache successful answer
    deps.cache[q_norm] = final_answer

    citations = [c.citation for c in retrieved_chunks]
    return Result(
        question=question,
        answer=final_answer,
        citations=citations,
        mode="llm" if final_answer != REFUSAL_TEXT else "refused",
        status="ok",
        first_token_ms=llm_resp.first_token_ms,
        total_ms=round(total_ms, 2),
        queue_wait_ms=admission.queue_wait_ms,
        retry_sleep_ms=round(total_retry_sleep_ms, 2),
        tokens_in=llm_resp.usage.prompt_tokens,
        tokens_out=llm_resp.usage.completion_tokens,
        retries=retries_executed,
        trace=trace,
    )


async def cli_main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Ask a question over the paper library")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--k", type=int, default=3, help="Number of passages to retrieve")
    parser.add_argument("--piece", type=int, default=200, help="Chunk whitespace token size")
    args = parser.parse_args()

    # Auto-detect client: Groq / xAI if keys exist, otherwise StandIn
    if os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY"):
        from llm.client import GroqAsyncClient
        client = GroqAsyncClient()
        source_name = "Live API (Groq/xAI)"
    else:
        from standin.app import StandInLLMClient
        client = StandInLLMClient()
        source_name = "Calibrated Stand-in"

    limiter = RateLimiter()
    deps = PipelineDependencies(
        retriever_fn=search,
        llm_client=client,
        limiter=limiter,
        top_k=args.k,
        piece_tokens=args.piece,
    )

    print(f"\n[Provider: {source_name}]")
    print(f"Query: {args.question}")
    result = await answer(args.question, deps)
    await client.close()

    print(f"\nAnswer ({result.mode}):")
    print(f"  {result.answer}\n")
    if result.citations:
        print("Citations:")
        for c in result.citations:
            print(f"  - {c}")
    print(f"\nMetrics: TTFT={result.first_token_ms}ms, Total={result.total_ms}ms, Tokens In/Out={result.tokens_in}/{result.tokens_out}")


if __name__ == "__main__":
    asyncio.run(cli_main())
