"""Stand-in LLM Client for offline simulation and stress testing.

Matches GroqAsyncClient's interface (stream_chat) and returns StreamResponse.
Uses StandInSimulator to model lognormal TTFT, non-linear concurrency slowdown,
sliding-window rate limits, and fault injection.
"""

import asyncio
import os
import re
import time
from typing import Dict, List, Optional

from llm.client import StreamResponse, UsageInfo
from standin.simulator import StandInSimulator


class StandInLLMClient:
    def __init__(self, simulator: Optional[StandInSimulator] = None, config_path: str = "standin/config.yaml"):
        self.simulator = simulator or StandInSimulator(config_path)

    async def close(self):
        pass

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> StreamResponse:
        t_start = time.perf_counter()

        # Extract system prompt and user prompt
        system_content = ""
        user_content = ""
        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content", "")
            elif m.get("role") == "user":
                user_content = m.get("content", "")

        # Estimate prompt tokens via whitespace
        prompt_tokens = len(system_content.split()) + len(user_content.split())

        # Check simulator limits
        admitted, err_type, retry_after = await self.simulator.admit_request(prompt_tokens)
        if not admitted:
            t_now = time.perf_counter()
            elapsed_ms = (t_now - t_start) * 1000.0
            status_code = 429
            error_msg = f"Rate limit exceeded: {err_type}"
            if err_type == "daily_quota_exceeded":
                error_msg = "daily request or token limit reached"
            return StreamResponse(
                content="",
                time_to_headers_ms=elapsed_ms,
                first_token_ms=0.0,
                total_ms=elapsed_ms,
                status="rate_limited",
                status_code=status_code,
                usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
                rate_limit_headers={"retry-after": str(retry_after or 1.0)},
                retry_after_s=retry_after,
                error_body=error_msg,
            )

        try:
            # Check for simulated 500 error
            if self.simulator.rng.random() < self.simulator.p_500:
                t_now = time.perf_counter()
                elapsed_ms = (t_now - t_start) * 1000.0
                return StreamResponse(
                    content="",
                    time_to_headers_ms=elapsed_ms,
                    first_token_ms=0.0,
                    total_ms=elapsed_ms,
                    status="error",
                    status_code=500,
                    usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens),
                    error_body="Simulated internal server error (500)",
                )

            # Sample TTFT
            ttft_ms = self.simulator.sample_ttft()
            sleep_ttft_s = (ttft_ms / 1000.0) * self.simulator.time_dilation
            await asyncio.sleep(sleep_ttft_s)

            t_first_token = time.perf_counter()
            first_token_ms = (t_first_token - t_start) * 1000.0

            # Generate synthetic answer grounded on input passages
            answer_text = self._synthesize_answer(user_content)

            # Simulate token generation streaming time
            completion_tokens = len(answer_text.split())
            if self.simulator.tokens_per_second > 0:
                gen_time_s = (completion_tokens / self.simulator.tokens_per_second) * self.simulator.time_dilation
                await asyncio.sleep(gen_time_s)

            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0

            return StreamResponse(
                content=answer_text,
                time_to_headers_ms=first_token_ms * 0.8,
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                status="ok",
                status_code=200,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                rate_limit_headers={
                    "x-ratelimit-remaining-requests": str(max(0, self.simulator.rpm_limit - len(self.simulator.request_window))),
                    "x-ratelimit-remaining-tokens": str(max(0, self.simulator.tpm_limit - sum(tok for _, tok in self.simulator.token_window))),
                },
            )
        finally:
            self.simulator.release_request()

    def _synthesize_answer(self, user_prompt: str) -> str:
        """Deterministically crafts an answer citing retrieved passages or refusing."""
        # Find passage tags: [P1], [P2], etc.
        passage_matches = re.findall(r"\[P\d+\]\s*(?:\[[^\]]+\]|\([^\)]+\)):\s*([^\n]+)", user_prompt)
        
        # Extract the question text specifically (do not check cues against retrieved passages)
        q_match = re.search(r"Question:\s*(.*?)(?:\n\nAnswer:|$)", user_prompt, re.DOTALL)
        question_text = q_match.group(1).strip() if q_match else user_prompt
        q_lower = question_text.lower()

        unanswerable_cues = [
            "quantum", "surface code", "direct preference optimization", r"\bdpo\b",
            "bitnet", "1.58", "tpu v5e", "french-to-german", "rope scaling",
        ]
        # Moshi subtle negative: Moshi evaluated on WMT? No.
        if "moshi" in q_lower and "wmt" in q_lower:
            return "The papers do not say."

        for cue in unanswerable_cues:
            if re.search(cue, q_lower):
                return "The papers do not say."

        if not passage_matches:
            return "The papers do not say."

        # Extract top passage info
        top_tag = "[P1]"
        top_snippet = passage_matches[0].strip()
        # Truncate to first two sentences
        sentences = [s.strip() for s in top_snippet.split(".") if len(s.strip()) > 10]
        if sentences:
            lead = ". ".join(sentences[:2]) + "."
            return f"{lead} {top_tag}"
        return f"According to the text, {top_snippet[:150]}... {top_tag}"
