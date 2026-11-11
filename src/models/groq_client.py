import json
import os
import time
from typing import Dict, Iterator, Optional
import requests

from src.models.base import BaseLLMClient, LLMResponse


class GroqClient(BaseLLMClient):
    """Client for Groq Chat Completions API with rate-limit and latency tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.last_rate_limit_headers: Dict[str, str] = {}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call Groq API with streaming to measure TTFT and handle retries."""
        if not self.api_key:
            return LLMResponse(
                content="",
                first_token_ms=0.0,
                total_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                status="error",
                retries=0,
                error_message="GROQ_API_KEY is missing",
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "stream": True,
        }

        retries = 0
        while retries <= self.max_retries:
            start_wall = time.perf_counter()
            first_token_wall: Optional[float] = None
            collected_chunks = []

            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                )

                # Capture rate limit headers
                self.last_rate_limit_headers = {
                    k.lower(): v for k, v in response.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"
                }

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 1.0))
                    if retries < self.max_retries:
                        retries += 1
                        time.sleep(retry_after)
                        continue
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return LLMResponse(
                        content="",
                        first_token_ms=0.0,
                        total_ms=total_ms,
                        tokens_in=len(prompt.split()),
                        tokens_out=0,
                        status="rate_limited",
                        retries=retries,
                        error_message=f"Rate limit exceeded (429): {response.text[:200]}",
                        headers=self.last_rate_limit_headers,
                    )

                if response.status_code >= 500:
                    if retries < self.max_retries:
                        retries += 1
                        time.sleep(self.backoff_factor * (2 ** retries))
                        continue
                    total_ms = (time.perf_counter() - start_wall) * 1000.0
                    return LLMResponse(
                        content="",
                        first_token_ms=0.0,
                        total_ms=total_ms,
                        tokens_in=len(prompt.split()),
                        tokens_out=0,
                        status="error",
                        retries=retries,
                        error_message=f"Server error {response.status_code}: {response.text[:200]}",
                        headers=self.last_rate_limit_headers,
                    )

                response.raise_for_status()

                # Process SSE stream
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_part = line_str[6:].strip()
                        if data_part == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_part)
                            delta = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                if first_token_wall is None:
                                    first_token_wall = time.perf_counter()
                                collected_chunks.append(delta)
                        except json.JSONDecodeError:
                            continue

                end_wall = time.perf_counter()
                total_ms = (end_wall - start_wall) * 1000.0
                first_token_ms = (
                    (first_token_wall - start_wall) * 1000.0
                    if first_token_wall is not None
                    else total_ms
                )
                full_text = "".join(collected_chunks)
                tokens_in = int(len(prompt.split()) * 1.3)
                tokens_out = int(len(full_text.split()) * 1.3)

                return LLMResponse(
                    content=full_text,
                    first_token_ms=round(first_token_ms, 2),
                    total_ms=round(total_ms, 2),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    status="ok",
                    retries=retries,
                    headers=self.last_rate_limit_headers,
                )

            except requests.Timeout:
                if retries < self.max_retries:
                    retries += 1
                    time.sleep(self.backoff_factor)
                    continue
                total_ms = (time.perf_counter() - start_wall) * 1000.0
                return LLMResponse(
                    content="",
                    first_token_ms=0.0,
                    total_ms=total_ms,
                    tokens_in=len(prompt.split()),
                    tokens_out=0,
                    status="timeout",
                    retries=retries,
                    error_message="Request timed out",
                )

            except Exception as e:
                total_ms = (time.perf_counter() - start_wall) * 1000.0
                return LLMResponse(
                    content="",
                    first_token_ms=0.0,
                    total_ms=total_ms,
                    tokens_in=len(prompt.split()),
                    tokens_out=0,
                    status="error",
                    retries=retries,
                    error_message=str(e),
                )

        return LLMResponse(
            content="",
            first_token_ms=0.0,
            total_ms=0.0,
            tokens_in=0,
            tokens_out=0,
            status="error",
            retries=retries,
            error_message="Exceeded max retries",
        )

