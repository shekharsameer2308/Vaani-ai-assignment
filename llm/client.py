"""Async Groq LLM client using a single shared httpx.AsyncClient."""

from dataclasses import dataclass, field
import json
import os
import time
from typing import AsyncIterator, Dict, Optional, Tuple

import httpx

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StreamResponse:
    content: str
    time_to_headers_ms: float
    first_token_ms: float
    total_ms: float
    status: str  # "ok", "rate_limited", "timeout", "error"
    status_code: int
    usage: UsageInfo
    rate_limit_headers: Dict[str, str] = field(default_factory=dict)
    retry_after_s: Optional[float] = None
    error_body: Optional[str] = None


class GroqAsyncClient:
    """Shared async client for Groq API."""

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._external_client = client is not None
        self._client = client

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=GROQ_BASE_URL, timeout=30.0)
        return self._client

    async def close(self):
        if not self._external_client and self._client and not self._client.is_closed:
            await self._client.aclose()

    async def list_models(self) -> Tuple[int, Dict[str, any], Dict[str, str]]:
        """GET /models, returns status_code, body, and headers."""
        client = await self.get_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = await client.get("/models", headers=headers)
        return resp.status_code, resp.json(), dict(resp.headers)

    async def stream_chat(
        self,
        messages: list,
        model: str,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> StreamResponse:
        """Single streaming completion call. Measures exact network timestamps and parses usage block."""
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        t_start = time.perf_counter()
        t_headers: Optional[float] = None
        t_first_token: Optional[float] = None
        chunks = []
        usage = UsageInfo()
        rate_headers = {}
        retry_after = None

        try:
            req = client.build_request("POST", "/chat/completions", headers=headers, json=payload, timeout=timeout_s)
            resp = await client.send(req, stream=True)
            t_headers = time.perf_counter()

            # Capture rate-limit headers
            for k, v in resp.headers.items():
                k_low = k.lower()
                if "ratelimit" in k_low or k_low in ("retry-after", "date"):
                    rate_headers[k_low] = v
                if k_low == "retry-after":
                    try:
                        retry_after = float(v)
                    except ValueError:
                        pass

            if resp.status_code == 429:
                err_text = await resp.aread()
                t_end = time.perf_counter()
                return StreamResponse(
                    content="",
                    time_to_headers_ms=round((t_headers - t_start) * 1000.0, 2),
                    first_token_ms=0.0,
                    total_ms=round((t_end - t_start) * 1000.0, 2),
                    status="rate_limited",
                    status_code=429,
                    usage=usage,
                    rate_limit_headers=rate_headers,
                    retry_after_s=retry_after,
                    error_body=err_text.decode("utf-8", errors="replace"),
                )

            if resp.status_code >= 400:
                err_text = await resp.aread()
                t_end = time.perf_counter()
                return StreamResponse(
                    content="",
                    time_to_headers_ms=round((t_headers - t_start) * 1000.0, 2),
                    first_token_ms=0.0,
                    total_ms=round((t_end - t_start) * 1000.0, 2),
                    status="error",
                    status_code=resp.status_code,
                    usage=usage,
                    rate_limit_headers=rate_headers,
                    retry_after_s=retry_after,
                    error_body=err_text.decode("utf-8", errors="replace"),
                )

            # Stream response chunks
            async for line in resp.aiter_lines():
                if not line:
                    continue
                line_str = line.strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        # Extract delta text
                        choices = chunk_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                            if delta:
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                chunks.append(delta)
                        # Extract usage block if present
                        if "usage" in chunk_json and chunk_json["usage"]:
                            u = chunk_json["usage"]
                            usage = UsageInfo(
                                prompt_tokens=u.get("prompt_tokens", 0),
                                completion_tokens=u.get("completion_tokens", 0),
                                total_tokens=u.get("total_tokens", 0),
                            )
                    except json.JSONDecodeError:
                        continue

            t_end = time.perf_counter()
            full_content = "".join(chunks)

            # If usage was not emitted in stream, fallback to whitespace word count
            if usage.total_tokens == 0:
                prompt_text = " ".join(m.get("content", "") for m in messages)
                usage = UsageInfo(
                    prompt_tokens=len(prompt_text.split()),
                    completion_tokens=len(full_content.split()),
                    total_tokens=len(prompt_text.split()) + len(full_content.split()),
                )

            headers_ms = (t_headers - t_start) * 1000.0
            first_tok_ms = (t_first_token - t_start) * 1000.0 if t_first_token else (t_end - t_start) * 1000.0
            total_ms = (t_end - t_start) * 1000.0

            return StreamResponse(
                content=full_content,
                time_to_headers_ms=round(headers_ms, 2),
                first_token_ms=round(first_tok_ms, 2),
                total_ms=round(total_ms, 2),
                status="ok",
                status_code=resp.status_code,
                usage=usage,
                rate_limit_headers=rate_headers,
                retry_after_s=retry_after,
            )

        except httpx.TimeoutException:
            t_end = time.perf_counter()
            return StreamResponse(
                content="",
                time_to_headers_ms=0.0,
                first_token_ms=0.0,
                total_ms=round((t_end - t_start) * 1000.0, 2),
                status="timeout",
                status_code=408,
                usage=usage,
                rate_limit_headers=rate_headers,
                error_body="Request timed out",
            )
        except Exception as e:
            t_end = time.perf_counter()
            return StreamResponse(
                content="",
                time_to_headers_ms=0.0,
                first_token_ms=0.0,
                total_ms=round((t_end - t_start) * 1000.0, 2),
                status="error",
                status_code=500,
                usage=usage,
                rate_limit_headers=rate_headers,
                error_body=str(e),
            )
