"""Fallback extractive and cached response generators when LLM service is saturated or exhausted."""

from typing import Dict, List, Optional
from retrieval.chunk import Chunk


def build_extractive_answer(top_chunk: Chunk, mode: str = "extractive") -> str:
    """Construct a grounded extractive answer using the top retrieved passage."""
    snippet = top_chunk.text.strip()

    if mode == "daily_fallback":
        disclaimer = "Answered without the language model because the daily budget has been reached."
    else:
        disclaimer = "Answered without the language model because the service is busy."

    answer = f'"{snippet}"\n\n{top_chunk.citation}\n\nNote: {disclaimer}'
    return answer


def get_fallback_answer(
    top_chunk: Optional[Chunk],
    cached_answer: Optional[str] = None,
    mode: str = "extractive",
) -> str:
    """Return cached answer if available; otherwise quote the top passage. Never return empty."""
    if cached_answer:
        return cached_answer

    if top_chunk is not None:
        return build_extractive_answer(top_chunk, mode=mode)

    return (
        "The service is currently unavailable and no relevant paper passages could be retrieved to answer this question."
    )
