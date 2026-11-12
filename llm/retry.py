"""Retry decision policy for LLM calls (decisions only, no HTTP)."""

from dataclasses import dataclass
import random
from typing import Optional


@dataclass
class RetryDecision:
    should_retry: bool
    sleep_s: float
    reason: str


def evaluate_retry(
    status_code: int,
    retry_count: int,
    first_token_received: bool,
    retry_after_header: Optional[float] = None,
    max_retries: int = 2,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 10.0,
) -> RetryDecision:
    """Determine whether to retry a failed LLM call based on strict assignment policy.
    
    Rules:
    - Never retry after the first token has already arrived.
    - Never retry on 400, 401, 403.
    - 429: sleep retry-after or exponential backoff with full jitter, max 2 retries.
    - 5xx or connect/read timeout before first token: retry once.
    """
    if first_token_received:
        return RetryDecision(
            should_retry=False,
            sleep_s=0.0,
            reason="Failure occurred after first token was already streamed; stream aborted",
        )

    if status_code in (400, 401, 403):
        return RetryDecision(
            should_retry=False,
            sleep_s=0.0,
            reason=f"Client error {status_code} is permanent; no retry",
        )

    if status_code == 429:
        if retry_count >= max_retries:
            return RetryDecision(
                should_retry=False,
                sleep_s=0.0,
                reason=f"Rate limit retry limit exceeded ({retry_count}/{max_retries})",
            )
        # Use retry-after header if present, otherwise exponential backoff with full jitter
        if retry_after_header is not None and retry_after_header > 0:
            sleep_time = min(max_backoff_s, retry_after_header + random.uniform(0.1, 0.5))
        else:
            calc_backoff = base_backoff_s * (2 ** retry_count)
            sleep_time = random.uniform(0.0, min(max_backoff_s, calc_backoff))
        return RetryDecision(
            should_retry=True,
            sleep_s=round(sleep_time, 3),
            reason=f"HTTP 429 rate limit backoff (attempt {retry_count + 1})",
        )

    if status_code >= 500 or status_code == 408:  # Server error or timeout
        if retry_count >= 1:
            return RetryDecision(
                should_retry=False,
                sleep_s=0.0,
                reason="Server error / timeout allows max 1 retry; limit exhausted",
            )
        sleep_time = random.uniform(0.2, 1.0)
        return RetryDecision(
            should_retry=True,
            sleep_s=round(sleep_time, 3),
            reason=f"Transient server error / timeout {status_code} retry",
        )

    return RetryDecision(
        should_retry=False,
        sleep_s=0.0,
        reason=f"Unhandled status code {status_code}; no retry",
    )
