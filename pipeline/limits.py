"""Token bucket rate limiter, concurrency semaphore, and daily budget tracking."""

import asyncio
from dataclasses import dataclass
import os
import time
from typing import Optional, Tuple
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "limits.yaml")


@dataclass
class AdmissionResult:
    admitted: bool
    reason: str  # "ok", "rpm_blocked", "tpm_blocked", "daily_exhausted", "timeout"
    queue_wait_ms: float
    mode: str    # "llm", "extractive", "daily_fallback"


class RateLimiter:
    """Token bucket rate limiter with concurrency semaphore and daily budget counter."""

    def __init__(
        self,
        config_path: str = CONFIG_PATH,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        daily_requests_limit: Optional[int] = None,
        daily_tokens_limit: Optional[int] = None,
        in_flight_limit: Optional[int] = None,
        max_wait_seconds: Optional[float] = None,
    ):
        self.config_path = config_path
        self._load_config()

        if requests_per_minute is not None:
            self.rpm_limit = requests_per_minute
        if tokens_per_minute is not None:
            self.tpm_limit = tokens_per_minute
        if daily_requests_limit is not None:
            self.daily_requests_limit = daily_requests_limit
        if daily_tokens_limit is not None:
            self.daily_tokens_limit = daily_tokens_limit
        if in_flight_limit is not None:
            self.in_flight_limit = in_flight_limit
        if max_wait_seconds is not None:
            self.max_wait_s = max_wait_seconds

        # Semaphore for in-flight requests
        self._semaphore = asyncio.Semaphore(self.in_flight_limit)

        # Token bucket state (sliding-window / leak rate)
        self._last_rpm_update = time.monotonic()
        self._rpm_tokens = float(self.rpm_limit)

        self._last_tpm_update = time.monotonic()
        self._tpm_tokens = float(self.tpm_limit)

        # Daily usage
        self.daily_requests_used = 0
        self.daily_tokens_used = 0
        self._daily_exhausted = False

        self._lock = asyncio.Lock()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rl = cfg.get("rate_limits", {})
        self.rpm_limit = rl.get("requests_per_minute", 30)
        self.tpm_limit = rl.get("tokens_per_minute", 14400)
        self.daily_requests_limit = rl.get("daily_requests_limit", 14400)
        self.daily_tokens_limit = rl.get("daily_tokens_limit", 500000)

        pl = cfg.get("pipeline", {})
        self.max_wait_s = pl.get("max_wait_seconds", 5.0)
        self.in_flight_limit = pl.get("in_flight_semaphore", 5)

    def mark_daily_exhausted(self):
        """Called when a 429 error body explicitly names the daily limit."""
        self._daily_exhausted = True

    def is_daily_exhausted(self) -> bool:
        return (
            self._daily_exhausted
            or self.daily_requests_used >= self.daily_requests_limit
            or self.daily_tokens_used >= self.daily_tokens_limit
        )

    def _replenish(self, now: float):
        # Replenish RPM bucket (rate = limit / 60.0 per sec)
        delta_rpm = now - self._last_rpm_update
        self._rpm_tokens = min(float(self.rpm_limit), self._rpm_tokens + delta_rpm * (self.rpm_limit / 60.0))
        self._last_rpm_update = now

        # Replenish TPM bucket (rate = limit / 60.0 per sec)
        delta_tpm = now - self._last_tpm_update
        self._tpm_tokens = min(float(self.tpm_limit), self._tpm_tokens + delta_tpm * (self.tpm_limit / 60.0))
        self._last_tpm_update = now

    async def acquire(self, estimated_tokens: int) -> AdmissionResult:
        """Attempt to acquire permission to make an LLM call.
        
        Waits up to max_wait_s. If blocked or daily budget is gone, returns appropriate mode.
        """
        start = time.monotonic()

        if self.is_daily_exhausted():
            return AdmissionResult(
                admitted=False,
                reason="daily_exhausted",
                queue_wait_ms=0.0,
                mode="daily_fallback",
            )

        deadline = start + self.max_wait_s

        while time.monotonic() < deadline:
            async with self._lock:
                now = time.monotonic()
                self._replenish(now)

                if self._rpm_tokens >= 1.0 and self._tpm_tokens >= estimated_tokens:
                    self._rpm_tokens -= 1.0
                    self._tpm_tokens -= estimated_tokens
                    self.daily_requests_used += 1
                    self.daily_tokens_used += estimated_tokens

                    wait_ms = (time.monotonic() - start) * 1000.0
                    return AdmissionResult(
                        admitted=True,
                        reason="ok",
                        queue_wait_ms=round(wait_ms, 2),
                        mode="llm",
                    )

            # Wait briefly before checking bucket replenishment
            await asyncio.sleep(0.1)

        wait_ms = (time.monotonic() - start) * 1000.0
        return AdmissionResult(
            admitted=False,
            reason="rpm_tpm_saturated",
            queue_wait_ms=round(wait_ms, 2),
            mode="extractive",
        )
