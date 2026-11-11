import random
import threading
import time
from typing import Dict, Iterator, List, Optional
from collections import deque

from src.models.base import BaseLLMClient, LLMResponse


class StandInModelService(BaseLLMClient):
    """Calibrated stand-in model service simulating real-world LLM inference dynamics.
    
    Models:
    - Base latency & lognormal tail spikes (p50: ~220ms, p95: ~1400ms, p99: ~2800ms)
    - Progressive slowdown under concurrent in-flight load (queueing model)
    - Sliding-window rate limit enforcement (RPM & TPM)
    - Daily request and token quotas
    - Transient server failures triggering retries
    - Token streaming generation
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        tokens_per_minute: int = 14400,
        daily_request_limit: int = 14400,
        daily_token_limit: int = 500000,
        base_ttft_ms: float = 180.0,
        tokens_per_second: float = 240.0,
        error_rate: float = 0.03,
        slowdown_factor_per_in_flight: float = 0.15,
        max_in_flight: int = 25,
        sim_speedup: float = 0.02,
    ):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.daily_request_limit = daily_request_limit
        self.daily_token_limit = daily_token_limit

        self.base_ttft_ms = base_ttft_ms
        self.tokens_per_second = tokens_per_second
        self.error_rate = error_rate
        self.slowdown_factor = slowdown_factor_per_in_flight
        self.max_in_flight = max_in_flight
        self.sim_speedup = sim_speedup

        self.in_flight_count = 0
        self.peak_queue_length = 0
        self.daily_requests_used = 0
        self.daily_tokens_used = 0

        # Sliding window timestamps
        self.request_timestamps: deque = deque()
        self.token_log: deque = deque()  # (timestamp, token_count)

        self.lock = threading.Lock()

    def set_limits(self, rpm: Optional[int] = None, tpm: Optional[int] = None, daily_req: Optional[int] = None):
        """Dynamic limit reconfiguration (e.g. For halved per-minute limits)."""
        with self.lock:
            if rpm is not None:
                self.rpm_limit = rpm
            if tpm is not None:
                self.tpm_limit = tpm
            if daily_req is not None:
                self.daily_request_limit = daily_req

    def exhaust_daily_budget(self):
        """Immediately trigger daily budget exhaustion."""
        with self.lock:
            self.daily_requests_used = self.daily_request_limit + 1
            self.daily_tokens_used = self.daily_token_limit + 1

    def _clean_sliding_windows(self, now: float):
        """Evict events older than 60 seconds."""
        cutoff = now - 60.0
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        while self.token_log and self.token_log[0][0] < cutoff:
            self.token_log.popleft()

    def _check_rate_limits(self, estimated_tokens: int) -> Optional[str]:
        """Check rate limits; returns error reason if exceeded, else None."""
        now = time.time()
        self._clean_sliding_windows(now)

        if self.daily_requests_used >= self.daily_request_limit:
            return "daily_request_limit_exhausted"

        if self.daily_tokens_used + estimated_tokens > self.daily_token_limit:
            return "daily_token_limit_exhausted"

        if len(self.request_timestamps) >= self.rpm_limit:
            return "rate_limited_rpm"

        current_tokens_in_window = sum(count for _, count in self.token_log)
        if current_tokens_in_window + estimated_tokens > self.tpm_limit:
            return "rate_limited_tpm"

        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        mock_output: Optional[str] = None,
    ) -> LLMResponse:
        """Execute simulated request with concurrency, latency, rate limits, and retries."""
        prompt_words = len(prompt.split())
        tokens_in = int(prompt_words * 1.3) + 20
        estimated_tokens_out = int(len((mock_output or prompt).split()) * 0.8) + 40
        total_estimated = tokens_in + estimated_tokens_out

        retries = 0
        overall_start = time.perf_counter()

        while retries <= max_retries:
            with self.lock:
                limit_breach = self._check_rate_limits(total_estimated)
                if limit_breach:
                    # If daily budget is exhausted, retrying will not help
                    if "daily" in limit_breach:
                        total_ms = (time.perf_counter() - overall_start) * 1000.0
                        return LLMResponse(
                            content="",
                            first_token_ms=0.0,
                            total_ms=round(total_ms, 2),
                            tokens_in=tokens_in,
                            tokens_out=0,
                            status="rate_limited",
                            retries=retries,
                            error_message=f"HTTP 429: {limit_breach}",
                        )
                    # Per-minute rate limit: retry if allowed
                    if retries < max_retries:
                        retries += 1
                        if self.sim_speedup > 0:
                            time.sleep(0.05 * retries)
                        continue
                    else:
                        sim_ms = 4500.0 + (retries * 1000.0)
                        return LLMResponse(
                            content="",
                            first_token_ms=0.0,
                            total_ms=round(sim_ms, 2),
                            tokens_in=tokens_in,
                            tokens_out=0,
                            status="rate_limited",
                            retries=retries,
                            error_message="HTTP 429: Rate limit exceeded (RPM/TPM)",
                        )

                # Acquire in-flight slot
                self.in_flight_count += 1
                if self.in_flight_count > self.peak_queue_length:
                    self.peak_queue_length = self.in_flight_count
                in_flight_current = self.in_flight_count

            # Simulate queue & concurrency slowdown
            load_factor = 1.0 + (in_flight_current * self.slowdown_factor)

            # Lognormal latency spike distribution
            r = random.random()
            if r > 0.97:
                spike_multiplier = random.uniform(3.5, 6.0)
            elif r > 0.85:
                spike_multiplier = random.uniform(1.8, 2.8)
            else:
                spike_multiplier = random.uniform(0.9, 1.25)

            # Simulated transient 503 / socket drop
            if random.random() < self.error_rate and retries < max_retries:
                with self.lock:
                    self.in_flight_count -= 1
                retries += 1
                if self.sim_speedup > 0:
                    time.sleep(0.02 * (2 ** retries))
                continue

            sim_ttft_ms = self.base_ttft_ms * load_factor * spike_multiplier
            tokens_out = estimated_tokens_out
            sim_gen_ms = (tokens_out / self.tokens_per_second) * 1000.0 * load_factor
            retry_penalty_ms = retries * 650.0

            first_token_ms = sim_ttft_ms + retry_penalty_ms
            total_ms = sim_ttft_ms + sim_gen_ms + retry_penalty_ms

            if self.sim_speedup > 0:
                time.sleep((first_token_ms / 1000.0 + sim_gen_ms / 1000.0) * self.sim_speedup)

            # Commit usage
            with self.lock:
                now = time.time()
                self.request_timestamps.append(now)
                self.token_log.append((now, tokens_in + tokens_out))
                self.daily_requests_used += 1
                self.daily_tokens_used += (tokens_in + tokens_out)
                self.in_flight_count -= 1

            generated_text = mock_output or (
                "Based on the provided research papers, the architecture utilizes multi-head attention "
                "with dimension d_k=64 and 8 parallel heads, achieving 28.4 BLEU on English-to-German translation."
            )

            return LLMResponse(
                content=generated_text,
                first_token_ms=round(first_token_ms, 2),
                total_ms=round(total_ms, 2),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                status="ok",
                retries=retries,
            )

        sim_ms = 3500.0 + (retries * 800.0)
        return LLMResponse(
            content="",
            first_token_ms=0.0,
            total_ms=round(sim_ms, 2),
            tokens_in=tokens_in,
            tokens_out=0,
            status="error",
            retries=retries,
            error_message="Max retries exhausted",
        )
