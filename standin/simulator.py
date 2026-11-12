"""Stand-in LLM simulator calibrated to reproduce real Groq API dynamics.

Simulates:
- Lognormal TTFT distribution (mu, sigma)
- Non-linear progressive slowdown under concurrency load:
    TTFT_effective = base_ttft * (1 + alpha * max(0, in_flight - knee)^beta)
- Sliding-window token buckets for RPM and TPM matching real limits
- Daily quota tracking and exhaustion
- Fault injection (drops, 500s, 429s, sudden tail slowdowns)
- Deterministic random seed for reproducibility
"""

import asyncio
import math
import random
import time
from typing import AsyncGenerator, Dict, Optional, Tuple
import yaml


class StandInSimulator:
    def __init__(self, config_path: str = "standin/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        ttft_cfg = self.config.get("ttft_distribution", {})
        self.mu: float = float(ttft_cfg.get("mu", 5.25))
        self.sigma: float = float(ttft_cfg.get("sigma", 0.35))

        gen_cfg = self.config.get("generation", {})
        self.tokens_per_second: float = float(gen_cfg.get("tokens_per_second", 240.0))

        rate_cfg = self.config.get("rate_limits", {})
        self.rpm_limit: int = int(rate_cfg.get("requests_per_minute", 30))
        self.tpm_limit: int = int(rate_cfg.get("tokens_per_minute", 14400))
        self.daily_requests_limit: int = int(rate_cfg.get("daily_requests_limit", 14400))
        self.daily_tokens_limit: int = int(rate_cfg.get("daily_tokens_limit", 500000))

        load_cfg = self.config.get("load_curve", {})
        self.knee: int = int(load_cfg.get("knee", 4))
        self.alpha: float = float(load_cfg.get("alpha", 0.25))
        self.beta: float = float(load_cfg.get("beta", 1.5))

        fault_cfg = self.config.get("fault_injection", {})
        self.p_drop: float = float(fault_cfg.get("p_drop", 0.0))
        self.p_500: float = float(fault_cfg.get("p_500", 0.01))
        self.p_slowdown: float = float(fault_cfg.get("p_slowdown", 0.05))
        self.p_rate_limit: float = float(fault_cfg.get("p_rate_limit", 0.0))

        sim_cfg = self.config.get("simulation", {})
        self.time_dilation: float = float(sim_cfg.get("time_dilation", 1.0))
        seed: int = int(sim_cfg.get("seed", 42))
        self.rng = random.Random(seed)

        # Operational state
        self.in_flight: int = 0
        self.daily_requests_used: int = 0
        self.daily_tokens_used: int = 0

        # Sliding window buckets: lists of (timestamp, value)
        self.request_window: list = []
        self.token_window: list = []
        self.lock = asyncio.Lock()

    def set_limits(self, rpm: Optional[int] = None, tpm: Optional[int] = None):
        """Allows test scenarios to dynamically scale rate limits (e.g. rpm_halved)."""
        if rpm is not None:
            self.rpm_limit = rpm
        if tpm is not None:
            self.tpm_limit = tpm

    def exhaust_daily_budget(self):
        """Triggers immediate daily budget exhaustion."""
        self.daily_tokens_used = self.daily_tokens_limit + 1
        self.daily_requests_used = self.daily_requests_limit + 1

    def _clean_windows(self, now: float):
        cutoff = now - 60.0
        self.request_window = [t for t in self.request_window if t >= cutoff]
        self.token_window = [(t, tok) for t, tok in self.token_window if t >= cutoff]

    async def admit_request(self, estimated_tokens: int) -> Tuple[bool, Optional[str], Optional[float]]:
        """Checks limits before dispatching request. Returns (admitted, error_type, retry_after)."""
        async with self.lock:
            now = time.time()
            self._clean_windows(now)

            if self.daily_requests_used >= self.daily_requests_limit or self.daily_tokens_used >= self.daily_tokens_limit:
                return False, "daily_quota_exceeded", 86400.0

            if len(self.request_window) >= self.rpm_limit:
                earliest = self.request_window[0]
                retry_after = max(0.5, 60.0 - (now - earliest))
                return False, "rate_limit_rpm", retry_after

            current_tpm = sum(tok for _, tok in self.token_window)
            if current_tpm + estimated_tokens > self.tpm_limit:
                earliest = self.token_window[0][0] if self.token_window else now
                retry_after = max(0.5, 60.0 - (now - earliest))
                return False, "rate_limit_tpm", retry_after

            # Check fault injection for synthetic 429
            if self.rng.random() < self.p_rate_limit:
                return False, "injected_rate_limit", 2.0

            # Admit request
            self.request_window.append(now)
            self.token_window.append((now, estimated_tokens))
            self.daily_requests_used += 1
            self.daily_tokens_used += estimated_tokens
            self.in_flight += 1
            return True, None, None

    def release_request(self):
        self.in_flight = max(0, self.in_flight - 1)

    def sample_ttft(self) -> float:
        """Sample TTFT in milliseconds based on lognormal distribution and load curve."""
        # Base lognormal sample in ms
        base_ttft_ms = math.exp(self.rng.gauss(self.mu, self.sigma))

        # Apply progressive slowdown under concurrency
        excess_in_flight = max(0, self.in_flight - self.knee)
        slowdown_multiplier = 1.0 + self.alpha * (excess_in_flight ** self.beta)

        # Fault injection for sudden tail slowdown spike
        if self.rng.random() < self.p_slowdown:
            slowdown_multiplier *= 3.0

        return base_ttft_ms * slowdown_multiplier
