from dataclasses import dataclass, field
from typing import Iterator, Optional, Dict, Any


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    first_token_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    status: str  # "ok", "rate_limited", "timeout", "error"
    retries: int = 0
    error_message: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


class BaseLLMClient:
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        raise NotImplementedError

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        raise NotImplementedError

