from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """A model plus explicit factors used for factorial analysis."""

    name: str
    provider: str = "mock"
    path: str | None = None
    quantization: str = "bf16"
    modification: str = "original"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=str(value["name"]),
            provider=str(value.get("provider", "mock")),
            path=value.get("path"),
            quantization=str(value.get("quantization", "bf16")),
            modification=str(value.get("modification", "original")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Distribution:
    """A next-token distribution represented by token -> probability."""

    probabilities: dict[str, float]


class ModelAdapter:
    """Small adapter protocol shared by behavioral and distribution evaluators."""

    def generate(self, prompt: str, *, temperature: float = 0.0,
                 max_tokens: int = 256, system: str | None = None) -> Generation:
        raise NotImplementedError

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
             max_tokens: int = 256) -> Generation:
        """Generate from a complete conversation, preserving assistant history."""
        system = next((message["content"] for message in messages if message["role"] == "system"), None)
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        return self.generate(user_messages[-1] if user_messages else "", temperature=temperature,
                             max_tokens=max_tokens, system=system)

    def distribution(self, prompt: str) -> Distribution | None:
        return None
