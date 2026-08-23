from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ModelSpec


@dataclass(frozen=True)
class RunConfig:
    name: str
    reference: ModelSpec
    candidates: tuple[ModelSpec, ...]
    tasks: tuple[str, ...] = ("smoke",)
    sessions: tuple[str, ...] = ()
    output_dir: str = "results"
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 256
    context_size: int = 4096
    repeats: int = 1
    baseline_repeats: int = 3
    session_repeats: int = 1
    session_scoring: str = "reference_judge"
    fail_threshold: float = 0.10
    warn_threshold: float = 0.03
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunConfig":
        reference = ModelSpec.from_dict(value["reference"])
        candidates = tuple(ModelSpec.from_dict(item) for item in value.get("candidates", []))
        if not candidates:
            raise ValueError("config must define at least one candidate")
        return cls(
            name=str(value.get("name", "quantbenchie-run")),
            reference=reference,
            candidates=candidates,
            tasks=tuple(value.get("tasks", ["smoke"])),
            sessions=tuple(value.get("sessions", [])),
            output_dir=str(value.get("output_dir", "results")),
            seed=int(value.get("seed", 0)),
            temperature=float(value.get("temperature", 0.0)),
            max_tokens=int(value.get("max_tokens", 256)),
            context_size=int(value.get("context_size", 4096)),
            repeats=int(value.get("repeats", 1)),
            baseline_repeats=int(value.get("baseline_repeats", 3)),
            session_repeats=int(value.get("session_repeats", 1)),
            session_scoring=str(value.get("session_scoring", "reference_judge")),
            fail_threshold=float(value.get("fail_threshold", 0.10)),
            warn_threshold=float(value.get("warn_threshold", 0.03)),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reference": self.reference.to_dict(),
            "candidates": [model.to_dict() for model in self.candidates],
            "tasks": list(self.tasks),
            "sessions": list(self.sessions),
            "output_dir": self.output_dir,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_size": self.context_size,
            "repeats": self.repeats,
            "baseline_repeats": self.baseline_repeats,
            "session_repeats": self.session_repeats,
            "session_scoring": self.session_scoring,
            "fail_threshold": self.fail_threshold,
            "warn_threshold": self.warn_threshold,
            "metadata": self.metadata,
        }


def load_config(path: str | Path) -> RunConfig:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config needs optional dependency: pip install quantbenchie[config]") from exc
        value = yaml.safe_load(raw)
    else:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return RunConfig.from_dict(value)
