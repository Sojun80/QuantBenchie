from __future__ import annotations

import math
from statistics import mean
from typing import Iterable


def kl_divergence(reference: dict[str, float], candidate: dict[str, float], epsilon: float = 1e-12) -> float:
    """Forward KL, KL(reference || candidate), in nats."""
    keys = set(reference) | set(candidate)
    return sum(
        p * math.log(max(p, epsilon) / max(candidate.get(key, 0.0), epsilon))
        for key in keys
        if (p := max(reference.get(key, 0.0), 0.0)) > 0
    )


def top_token(probabilities: dict[str, float]) -> str | None:
    return max(probabilities, key=probabilities.get) if probabilities else None


def top_k_agreement(reference: dict[str, float], candidate: dict[str, float], k: int = 5) -> float:
    ref = {token for token, _ in sorted(reference.items(), key=lambda x: x[1], reverse=True)[:k]}
    cmp = {token for token, _ in sorted(candidate.items(), key=lambda x: x[1], reverse=True)[:k]}
    return len(ref & cmp) / max(k, 1)


def summarize_distribution_rows(rows: Iterable[dict[str, float | int | bool]]) -> dict[str, float | int]:
    rows = list(rows)
    if not rows:
        return {"positions": 0, "mean_kl": 0.0, "p95_kl": 0.0, "top1_agreement": 0.0}
    kls = sorted(float(row["kl"]) for row in rows)
    p95 = kls[min(len(kls) - 1, math.ceil(len(kls) * 0.95) - 1)]
    return {
        "positions": len(rows),
        "mean_kl": mean(kls),
        "p95_kl": p95,
        "top1_agreement": mean(float(row["top1_agreement"]) for row in rows),
    }
