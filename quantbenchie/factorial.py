from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FactorialEffect:
    quantization: str
    modification: str
    observed_delta: float
    expected_additive_delta: float | None
    interaction_delta: float | None
    interpretation: str


def interaction_effects(rows: list[dict[str, Any]], reference_name: str) -> list[FactorialEffect]:
    """Estimate interaction from aggregate behavioral loss rows.

    Rows must contain model, quantization, modification, and loss (higher is worse).
    A combined cell is compared with matching one-factor cells when available.
    """
    by_cell = {(row["quantization"], row["modification"]): float(row["loss"]) for row in rows}
    reference = next((float(row["loss"]) for row in rows if row["model"] == reference_name), 0.0)
    original_q = {(q, "original"): value for (q, m), value in by_cell.items() if m == "original"}
    bf16_mod = {("bf16", m): value for (q, m), value in by_cell.items() if q == "bf16"}
    results: list[FactorialEffect] = []
    for (quant, modification), observed in by_cell.items():
        if quant == "bf16" or modification == "original":
            continue
        q_only = original_q.get((quant, "original"))
        mod_only = bf16_mod.get(("bf16", modification))
        if q_only is None or mod_only is None:
            expected = interaction = None
            interpretation = "insufficient matched cells for interaction estimate"
        else:
            q_delta = q_only - reference
            mod_delta = mod_only - reference
            expected = reference + q_delta + mod_delta
            interaction = observed - expected
            interpretation = "interaction suggests compounded regression" if interaction > 0 else "no positive interaction detected"
        results.append(FactorialEffect(quant, modification, observed, expected, interaction, interpretation))
    return results
