from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
from typing import Any

from .adapters import create_adapter
from .config import RunConfig
from .factorial import interaction_effects
from .metrics import kl_divergence, summarize_distribution_rows, top_token, top_k_agreement
from .models import ModelAdapter, ModelSpec
from .tasks import CaseScore, resolve_tasks, score_case


def _loss(scores: list[CaseScore]) -> float:
    return 1.0 - (sum(score.score for score in scores) / max(len(scores), 1))


def evaluate_model(model: ModelSpec, cases: tuple, config: RunConfig, adapter: ModelAdapter | None = None,
                   reference_adapter: ModelAdapter | None = None) -> dict[str, Any]:
    adapter = adapter or create_adapter(model)
    scores: list[CaseScore] = []
    for _ in range(max(1, config.repeats)):
        for case in cases:
            generation = adapter.generate(case.prompt, temperature=config.temperature, max_tokens=config.max_tokens, system=case.system)
            scores.append(score_case(case, generation.text))
    by_category: dict[str, list[float]] = {}
    for score in scores:
        by_category.setdefault(score.category, []).append(score.score)
    category_scores = {category: sum(values) / len(values) for category, values in by_category.items()}
    dist_rows: list[dict[str, float | int | bool]] = []
    if reference_adapter is not None:
        for index, case in enumerate(cases):
            reference_dist = reference_adapter.distribution(case.prompt)
            candidate_dist = adapter.distribution(case.prompt)
            if reference_dist and candidate_dist:
                dist_rows.append({"position": index, "kl": kl_divergence(reference_dist.probabilities, candidate_dist.probabilities),
                                  "top1_agreement": float(top_token(reference_dist.probabilities) == top_token(candidate_dist.probabilities)),
                                  "top_k_agreement": top_k_agreement(reference_dist.probabilities, candidate_dist.probabilities)})
    return {
        "model": model.name,
        "quantization": model.quantization,
        "modification": model.modification,
        "provider": model.provider,
        "samples": [asdict(score) for score in scores],
        "overall_score": 1.0 - _loss(scores),
        "loss": _loss(scores),
        "category_scores": category_scores,
        "distribution": summarize_distribution_rows(dist_rows),
    }


def compare_distribution(reference: ModelSpec, candidate: ModelSpec, prompts: list[str]) -> dict[str, Any]:
    ref_adapter = create_adapter(reference)
    cmp_adapter = create_adapter(candidate)
    rows = []
    for index, prompt in enumerate(prompts):
        ref = ref_adapter.distribution(prompt)
        cmp = cmp_adapter.distribution(prompt)
        if not ref or not cmp:
            continue
        rows.append({"position": index, "kl": kl_divergence(ref.probabilities, cmp.probabilities),
                     "top1_agreement": float(top_token(ref.probabilities) == top_token(cmp.probabilities)),
                     "top_k_agreement": top_k_agreement(ref.probabilities, cmp.probabilities)})
    return {"reference": reference.name, "candidate": candidate.name, "rows": rows,
            "summary": summarize_distribution_rows(rows)}


def run(config: RunConfig) -> dict[str, Any]:
    random.seed(config.seed)
    cases = resolve_tasks(config.tasks)
    reference_adapter = create_adapter(config.reference)
    reference = evaluate_model(config.reference, cases, config, adapter=reference_adapter)
    candidates = [evaluate_model(model, cases, config, reference_adapter=reference_adapter) for model in config.candidates]
    all_rows = [reference, *candidates]
    effects = interaction_effects(all_rows, config.reference.name)
    verdicts = []
    for result in candidates:
        regression = result["loss"] - reference["loss"]
        result["regression_vs_reference"] = regression
        result["verdict"] = "FAIL" if regression >= config.fail_threshold else "WARN" if regression >= config.warn_threshold else "PASS"
        verdicts.append(result["verdict"])
    output = {
        "schema_version": "0.1",
        "run": {"name": config.name, "started_at": datetime.now(timezone.utc).isoformat(),
                "seed": config.seed, "platform": platform.platform(), "config": config.to_dict()},
        "reference": reference,
        "candidates": candidates,
        "factorial_interactions": [asdict(effect) for effect in effects],
        "summary": {"pass": verdicts.count("PASS"), "warn": verdicts.count("WARN"), "fail": verdicts.count("FAIL")},
    }
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (output_path / "report.md").write_text(render_report(output), encoding="utf-8")
    return output


def render_report(result: dict[str, Any]) -> str:
    lines = [f"# QuantBenchie integrity report: {result['run']['name']}", "", f"Schema: `{result['schema_version']}`", "",
             "## Model comparison", "", "| Model | Quantization | Modification | Score | Regression | Verdict |", "|---|---|---|---:|---:|---|"]
    reference = result["reference"]
    lines.append(f"| {reference['model']} | {reference['quantization']} | {reference['modification']} | {reference['overall_score']:.3f} | — | CONTROL |")
    for candidate in result["candidates"]:
        lines.append(f"| {candidate['model']} | {candidate['quantization']} | {candidate['modification']} | {candidate['overall_score']:.3f} | {candidate['regression_vs_reference']:+.3f} | **{candidate['verdict']}** |")
    lines.extend(["", "## Category scores", "", "| Model | " + " | ".join(sorted({category for model in [reference, *result['candidates']] for category in model['category_scores']})) + " |", "|---|" + "---:|" * len({category for model in [reference, *result['candidates']] for category in model['category_scores']})])
    categories = sorted({category for model in [reference, *result["candidates"]] for category in model["category_scores"]})
    for model in [reference, *result["candidates"]]:
        lines.append("| " + model["model"] + " | " + " | ".join(f"{model['category_scores'].get(category, 0.0):.3f}" for category in categories) + " |")
    if result["factorial_interactions"]:
        lines.extend(["", "## Factorial interaction estimates", "", "| Quantization | Modification | Observed loss | Additive expectation | Interaction | Interpretation |", "|---|---|---:|---:|---:|---|"])
        for effect in result["factorial_interactions"]:
            expected = "—" if effect["expected_additive_delta"] is None else f"{effect['expected_additive_delta']:.3f}"
            interaction = "—" if effect["interaction_delta"] is None else f"{effect['interaction_delta']:+.3f}"
            lines.append(f"| {effect['quantization']} | {effect['modification']} | {effect['observed_delta']:.3f} | {expected} | {interaction} | {effect['interpretation']} |")
    lines.extend(["", "## Interpretation", "", "Behavioral scores are task-probe results, not a general intelligence score. A FAIL means the candidate regressed beyond the configured threshold versus the paired control; inspect individual samples before drawing causal conclusions.", ""])
    return "\n".join(lines)
