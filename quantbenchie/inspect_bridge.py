"""Optional Inspect AI bridge for running QuantBenchie probes as Inspect tasks.

Install the extra with ``pip install -e '.[inspect]'`` and run, for example:

    inspect eval quantbenchie/inspect_bridge.py@smoke --model ollama/qwen

The bridge deliberately reuses QuantBenchie's case definitions and scorers. It
does not import Inspect at normal package import time, so the base install stays
lightweight.
"""

from __future__ import annotations

from .tasks import resolve_tasks, score_case


def build_task(suite: str = "smoke"):
    try:
        from inspect_ai import Task
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import Score, scorer
        from inspect_ai.solver import generate
    except ImportError as exc:
        raise RuntimeError("Inspect bridge requires: pip install quantbenchie[inspect]") from exc

    cases = resolve_tasks((suite,))
    cases_by_id = {case.id: case for case in cases}

    @scorer
    def quantbenchie_scorer():
        async def score(state, target):
            case = cases_by_id[str(state.sample_id)]
            result = score_case(case, state.output.completion)
            return Score(value=result.score, answer=result.output, explanation=result.reason,
                         metadata={"quantbenchie_category": result.category, "quantbenchie_passed": result.passed})

        return score

    return Task(
        dataset=[Sample(input=case.prompt, target=str(case.expected or ""), id=case.id,
                        metadata={"category": case.category, "evaluator": case.evaluator}) for case in cases],
        solver=generate(),
        scorer=quantbenchie_scorer(),
        name=f"quantbenchie_{suite}",
        version="0.1",
    )


try:
    from inspect_ai import task
except ImportError:
    task = None

if task is not None:
    @task
    def smoke():
        return build_task("smoke")

    @task
    def integrity():
        return build_task("integrity")
