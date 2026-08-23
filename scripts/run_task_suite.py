from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from quantbenchie.adapters import create_adapter
from quantbenchie.models import ModelSpec
from semantic_compare import classify, make_tasks


def run_model(url: str, name: str, output: str) -> dict:
    adapter = create_adapter(ModelSpec(name=name, provider="openai-compatible", path=url,
                                       metadata={"served_model": name, "timeout": 180}))
    records = []
    for task in make_tasks():
        started = time.perf_counter()
        generation = adapter.generate(task.prompt + "\n/no_think", temperature=0.0, max_tokens=256)
        correct, answer_class = classify(task, generation.text)
        records.append({
            "task_id": task.task_id,
            "category": task.category,
            "prompt": task.prompt,
            "expected": task.expected,
            "answer": generation.text,
            "correct": correct,
            "class": answer_class,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "finish_reason": generation.finish_reason,
        })
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({"model": name, "records": records}, indent=2), encoding="utf-8")
        print(json.dumps({"task": task.task_id, "correct": correct, "class": answer_class}), flush=True)
    return {"model": name, "records": records}


def compare(reference_path: str, candidate_path: str, output: str) -> dict:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    ref_by_id = {item["task_id"]: item for item in reference["records"]}
    records = []
    for item in candidate["records"]:
        ref = ref_by_id[item["task_id"]]
        records.append({
            "task_id": item["task_id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "expected": item["expected"],
            "official": {"answer": ref["answer"], "correct": ref["correct"], "class": ref["class"]},
            "heretic_f16": {"answer": item["answer"], "correct": item["correct"], "class": item["class"]},
            "official_agreement": ref["class"] == item["class"],
        })
    result = {"official_model": reference["model"], "candidate_model": candidate["model"], "records": records}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "compare"))
    parser.add_argument("--url")
    parser.add_argument("--name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    if args.mode == "run":
        if not args.url or not args.name:
            parser.error("run requires --url and --name")
        run_model(args.url, args.name, args.output)
    else:
        if not args.reference or not args.candidate:
            parser.error("compare requires --reference and --candidate")
        compare(args.reference, args.candidate, args.output)
