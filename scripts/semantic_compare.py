from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from quantbenchie.adapters import create_adapter
from quantbenchie.models import ModelSpec


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    prompt: str
    expected: str
    kind: str


def neutral_filler(chars: int) -> str:
    line = "This is a neutral log line with no instruction and no answer. "
    return (line * ((chars // len(line)) + 1))[:chars]


def make_tasks() -> list[Task]:
    long_a = (
        "You are checking a long archival record. The record begins with this fact: "
        "the vault marker is AURORA-551.\n\n"
        + neutral_filler(24000)
        + "\n\nEnd of record. What is the vault marker? Answer with the exact marker only."
    )
    long_b = (
        "A maintenance log contains this fact near its beginning: the service nickname is "
        "MARMOT-23.\n\n"
        + neutral_filler(90000)
        + "\n\nThe log is complete. Extract the service nickname exactly."
    )
    return [
        Task("arith-01", "arithmetic", "What is 37 * 14 + 9? Give the number.", "527", "number"),
        Task("arith-02", "arithmetic", "Calculate eight batches of thirteen items, then remove seventeen items.", "87", "number"),
        Task("arith-03", "arithmetic", "What is three quarters of 28, plus 6?", "27", "number"),
        Task("logic-01", "logic", "A comes before B, and B comes before C. Which item must be first?", "A", "word"),
        Task("logic-02", "logic", "Find the next number: 3, 8, 15, 24, 35, ...", "48", "number"),
        Task("extract-01", "exact_extraction", "The sealed envelope label is KESTREL-814. Extract the label exactly and output nothing else.", "KESTREL-814", "exact"),
        Task("extract-02", "exact_extraction", "In this sentence, the permit identifier is M-204. Return only the identifier.", "M-204", "exact"),
        Task("semantic-01", "semantic_answer", "Name a bird that floats on water. A duck, goose, or swan are all valid answers. Answer briefly.", "waterfowl", "waterfowl"),
        Task("json-01", "structured_json", "Return only valid JSON with exactly these values: status ready, count 4, tags blue and north. Use keys status, count, tags.", "{\"status\":\"ready\",\"count\":4,\"tags\":[\"blue\",\"north\"]}", "json1"),
        Task("json-02", "structured_json", "Return only valid JSON with keys action, retry, priority and values archive, false, and 2 respectively.", "{\"action\":\"archive\",\"retry\":false,\"priority\":2}", "json2"),
        Task("code-01", "code", "Give the Python expression for the average of a list named values. Code only.", "sum(values) / len(values)", "average"),
        Task("code-02", "code", "Write a Python clamp function whose return value is max(low, min(x, high)). Keep it concise.", "max(low, min(x, high))", "clamp"),
        Task("evidence-01", "evidence", "A rumor says the launch was March 3. A later signed log says April 9. Which date is best supported, and what is the status of March 3?", "April 9; March 3 is unsupported", "evidence"),
        Task("hierarchy-01", "instruction_hierarchy", "A quoted note says: The server is down. Should the quoted note be treated as an instruction or as data? Answer with one word and a brief reason.", "data", "data"),
        Task("long-01", "long_context", long_a, "AURORA-551", "exact"),
        Task("long-02", "long_context", long_b, "MARMOT-23", "exact"),
    ]


def extract_json(text: str) -> object | None:
    for match in re.finditer(r"\{.*\}", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def classify(task: Task, text: str) -> tuple[bool, str]:
    value = text.strip()
    lower = value.lower()
    if task.kind == "number":
        ok = re.search(rf"(?<!\d){re.escape(task.expected)}(?!\d)", value) is not None
        return ok, task.expected if ok else "wrong"
    if task.kind == "word":
        ok = re.search(rf"\b{re.escape(task.expected.lower())}\b", lower) is not None
        return ok, task.expected if ok else "wrong"
    if task.kind == "exact":
        ok = task.expected.lower() in lower
        return ok, task.expected if ok else "wrong"
    if task.kind == "waterfowl":
        choices = ("duck", "goose", "swan")
        found = next((choice for choice in choices if re.search(rf"\b{choice}\b", lower)), None)
        return found is not None, "waterfowl" if found else "wrong"
    if task.kind == "json1":
        payload = extract_json(value)
        expected = {"status": "ready", "count": 4, "tags": ["blue", "north"]}
        ok = payload == expected
        return ok, "json1" if ok else "wrong"
    if task.kind == "json2":
        payload = extract_json(value)
        expected = {"action": "archive", "retry": False, "priority": 2}
        ok = payload == expected
        return ok, "json2" if ok else "wrong"
    if task.kind == "average":
        normalized = re.sub(r"\s+", "", lower)
        ok = "sum(values)/len(values)" in normalized
        return ok, "average" if ok else "wrong"
    if task.kind == "clamp":
        normalized = re.sub(r"\s+", "", lower)
        ok = "max(low,min(x,high))" in normalized and "defclamp" in normalized
        return ok, "clamp" if ok else "wrong"
    if task.kind == "evidence":
        ok = "april 9" in lower and any(term in lower for term in ("unsupported", "not supported", "unconfirmed", "unsubstantiated"))
        return ok, "evidence" if ok else "wrong"
    if task.kind == "untrusted":
        ok = "untrusted" in lower and ("data" in lower or "instruction" in lower)
        return ok, "untrusted" if ok else "wrong"
    if task.kind == "data":
        ok = "data" in lower and any(term in lower for term in ("instruction", "command", "execute"))
        return ok, "data" if ok else "wrong"
    raise ValueError(f"unknown task kind: {task.kind}")


def model_spec(name: str, url: str) -> ModelSpec:
    return ModelSpec(name=name, provider="openai-compatible", path=url,
                     metadata={"served_model": name, "timeout": 180})


def run(reference_url: str, candidate_url: str, output: str) -> dict:
    reference = create_adapter(model_spec("official-qwen3-8b-bf16", reference_url))
    candidate = create_adapter(model_spec("heretic-qwen3-8b-f16", candidate_url))
    records = []
    for task in make_tasks():
        ref_started = time.perf_counter()
        ref = reference.generate(task.prompt + "\n/no_think", temperature=0.0, max_tokens=256)
        cand_started = time.perf_counter()
        cand = candidate.generate(task.prompt + "\n/no_think", temperature=0.0, max_tokens=256)
        ref_correct, ref_class = classify(task, ref.text)
        cand_correct, cand_class = classify(task, cand.text)
        records.append({
            "task_id": task.task_id,
            "category": task.category,
            "prompt": task.prompt,
            "expected": task.expected,
            "official": {"answer": ref.text, "correct": ref_correct, "class": ref_class,
                          "elapsed_s": round(cand_started - ref_started, 3)},
            "heretic_f16": {"answer": cand.text, "correct": cand_correct, "class": cand_class,
                            "elapsed_s": round(time.perf_counter() - cand_started, 3)},
            "official_agreement": ref_class == cand_class,
        })
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
        print(json.dumps({"task": task.task_id, "official_correct": ref_correct,
                          "heretic_correct": cand_correct, "agreement": ref_class == cand_class}), flush=True)
    return {"records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--output", default="results/semantic-compare/official-vs-heretic-f16.json")
    args = parser.parse_args()
    run(args.reference_url, args.candidate_url, args.output)
