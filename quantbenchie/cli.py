from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from .config import load_config
from .answer_sheet import generate_answer_sheet, load_answer_sheet, run_answer_sheet_experiment, save_answer_sheet, render_session_report
from .runner import render_report, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantbenchie", description="Compare local LLM variants for integrity regressions.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="write a starter experiment config")
    init.add_argument("path", nargs="?", default="quantbenchie.json")
    run_parser = sub.add_parser("run", help="execute behavioral probes and write JSON + Markdown")
    run_parser.add_argument("config")
    report = sub.add_parser("report", help="render Markdown from a saved results.json")
    report.add_argument("results")
    validate = sub.add_parser("validate", help="validate a config without loading a model")
    validate.add_argument("config")
    baseline = sub.add_parser("baseline", help="generate repeated reference-model transcripts as an answer sheet")
    baseline.add_argument("config")
    baseline.add_argument("--output", help="answer-sheet path; defaults to <output_dir>/answer-sheet.json")
    compare = sub.add_parser("compare", help="replay long sessions and compare candidates to a reference answer sheet")
    compare.add_argument("config")
    compare.add_argument("--answer-sheet", help="existing answer-sheet path; defaults to <output_dir>/answer-sheet.json")
    return parser


def starter_config() -> dict:
    return {
        "name": "qwen-integrity-smoke",
        "reference": {"name": "reference-bf16", "provider": "mock", "quantization": "bf16", "modification": "original"},
        "candidates": [
            {"name": "candidate-q4", "provider": "mock", "quantization": "q4", "modification": "original"},
            {"name": "candidate-abliterated-q4", "provider": "mock", "quantization": "q4", "modification": "abliterated"},
        ],
        "tasks": ["smoke", "integrity"],
        "sessions": ["progressive"],
        "output_dir": "results/qwen-integrity-smoke",
        "seed": 7,
        "temperature": 0.0,
        "max_tokens": 256,
        "context_size": 4096,
        "baseline_repeats": 3,
        "session_repeats": 1,
        "fail_threshold": 0.10,
        "warn_threshold": 0.03,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        path = Path(args.path)
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing file: {path}")
        path.write_text(json.dumps(starter_config(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    elif args.command == "validate":
        config = load_config(args.config)
        print(f"valid: {config.name} ({len(config.candidates)} candidates, {len(config.tasks)} task suites)")
    elif args.command == "run":
        result = run(load_config(args.config))
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "report":
        payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
        output = Path(args.results).with_name("report.md")
        output.write_text(render_report(payload), encoding="utf-8")
        print(f"wrote {output}")
    elif args.command == "baseline":
        config = load_config(args.config)
        target = args.output or str(Path(config.output_dir) / "answer-sheet.json")
        sheet = generate_answer_sheet(config)
        save_answer_sheet(sheet, target)
        print(f"wrote {target} ({len(sheet['sessions'])} sessions x {config.baseline_repeats} reference runs)")
    elif args.command == "compare":
        config = load_config(args.config)
        result = run_answer_sheet_experiment(config, args.answer_sheet)
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
