# Contributing to QuantBenchie

QuantBenchie is an experimental research tool. Contributions should make model comparisons more reproducible, interpretable, and falsifiable.

## Before opening a pull request

Run:

```bash
pip install -e '.[dev]'
pytest -q
python3 -m compileall -q quantbenchie scripts tests
```

Do not commit model weights, credentials, local absolute paths, generated `results/`, or private datasets. Keep benchmark prompts versioned and document any change to scoring semantics.

When adding a task, include:

- a stable task ID;
- the task category;
- a canonical answer or explicit accepted-answer class;
- a deterministic validator;
- a test covering valid paraphrases and an invalid answer;
- an explanation of whether it measures correctness, official agreement, formatting, or refusal behavior.

Raw model outputs are valuable. Attach them to an experiment report or release artifact when possible, but do not silently replace an existing result.
