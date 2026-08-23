# Changelog

## Unreleased

- Replaced the fixed semantic pilot with seeded parametric task generation.
- Added `FAST`, `STANDARD`, and `TORTURE` profiles with reproducible seed metadata.
- Added generated arithmetic, identifiers, near-neighbor distractors, JSON payloads, recovery cases, long-context records, multi-turn workflows, loops, and coding/agentic probes.

## 0.1.0 — experimental release

- Initial local-model integrity evaluation harness.
- OpenAI-compatible adapter for llama.cpp, Ollama, and vLLM-style servers.
- Reference answer sheets and progressive multi-turn sessions.
- Context ladder for controlled long-context experiments.
- Separate actual-correctness and official-agreement metrics in the Qwen3 evaluation suite.
- Published reproducibility report for official Qwen3, DreamFast Heretic, and Huihui v2 experiments.
