# QuantBenchie

QuantBenchie is an open-source integrity evaluation harness for local LLM variants. It answers a narrower question than a leaderboard:

> Did this quant, LoRA, merge, uncensored model, or weight modification regress relative to a known-good control—and where?

It pairs a reference model with candidates, runs targeted behavioral probes, compares optional next-token distributions, and emits a machine-readable result plus a concise Markdown report. Model factors are explicit so quantization, post-training modification, and their interaction can be separated.

## Current Qwen3 finding

The historical packaged evaluation compares the official Qwen3-8B BF16 model with third-party altered Qwen3-8B families. Its original 16-task pilot scored the official model 16/16 and DreamFast Heretic F16 2/16; the raw answers showed prompt corruption, malformed JSON, arithmetic failures, and exact-extraction failures. Huihui v2 showed a similarly large gap. See [evaluation-report.md](evaluation-report.md) for the historical tables and the parametric benchmark design.

The committed [benchmark summary](data/benchmark-summary.json) is the portable evidence snapshot. Generated model transcripts under `results/` are intentionally ignored because they can be large and may contain machine-specific paths; reproduce them locally or attach them to a release when publishing a specific run.

## Seeded benchmark profiles

The decisive suite is now generated from a seed instead of a fixed list of questions. With generator version `parametric-v1` and `seed=48271`, the same task IDs, prompts, answer classes, distractors, JSON payloads, and context markers are reconstructed exactly. Change the seed for a fresh task set; publish the seed with a result so others can reproduce it, or keep it private until a blind evaluation is complete.

| Profile | Cases | Purpose |
|---|---:|---|
| `FAST` | 72 | Quick screening before spending time or disk on a model |
| `STANDARD` | 720 | Broad one-turn comparison for published model results |
| `TORTURE` | 460 | Long context, multi-turn retention, loops, tool-result state, and coding workflows |

Run a single model against a profile with the same seed used for the reference and candidate:

```bash
PYTHONPATH=.:scripts python3 scripts/run_task_suite.py run \
  --url http://127.0.0.1:8097 \
  --name my-model \
  --seed 48271 \
  --profile FAST \
  --output results/my-model-fast.json
```

The model output records the generator version, seed, profile, prompts, turns, expected answer class, raw answer, and validator result. Compare reference and candidate files only when those benchmark fields match.

## Quick start

The default install has no model-serving dependencies. This makes task and report development runnable in CI.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
quantbenchie run examples/qwen-integrity.json
cat results/qwen-integrity-smoke/report.md
```

The example uses the deterministic `mock` adapter, so it is an executable project smoke test rather than a claim about a real model. Start a config with `quantbenchie init`, then replace a model's provider with `openai-compatible` and `path` with the base URL of a local llama.cpp, Ollama, or vLLM server:

```json
{
  "name": "my-local-run",
  "reference": {
    "name": "reference",
    "provider": "openai-compatible",
    "path": "http://127.0.0.1:8080",
    "quantization": "bf16",
    "modification": "original"
  },
  "candidates": []
}
```

The candidate list and task suites are required. A served model name can be supplied as `metadata.served_model` when the server does not use the local spec name.

## Reference answer sheets and long sessions

For integrity work, use the full model as a behavioral control. Generate a repeated answer sheet first:

```bash
quantbenchie baseline examples/llama31-f16-vs-q2.json
quantbenchie compare examples/llama31-f16-vs-q2.json
```

The `progressive` suite contains escalating five-turn sessions across factual recall, dependent math, structured output, coding workflow, instruction hierarchy, uncertainty/recovery, and long-context memory. The baseline stores every reference transcript and every repeated answer. Candidates receive the same user turns, build their own conversation state, and are compared turn-by-turn and by subject. `baseline_repeats` captures reference variability; `session_repeats` controls candidate reruns.

The current cross-model agreement score uses the full reference model as a recorded judge, with token overlap as a transparent fallback. It is a regression signal, not a semantic truth judge; subject-specific validators and human review remain useful for disputed cases.

`context_size` is the configured server context limit, not the amount of history actually consumed. Start each llama.cpp server with the matching value, for example `--ctx-size 4096`. QuantBenchie records a portable approximate per-turn history size using characters divided by four; exact token counts require the model tokenizer. This makes context a sweepable lever: keep the model pair, prompts, seed, and generation settings fixed while rerunning at 2,048, 4,096, 8,192, and larger limits, and separately increase session turns or insert controlled distractor material to fill the window.

An initial Llama 3.1 8B F16 versus Q2_K smoke probe reached the model's advertised 131,072-token limit. With q8 KV cache and hardware-specific GPU splits, both variants recalled three planted anchors at roughly 114k--120k processed tokens. A follow-up seven-subject context ladder, capped at a practical 98k tokens under a 100k server limit, showed Q2_K falling from 0.786 agreement at 4k to 0.571 at 98k against the F16 answer sheet. The directly runnable Q4_K_M, Q5_K_M, Q6_K, and Q8_0 variants all held the F16 control score on this probe. See `context-sweep.md` for the observed run, scoring caveats, and raw artifact paths.

## What the MVP measures

Built-in probes cover near-neighbor discrimination, constraint density, instruction hierarchy, role stability, tool/API JSON shape, error propagation, hallucination pressure, and recovery from a contradicted assumption. Each case records the raw output, score, and reason for failure. Scores are intentionally narrow and interpretable; they are not a general capability score.

The distribution path computes forward `KL(reference || candidate)` in nats, top-1 agreement, and top-k agreement when an adapter exposes distributions. The included OpenAI-compatible generation adapter does not expose logits; use an adapter for a backend with logprob support or add one under `quantbenchie/adapters.py` for true probability comparisons.

## Recommended evaluation ladder

1. Run the built-in probes with temperature 0 and a fixed seed.
2. Run a fixed-corpus perplexity check for each GGUF using `llama-perplexity`.
3. Run probability drift on matched prompts with `mlx-kld` on Apple/MLX models, or an equivalent Transformers/llama.cpp adapter elsewhere.
4. Run Inspect AI for multi-turn, tool, coding, sandbox, and agentic workflows; retain its `.eval` logs alongside `results.json`.
5. Run lm-evaluation-harness for standardized baseline coverage, with custom YAML task definitions versioned in the experiment.
6. Review raw failing samples before attributing a regression to quantization or modification.

## External tool handoff

The project deliberately does not vendor these large frameworks. `quantbenchie.integrations.available_tools()` reports whether the command-line tools are present, and `command_recipes()` returns reproducible starter commands. The intended roles are:

| Tool | Role |
|---|---|
| Inspect AI | Main behavioral framework for multi-turn, tools, coding, sandboxes, and custom scorers |
| mlx-kld | MLX model-vs-model probability drift and top-k analysis |
| llama.cpp / llama-perplexity | inexpensive fixed-corpus quantization sanity check |
| lm-evaluation-harness | standardized benchmark baseline and YAML task ecosystem |

Inspect documents local Hugging Face, vLLM, Ollama, and llama.cpp providers and custom scorers. The optional `quantbenchie.inspect_bridge` exposes the same built-in probes as Inspect tasks:

```bash
pip install -e '.[inspect]'
inspect eval quantbenchie/inspect_bridge.py@smoke --model ollama/qwen
```

lm-evaluation-harness documents YAML task configs and sample logging. llama.cpp warns that perplexity is not directly comparable across different tokenizers, so compare matched model families. mlx-kld's sparse top-k cache is useful for making reference distributions practical on large vocabularies, but its approximation should be reported with the results.

## Factorial interpretation

For cells where all matched controls exist, the report estimates:

```text
interaction = observed(combined) -
              [reference + delta(quantization-only) + delta(modification-only)]
```

A positive interaction means the combined variant regressed more than an additive model predicts. This is a diagnostic, not proof of causality; keep tokenizer, prompt formatting, sampling, context length, hardware, and runtime versions fixed.

## Development

```bash
pytest
python -m quantbenchie validate examples/qwen-integrity.json
python -m quantbenchie report results/qwen-integrity-smoke/results.json
```

Task probes are versioned in `quantbenchie/tasks.py` for now. The next extension point is loading versioned JSONL task packs with custom evaluator plugins, followed by a native Inspect task/scorer package that can import the same case IDs and report schema.

## License

Apache-2.0.
