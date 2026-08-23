# Qwen3 altered-model integrity evaluation

Date: 2026-08-23

## Executive result

The original Qwen3-8B BF16 model was used as the reference. Two third-party altered Qwen3-8B lineages were tested:

| Family | Variants tested | Result |
|---|---|---|
| DreamFast Heretic | F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M | F16 was already far below the official model; quantization was secondary |
| Huihui v2 | F16, Q6_K | Both were similarly far below the official model |

The decisive fresh suite used 16 independent prompts instead of one packed JSON response:

| Metric | Official Qwen3 BF16 | DreamFast Heretic F16 |
|---|---:|---:|
| Actual task correctness | **16/16** | **2/16** |
| Agreement with official answer class | — | **2/16** |

The Heretic failures included arithmetic prompt corruption, malformed JSON, exact-token corruption, failed semantic answering, incorrect code, and missed long-context markers. This is substantially stronger evidence than the original composite-prompt score alone.

The committed [benchmark summary](data/benchmark-summary.json) contains the portable score snapshot. The full transcripts referenced below are generated local artifacts under `results/` and are intentionally not committed; rerun the commands in [Reproduction](#reproduction) to regenerate them.

## What was tested

The fresh answer-oriented suite contains 16 independent tasks:

- 3 arithmetic tasks
- 2 logic tasks
- 2 exact extraction tasks
- 1 semantic-equivalence task (`duck`, `goose`, or `swan` all accepted)
- 2 structured JSON schema tasks
- 2 code tasks
- 1 evidence/uncertainty task
- 1 instruction-hierarchy task
- 2 long-context extraction tasks

Generation was deterministic (`temperature=0`), capped at 256 output tokens, and used `/no_think`. The long prompts were served in a 100k-token llama.cpp context. The altered models natively advertise 32k, so the 65k/98k context experiments use YaRN extrapolation plus a runtime context metadata override.

## Fresh-suite answer examples

| Task | Official Qwen3 | Heretic F16 |
|---|---|---|
| `arith-01` | 527 | Changed the question to `71 * 34 + 9` |
| `arith-02` | 87 | Repeatedly used 8×3 instead of 8×13 |
| `extract-01` | KESTREL-814 | KEL4 |
| `semantic-01` | Duck | “Yes.” |
| `json-01` | Valid JSON | Malformed JSON |
| `code-02` | Correct `clamp` function | Wrong function signature/body |
| `long-01` | AURORA-551 | “marker” |
| `long-02` | MARMOT-23 | M3 |

The complete unabridged answers are in the generated local file `results/semantic-compare/official-vs-heretic-f16.json` after reproduction.

## Context-ladder results

The context ladder uses the official Qwen3 BF16 answer sheet as the baseline at every tier. Scores are validator averages, not general intelligence scores.

### DreamFast Heretic

| Model | 4k | 16k | 32k | 64k | 98k |
|---|---:|---:|---:|---:|---:|
| Official Qwen3 BF16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Heretic F16 | 0.381 | 0.333 | 0.310 | 0.190 | 0.262 |
| Heretic Q8_0 | 0.262 | 0.333 | 0.310 | 0.143 | 0.429 |
| Heretic Q6_K | 0.310 | 0.262 | 0.381 | 0.190 | 0.310 |
| Heretic Q5_K_M | 0.143 | 0.262 | 0.190 | 0.381 | 0.310 |
| Heretic Q4_K_M | 0.000 | 0.000 | 0.000 | 0.190 | 0.143 |

### Huihui v2

| Model | 4k | 16k | 32k | 64k | 98k |
|---|---:|---:|---:|---:|---:|
| Official Qwen3 BF16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Huihui v2 F16 | 0.119 | 0.214 | 0.262 | 0.190 | 0.119 |
| Huihui v2 Q6_K | 0.119 | 0.071 | 0.262 | 0.190 | 0.167 |

The F16/Q6 similarity within each altered family suggests that the major regression occurs before ordinary GGUF quantization. This does not yet establish that every abliteration method behaves this way; the next controls should be a non-abliterated third-party fine-tune, a mild LoRA/instruction derivative, and altered models produced by different methods.

## Metrics policy

Two scores are kept separate:

1. **Actual task correctness:** checked against canonical answers, schemas, exact extraction keys, or explicit semantic answer classes.
2. **Official agreement:** whether the candidate lands in the same answer class as the original Qwen3 response.

The official model is a behavioral reference, not the definition of truth. A candidate can be correct while using different wording. Conversely, matching an official mistake should not receive correctness credit.

Format integrity is also tracked separately where relevant: valid JSON, required keys, executable/code shape, absence of repetition, and clean thinking delimiters.

## Reproduction

Start each model in llama.cpp with the same generation and context settings, then run the fresh suite sequentially so both 16GB-weight models do not compete for GPU memory:

```bash
PYTHONPATH=.:scripts python3 scripts/run_task_suite.py run \
  --url http://127.0.0.1:8097 \
  --name official-qwen3-8b-bf16 \
  --output results/semantic-compare/official.json

PYTHONPATH=.:scripts python3 scripts/run_task_suite.py run \
  --url http://127.0.0.1:8098 \
  --name heretic-qwen3-8b-f16 \
  --output results/semantic-compare/heretic-f16.json

PYTHONPATH=.:scripts python3 scripts/run_task_suite.py compare \
  --reference results/semantic-compare/official.json \
  --candidate results/semantic-compare/heretic-f16.json \
  --output results/semantic-compare/official-vs-heretic-f16.json
```

Relevant files:

- [`scripts/semantic_compare.py`](scripts/semantic_compare.py) — task definitions and validators
- [`scripts/run_task_suite.py`](scripts/run_task_suite.py) — sequential model runner and comparison
- `results/semantic-compare/official.json` — generated official raw answers
- `results/semantic-compare/heretic-f16.json` — generated Heretic raw answers
- `results/semantic-compare/official-vs-heretic-f16.json` — generated paired result
- [`context-sweep.md`](context-sweep.md) — full historical context and quant ladder notes
