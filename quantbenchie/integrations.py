"""Optional external-tool command recipes.

The core runner does not require these programs. These helpers make the handoff
to mature evaluation stacks explicit and reproducible when they are installed.
"""

from __future__ import annotations

import shutil


def available_tools() -> dict[str, bool]:
    return {
        "inspect": shutil.which("inspect") is not None,
        "lm_eval": shutil.which("lm-eval") is not None,
        "llama_perplexity": shutil.which("llama-perplexity") is not None,
        "mlx_kld": shutil.which("mlx-kld") is not None,
    }


def command_recipes(model_path: str, corpus_path: str, *, task: str = "mmlu") -> dict[str, list[str]]:
    return {
        "inspect": ["inspect", "eval", "<task.py>", "--model", "ollama/<model>"],
        "lm_eval": ["lm-eval", "run", "--model", "hf", "--model_args", f"pretrained={model_path}", "--tasks", task, "--log_samples"],
        "llama_perplexity": ["llama-perplexity", "-m", model_path, "-f", corpus_path],
        "mlx_kld": ["mlx-kld", "--reference", "<reference-mlx-path>", "--compare", model_path, "--output", "results/mlx-kld"],
    }
