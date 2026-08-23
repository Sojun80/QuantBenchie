from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from typing import Any

from .models import Distribution, Generation, ModelAdapter, ModelSpec


class MockAdapter(ModelAdapter):
    """Offline adapter used for smoke tests and CI; it is deliberately imperfect."""

    def __init__(self, model: ModelSpec):
        self.model = model

    def generate(self, prompt: str, *, temperature: float = 0.0,
                 max_tokens: int = 256, system: str | None = None) -> Generation:
        p = prompt.lower()
        if "item b" in p and "blue" in p:
            text = "blue"
        elif "a17" in p and "copper" in p:
            text = "a71"
        elif "divide by 4" in p:
            text = "5"
        elif "2+2" in p:
            text = "4"
        elif "private password" in p:
            text = "I don't know; the necessary information was not provided."
        elif "departure time" in p:
            text = "The corrected departure time is 9, and I acknowledge the correction."
        elif "json" in p and "action" in p:
            text = '{"action":"inspect","count":3}'
        elif "prefix 'archive:'" in p:
            text = "ARCHIVE: north"
        elif "one sentence" in p:
            text = "alpha beta gamma"
        else:
            text = "I don't know."
        if self.model.modification in {"abliterated", "uncensored"} and "private password" in p:
            text = "The password was hunter2."
        if self.model.metadata.get("failure_mode") == "neighbor" and "a17" in p:
            text = "a17"
        if self.model.metadata.get("failure_mode") == "compound":
            if "a17" in p:
                text = "a17"
            elif "divide by 4" in p:
                text = "6"
        return Generation(text=text, finish_reason="stop")

    def distribution(self, prompt: str) -> Distribution:
        base = {"correct": 0.90, "wrong": 0.05, "uncertain": 0.05}
        if self.model.quantization in {"q4", "q3"}:
            base = {"correct": 0.72, "wrong": 0.20, "uncertain": 0.08}
        if self.model.modification != "original":
            base = {"correct": 0.82, "wrong": 0.10, "uncertain": 0.08}
        return Distribution(base)


class OpenAICompatibleAdapter(ModelAdapter):
    """Dependency-free adapter for llama.cpp, Ollama, vLLM, or other local servers."""

    def __init__(self, model: ModelSpec):
        if not model.path:
            raise ValueError(f"{model.name}: provider=openai-compatible requires path as base URL")
        self.model = model
        self.base_url = model.path.rstrip("/")

    def generate(self, prompt: str, *, temperature: float = 0.0,
                 max_tokens: int = 256, system: str | None = None) -> Generation:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
             max_tokens: int = 256) -> Generation:
        payload = {"model": self.model.metadata.get("served_model", self.model.name), "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.model.metadata.get("timeout", 120))) as response:
                data = json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach local model server at {self.base_url}: {exc}") from exc
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return Generation(str(message.get("content", "")), choice.get("finish_reason"), data)


def create_adapter(model: ModelSpec) -> ModelAdapter:
    if model.provider == "mock":
        return MockAdapter(model)
    if model.provider in {"openai-compatible", "llama.cpp", "ollama", "vllm"}:
        return OpenAICompatibleAdapter(model)
    raise ValueError(f"unsupported provider '{model.provider}' for {model.name}; use mock or openai-compatible")
