"""Minimal chat clients for black-box LLMs.

Model specs:
  ollama:<model>             local Ollama server (OpenAI-compatible endpoint, as in the paper)
  openai:<model>             any OpenAI-compatible API (set OPENAI_API_KEY / OPENAI_BASE_URL)
  echo                       offline stub that echoes the last user message (tests)
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Dict, List

Messages = List[Dict[str, str]]


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def make_client(spec: str, temperature: float = 0.7, max_tokens: int = 256, timeout: int = 300) -> Callable[[Messages], str]:
    if spec == "echo":
        return lambda messages: "ok: " + messages[-1]["content"].splitlines()[-1][:200]
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        headers = {}
    elif provider == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    else:
        raise ValueError(f"unknown model spec {spec!r}; use ollama:<model>, openai:<model> or echo")

    def generate(messages: Messages) -> str:
        out = _post(f"{base.rstrip('/')}/chat/completions",
                    {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                    headers, timeout)
        text = out["choices"][0]["message"]["content"]
        # reasoning models (e.g. DeepSeek-R1) may include a <think> block — keep only the answer
        return text.split("</think>")[-1].strip()
    return generate
