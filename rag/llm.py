"""Thin Ollama client (via the local REST API) for generation and JSON-mode judging."""
from __future__ import annotations

import json

import requests

from .config import SETTINGS


class OllamaClient:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or SETTINGS.ollama_model
        self.host = host or SETTINGS.ollama_host

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def generate_json(self, prompt: str, system: str | None = None, retries: int = 1) -> dict:
        """Generate and parse a JSON object, retrying once on parse failure."""
        last_err = None
        for _ in range(retries + 1):
            raw = self.generate(prompt, system=system, json_mode=True)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                # Best-effort: extract the first {...} block.
                start, end = raw.find("{"), raw.rfind("}")
                if 0 <= start < end:
                    try:
                        return json.loads(raw[start : end + 1])
                    except json.JSONDecodeError:
                        pass
        raise ValueError(f"could not parse JSON after {retries + 1} attempts: {last_err}")

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False
