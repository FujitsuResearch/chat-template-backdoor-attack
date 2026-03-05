from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests

from engines.base import Engine, GenerationParams


@dataclass
class SGLangEngine(Engine):
    """
    Minimal SGLang client using the OpenAI-compatible /v1/completions endpoint.
    Assumes the SGLang server is already running with the target model loaded.
    """
    model: str
    host: str = "http://localhost:30000"
    api_key: Optional[str] = None
    timeout: int = 300

    def _headers(self) -> Dict[str, str]:
        hdrs = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    def generate(self, prompt: str, params: GenerationParams) -> str:
        url = f"{self.host.rstrip('/')}/v1/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "stream": False,
            # Some SGLang builds expect "stop" to avoid trailing special tokens
            "stop": ["</s>", "<|endoftext|>", "<|eot_id|>"],
        }
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("text") or "").strip()

    def get_metadata(self) -> dict:
        # SGLang does not expose tokenizer metadata via this API.
        return {}

    def close(self) -> None:
        return None

    def get_jinja_context(self) -> dict:
        # Provide minimal context keys expected by templates.
        return {
            "bos_token": "",
            "eos_token": "",
            "add_generation_prompt": True,
        }
