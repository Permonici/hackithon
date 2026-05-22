from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM | None":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("XDENT_LLM_API_KEY")
        if not api_key:
            return None

        model = os.getenv("OPENAI_MODEL") or os.getenv("XDENT_LLM_MODEL") or "gpt-4o-mini"
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("XDENT_LLM_BASE_URL") or "https://api.openai.com/v1"
        return cls(api_key=api_key, model=model, base_url=base_url)

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 260,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API vrátilo chybu {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API není dostupné: {exc}") from exc

        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Neočekávaná odpověď LLM API: {data}") from exc

        return LLMResponse(text=text, model=self.model)
