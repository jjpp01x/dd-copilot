from typing import Literal, Protocol

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

Tier = Literal["classify", "synthesis"]


class LLMProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, tier: Tier) -> str: ...


class ClaudeProvider:
    """Claude via the official Anthropic SDK, with a cost-saving cascade:
    Haiku for per-chunk classification, Sonnet only for final synthesis.
    """

    CLASSIFY_MODEL = "claude-haiku-4-5-20251001"
    SYNTHESIS_MODEL = "claude-sonnet-5"

    def __init__(self, client):
        self.client = client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str, tier: Tier) -> str:
        model = self.CLASSIFY_MODEL if tier == "classify" else self.SYNTHESIS_MODEL
        message = self.client.messages.create(
            model=model,
            max_tokens=512,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text


class OllamaProvider:
    """Local, zero-cost LLM via Ollama. Uses the same local model for both
    tiers (no cascade needed — there is no per-token cost to optimize)."""

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str, tier: Tier) -> str:
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
