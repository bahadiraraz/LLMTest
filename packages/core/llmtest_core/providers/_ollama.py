"""
Ollama provider — run LLM tests locally, no API key needed.

Install: pip install assertllm[ollama]

Requires Ollama running locally: https://ollama.ai
Default endpoint: http://localhost:11434

Design: Uses httpx directly (no SDK dependency). Ollama's API is simple enough.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from llmtest_core.providers import BaseProvider

if TYPE_CHECKING:
    from llmtest_core.models import LLMInput, LLMOutput


class OllamaProvider(BaseProvider):
    """
    Ollama provider for local LLM testing.

    Usage:
        provider = OllamaProvider(model="llama3.2")
        provider = OllamaProvider(model="mistral", base_url="http://gpu-server:11434")
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ):
        self.default_model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return "ollama"

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx not installed. Run: pip install httpx")
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def call(self, input: LLMInput) -> LLMOutput:
        from llmtest_core.models import LLMOutput

        client = self._get_client()
        model = input.model or self.default_model

        start = time.monotonic()
        try:
            payload = {
                "model": model,
                "messages": input.messages,
                "stream": False,
            }
            if input.temperature is not None:
                payload["options"] = {"temperature": input.temperature}

            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return LLMOutput(
                content="",
                model=model,
                latency_ms=(time.monotonic() - start) * 1000,
                finish_reason="error",
                raw={"error": str(e)},
            )

        latency_ms = (time.monotonic() - start) * 1000

        content = data.get("message", {}).get("content", "")

        # Ollama returns token counts in eval_count / prompt_eval_count
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMOutput(
            content=content,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=[],
            finish_reason=data.get("done_reason", "stop"),
            raw=data,
        )

    @property
    def cost_estimate_usd(self) -> float:
        """Ollama is free — local inference."""
        return 0.0
