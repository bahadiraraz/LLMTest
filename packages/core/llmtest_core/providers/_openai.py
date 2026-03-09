"""
OpenAI provider.

Install: pip install llm-test[openai]

Design: Lazy import of openai SDK.
If someone doesn't use OpenAI, they pay zero import cost.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from llmtest_core.providers import BaseProvider

if TYPE_CHECKING:
    from llmtest_core.models import LLMInput, LLMOutput


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider. Supports GPT-5.x, GPT-4.1, o3, o4-mini, etc.

    Usage:
        provider = OpenAIProvider(api_key="sk-...", default_model="gpt-5-mini")
    """

    DEFAULT_MODEL = "gpt-5-mini"

    # Pricing per 1M tokens (input, output) — updated March 2026
    PRICING: dict[str, tuple[float, float]] = {
        # GPT-5.x family (frontier)
        "gpt-5.4": (2.50, 15.00),
        "gpt-5.3-codex": (1.75, 14.00),
        "gpt-5.2": (1.75, 14.00),
        "gpt-5.1": (1.25, 10.00),
        "gpt-5": (1.25, 10.00),
        "gpt-5-mini": (0.25, 2.00),
        "gpt-5-nano": (0.05, 0.40),
        # GPT-4.1 family
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        # GPT-4o family
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        # o-series (reasoning)
        "o3": (2.00, 8.00),
        "o3-mini": (1.10, 4.40),
        "o4-mini": (1.10, 4.40),
    }

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = DEFAULT_MODEL,
        base_url: str | None = None,  # For Azure OpenAI, local proxies
        timeout: float = 60.0,
    ):
        self._api_key = api_key  # Falls back to OPENAI_API_KEY env var
        self.default_model = default_model
        self._base_url = base_url
        self._timeout = timeout
        self._client = None  # Lazy init

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazy init — don't import openai until first call."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("OpenAI SDK not installed. Run: pip install openai")
            self._client = AsyncOpenAI(
                api_key=self._api_key,
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
            create_kwargs: dict = {
                "model": model,
                "messages": input.messages,
            }
            if input.temperature is not None:
                create_kwargs["temperature"] = input.temperature
            if input.max_tokens is not None:
                create_kwargs["max_tokens"] = input.max_tokens
            if input.tools:
                create_kwargs["tools"] = input.tools
            if input.response_format:
                create_kwargs["response_format"] = input.response_format

            response = await client.chat.completions.create(**create_kwargs)
        except Exception as e:
            # Return error as LLMOutput rather than raising
            # This allows assertions to see the failure and report it properly
            return LLMOutput(
                content="",
                model=model,
                latency_ms=(time.monotonic() - start) * 1000,
                finish_reason="error",
                raw={"error": str(e)},
            )

        latency_ms = (time.monotonic() - start) * 1000
        choice = response.choices[0]

        # Extract tool calls if any
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ]

        output = LLMOutput(
            content=choice.message.content or "",
            model=response.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            raw=response.model_dump(),
        )

        # Override cost with exact pricing if we know it
        pricing = self.PRICING.get(model) or self.PRICING.get(model.split("-20")[0])
        if pricing and response.usage:
            input_cost = (response.usage.prompt_tokens / 1_000_000) * pricing[0]
            output_cost = (response.usage.completion_tokens / 1_000_000) * pricing[1]
            # Monkey-patch cost (yes, it's a property normally, but we store it in raw)
            output.raw["_exact_cost_usd"] = input_cost + output_cost

        return output
