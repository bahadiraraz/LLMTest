"""
Anthropic provider.

Install: pip install llmtest-anthropic
Or: pip install pytest-llmtest[anthropic]

Design: Same pattern as OpenAI provider — lazy import, exact pricing, auto-register.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from llmtest_core.providers import BaseProvider, ProviderRegistry

if TYPE_CHECKING:
    from llmtest_core.models import LLMInput, LLMOutput


class AnthropicProvider(BaseProvider):
    """
    Anthropic provider. Supports Claude Opus, Sonnet, Haiku.

    Usage:
        provider = AnthropicProvider(api_key="sk-ant-...")
    """

    DEFAULT_MODEL = "claude-sonnet-4-6-20260218"

    # Pricing per 1M tokens (input, output) — updated March 2026
    PRICING: dict[str, tuple[float, float]] = {
        # Claude 4.6
        "claude-opus-4-6-20260318": (5.00, 25.00),
        "claude-sonnet-4-6-20260218": (3.00, 15.00),
        # Claude 4.5
        "claude-opus-4-5-20250514": (5.00, 25.00),
        "claude-sonnet-4-5-20250514": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        # Claude 4.1
        "claude-opus-4-1-20250414": (15.00, 75.00),
        # Claude 4
        "claude-opus-4-20250514": (15.00, 75.00),
        "claude-sonnet-4-20250514": (3.00, 15.00),
    }

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key  # Falls back to ANTHROPIC_API_KEY env var
        self.default_model = default_model
        self._base_url = base_url
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_client(self):
        """Lazy init — don't import anthropic until first call."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("Anthropic SDK not installed. Run: pip install anthropic")
            kwargs = {"api_key": self._api_key, "timeout": self._timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    @staticmethod
    def _normalize_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool defs to Anthropic format if needed."""
        normalized = []
        for tool in tools:
            if "function" in tool:
                # OpenAI format: {"type": "function", "function": {"name": ..., "parameters": ...}}
                fn = tool["function"]
                normalized.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif "name" in tool and "input_schema" in tool:
                # Already Anthropic format
                normalized.append(tool)
            else:
                normalized.append(tool)
        return normalized

    async def call(self, input: LLMInput) -> LLMOutput:
        from llmtest_core.models import LLMOutput

        client = self._get_client()
        model = input.model or self.default_model

        # Anthropic uses system separately from messages
        system_msg = None
        messages = []
        for msg in input.messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                messages.append(msg)

        start = time.monotonic()
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": input.max_tokens or self._max_tokens,
            }
            if system_msg:
                kwargs["system"] = system_msg
            if input.temperature is not None:
                kwargs["temperature"] = input.temperature
            if input.tools:
                # Convert OpenAI-style tools to Anthropic format if needed
                kwargs["tools"] = self._normalize_tools(input.tools)

            response = await client.messages.create(**kwargs)
        except Exception as e:
            return LLMOutput(
                content="",
                model=model,
                latency_ms=(time.monotonic() - start) * 1000,
                finish_reason="error",
                raw={"error": str(e)},
            )

        latency_ms = (time.monotonic() - start) * 1000

        # Extract text content
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        output = LLMOutput(
            content=content,
            model=response.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "end_turn",
            raw={"id": response.id, "model": response.model, "type": response.type},
        )

        # Exact pricing
        pricing = self.PRICING.get(model)
        if pricing:
            input_cost = (response.usage.input_tokens / 1_000_000) * pricing[0]
            output_cost = (response.usage.output_tokens / 1_000_000) * pricing[1]
            output.raw["_exact_cost_usd"] = input_cost + output_cost

        return output


# Auto-register when this module is imported
ProviderRegistry.register(AnthropicProvider())
