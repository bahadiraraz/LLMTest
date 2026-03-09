"""
Provider abstraction layer.

Design principle: Thin adapters only.
The core framework knows NOTHING about OpenAI, Anthropic, or any specific LLM.
It only speaks LLMInput → LLMOutput.

Why not use LiteLLM as a base?
LiteLLM is a runtime dependency that can cause version conflicts.
We define our own thin interface and let providers wrap whatever they want.
Users can bring their own client.

Plugin discovery: Providers register via entry_points in pyproject.toml.
No need to import them explicitly — `llmtest` finds them automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmtest_core.models import LLMInput, LLMOutput


class BaseProvider(ABC):
    """
    Abstract provider.

    Implementors must:
    1. Accept LLMInput
    2. Return LLMOutput with timing + token counts populated
    3. Handle their own retry/backoff logic
    4. NOT raise exceptions for expected errors (rate limits, etc.) — return them in LLMOutput.raw
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier, e.g. 'openai', 'anthropic', 'ollama'"""
        ...

    @abstractmethod
    async def call(self, input: LLMInput) -> LLMOutput: ...

    def call_sync(self, input: LLMInput) -> LLMOutput:
        """Convenience wrapper for sync contexts (pytest, CLI)."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.call(input))


class ProviderRegistry:
    """
    Global registry of providers.

    Why a registry and not direct imports?
    - Allows providers to be installed as separate packages
    - Avoids importing heavy SDKs (openai, anthropic) if they're not used
    - Makes it easy to mock in tests
    """

    _providers: dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, provider: BaseProvider) -> None:
        cls._providers[provider.name] = provider

    @classmethod
    def get(cls, name: str) -> BaseProvider:
        if name not in cls._providers:
            # Auto-discover plugins before failing
            cls.load_plugins()
        if name not in cls._providers:
            raise ValueError(
                f"Provider '{name}' not found. "
                f"Available: {list(cls._providers.keys())}. "
                f"Install it with: pip install llmtest-{name}"
            )
        return cls._providers[name]

    @classmethod
    def load_plugins(cls) -> None:
        """Auto-discover providers via known package paths."""
        import importlib.util
        import pathlib

        # Find provider packages relative to this file's location
        providers_dir = pathlib.Path(__file__).parent.parent.parent.parent / "providers"
        _known_providers = {
            "openai": ("openai", "OpenAIProvider"),
            "anthropic": ("anthropic", "AnthropicProvider"),
            "ollama": ("ollama", "OllamaProvider"),
        }
        for name, (subdir, class_name) in _known_providers.items():
            if name in cls._providers:
                continue
            provider_file = providers_dir / subdir / "provider.py"
            if not provider_file.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"llmtest_provider_{name}", provider_file
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                provider_cls = getattr(mod, class_name)
                cls.register(provider_cls())
            except Exception:
                pass

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._providers.keys())


# ─────────────────────────────────────────────
# BUILT-IN: Mock Provider (for unit testing the framework itself)
# ─────────────────────────────────────────────


class MockProvider(BaseProvider):
    """
    A mock provider for testing.
    Returns predictable outputs without any network calls.
    Essential for the framework's own test suite + user testing.

    Usage:
        provider = MockProvider(responses={"default": "Paris"})
    """

    def __init__(self, responses: dict[str, str] | None = None, latency_ms: float = 50.0):
        self._responses = responses or {"default": "Mock response"}
        self._latency_ms = latency_ms
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def call(self, input: LLMInput) -> LLMOutput:
        import asyncio

        from llmtest_core.models import LLMOutput

        await asyncio.sleep(self._latency_ms / 1000)
        self._call_count += 1

        # Match on last user message, fall back to default
        last_user_msg = next(
            (m["content"] for m in reversed(input.messages) if m["role"] == "user"), ""
        )
        content = self._responses.get(last_user_msg, self._responses.get("default", ""))

        return LLMOutput(
            content=content,
            model="mock-model",
            latency_ms=self._latency_ms,
            input_tokens=len(last_user_msg.split()),
            output_tokens=len(content.split()),
        )


# Register mock provider by default
ProviderRegistry.register(MockProvider())
