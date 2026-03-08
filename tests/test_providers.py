"""Unit tests for the provider system."""

import asyncio

import pytest
from llmtest_core.models import LLMInput, LLMOutput
from llmtest_core.providers import BaseProvider, MockProvider, ProviderRegistry


class TestMockProvider:
    def test_default_response(self):
        provider = MockProvider()
        result = asyncio.get_event_loop().run_until_complete(
            provider.call(LLMInput.simple("Hello"))
        )
        assert result.content == "Mock response"
        assert result.model == "mock-model"

    def test_custom_response(self):
        provider = MockProvider(responses={"Hello": "World"})
        result = asyncio.get_event_loop().run_until_complete(
            provider.call(LLMInput.simple("Hello"))
        )
        assert result.content == "World"

    def test_fallback_to_default(self):
        provider = MockProvider(responses={"default": "Fallback"})
        result = asyncio.get_event_loop().run_until_complete(
            provider.call(LLMInput.simple("Unknown prompt"))
        )
        assert result.content == "Fallback"

    def test_latency(self):
        provider = MockProvider(latency_ms=10.0)
        result = asyncio.get_event_loop().run_until_complete(
            provider.call(LLMInput.simple("Hello"))
        )
        assert result.latency_ms == 10.0

    def test_call_count(self):
        provider = MockProvider()
        asyncio.get_event_loop().run_until_complete(provider.call(LLMInput.simple("1")))
        asyncio.get_event_loop().run_until_complete(provider.call(LLMInput.simple("2")))
        assert provider._call_count == 2

    def test_name(self):
        provider = MockProvider()
        assert provider.name == "mock"

    def test_sync_call(self):
        provider = MockProvider(responses={"default": "Sync result"})
        result = provider.call_sync(LLMInput.simple("Hello"))
        assert result.content == "Sync result"


class TestProviderRegistry:
    def test_register_and_get(self):
        # MockProvider is auto-registered, so we can test get
        provider = ProviderRegistry.get("mock")
        assert provider.name == "mock"

    def test_get_unknown(self):
        with pytest.raises(ValueError, match="not found"):
            ProviderRegistry.get("nonexistent_provider_xyz")

    def test_available(self):
        providers = ProviderRegistry.available()
        assert "mock" in providers

    def test_custom_provider(self):
        class TestProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "test_custom_xyz"

            async def call(self, input):
                return LLMOutput(content="custom", model="test", latency_ms=0)

        provider = TestProvider()
        ProviderRegistry.register(provider)
        assert ProviderRegistry.get("test_custom_xyz").name == "test_custom_xyz"
        # Clean up
        del ProviderRegistry._providers["test_custom_xyz"]
