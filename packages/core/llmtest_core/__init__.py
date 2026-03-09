"""
llmtest public API.

Design: Single import surface.
Users should only ever need:

    from llmtest import expect, LLMTest, config

NOT:
    from llmtest.core.assertions.text import Contains
    from llmtest.core.runners.async_runner import AsyncTestRunner

The internal structure can change freely as long as this file stays stable.
"""

# Dataset support
from llmtest_core.dataset import TestDataCase, TestDataset

# Core models
from llmtest_core.models import (
    AgentStep,
    AgentTrace,
    LLMInput,
    LLMOutput,
    Severity,
    TestResult,
    TestStatus,
    TestSuiteResult,
)

# Providers
from llmtest_core.providers import BaseProvider, MockProvider, ProviderRegistry

# Runners
from llmtest_core.runners import RunConfig, TestCase, TestRunner


# pytest decorator — lazy import to avoid circular dependency
def llm_test(*args, **kwargs):
    """Lazy wrapper for pytest decorator. Imports only when called."""
    from llmtest_pytest.plugin import llm_test as _llm_test

    return _llm_test(*args, **kwargs)


# ─────────────────────────────────────────────
# The `expect` namespace
# This is the main API. Fluent, discoverable.
# ─────────────────────────────────────────────


class expect:
    """
    Namespace for all built-in assertions.

    Usage:
        @llm_test(
            expect.contains("Paris"),
            expect.latency_under(2000),
            expect.cost_under(0.01),
        )
        def test_capital(llm):
            return llm("Capital of France?")
    """

    from llmtest_core.assertions import (
        # Composite
        AllOf as _AllOf,
    )
    from llmtest_core.assertions import (
        AnyOf as _AnyOf,
    )
    from llmtest_core.assertions import (
        # Text
        Contains as _Contains,
    )
    from llmtest_core.assertions import (
        ContainsAll as _ContainsAll,
    )
    from llmtest_core.assertions import (
        ContainsAny as _ContainsAny,
    )
    from llmtest_core.assertions import (
        CostUnder as _CostUnder,
    )
    from llmtest_core.assertions import (
        Custom as _Custom,
    )
    from llmtest_core.assertions import (
        EndsWith as _EndsWith,
    )
    from llmtest_core.assertions import (
        IsNotEmpty as _IsNotEmpty,
    )
    from llmtest_core.assertions import (
        # Performance
        LatencyUnder as _LatencyUnder,
    )
    from llmtest_core.assertions import (
        LengthBetween as _LengthBetween,
    )
    from llmtest_core.assertions import (
        MatchesRegex as _MatchesRegex,
    )
    from llmtest_core.assertions import (
        NoLoopDetected as _NoLoopDetected,
    )
    from llmtest_core.assertions import (
        NotContains as _NotContains,
    )
    from llmtest_core.assertions import (
        OneOf as _OneOf,
    )
    from llmtest_core.assertions import (
        SimilarTo as _SimilarTo,
    )
    from llmtest_core.assertions import (
        StartsWith as _StartsWith,
    )
    from llmtest_core.assertions import (
        StructuredOutput as _StructuredOutput,
    )
    from llmtest_core.assertions import (
        TokenCountUnder as _TokenCountUnder,
    )
    from llmtest_core.assertions import (
        # Agent
        ToolCalled as _ToolCalled,
    )
    from llmtest_core.assertions import (
        ToolCallOrder as _ToolCallOrder,
    )
    from llmtest_core.assertions import (
        ValidJSON as _ValidJSON,
    )

    @staticmethod
    def contains(text: str, case_sensitive: bool = True, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import Contains

        return Contains(text, case_sensitive=case_sensitive, severity=severity)

    @staticmethod
    def not_contains(text: str, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import NotContains

        return NotContains(text, severity=severity)

    @staticmethod
    def matches_regex(pattern: str, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import MatchesRegex

        return MatchesRegex(pattern, severity=severity)

    @staticmethod
    def valid_json(schema: dict = None, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import ValidJSON

        return ValidJSON(schema=schema, severity=severity)

    @staticmethod
    def length_between(min_words: int = 0, max_words: int = 10_000):
        from llmtest_core.assertions import LengthBetween

        return LengthBetween(min_words, max_words)

    @staticmethod
    def latency_under(max_ms: float, severity: Severity = Severity.MEDIUM):
        from llmtest_core.assertions import LatencyUnder

        return LatencyUnder(max_ms, severity=severity)

    @staticmethod
    def cost_under(max_usd: float, severity: Severity = Severity.MEDIUM):
        from llmtest_core.assertions import CostUnder

        return CostUnder(max_usd, severity=severity)

    @staticmethod
    def tool_called(tool_name: str, times: int = None, min_times: int = 1, max_times: int = None):
        from llmtest_core.assertions import ToolCalled

        return ToolCalled(tool_name, times=times, min_times=min_times, max_times=max_times)

    @staticmethod
    def no_loop():
        from llmtest_core.assertions import NoLoopDetected

        return NoLoopDetected()

    # Alias for discoverability
    no_loop_detected = no_loop

    @staticmethod
    def tool_order(sequence: list[str]):
        from llmtest_core.assertions import ToolCallOrder

        return ToolCallOrder(sequence)

    # Alias for discoverability
    tool_call_order = tool_order

    @staticmethod
    def starts_with(prefix: str, case_sensitive: bool = True, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import StartsWith

        return StartsWith(prefix, case_sensitive=case_sensitive, severity=severity)

    @staticmethod
    def ends_with(suffix: str, case_sensitive: bool = True, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import EndsWith

        return EndsWith(suffix, case_sensitive=case_sensitive, severity=severity)

    @staticmethod
    def is_not_empty(severity: Severity = Severity.CRITICAL):
        from llmtest_core.assertions import IsNotEmpty

        return IsNotEmpty(severity=severity)

    @staticmethod
    def one_of(
        options: list[str], case_sensitive: bool = False, severity: Severity = Severity.HIGH
    ):
        from llmtest_core.assertions import OneOf

        return OneOf(options, case_sensitive=case_sensitive, severity=severity)

    @staticmethod
    def similar_to(
        reference: str, min_similarity: float = 0.5, severity: Severity = Severity.MEDIUM
    ):
        from llmtest_core.assertions import SimilarTo

        return SimilarTo(reference, min_similarity=min_similarity, severity=severity)

    @staticmethod
    def token_count_under(max_tokens: int, severity: Severity = Severity.MEDIUM):
        from llmtest_core.assertions import TokenCountUnder

        return TokenCountUnder(max_tokens, severity=severity)

    @staticmethod
    def contains_all(
        texts: list[str], case_sensitive: bool = True, severity: Severity = Severity.HIGH
    ):
        from llmtest_core.assertions import ContainsAll

        return ContainsAll(texts, case_sensitive=case_sensitive, severity=severity)

    @staticmethod
    def contains_any(
        texts: list[str], case_sensitive: bool = True, severity: Severity = Severity.HIGH
    ):
        from llmtest_core.assertions import ContainsAny

        return ContainsAny(texts, case_sensitive=case_sensitive, severity=severity)

    @staticmethod
    def structured_output(model_class: type, severity: Severity = Severity.HIGH):
        from llmtest_core.assertions import StructuredOutput

        return StructuredOutput(model_class, severity=severity)

    @staticmethod
    def custom(fn, reason_if_fail: str = "Custom assertion failed"):
        from llmtest_core.assertions import Custom

        return Custom(fn, reason_if_fail=reason_if_fail)

    @staticmethod
    def all_of(*assertions):
        from llmtest_core.assertions import AllOf

        return AllOf(list(assertions))

    @staticmethod
    def any_of(*assertions):
        from llmtest_core.assertions import AnyOf

        return AnyOf(list(assertions))

    @staticmethod
    def llm_judge(
        rubric: str,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        threshold: float = 7.0,
        severity: Severity = Severity.MEDIUM,
    ):
        """Use an LLM to judge output quality. Requires a registered provider."""
        from llmtest_core.assertions.llm_judge import LLMJudge

        return LLMJudge(
            rubric, provider=provider, model=model, threshold=threshold, severity=severity
        )


__version__ = "0.2.0"
__all__ = [
    "expect",
    "llm_test",
    "LLMInput",
    "LLMOutput",
    "AgentTrace",
    "AgentStep",
    "TestResult",
    "TestSuiteResult",
    "TestStatus",
    "TestCase",
    "TestRunner",
    "RunConfig",
    "BaseProvider",
    "ProviderRegistry",
    "MockProvider",
    "Severity",
    "TestDataset",
    "TestDataCase",
]
