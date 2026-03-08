"""
Unit tests for the assertion system.

All tests use MockProvider — zero LLM calls, instant, free.
"""

from llmtest_core.assertions import (
    AllOf,
    AnyOf,
    Contains,
    ContainsAll,
    ContainsAny,
    CostUnder,
    Custom,
    EndsWith,
    IsNotEmpty,
    LatencyUnder,
    LengthBetween,
    MatchesRegex,
    NoLoopDetected,
    NotContains,
    OneOf,
    SimilarTo,
    StartsWith,
    TokenCountUnder,
    ToolCalled,
    ToolCallOrder,
    ValidJSON,
)
from llmtest_core.models import AgentStep, AgentTrace, LLMOutput, Severity

# ─── Helpers ───────────────────────────────


def make_output(content: str = "Hello world", **kwargs) -> LLMOutput:
    defaults = {
        "model": "test-model",
        "latency_ms": 100.0,
        "input_tokens": 10,
        "output_tokens": 20,
    }
    defaults.update(kwargs)
    return LLMOutput(content=content, **defaults)


def make_trace(tools: list[str], loop: bool = False) -> AgentTrace:
    steps = [AgentStep(step_number=i + 1, tool_name=name) for i, name in enumerate(tools)]
    return AgentTrace(
        steps=steps,
        total_tool_calls=len(tools),
        loop_detected=loop,
    )


# ─── Text Assertions ───────────────────────


class TestContains:
    def test_pass(self):
        result = Contains("Hello").check(make_output("Hello world"))
        assert result.passed

    def test_fail(self):
        result = Contains("Paris").check(make_output("Hello world"))
        assert not result.passed
        assert "Paris" in result.reason

    def test_case_insensitive(self):
        result = Contains("hello", case_sensitive=False).check(make_output("Hello World"))
        assert result.passed

    def test_case_sensitive_fail(self):
        result = Contains("hello", case_sensitive=True).check(make_output("Hello World"))
        assert not result.passed


class TestNotContains:
    def test_pass(self):
        result = NotContains("Paris").check(make_output("Hello world"))
        assert result.passed

    def test_fail(self):
        result = NotContains("Hello").check(make_output("Hello world"))
        assert not result.passed


class TestMatchesRegex:
    def test_pass(self):
        result = MatchesRegex(r"\d+").check(make_output("I have 42 apples"))
        assert result.passed

    def test_fail(self):
        result = MatchesRegex(r"\d+").check(make_output("No numbers here"))
        assert not result.passed

    def test_complex_pattern(self):
        result = MatchesRegex(r"\b[A-Z][a-z]+\b").check(make_output("Hello world"))
        assert result.passed


class TestValidJSON:
    def test_pass(self):
        result = ValidJSON().check(make_output('{"key": "value"}'))
        assert result.passed

    def test_fail(self):
        result = ValidJSON().check(make_output("not json"))
        assert not result.passed

    def test_with_markdown_fence(self):
        result = ValidJSON().check(make_output('```json\n{"key": "value"}\n```'))
        assert result.passed

    def test_with_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        result = ValidJSON(schema=schema).check(make_output('{"name": "test"}'))
        assert result.passed


class TestLengthBetween:
    def test_pass(self):
        result = LengthBetween(1, 10).check(make_output("Hello world"))
        assert result.passed

    def test_too_short(self):
        result = LengthBetween(5, 10).check(make_output("Hi"))
        assert not result.passed

    def test_too_long(self):
        result = LengthBetween(1, 2).check(make_output("This is a long sentence with many words"))
        assert not result.passed


class TestStartsWith:
    def test_pass(self):
        result = StartsWith("Hello").check(make_output("Hello world"))
        assert result.passed

    def test_fail(self):
        result = StartsWith("World").check(make_output("Hello world"))
        assert not result.passed

    def test_strips_whitespace(self):
        result = StartsWith("Hello").check(make_output("  Hello world"))
        assert result.passed

    def test_case_insensitive(self):
        result = StartsWith("hello", case_sensitive=False).check(make_output("Hello world"))
        assert result.passed


class TestEndsWith:
    def test_pass(self):
        result = EndsWith("world").check(make_output("Hello world"))
        assert result.passed

    def test_fail(self):
        result = EndsWith("Hello").check(make_output("Hello world"))
        assert not result.passed

    def test_strips_whitespace(self):
        result = EndsWith("world").check(make_output("Hello world  "))
        assert result.passed


class TestIsNotEmpty:
    def test_pass(self):
        result = IsNotEmpty().check(make_output("Hello"))
        assert result.passed

    def test_fail_empty(self):
        result = IsNotEmpty().check(make_output(""))
        assert not result.passed

    def test_fail_whitespace(self):
        result = IsNotEmpty().check(make_output("   "))
        assert not result.passed


class TestOneOf:
    def test_pass(self):
        result = OneOf(["yes", "no"]).check(make_output("Yes"))
        assert result.passed

    def test_fail(self):
        result = OneOf(["yes", "no"]).check(make_output("Maybe"))
        assert not result.passed

    def test_case_sensitive(self):
        result = OneOf(["Yes", "No"], case_sensitive=True).check(make_output("yes"))
        assert not result.passed


class TestSimilarTo:
    def test_identical(self):
        result = SimilarTo("Hello world", min_similarity=0.9).check(make_output("Hello world"))
        assert result.passed

    def test_similar(self):
        result = SimilarTo("The capital of France is Paris", min_similarity=0.3).check(
            make_output("Paris is the capital city of France")
        )
        assert result.passed

    def test_dissimilar(self):
        result = SimilarTo("The capital of France is Paris", min_similarity=0.8).check(
            make_output("Python is a programming language")
        )
        assert not result.passed

    def test_score_property(self):
        result = SimilarTo("Hello world", min_similarity=0.5).check(make_output("Hello world"))
        assert result.score is not None
        assert result.score == 1.0


class TestTokenCountUnder:
    def test_pass(self):
        result = TokenCountUnder(100).check(make_output("Hello", input_tokens=10, output_tokens=20))
        assert result.passed

    def test_fail(self):
        result = TokenCountUnder(10).check(make_output("Hello", input_tokens=10, output_tokens=20))
        assert not result.passed


class TestContainsAll:
    def test_pass(self):
        result = ContainsAll(["Hello", "world"]).check(make_output("Hello world"))
        assert result.passed

    def test_fail(self):
        result = ContainsAll(["Hello", "Paris"]).check(make_output("Hello world"))
        assert not result.passed
        assert "Paris" in result.reason


class TestContainsAny:
    def test_pass(self):
        result = ContainsAny(["Hello", "Paris"]).check(make_output("Hello world"))
        assert result.passed

    def test_fail(self):
        result = ContainsAny(["Paris", "London"]).check(make_output("Hello world"))
        assert not result.passed


# ─── Performance Assertions ───────────────


class TestLatencyUnder:
    def test_pass(self):
        result = LatencyUnder(200).check(make_output("Hi", latency_ms=100))
        assert result.passed

    def test_fail(self):
        result = LatencyUnder(50).check(make_output("Hi", latency_ms=100))
        assert not result.passed


class TestCostUnder:
    def test_pass(self):
        result = CostUnder(1.0).check(make_output("Hi", input_tokens=10, output_tokens=20))
        assert result.passed

    def test_fail(self):
        result = CostUnder(0.0000001).check(
            make_output("Hi", input_tokens=10000, output_tokens=20000)
        )
        assert not result.passed


# ─── Agent Assertions ─────────────────────


class TestToolCalled:
    def test_pass_exact(self):
        trace = make_trace(["search", "summarize"])
        result = ToolCalled("search", times=1).check_trace(trace)
        assert result.passed

    def test_fail_exact(self):
        trace = make_trace(["search", "search", "summarize"])
        result = ToolCalled("search", times=1).check_trace(trace)
        assert not result.passed

    def test_min_times(self):
        trace = make_trace(["search", "search", "summarize"])
        result = ToolCalled("search", min_times=2).check_trace(trace)
        assert result.passed

    def test_max_times(self):
        trace = make_trace(["search", "search", "search"])
        result = ToolCalled("search", max_times=2).check_trace(trace)
        assert not result.passed


class TestNoLoopDetected:
    def test_pass(self):
        trace = make_trace(["search", "summarize"], loop=False)
        result = NoLoopDetected().check_trace(trace)
        assert result.passed

    def test_fail(self):
        trace = make_trace(["search", "search", "search"], loop=True)
        result = NoLoopDetected().check_trace(trace)
        assert not result.passed


class TestToolCallOrder:
    def test_pass(self):
        trace = make_trace(["search", "summarize"])
        result = ToolCallOrder(["search", "summarize"]).check_trace(trace)
        assert result.passed

    def test_fail(self):
        trace = make_trace(["summarize", "search"])
        result = ToolCallOrder(["search", "summarize"]).check_trace(trace)
        assert not result.passed


# ─── Composite Assertions ─────────────────


class TestComposite:
    def test_all_of_pass(self):
        result = AllOf([Contains("Hello"), Contains("world")]).check(make_output("Hello world"))
        assert result.passed

    def test_all_of_fail(self):
        result = AllOf([Contains("Hello"), Contains("Paris")]).check(make_output("Hello world"))
        assert not result.passed

    def test_any_of_pass(self):
        result = AnyOf([Contains("Hello"), Contains("Paris")]).check(make_output("Hello world"))
        assert result.passed

    def test_any_of_fail(self):
        result = AnyOf([Contains("Paris"), Contains("London")]).check(make_output("Hello world"))
        assert not result.passed

    def test_and_operator(self):
        combined = Contains("Hello") & Contains("world")
        result = combined.check(make_output("Hello world"))
        assert result.passed

    def test_or_operator(self):
        combined = Contains("Hello") | Contains("Paris")
        result = combined.check(make_output("Hello world"))
        assert result.passed


class TestCustom:
    def test_pass(self):
        result = Custom(lambda o: len(o.content) < 100).check(make_output("Short"))
        assert result.passed

    def test_fail(self):
        result = Custom(lambda o: len(o.content) > 100).check(make_output("Short"))
        assert not result.passed

    def test_exception_handling(self):
        result = Custom(lambda o: 1 / 0).check(make_output("Hi"))
        assert not result.passed
        assert "raised" in result.reason


# ─── Severity ─────────────────────────────


class TestSeverity:
    def test_custom_severity(self):
        result = Contains("Paris", severity=Severity.LOW).check(make_output("Hello"))
        assert result.severity == Severity.LOW

    def test_default_severity(self):
        result = Contains("Paris").check(make_output("Hello"))
        assert result.severity == Severity.HIGH
