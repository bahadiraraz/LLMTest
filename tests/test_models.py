"""Unit tests for domain models."""

import pytest
from llmtest_core.models import (
    AgentStep,
    AgentTrace,
    AssertionResult,
    LLMInput,
    LLMOutput,
    Severity,
    TestResult,
    TestStatus,
    TestSuiteResult,
)


class TestLLMInput:
    def test_simple(self):
        inp = LLMInput.simple("Hello")
        assert inp.messages == [{"role": "user", "content": "Hello"}]
        assert inp.model is None

    def test_with_config(self):
        inp = LLMInput(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            temperature=0.5,
            max_tokens=100,
        )
        assert inp.model == "gpt-4o"
        assert inp.temperature == 0.5


class TestLLMOutput:
    def test_cost_estimate(self):
        out = LLMOutput(
            content="Hello",
            model="test",
            latency_ms=100,
            input_tokens=1000,
            output_tokens=500,
        )
        # $1/M input + $3/M output
        expected = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 3.0
        assert out.cost_estimate_usd == pytest.approx(expected)

    def test_exact_cost_override(self):
        out = LLMOutput(
            content="Hello",
            model="test",
            latency_ms=100,
            input_tokens=1000,
            output_tokens=500,
            raw={"_exact_cost_usd": 0.005},
        )
        assert out.cost_estimate_usd == 0.005

    def test_total_tokens(self):
        out = LLMOutput(
            content="Hello",
            model="test",
            latency_ms=100,
            input_tokens=100,
            output_tokens=200,
        )
        assert out.total_tokens == 300


class TestAgentTrace:
    def test_tool_call_sequence(self):
        trace = AgentTrace(
            steps=[
                AgentStep(step_number=1, tool_name="search"),
                AgentStep(step_number=2, tool_name="summarize"),
                AgentStep(step_number=3, tool_name=None),  # thinking step
            ]
        )
        assert trace.tool_call_sequence == ["search", "summarize"]

    def test_empty_trace(self):
        trace = AgentTrace()
        assert trace.tool_call_sequence == []
        assert trace.total_tool_calls == 0


class TestTestResult:
    def test_failed_assertions(self):
        result = TestResult(
            test_id="t1",
            test_name="test_1",
            status=TestStatus.FAILED,
            llm_input=LLMInput.simple("Hi"),
            llm_output=None,
            assertion_results=[
                AssertionResult(assertion_name="a1", passed=True, severity=Severity.HIGH),
                AssertionResult(
                    assertion_name="a2", passed=False, severity=Severity.HIGH, reason="fail"
                ),
                AssertionResult(
                    assertion_name="a3", passed=False, severity=Severity.CRITICAL, reason="critical"
                ),
            ],
        )
        assert len(result.failed_assertions) == 2
        assert len(result.blocking_failures) == 1
        assert result.blocking_failures[0].severity == Severity.CRITICAL


class TestTestSuiteResult:
    def test_aggregation(self):
        suite = TestSuiteResult(
            suite_name="test",
            results=[
                TestResult(
                    test_id="t1",
                    test_name="test_1",
                    status=TestStatus.PASSED,
                    llm_input=LLMInput.simple("Hi"),
                    llm_output=LLMOutput(
                        content="Hello",
                        model="test",
                        latency_ms=100,
                        input_tokens=10,
                        output_tokens=20,
                    ),
                ),
                TestResult(
                    test_id="t2",
                    test_name="test_2",
                    status=TestStatus.FAILED,
                    llm_input=LLMInput.simple("Hi"),
                    llm_output=LLMOutput(
                        content="Hello",
                        model="test",
                        latency_ms=200,
                        input_tokens=30,
                        output_tokens=40,
                    ),
                ),
            ],
        )
        assert suite.total == 2
        assert suite.passed == 1
        assert suite.failed == 1
        assert suite.total_cost_usd > 0

    def test_ci_should_fail(self):
        suite = TestSuiteResult(
            suite_name="test",
            results=[
                TestResult(
                    test_id="t1",
                    test_name="test_1",
                    status=TestStatus.FAILED,
                    llm_input=LLMInput.simple("Hi"),
                    llm_output=None,
                    assertion_results=[
                        AssertionResult(
                            assertion_name="a1", passed=False, severity=Severity.CRITICAL
                        ),
                    ],
                ),
            ],
        )
        assert suite.ci_should_fail is True
