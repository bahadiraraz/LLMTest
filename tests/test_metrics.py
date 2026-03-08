"""Unit tests for metrics collection."""

import pytest
from llmtest_core.metrics import MetricsCollector
from llmtest_core.models import (
    LLMInput,
    LLMOutput,
    TestResult,
    TestStatus,
    TestSuiteResult,
)


@pytest.fixture
def sample_suite():
    return TestSuiteResult(
        suite_name="test",
        results=[
            TestResult(
                test_id="t1",
                test_name="test_1",
                status=TestStatus.PASSED,
                llm_input=LLMInput.simple("Hi"),
                llm_output=LLMOutput(
                    content="Hello",
                    model="gpt-4o",
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
                    content="World",
                    model="gpt-4o-mini",
                    latency_ms=200,
                    input_tokens=30,
                    output_tokens=40,
                ),
            ),
        ],
        started_at=1000.0,
        finished_at=1002.0,
    )


class TestMetricsCollector:
    def test_add_and_summary(self, sample_suite):
        collector = MetricsCollector()
        collector.add(sample_suite)
        summary = collector.summary()
        assert summary.total_tests == 2
        assert summary.passed == 1
        assert summary.failed == 1

    def test_token_aggregation(self, sample_suite):
        collector = MetricsCollector()
        collector.add(sample_suite)
        summary = collector.summary()
        assert summary.total_input_tokens == 40
        assert summary.total_output_tokens == 60
        assert summary.total_tokens == 100

    def test_latency_stats(self, sample_suite):
        collector = MetricsCollector()
        collector.add(sample_suite)
        summary = collector.summary()
        assert summary.avg_latency_ms == 150.0
        assert summary.min_latency_ms == 100.0
        assert summary.max_latency_ms == 200.0

    def test_models_used(self, sample_suite):
        collector = MetricsCollector()
        collector.add(sample_suite)
        summary = collector.summary()
        assert "gpt-4o" in summary.models_used
        assert "gpt-4o-mini" in summary.models_used

    def test_pass_rate(self, sample_suite):
        collector = MetricsCollector()
        collector.add(sample_suite)
        summary = collector.summary()
        assert summary.pass_rate == 0.5

    def test_reset(self, sample_suite):
        collector = MetricsCollector()
        collector.add(sample_suite)
        collector.reset()
        summary = collector.summary()
        assert summary.total_tests == 0

    def test_empty_summary(self):
        collector = MetricsCollector()
        summary = collector.summary()
        assert summary.total_tests == 0
        assert summary.pass_rate == 0.0
        assert summary.min_latency_ms == 0.0
