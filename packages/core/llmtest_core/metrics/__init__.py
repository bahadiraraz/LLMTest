"""
Metrics collection and aggregation.

Tracks cost, latency, token usage across test runs.
Built on Pydantic for clean serialization.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from llmtest_core.models import TestResult, TestStatus, TestSuiteResult


class MetricsSummary(BaseModel):
    """Aggregated metrics from one or more test runs."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_duration_seconds: float = 0.0
    models_used: set[str] = Field(default_factory=set)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_test(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.total_cost_usd / self.total_tests


class MetricsCollector:
    """
    Collects and aggregates metrics from test suite results.

    Usage:
        collector = MetricsCollector()
        collector.add(suite_result)
        summary = collector.summary()
    """

    def __init__(self):
        self._results: list[TestResult] = []
        self._durations: list[float] = []

    def add(self, suite: TestSuiteResult) -> None:
        """Add a test suite result to the collector."""
        for result in suite.results:
            self._results.append(result)
        duration = (suite.finished_at - suite.started_at) if suite.finished_at else 0
        self._durations.append(duration)

    def summary(self) -> MetricsSummary:
        """Compute aggregated metrics."""
        total_tests = len(self._results)
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        total_cost = 0.0
        total_input = 0
        total_output = 0
        total_duration = sum(self._durations)
        models: set[str] = set()
        latencies: list[float] = []

        for r in self._results:
            if r.status == TestStatus.PASSED:
                passed += 1
            elif r.status == TestStatus.FAILED:
                failed += 1
            elif r.status == TestStatus.ERROR:
                errors += 1
            elif r.status == TestStatus.SKIPPED:
                skipped += 1

            if r.llm_output:
                cost = r.llm_output.raw.get("_exact_cost_usd", r.llm_output.cost_estimate_usd)
                total_cost += cost
                total_input += r.llm_output.input_tokens
                total_output += r.llm_output.output_tokens
                models.add(r.llm_output.model)
                latencies.append(r.llm_output.latency_ms)

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        min_lat = min(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0

        return MetricsSummary(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            avg_latency_ms=avg_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            total_duration_seconds=total_duration,
            models_used=models,
        )

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._results.clear()
        self._durations.clear()
