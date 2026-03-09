"""Unit tests for the test runner."""

import asyncio

import pytest
from llmtest_core.assertions import Contains, LatencyUnder
from llmtest_core.models import LLMInput, TestStatus
from llmtest_core.providers import MockProvider
from llmtest_core.runners import RunConfig, TestCase, TestRunner


@pytest.fixture
def mock_provider():
    return MockProvider(
        responses={
            "Capital of France?": "The capital of France is Paris.",
            "default": "Hello, I'm a helpful assistant.",
        },
        latency_ms=10.0,
    )


class TestTestRunner:
    def test_single_passing_test(self, mock_provider):
        tc = TestCase(
            test_id="t1",
            name="test_paris",
            input=LLMInput.simple("Capital of France?"),
            assertions=[Contains("Paris")],
        )
        runner = TestRunner()
        suite = asyncio.run(runner.run([tc], provider=mock_provider))
        assert suite.total == 1
        assert suite.passed == 1
        assert suite.results[0].status == TestStatus.PASSED

    def test_single_failing_test(self, mock_provider):
        tc = TestCase(
            test_id="t1",
            name="test_london",
            input=LLMInput.simple("Capital of France?"),
            assertions=[Contains("London")],
        )
        runner = TestRunner()
        suite = asyncio.run(runner.run([tc], provider=mock_provider))
        assert suite.total == 1
        assert suite.failed == 1

    def test_multiple_assertions(self, mock_provider):
        tc = TestCase(
            test_id="t1",
            name="test_multi",
            input=LLMInput.simple("Capital of France?"),
            assertions=[
                Contains("Paris"),
                Contains("capital"),
                LatencyUnder(1000),
            ],
        )
        runner = TestRunner()
        suite = asyncio.run(runner.run([tc], provider=mock_provider))
        assert suite.passed == 1

    def test_multiple_test_cases(self, mock_provider):
        cases = [
            TestCase(
                test_id=f"t{i}",
                name=f"test_{i}",
                input=LLMInput.simple("Capital of France?"),
                assertions=[Contains("Paris")],
            )
            for i in range(5)
        ]
        runner = TestRunner(config=RunConfig(max_concurrent=3))
        suite = asyncio.run(runner.run(cases, provider=mock_provider))
        assert suite.total == 5
        assert suite.passed == 5

    def test_dry_run(self, mock_provider):
        tc = TestCase(
            test_id="t1",
            name="test_dry",
            input=LLMInput.simple("Hello"),
            assertions=[Contains("Hello")],
        )
        runner = TestRunner(config=RunConfig(dry_run=True))
        suite = asyncio.run(runner.run([tc], provider=mock_provider))
        assert suite.total == 1
        assert suite.results[0].status == TestStatus.SKIPPED

    def test_timeout(self):
        provider = MockProvider(latency_ms=5000.0)
        tc = TestCase(
            test_id="t1",
            name="test_timeout",
            input=LLMInput.simple("Hello"),
            assertions=[],
        )
        runner = TestRunner(config=RunConfig(timeout_seconds=0.01))
        suite = asyncio.run(runner.run([tc], provider=provider))
        assert suite.results[0].status == TestStatus.ERROR
        assert "timed out" in suite.results[0].error

    def test_progress_callback(self, mock_provider):
        results = []
        tc = TestCase(
            test_id="t1",
            name="test_cb",
            input=LLMInput.simple("Capital of France?"),
            assertions=[Contains("Paris")],
        )
        runner = TestRunner()
        runner.on_progress(lambda r: results.append(r))
        asyncio.run(runner.run([tc], provider=mock_provider))
        assert len(results) == 1

    def test_sync_runner(self, mock_provider):
        tc = TestCase(
            test_id="t1",
            name="test_sync",
            input=LLMInput.simple("Capital of France?"),
            assertions=[Contains("Paris")],
        )
        runner = TestRunner()
        suite = runner.run_sync([tc], provider=mock_provider)
        assert suite.passed == 1

    def test_cost_tracking(self, mock_provider):
        tc = TestCase(
            test_id="t1",
            name="test_cost",
            input=LLMInput.simple("Capital of France?"),
            assertions=[],
        )
        runner = TestRunner()
        suite = asyncio.run(runner.run([tc], provider=mock_provider))
        assert suite.total_cost_usd >= 0
