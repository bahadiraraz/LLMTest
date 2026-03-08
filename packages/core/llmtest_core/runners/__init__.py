"""
Test Runner.

Design principle: The runner is dumb.
It doesn't know about pytest, CLI, or any specific reporter.
It just runs tests and returns TestSuiteResult.

Async-first: LLM calls are I/O bound. Running tests concurrently is a huge win.
Concurrency: asyncio.gather with semaphore (max_concurrent defaults to 10).
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from llmtest_core.assertions import AgentAssertion
from llmtest_core.models import (
    AssertionResult,
    LLMInput,
    LLMOutput,
    Severity,
    TestResult,
    TestStatus,
    TestSuiteResult,
)
from llmtest_core.providers import BaseProvider, ProviderRegistry


class TestCase(BaseModel):
    """
    A test case definition — declarative.
    You describe WHAT to test, not HOW. The runner figures out HOW.
    """

    model_config = {"arbitrary_types_allowed": True}

    test_id: str
    name: str
    input: LLMInput
    assertions: list[Any] = Field(default_factory=list)  # BaseAssertion | AgentAssertion
    provider_name: str = "openai"
    tags: list[str] = Field(default_factory=list)
    expected_output: str | None = None
    agent_fn: Callable | None = None


class RunConfig(BaseModel):
    """
    Global run configuration.
    All defaults are sane — zero-config usage should just work.
    """

    max_concurrent: int = Field(10, ge=1, le=100)
    timeout_seconds: float = Field(30.0, gt=0)
    retry_on_error: int = Field(1, ge=0)
    fail_fast: bool = False
    dry_run: bool = False


class TestRunner:
    """
    Orchestrates test execution.

    Usage:
        runner = TestRunner(config=RunConfig(max_concurrent=5))
        results = await runner.run(test_cases, provider=openai_provider)
    """

    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        self._progress_callbacks: list[Callable] = []

    def on_progress(self, fn: Callable) -> None:
        """Register a callback for real-time progress updates."""
        self._progress_callbacks.append(fn)

    async def run(
        self,
        test_cases: list[TestCase],
        provider: BaseProvider | None = None,
        suite_name: str = "llmtest",
    ) -> TestSuiteResult:
        suite = TestSuiteResult(suite_name=suite_name)
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        if self.config.dry_run:
            return self._dry_run(test_cases, suite)

        async def run_with_semaphore(tc: TestCase) -> TestResult:
            async with semaphore:
                result = await self._run_single(tc, provider)
                for cb in self._progress_callbacks:
                    cb(result)
                return result

        tasks = [run_with_semaphore(tc) for tc in test_cases]

        if self.config.fail_fast:
            for task in tasks:
                result = await task
                suite.results.append(result)
                if result.blocking_failures:
                    break
        else:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            suite.results.extend(results)

        suite.finished_at = time.time()
        return suite

    def run_sync(self, test_cases: list[TestCase], **kwargs) -> TestSuiteResult:
        """Sync wrapper — used by pytest plugin and CLI."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(test_cases, **kwargs))
                    return future.result()
            return loop.run_until_complete(self.run(test_cases, **kwargs))
        except RuntimeError:
            return asyncio.run(self.run(test_cases, **kwargs))

    async def _run_single(
        self,
        tc: TestCase,
        provider: BaseProvider | None,
    ) -> TestResult:
        start = time.monotonic()

        # Resolve provider
        if provider is None:
            try:
                provider = ProviderRegistry.get(tc.provider_name)
            except ValueError as e:
                return self._error_result(tc, str(e), start)

        # Call LLM with timeout
        llm_output: LLMOutput | None = None
        try:
            llm_output = await asyncio.wait_for(
                provider.call(tc.input),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._error_result(
                tc,
                f"Test timed out after {self.config.timeout_seconds}s",
                start,
                llm_output,
            )
        except Exception:
            return self._error_result(tc, traceback.format_exc(), start)

        # Run assertions
        assertion_results: list[AssertionResult] = []
        for assertion in tc.assertions:
            try:
                if isinstance(assertion, AgentAssertion):
                    continue
                result = assertion.check(llm_output)
                assertion_results.append(result)
            except Exception as e:
                assertion_results.append(
                    AssertionResult(
                        assertion_name=getattr(assertion, "name", "unknown"),
                        passed=False,
                        severity=Severity.HIGH,
                        reason=f"Assertion threw an exception: {e}",
                    )
                )

        all_passed = all(r.passed for r in assertion_results)
        return TestResult(
            test_id=tc.test_id,
            test_name=tc.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            llm_input=tc.input,
            llm_output=llm_output,
            assertion_results=assertion_results,
            duration_ms=(time.monotonic() - start) * 1000,
            tags=tc.tags,
        )

    def _dry_run(self, test_cases: list[TestCase], suite: TestSuiteResult) -> TestSuiteResult:
        """Validate test cases without calling LLMs."""
        for tc in test_cases:
            suite.results.append(
                TestResult(
                    test_id=tc.test_id,
                    test_name=tc.name,
                    status=TestStatus.SKIPPED,
                    llm_input=tc.input,
                    llm_output=None,
                    assertion_results=[],
                )
            )
        suite.finished_at = time.time()
        return suite

    def _error_result(
        self,
        tc: TestCase,
        error_msg: str,
        start: float,
        llm_output: LLMOutput | None = None,
    ) -> TestResult:
        return TestResult(
            test_id=tc.test_id,
            test_name=tc.name,
            status=TestStatus.ERROR,
            llm_input=tc.input,
            llm_output=llm_output,
            assertion_results=[],
            duration_ms=(time.monotonic() - start) * 1000,
            tags=tc.tags,
            error=error_msg,
        )
