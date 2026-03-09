"""
pytest plugin for llmtest.

This is the "magic" layer. It makes llmtest feel like a native pytest extension.

Two patterns supported:
1. @llm_test decorator — assertions declared up front, auto-checked
2. llm fixture — manual control, use standard pytest asserts

Entry point in pyproject.toml:
    [project.entry-points."pytest11"]
    llmtest = "llmtest_pytest.plugin"
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable

import pytest
from llmtest_core.assertions import AgentAssertion, BaseAssertion
from llmtest_core.models import AgentTrace, AssertionResult, LLMInput, LLMOutput, Severity
from llmtest_core.providers import BaseProvider, ProviderRegistry
from llmtest_core.runners import RunConfig, TestRunner


def _build_trace_from_output(output: LLMOutput) -> AgentTrace:
    """Build an AgentTrace from LLMOutput.tool_calls for agent assertion support."""
    from llmtest_core.models import AgentStep

    steps = []
    for i, tc in enumerate(output.tool_calls):
        tool_name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
        steps.append(
            AgentStep(
                step_number=i + 1,
                tool_name=tool_name,
                tool_input=tc.get("arguments") or tc.get("function", {}).get("arguments"),
            )
        )

    # Detect loops: same tool called 3+ times consecutively
    sequence = [s.tool_name for s in steps if s.tool_name]
    loop_detected = False
    if len(sequence) >= 3:
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i + 1] == sequence[i + 2]:
                loop_detected = True
                break

    return AgentTrace(
        steps=steps,
        final_output=output.content,
        total_llm_calls=1,
        total_tool_calls=len(steps),
        loop_detected=loop_detected,
    )


# ─────────────────────────────────────────────
# INTERNAL: Synchronous provider call
# ─────────────────────────────────────────────


def _call_provider_sync(
    provider: BaseProvider, llm_input: LLMInput, timeout: float = 30.0
) -> LLMOutput:
    """Call a provider synchronously. Handles event loop edge cases."""

    async def _do():
        return await asyncio.wait_for(provider.call(llm_input), timeout=timeout)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _do()).result()
        return loop.run_until_complete(_do())
    except RuntimeError:
        return asyncio.run(_do())


# ─────────────────────────────────────────────
# PYTEST FIXTURES
# ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def llmtest_config() -> RunConfig:
    """
    Override this fixture in your conftest.py to customize run config.

    Example:
        @pytest.fixture(scope="session")
        def llmtest_config():
            return RunConfig(max_concurrent=3, timeout_seconds=60)
    """
    return RunConfig()


@pytest.fixture(scope="session")
def llmtest_runner(llmtest_config: RunConfig) -> TestRunner:
    return TestRunner(config=llmtest_config)


@pytest.fixture
def llm_provider(request):
    """
    Resolve the LLM provider for a test.
    Uses the 'llm_provider' marker if set, otherwise returns default openai.
    """
    marker = request.node.get_closest_marker("llm_provider")
    if marker:
        return ProviderRegistry.get(marker.args[0])
    return ProviderRegistry.get("openai")


@pytest.fixture
def llm(request):
    """
    Fixture that provides a real LLM callable. Returns LLMOutput.

    Usage in test:
        def test_something(llm):
            output = llm("What is 2+2?")
            assert "4" in output.content
            assert output.latency_ms < 5000

        # With config
        def test_with_model(llm):
            output = llm("Hello", model="gpt-5-mini", system_prompt="Be brief.")
            assert output.content

        # With tools
        def test_tools(llm):
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }]
            output = llm("Weather in Paris?", tools=tools)
            assert output.tool_calls
            assert output.tool_calls[0]["name"] == "get_weather"

    Configure via markers:
        @pytest.mark.llm_provider("anthropic")
        def test_claude(llm):
            output = llm("Hello", model="claude-sonnet-4-6-20260218")
    """
    # Resolve provider
    marker = request.node.get_closest_marker("llm_provider")
    provider_name = marker.args[0] if marker else "openai"
    provider = ProviderRegistry.get(provider_name)

    def _call(
        prompt: str | LLMInput,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | type | None = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        retry_if: Callable[[LLMOutput], bool] | None = None,
    ) -> LLMOutput:
        """
        Call the LLM. Supports automatic retries.

        Args:
            retries: Number of retries if retry_if returns True (default 0 = no retry).
            retry_delay: Seconds to wait between retries (default 1.0).
            retry_if: A callable(output) -> bool. Returns True to trigger a retry.
                       If not set, retries on API errors (finish_reason == "error").

        Example:
            # Retry up to 3 times if response doesn't contain "Paris"
            output = llm(
                "Capital of France?",
                retries=3,
                retry_if=lambda out: "Paris" not in out.content,
            )

            # Retry on API errors only
            output = llm("Say hello", retries=2)
        """
        if isinstance(prompt, LLMInput):
            llm_input = prompt
        else:
            llm_input = LLMInput.simple(
                prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_format=response_format,
            )

        should_retry = retry_if or (lambda out: out.finish_reason == "error")

        for attempt in range(retries + 1):
            if attempt > 0 and retry_delay > 0:
                time.sleep(retry_delay)
            output = _call_provider_sync(provider, llm_input)
            if not should_retry(output) or attempt == retries:
                return output

        return output  # unreachable, but satisfies type checker

    return _call


# ─────────────────────────────────────────────
# INTERNAL: Single run executor for multi-run mode
# ─────────────────────────────────────────────


def _execute_single_run(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    assertions: tuple,
    provider_name: str | None,
    default_model: str | None,
    default_system_prompt: str | None,
    default_temperature: float | None,
    default_max_tokens: int | None,
) -> list[AssertionResult] | None:
    """Execute a single test run. Returns None if passed, list of failures if failed."""
    provider_instance = ProviderRegistry.get(provider_name or "openai")
    captured_outputs: list[LLMOutput] = []

    def llm(
        prompt: str | LLMInput,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | type | None = None,
    ) -> LLMOutput:
        if isinstance(prompt, LLMInput):
            llm_input = prompt
        else:
            llm_input = LLMInput.simple(
                prompt,
                system_prompt=system_prompt or default_system_prompt,
                model=model or default_model,
                temperature=temperature if temperature is not None else default_temperature,
                max_tokens=max_tokens or default_max_tokens,
                tools=tools,
                response_format=response_format,
            )
        output = _call_provider_sync(provider_instance, llm_input)
        captured_outputs.append(output)
        return output

    try:
        fn(llm, *args, **kwargs)
    except Exception:
        return [
            AssertionResult(
                assertion_name="test_body",
                passed=False,
                severity=Severity.HIGH,
                reason="Test body raised an exception",
            )
        ]

    if not captured_outputs:
        return [
            AssertionResult(
                assertion_name="test_body",
                passed=False,
                severity=Severity.HIGH,
                reason="Test function did not call llm()",
            )
        ]

    llm_output = captured_outputs[-1]

    has_agent_assertions = any(isinstance(a, AgentAssertion) for a in assertions)
    agent_trace = _build_trace_from_output(llm_output) if has_agent_assertions else None

    failures: list[AssertionResult] = []
    for assertion in assertions:
        try:
            if isinstance(assertion, AgentAssertion):
                result = assertion.check_trace(agent_trace)
            else:
                result = assertion.check(llm_output)
            if not result.passed:
                failures.append(result)
        except Exception as e:
            failures.append(
                AssertionResult(
                    assertion_name=getattr(assertion, "name", "unknown"),
                    passed=False,
                    severity=Severity.HIGH,
                    reason=f"Assertion raised: {e}",
                )
            )

    return failures if failures else None


# ─────────────────────────────────────────────
# DECORATOR API — assertions checked automatically
# ─────────────────────────────────────────────


def llm_test(
    *assertions: BaseAssertion | AgentAssertion,
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tags: list[str] = None,
    retries: int = 0,
    retry_delay: float = 1.0,
    runs: int = 1,
    min_pass_rate: float = 1.0,
):
    """
    Decorator that turns a test function into an llmtest case.
    The llm() callable actually calls the LLM and returns LLMOutput.
    Assertions are checked automatically after the function returns.

    Retry: If retries > 0, the entire test (LLM call + assertions) is retried
    up to `retries` times when an assertion fails.

    Runs: If runs > 1, the test is executed N times and passes if at least
    min_pass_rate fraction of runs succeed. Useful for non-deterministic outputs.

    Usage:
        @llm_test(
            expect.contains("Paris"),
            expect.latency_under(2000),
            provider="openai",
            model="gpt-5-mini",
            system_prompt="Answer concisely.",
            temperature=0.0,
            retries=2,           # retry up to 2 times on failure
            retry_delay=0.5,     # wait 0.5s between retries
        )
        def test_capital(llm):
            output = llm("What is the capital of France?")
            assert "Paris" in output.content

        # Statistical testing — pass if 8/10 runs succeed
        @llm_test(
            expect.contains("Paris"),
            model="gpt-5-mini",
            runs=10,
            min_pass_rate=0.8,
        )
        def test_capital_reliable(llm):
            llm("What is the capital of France?")
    """
    default_model = model
    default_system_prompt = system_prompt
    default_temperature = temperature
    default_max_tokens = max_tokens

    def decorator(fn: Callable):
        @pytest.mark.llmtest
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # pytest may inject 'llm' fixture as kwarg — remove it,
            # the decorator provides its own llm callable
            kwargs.pop("llm", None)

            # Multi-run mode: run N times, check pass rate
            if runs > 1:
                passed_count = 0
                last_failures = []
                for run_idx in range(runs):
                    run_failures = _execute_single_run(
                        fn,
                        args,
                        kwargs,
                        assertions,
                        provider,
                        default_model,
                        default_system_prompt,
                        default_temperature,
                        default_max_tokens,
                    )
                    if run_failures is None:
                        passed_count += 1
                    else:
                        last_failures = run_failures

                pass_rate = passed_count / runs
                if pass_rate < min_pass_rate:
                    needed = int(min_pass_rate * runs)
                    msg = (
                        f"\n  ! Pass rate {passed_count}/{runs} "
                        f"({pass_rate:.0%}) below threshold "
                        f"({min_pass_rate:.0%}, need {needed}/{runs})"
                    )
                    for f in last_failures:
                        msg += f"\n  x [{f.severity.value.upper()}] {f.assertion_name}"
                        if f.reason:
                            msg += f"\n    Last failure: {f.reason}"
                    pytest.fail("LLM assertions failed:" + msg)

                if passed_count < runs:
                    import warnings

                    warnings.warn(
                        f"llmtest: {passed_count}/{runs} runs passed ({pass_rate:.0%})",
                        stacklevel=2,
                    )
                return

            # Single-run mode with optional retries
            for attempt in range(retries + 1):
                if attempt > 0 and retry_delay > 0:
                    time.sleep(retry_delay)

                # Resolve provider
                provider_instance = ProviderRegistry.get(provider or "openai")
                captured_outputs: list[LLMOutput] = []
                captured_inputs: list[LLMInput] = []

                def llm(
                    prompt: str | LLMInput,
                    *,
                    model: str | None = None,
                    system_prompt: str | None = None,
                    temperature: float | None = None,
                    max_tokens: int | None = None,
                    tools: list[dict] | None = None,
                    response_format: dict | type | None = None,
                ) -> LLMOutput:
                    if isinstance(prompt, LLMInput):
                        llm_input = prompt
                    else:
                        llm_input = LLMInput.simple(
                            prompt,
                            system_prompt=system_prompt or default_system_prompt,
                            model=model or default_model,
                            temperature=temperature
                            if temperature is not None
                            else default_temperature,
                            max_tokens=max_tokens or default_max_tokens,
                            tools=tools,
                            response_format=response_format,
                        )
                    captured_inputs.append(llm_input)

                    # Actually call the LLM
                    output = _call_provider_sync(provider_instance, llm_input)
                    captured_outputs.append(output)
                    return output

                # Run the test function — llm() calls are real now
                try:
                    fn(llm, *args, **kwargs)
                except Exception:
                    # Test body assertion (e.g. bare `assert`) failed
                    if attempt < retries:
                        continue
                    raise

                if not captured_outputs:
                    pytest.fail("Test function did not call llm()")

                # Run automatic assertions on the last output
                llm_output = captured_outputs[-1]

                # Build agent trace from tool_calls if any agent assertions exist
                has_agent_assertions = any(isinstance(a, AgentAssertion) for a in assertions)
                agent_trace = _build_trace_from_output(llm_output) if has_agent_assertions else None

                failures: list[AssertionResult] = []
                for assertion in assertions:
                    try:
                        if isinstance(assertion, AgentAssertion):
                            result = assertion.check_trace(agent_trace)
                        else:
                            result = assertion.check(llm_output)
                        if not result.passed:
                            failures.append(result)
                    except Exception as e:
                        failures.append(
                            AssertionResult(
                                assertion_name=getattr(assertion, "name", "unknown"),
                                passed=False,
                                severity=Severity.HIGH,
                                reason=f"Assertion raised: {e}",
                            )
                        )

                if failures:
                    if attempt < retries:
                        # Still have retries left — try again
                        continue

                    # Final attempt failed — report
                    failure_messages = []
                    if retries > 0:
                        failure_messages.append(f"\n  ! Failed after {retries + 1} attempts")
                    for f in failures:
                        msg = f"\n  x [{f.severity.value.upper()}] {f.assertion_name}"
                        if f.reason:
                            msg += f"\n    Reason: {f.reason}"
                        if f.expected:
                            msg += f"\n    Expected: {f.expected}"
                        if f.actual:
                            msg += f"\n    Actual: {f.actual}"
                        if f.suggestions:
                            msg += "\n    Suggestions:"
                            for s in f.suggestions:
                                msg += f"\n      -> {s}"
                        failure_messages.append(msg)

                    failure_messages.append(
                        f"\n  i  Latency: {llm_output.latency_ms:.0f}ms | "
                        f"Tokens: {llm_output.input_tokens} up {llm_output.output_tokens} down | "
                        f"Est. cost: ${llm_output.cost_estimate_usd:.5f}"
                    )

                    pytest.fail("LLM assertions failed:" + "".join(failure_messages))

                # All passed
                if attempt > 0:
                    import warnings

                    warnings.warn(
                        f"llmtest: passed on attempt {attempt + 1}/{retries + 1}",
                        stacklevel=2,
                    )
                return

        return wrapper

    return decorator


# ─────────────────────────────────────────────
# PYTEST HOOKS
# ─────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    """Register our custom markers."""
    config.addinivalue_line("markers", "llmtest: marks a test as an LLM test")
    config.addinivalue_line("markers", "llm_provider(name): specify the LLM provider to use")
    config.addinivalue_line("markers", "llm_tags(*tags): tag an LLM test for filtering")


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """
    Add a summary section to pytest output.
    Shows: total LLM calls, total cost, total tokens.
    """
    llm_reports = [
        r
        for r in terminalreporter.stats.get("failed", []) + terminalreporter.stats.get("passed", [])
        if r.keywords.get("llmtest")
    ]

    if not llm_reports:
        return

    terminalreporter.write_sep("=", "llmtest summary")
    terminalreporter.write_line(f"  LLM tests run: {len(llm_reports)}")
