"""
Core domain models — built on Pydantic.

Using Pydantic BaseModel gives us:
- Auto-validation at construction time
- JSON serialization / deserialization for free
- Schema generation for docs and structured output testing
- Type coercion and clean error messages
- Works with any LLM provider
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class Severity(str, Enum):
    """How bad is a failure? Used for CI gate decisions."""

    CRITICAL = "critical"  # Block the pipeline
    HIGH = "high"  # Warn + report
    MEDIUM = "medium"  # Report only
    LOW = "low"  # FYI


class LLMInput(BaseModel):
    """
    Represents a single interaction with an LLM.
    Keeping it simple: messages + optional config.
    Providers handle their own quirks.
    """

    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | type | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def simple(
        cls,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | type | None = None,
    ) -> LLMInput:
        """Shorthand for single-turn inputs.

        Usage:
            LLMInput.simple("What is 2+2?")
            LLMInput.simple("Capital of France?", model="gpt-5-mini", system_prompt="Be concise.")
            LLMInput.simple("Weather?", tools=[...], model="gpt-5-mini")
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return cls(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        )

    @classmethod
    def chat(
        cls,
        *messages: tuple[str, str] | dict[str, str],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | type | None = None,
    ) -> LLMInput:
        """Multi-turn conversation shorthand.

        Accepts tuples or dicts:
            LLMInput.chat(
                ("system", "You are helpful."),
                ("user", "Hi"),
                model="gpt-5-mini",
                tools=[{"type": "function", "function": {...}}],
            )
        """
        parsed = []
        for msg in messages:
            if isinstance(msg, dict):
                parsed.append(msg)
            else:
                role, content = msg
                parsed.append({"role": role, "content": content})
        return cls(
            messages=parsed,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        )


class LLMOutput(BaseModel):
    """
    Raw output from an LLM call.
    Tracks cost/latency so assertions can reason about them.
    """

    model_config = {"arbitrary_types_allowed": True}

    content: str
    model: str
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_estimate_usd(self) -> float:
        """
        Cost estimate. Uses exact pricing from provider if available,
        otherwise falls back to a rough estimate.
        """
        if "_exact_cost_usd" in self.raw:
            return self.raw["_exact_cost_usd"]
        return (self.input_tokens / 1_000_000) * 1.0 + (self.output_tokens / 1_000_000) * 3.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total token count (input + output)."""
        return self.input_tokens + self.output_tokens


class AgentStep(BaseModel):
    """Single step in an agent execution trace."""

    step_number: int = Field(ge=1)
    llm_output: LLMOutput | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    reasoning: str | None = None


class AgentTrace(BaseModel):
    """
    Full trace of agent execution.
    This is what makes llmtest different from RAG-only testing tools.
    """

    steps: list[AgentStep] = Field(default_factory=list)
    final_output: str = ""
    total_llm_calls: int = Field(0, ge=0)
    total_tool_calls: int = Field(0, ge=0)
    loop_detected: bool = False
    goal_achieved: bool | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_call_sequence(self) -> list[str]:
        """Ordered list of tool names called during execution."""
        return [step.tool_name for step in self.steps if step.tool_name]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unique_tools_used(self) -> list[str]:
        """Unique tools used during execution."""
        return list(dict.fromkeys(self.tool_call_sequence))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_latency_ms(self) -> float:
        """Sum of all LLM call latencies."""
        return sum(step.llm_output.latency_ms for step in self.steps if step.llm_output)


class AssertionResult(BaseModel):
    """Result of a single assertion check."""

    assertion_name: str
    passed: bool
    severity: Severity
    score: float | None = Field(None, ge=0.0, le=1.0)
    expected: Any | None = None
    actual: Any | None = None
    reason: str = ""
    suggestions: list[str] = Field(default_factory=list)


class TestResult(BaseModel):
    """The result of running a single test case."""

    test_id: str
    test_name: str
    status: TestStatus
    llm_input: LLMInput
    llm_output: LLMOutput | None = None
    assertion_results: list[AssertionResult] = Field(default_factory=list)
    agent_trace: AgentTrace | None = None
    duration_ms: float = 0.0
    tags: list[str] = Field(default_factory=list)
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_assertions(self) -> list[AssertionResult]:
        return [a for a in self.assertion_results if not a.passed]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocking_failures(self) -> list[AssertionResult]:
        """Failures that should block CI."""
        return [a for a in self.failed_assertions if a.severity == Severity.CRITICAL]


class TestSuiteResult(BaseModel):
    """Aggregated results for a full test run."""

    suite_name: str
    results: list[TestResult] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return len(self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> float:
        return sum(r.llm_output.cost_estimate_usd for r in self.results if r.llm_output)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ci_should_fail(self) -> bool:
        """Should this test run block a CI pipeline?"""
        return any(r.blocking_failures for r in self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all tests with output."""
        latencies = [r.llm_output.latency_ms for r in self.results if r.llm_output]
        return sum(latencies) / len(latencies) if latencies else 0.0
