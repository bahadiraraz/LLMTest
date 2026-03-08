"""
Assertion system.

Design principle: Every assertion is a pure function.
No side effects, no LLM calls (unless explicitly opted into with LLMJudge).

This is the key differentiator: most assertions work WITHOUT calling an LLM.
That means: fast, cheap, deterministic, offline-capable.

Architecture: BaseAssertion (abstract) → concrete assertions
Assertions are composable: use & and | operators to combine them.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmtest_core.models import AgentTrace, AssertionResult, LLMOutput
from llmtest_core.models import AssertionResult, Severity


class BaseAssertion(ABC):
    """
    Abstract base for all assertions.

    Why ABC and not Protocol?
    We want assertions to be composable (& and | operators).
    ABC lets us attach that behavior to the base class cleanly.
    """

    severity: Severity = Severity.HIGH
    name: str = "base_assertion"

    def __init__(self, severity: Severity = Severity.HIGH):
        self.severity = severity

    @abstractmethod
    def check(self, output: LLMOutput) -> AssertionResult: ...

    def __and__(self, other: BaseAssertion) -> AllOf:
        return AllOf([self, other])

    def __or__(self, other: BaseAssertion) -> AnyOf:
        return AnyOf([self, other])


# ─────────────────────────────────────────────
# TEXT ASSERTIONS — Zero LLM dependency
# ─────────────────────────────────────────────


class Contains(BaseAssertion):
    """Output must contain the given substring."""

    name = "contains"

    def __init__(self, text: str, case_sensitive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.case_sensitive = case_sensitive

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content if self.case_sensitive else output.content.lower()
        target = self.text if self.case_sensitive else self.text.lower()
        passed = target in content
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"output to contain '{self.text}'",
            actual=output.content[:200],
            reason="" if passed else f"Expected text '{self.text}' not found in output.",
            suggestions=["Check if the prompt clearly asks for this content"] if not passed else [],
        )


class NotContains(BaseAssertion):
    name = "not_contains"

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self.text = text

    def check(self, output: LLMOutput) -> AssertionResult:
        passed = self.text not in output.content
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="" if passed else f"Forbidden text '{self.text}' was found in output.",
        )


class MatchesRegex(BaseAssertion):
    name = "matches_regex"

    def __init__(self, pattern: str, flags: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.compiled = re.compile(pattern, flags)

    def check(self, output: LLMOutput) -> AssertionResult:
        match = self.compiled.search(output.content)
        passed = match is not None
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"output to match pattern: {self.pattern}",
            actual=output.content[:200],
            reason="" if passed else f"Pattern '{self.pattern}' did not match output.",
        )


class ValidJSON(BaseAssertion):
    """
    Asserts output is valid JSON.
    Optionally validates against a JSON schema.
    """

    name = "valid_json"

    def __init__(self, schema: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema

    def check(self, output: LLMOutput) -> AssertionResult:
        # Strip markdown code blocks that models often add
        content = output.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            return AssertionResult(
                assertion_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"Output is not valid JSON: {e}",
                suggestions=[
                    "Add 'Respond only with valid JSON.' to your system prompt",
                    "Consider using structured outputs / response_format "
                    "if your provider supports it",
                ],
            )

        if self.schema:
            try:
                import jsonschema

                jsonschema.validate(parsed, self.schema)
            except jsonschema.ValidationError as e:
                return AssertionResult(
                    assertion_name=self.name,
                    passed=False,
                    severity=self.severity,
                    reason=f"JSON schema validation failed: {e.message}",
                )

        return AssertionResult(assertion_name=self.name, passed=True, severity=self.severity)


class LengthBetween(BaseAssertion):
    """Word count or character count within range."""

    name = "length_between"

    def __init__(self, min_words: int = 0, max_words: int = 10_000, **kwargs):
        super().__init__(**kwargs)
        self.min_words = min_words
        self.max_words = max_words

    def check(self, output: LLMOutput) -> AssertionResult:
        word_count = len(output.content.split())
        passed = self.min_words <= word_count <= self.max_words
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"{self.min_words}–{self.max_words} words",
            actual=f"{word_count} words",
            reason=""
            if passed
            else f"Word count {word_count} outside range [{self.min_words}, {self.max_words}]",
        )


# ─────────────────────────────────────────────
# PERFORMANCE ASSERTIONS — No LLM needed
# ─────────────────────────────────────────────


class LatencyUnder(BaseAssertion):
    """Response must arrive within N milliseconds."""

    name = "latency_under"

    def __init__(self, max_ms: float, **kwargs):
        super().__init__(**kwargs)
        self.max_ms = max_ms

    def check(self, output: LLMOutput) -> AssertionResult:
        passed = output.latency_ms <= self.max_ms
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"≤ {self.max_ms}ms",
            actual=f"{output.latency_ms:.0f}ms",
            reason=""
            if passed
            else f"Latency {output.latency_ms:.0f}ms exceeded limit of {self.max_ms}ms",
            suggestions=["Consider using a faster model", "Cache frequent prompts"]
            if not passed
            else [],
        )


class CostUnder(BaseAssertion):
    """Single call must cost less than N USD."""

    name = "cost_under"

    def __init__(self, max_usd: float, **kwargs):
        super().__init__(**kwargs)
        self.max_usd = max_usd

    def check(self, output: LLMOutput) -> AssertionResult:
        cost = output.cost_estimate_usd
        passed = cost <= self.max_usd
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"≤ ${self.max_usd:.4f}",
            actual=f"${cost:.4f}",
            reason=""
            if passed
            else f"Estimated cost ${cost:.4f} exceeded limit of ${self.max_usd:.4f}",
        )


# ─────────────────────────────────────────────
# AGENT ASSERTIONS
# ─────────────────────────────────────────────


class AgentAssertion(ABC):
    """Separate base for assertions that operate on AgentTrace, not LLMOutput."""

    severity: Severity = Severity.HIGH
    name: str = "agent_assertion"

    @abstractmethod
    def check_trace(self, trace: AgentTrace) -> AssertionResult: ...


class ToolCalled(AgentAssertion):
    """Assert a specific tool was (or wasn't) called."""

    name = "tool_called"

    def __init__(
        self,
        tool_name: str,
        times: int | None = None,
        min_times: int = 1,
        max_times: int | None = None,
    ):
        self.tool_name = tool_name
        self.times = times
        self.min_times = min_times
        self.max_times = max_times

    def check_trace(self, trace: AgentTrace) -> AssertionResult:
        actual_calls = trace.tool_call_sequence.count(self.tool_name)

        if self.times is not None:
            passed = actual_calls == self.times
            reason = (
                f"Tool '{self.tool_name}' called {actual_calls} times, "
                f"expected exactly {self.times}"
            )
        else:
            too_few = actual_calls < self.min_times
            too_many = self.max_times is not None and actual_calls > self.max_times
            passed = not too_few and not too_many
            reason = (
                f"Tool '{self.tool_name}' called {actual_calls} times "
                f"(expected {self.min_times}–{self.max_times or '∞'})"
            )

        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="" if passed else reason,
        )


class NoLoopDetected(AgentAssertion):
    """Agent must not enter an infinite tool-calling loop."""

    name = "no_loop"

    def check_trace(self, trace: AgentTrace) -> AssertionResult:
        return AssertionResult(
            assertion_name=self.name,
            passed=not trace.loop_detected,
            severity=Severity.CRITICAL,
            reason=""
            if not trace.loop_detected
            else "Agent loop detected — same tool called repeatedly without progress",
            suggestions=[
                "Add a max_iterations guard to your agent",
                "Check tool output parsing logic",
            ],
        )


class ToolCallOrder(AgentAssertion):
    """Assert tools were called in a specific sequence."""

    name = "tool_call_order"

    def __init__(self, expected_sequence: list[str]):
        self.expected_sequence = expected_sequence

    def check_trace(self, trace: AgentTrace) -> AssertionResult:
        actual = trace.tool_call_sequence
        passed = actual == self.expected_sequence
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=str(self.expected_sequence),
            actual=str(actual),
            reason="" if passed else "Tool call order mismatch",
        )


# ─────────────────────────────────────────────
# EXTENDED TEXT ASSERTIONS
# ─────────────────────────────────────────────


class StartsWith(BaseAssertion):
    """Output must start with the given prefix."""

    name = "starts_with"

    def __init__(self, prefix: str, case_sensitive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.case_sensitive = case_sensitive

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content if self.case_sensitive else output.content.lower()
        target = self.prefix if self.case_sensitive else self.prefix.lower()
        # Strip leading whitespace for robustness
        passed = content.lstrip().startswith(target)
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"output to start with '{self.prefix}'",
            actual=output.content[:100],
            reason="" if passed else f"Output does not start with '{self.prefix}'",
        )


class EndsWith(BaseAssertion):
    """Output must end with the given suffix."""

    name = "ends_with"

    def __init__(self, suffix: str, case_sensitive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix
        self.case_sensitive = case_sensitive

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content if self.case_sensitive else output.content.lower()
        target = self.suffix if self.case_sensitive else self.suffix.lower()
        passed = content.rstrip().endswith(target)
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"output to end with '{self.suffix}'",
            actual=output.content[-100:],
            reason="" if passed else f"Output does not end with '{self.suffix}'",
        )


class IsNotEmpty(BaseAssertion):
    """Output must not be empty or whitespace-only."""

    name = "is_not_empty"

    def check(self, output: LLMOutput) -> AssertionResult:
        passed = bool(output.content and output.content.strip())
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="" if passed else "Output is empty or whitespace-only",
            suggestions=[
                "Check if the LLM returned an empty response",
                "Verify your prompt is complete",
            ],
        )


class OneOf(BaseAssertion):
    """Output must be exactly one of the given options (after stripping whitespace)."""

    name = "one_of"

    def __init__(self, options: list[str], case_sensitive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.options = options
        self.case_sensitive = case_sensitive

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content.strip()
        if not self.case_sensitive:
            content = content.lower()
            options = [o.lower() for o in self.options]
        else:
            options = self.options
        passed = content in options
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"one of {self.options}",
            actual=output.content[:200],
            reason="" if passed else f"Output '{output.content[:50]}' is not one of {self.options}",
        )


class SimilarTo(BaseAssertion):
    """
    Output must be similar to a reference text.
    Uses word overlap (Jaccard similarity) — no ML dependency.
    """

    name = "similar_to"

    def __init__(self, reference: str, min_similarity: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.reference = reference
        self.min_similarity = min_similarity

    def check(self, output: LLMOutput) -> AssertionResult:
        score = self._jaccard_similarity(output.content, self.reference)
        passed = score >= self.min_similarity
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            score=score,
            expected=f"similarity ≥ {self.min_similarity:.2f}",
            actual=f"similarity = {score:.2f}",
            reason=""
            if passed
            else f"Similarity {score:.2f} below threshold {self.min_similarity:.2f}",
        )

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


class TokenCountUnder(BaseAssertion):
    """Total token usage must be under a limit."""

    name = "token_count_under"

    def __init__(self, max_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    def check(self, output: LLMOutput) -> AssertionResult:
        total = output.input_tokens + output.output_tokens
        passed = total <= self.max_tokens
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"≤ {self.max_tokens} tokens",
            actual=f"{total} tokens ({output.input_tokens}↑ {output.output_tokens}↓)",
            reason="" if passed else f"Token count {total} exceeded limit of {self.max_tokens}",
            suggestions=["Use a more concise prompt", "Set max_tokens on the LLM call"]
            if not passed
            else [],
        )


class ContainsAll(BaseAssertion):
    """Output must contain ALL of the given texts."""

    name = "contains_all"

    def __init__(self, texts: list[str], case_sensitive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.texts = texts
        self.case_sensitive = case_sensitive

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content if self.case_sensitive else output.content.lower()
        missing = []
        for text in self.texts:
            target = text if self.case_sensitive else text.lower()
            if target not in content:
                missing.append(text)
        passed = len(missing) == 0
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"output to contain all of {self.texts}",
            actual=f"missing: {missing}" if missing else "all found",
            reason="" if passed else f"Missing text(s): {missing}",
        )


class ContainsAny(BaseAssertion):
    """Output must contain at least one of the given texts."""

    name = "contains_any"

    def __init__(self, texts: list[str], case_sensitive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.texts = texts
        self.case_sensitive = case_sensitive

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content if self.case_sensitive else output.content.lower()
        found = []
        for text in self.texts:
            target = text if self.case_sensitive else text.lower()
            if target in content:
                found.append(text)
        passed = len(found) > 0
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            expected=f"output to contain any of {self.texts}",
            actual=f"found: {found}" if found else "none found",
            reason="" if passed else f"None of {self.texts} found in output",
        )


class StructuredOutput(BaseAssertion):
    """
    Validate output against a Pydantic model.
    Parses the output as JSON and validates it against the provided model.
    """

    name = "structured_output"

    def __init__(self, model_class: type, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class

    def check(self, output: LLMOutput) -> AssertionResult:
        content = output.content.strip()
        # Strip markdown code blocks
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return AssertionResult(
                assertion_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"Output is not valid JSON: {e}",
                suggestions=["Add 'Respond only with valid JSON.' to your system prompt"],
            )

        try:
            self.model_class(**data) if isinstance(data, dict) else self.model_class.model_validate(
                data
            )
        except Exception as e:
            return AssertionResult(
                assertion_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"Output does not match {self.model_class.__name__}: {e}",
                suggestions=["Ensure output matches the expected Pydantic schema"],
            )

        return AssertionResult(
            assertion_name=self.name,
            passed=True,
            severity=self.severity,
        )


# ─────────────────────────────────────────────
# COMPOSITE ASSERTIONS
# ─────────────────────────────────────────────


class AllOf(BaseAssertion):
    """All assertions must pass. Equivalent to AND."""

    name = "all_of"

    def __init__(self, assertions: list[BaseAssertion]):
        self.assertions = assertions

    def check(self, output: LLMOutput) -> AssertionResult:
        results = [a.check(output) for a in self.assertions]
        all_passed = all(r.passed for r in results)
        failed = [r for r in results if not r.passed]
        return AssertionResult(
            assertion_name=self.name,
            passed=all_passed,
            severity=max(
                (r.severity for r in failed),
                default=Severity.LOW,
                key=lambda s: list(Severity).index(s),
            ),
            reason="; ".join(r.reason for r in failed),
        )


class AnyOf(BaseAssertion):
    """At least one assertion must pass. Equivalent to OR."""

    name = "any_of"

    def __init__(self, assertions: list[BaseAssertion]):
        self.assertions = assertions

    def check(self, output: LLMOutput) -> AssertionResult:
        results = [a.check(output) for a in self.assertions]
        any_passed = any(r.passed for r in results)
        return AssertionResult(
            assertion_name=self.name,
            passed=any_passed,
            severity=self.severity,
            reason="" if any_passed else "None of the assertions passed",
        )


# ─────────────────────────────────────────────
# CUSTOM ASSERTION — Escape hatch
# ─────────────────────────────────────────────


class Custom(BaseAssertion):
    """
    Bring-your-own assertion logic.

    Usage:
        Custom(lambda out: "Paris" in out.content and len(out.content) < 500)
    """

    name = "custom"

    def __init__(
        self,
        fn: Callable[[LLMOutput], bool],
        reason_if_fail: str = "Custom assertion failed",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fn = fn
        self.reason_if_fail = reason_if_fail

    def check(self, output: LLMOutput) -> AssertionResult:
        try:
            passed = self.fn(output)
        except Exception as e:
            return AssertionResult(
                assertion_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"Custom assertion raised: {e}",
            )
        return AssertionResult(
            assertion_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="" if passed else self.reason_if_fail,
        )
