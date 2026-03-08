"""
LLM-as-a-Judge evaluator.

Uses the llmtest provider system to grade LLM outputs against a rubric.
This is the "escape hatch" when deterministic assertions aren't enough.

Usage:
    @llm_test(
        expect.llm_judge("Response must be helpful and accurate"),
        expect.latency_under(5000),
    )
    def test_helpfulness(llm):
        return llm("Explain quantum computing simply")

Design: Intentionally separate from the zero-LLM assertions.
Users opt into LLM-based evaluation explicitly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from llmtest_core.assertions import BaseAssertion
from llmtest_core.models import AssertionResult, LLMInput

if TYPE_CHECKING:
    from llmtest_core.models import LLMOutput
    from llmtest_core.providers import BaseProvider


class LLMJudge(BaseAssertion):
    """
    Use an LLM to judge the quality of output against a rubric.

    Uses the llmtest provider system (no external framework dependency).

    Args:
        rubric: Natural language description of what makes a good output.
        provider: Provider name (e.g. "openai", "anthropic") or BaseProvider instance.
        model: Model identifier for the judge (default: 'gpt-5-mini').
        threshold: Minimum score (0-10) to pass (default: 7).
        include_input: Whether to show the original prompt to the judge.
    """

    name = "llm_judge"

    def __init__(
        self,
        rubric: str,
        provider: str | BaseProvider = "openai",
        model: str = "gpt-5-mini",
        threshold: float = 7.0,
        include_input: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.rubric = rubric
        self._provider = provider
        self.model = model
        self.threshold = threshold
        self.include_input = include_input

    def _resolve_provider(self) -> BaseProvider:
        from llmtest_core.providers import BaseProvider, ProviderRegistry

        if isinstance(self._provider, BaseProvider):
            return self._provider
        provider = ProviderRegistry.get(self._provider)
        if provider is None:
            raise ValueError(
                f"Provider '{self._provider}' not found. "
                f"Available: {ProviderRegistry.available()}. "
                f"Install the provider package (e.g. pip install llmtest[openai])."
            )
        return provider

    def check(self, output: LLMOutput) -> AssertionResult:
        """Synchronous check — runs the async judge inline."""
        import asyncio

        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self._async_check(output))
            return result
        except Exception as e:
            return AssertionResult(
                assertion_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"LLM Judge failed: {e}",
                suggestions=["Check your API key and model availability"],
            )
        finally:
            loop.close()

    async def _async_check(self, output: LLMOutput) -> AssertionResult:
        provider = self._resolve_provider()

        system_prompt = (
            "You are an expert evaluator. Grade the given LLM output against the rubric.\n"
            "Score from 0-10 where 10 is perfect.\n"
            "Be strict but fair.\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            '{"score": <int 0-10>, "reason": "<brief explanation>"}'
        )

        prompt_parts = [f"## Rubric\n{self.rubric}"]
        prompt_parts.append(f"\n## Threshold\nScore must be >= {self.threshold} to pass.")
        prompt_parts.append(f"\n## LLM Output to Evaluate\n{output.content}")

        judge_input = LLMInput.chat(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(prompt_parts)},
            model=self.model,
            temperature=0.0,
        )

        judge_output = await provider.call(judge_input)

        # Parse JSON response
        content = judge_output.content.strip()
        if content.startswith("```"):
            import re

            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        try:
            grading = json.loads(content)
            score = int(grading.get("score", 0))
            reason = grading.get("reason", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            return AssertionResult(
                assertion_name=self.name,
                passed=False,
                severity=self.severity,
                reason=f"LLM Judge returned unparseable response: {judge_output.content[:200]}",
                suggestions=["Try a different judge model"],
            )

        return AssertionResult(
            assertion_name=self.name,
            passed=score >= self.threshold,
            severity=self.severity,
            score=score / 10.0,
            expected=f"score >= {self.threshold}/10",
            actual=f"score = {score}/10",
            reason=reason if score < self.threshold else "",
            suggestions=[
                "Refine your prompt to better match the rubric",
                "Consider lowering the threshold if the rubric is subjective",
            ]
            if score < self.threshold
            else [],
        )
