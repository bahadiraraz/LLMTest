"""
Dataset support — Load test cases from YAML/JSON files.

Inspired by pydantic-evals' Dataset/Case pattern.
Enables data-driven testing without writing Python for each case.

Usage:

    # From YAML file
    dataset = TestDataset.from_yaml("tests/capitals.yml")
    suite = dataset.evaluate(provider)

    # From Python
    dataset = TestDataset(
        name="capitals",
        cases=[
            TestDataCase(
                name="france",
                prompt="Capital of France?",
                assertions=[expect.contains("Paris")],
            ),
        ],
    )

YAML format:
    name: capitals
    provider: openai
    model: gpt-5-mini
    system_prompt: "Answer concisely in one word."
    temperature: 0.0
    cases:
      - name: france
        prompt: "What is the capital of France?"
        expect_contains: ["Paris"]
        expect_not_contains: ["London"]
        max_latency_ms: 3000
        max_cost_usd: 0.01
      - name: germany
        prompt: "What is the capital of Germany?"
        model: gpt-5-nano  # override per-case
        expect_contains: ["Berlin"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmtest_core.assertions import (
    BaseAssertion,
    Contains,
    ContainsAll,
    ContainsAny,
    CostUnder,
    EndsWith,
    IsNotEmpty,
    LatencyUnder,
    LengthBetween,
    MatchesRegex,
    NotContains,
    OneOf,
    StartsWith,
    TokenCountUnder,
    ValidJSON,
)
from llmtest_core.models import LLMInput, TestSuiteResult
from llmtest_core.providers import BaseProvider, ProviderRegistry
from llmtest_core.runners import RunConfig, TestCase, TestRunner


class TestDataCase(BaseModel):
    """A single test case definition — can be loaded from YAML/JSON."""

    name: str
    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens_limit: int | None = None
    tags: list[str] = Field(default_factory=list)

    # Declarative assertions (for YAML/JSON)
    expect_contains: list[str] = Field(default_factory=list)
    expect_not_contains: list[str] = Field(default_factory=list)
    expect_contains_all: list[str] = Field(default_factory=list)
    expect_contains_any: list[str] = Field(default_factory=list)
    expect_starts_with: str | None = None
    expect_ends_with: str | None = None
    expect_matches_regex: str | None = None
    expect_one_of: list[str] = Field(default_factory=list)
    expect_valid_json: bool = False
    expect_not_empty: bool = True
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    min_words: int | None = None
    max_words: int | None = None

    # Programmatic assertions (for Python API)
    assertions: list[Any] = Field(default_factory=list, exclude=True)

    def build_assertions(self) -> list[BaseAssertion]:
        """Convert declarative fields into assertion objects."""
        result: list[BaseAssertion] = list(self.assertions)

        for text in self.expect_contains:
            result.append(Contains(text, case_sensitive=False))
        for text in self.expect_not_contains:
            result.append(NotContains(text))
        if self.expect_contains_all:
            result.append(ContainsAll(self.expect_contains_all, case_sensitive=False))
        if self.expect_contains_any:
            result.append(ContainsAny(self.expect_contains_any, case_sensitive=False))
        if self.expect_starts_with:
            result.append(StartsWith(self.expect_starts_with))
        if self.expect_ends_with:
            result.append(EndsWith(self.expect_ends_with))
        if self.expect_matches_regex:
            result.append(MatchesRegex(self.expect_matches_regex))
        if self.expect_one_of:
            result.append(OneOf(self.expect_one_of))
        if self.expect_valid_json:
            result.append(ValidJSON())
        if self.expect_not_empty:
            result.append(IsNotEmpty())
        if self.max_latency_ms is not None:
            result.append(LatencyUnder(self.max_latency_ms))
        if self.max_cost_usd is not None:
            result.append(CostUnder(self.max_cost_usd))
        if self.max_tokens is not None:
            result.append(TokenCountUnder(self.max_tokens))
        if self.min_words is not None or self.max_words is not None:
            result.append(
                LengthBetween(
                    self.min_words or 0,
                    self.max_words or 10_000,
                )
            )

        return result

    def to_test_case(self, index: int = 0) -> TestCase:
        """Convert to a TestCase for the runner."""
        return TestCase(
            test_id=f"{self.name}_{index}",
            name=self.name,
            input=LLMInput.simple(
                self.prompt,
                system_prompt=self.system_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens_limit,
            ),
            assertions=self.build_assertions(),
            tags=self.tags,
        )


class TestDataset(BaseModel):
    """
    A collection of test cases that can be loaded from YAML/JSON.

    Inspired by pydantic-evals Dataset pattern but optimized for LLM testing.
    """

    name: str = "llmtest"
    description: str = ""
    provider: str = "openai"
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    cases: list[TestDataCase] = Field(default_factory=list)
    config: RunConfig = Field(default_factory=RunConfig)

    # Dataset-level defaults
    default_max_latency_ms: float | None = None
    default_max_cost_usd: float | None = None

    def evaluate(
        self,
        provider: BaseProvider | None = None,
        config: RunConfig | None = None,
    ) -> TestSuiteResult:
        """Run all test cases and return results."""
        if provider is None:
            try:
                provider = ProviderRegistry.get(self.provider)
            except ValueError:
                provider = None

        test_cases = []
        for i, case in enumerate(self.cases):
            tc = case.to_test_case(index=i)

            # Apply dataset-level defaults
            if self.default_max_latency_ms and not case.max_latency_ms:
                tc.assertions.append(LatencyUnder(self.default_max_latency_ms))
            if self.default_max_cost_usd and not case.max_cost_usd:
                tc.assertions.append(CostUnder(self.default_max_cost_usd))

            if self.model and not case.model:
                tc.input.model = self.model
            if self.system_prompt and not case.system_prompt:
                # Prepend dataset-level system prompt
                if not any(m["role"] == "system" for m in tc.input.messages):
                    tc.input.messages.insert(0, {"role": "system", "content": self.system_prompt})
            if self.temperature is not None and case.temperature is None:
                tc.input.temperature = self.temperature

            test_cases.append(tc)

        runner = TestRunner(config=config or self.config)
        return runner.run_sync(test_cases, provider=provider)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TestDataset:
        """Load dataset from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> TestDataset:
        """Load dataset from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save dataset to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")

        path = Path(path)
        data = self.model_dump(exclude_defaults=True, exclude_none=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Save dataset to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=indent, exclude_defaults=True, exclude_none=True))

    def add_case(self, case: TestDataCase) -> None:
        """Add a test case to the dataset."""
        self.cases.append(case)
