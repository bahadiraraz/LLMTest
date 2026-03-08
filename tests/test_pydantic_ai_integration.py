"""
Real Pydantic AI integration tests with Anthropic.

Tests PydanticAIAdapter against the actual Claude API.
Run: ANTHROPIC_API_KEY=... pytest tests/test_pydantic_ai_integration.py -v
"""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core"))

from llmtest_core.assertions import (
    Contains,
    ContainsAny,
    IsNotEmpty,
    LatencyUnder,
    LengthBetween,
    StructuredOutput,
    TokenCountUnder,
)
from llmtest_core.integrations.pydantic_ai import PydanticAIAdapter
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

MODEL = "anthropic:claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────
# TEST 1: Basic text agent
# ─────────────────────────────────────────────


def test_basic_text_agent():
    agent = Agent(MODEL, system_prompt="Answer concisely.")
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("What is 2 + 2? Answer with just the number.")
    print(
        f"\n[basic] content={output.content!r}, model={output.model}, "
        f"latency={output.latency_ms:.0f}ms"
    )

    assert IsNotEmpty().check(output).passed
    assert "4" in output.content
    assert output.input_tokens > 0
    assert output.output_tokens > 0


# ─────────────────────────────────────────────
# TEST 2: Structured output (Pydantic model)
# ─────────────────────────────────────────────


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str


def test_structured_output():
    agent = Agent(
        MODEL,
        output_type=MovieReview,
        system_prompt="Extract movie review info accurately.",
    )
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("Review: Inception is a 9/10 masterpiece about dreams within dreams.")
    print(f"\n[structured] content={output.content[:200]!r}")

    result = StructuredOutput(MovieReview).check(output)
    assert result.passed, f"StructuredOutput failed: {result.reason}"

    # Parse and verify
    data = json.loads(output.content)
    assert data["title"] == "Inception"
    assert data["rating"] == 9


# ─────────────────────────────────────────────
# TEST 3: Agent with tools
# ─────────────────────────────────────────────


def test_agent_with_tools():
    agent = Agent(
        MODEL,
        system_prompt="Always use the add tool for math calculations. Never calculate mentally.",
    )

    @agent.tool_plain
    def add(a: float, b: float) -> str:
        """Add two numbers together."""
        return str(a + b)

    adapter = PydanticAIAdapter(agent)
    output = adapter.run_sync("What is 42 + 58?")

    print(f"\n[tools] content={output.content!r}, tool_calls={output.tool_calls}")

    assert "100" in output.content, f"Expected 100 in response, got: {output.content}"
    assert output.tool_calls, "Expected tool calls"
    assert any(tc["name"] == "add" for tc in output.tool_calls)


# ─────────────────────────────────────────────
# TEST 4: Agent with multiple tools — right choice
# ─────────────────────────────────────────────


def test_tool_selection():
    agent = Agent(
        MODEL,
        system_prompt="Use the appropriate tool. Do not answer without a tool.",
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny, 25°C in {city}"

    @agent.tool_plain
    def get_time(timezone: str) -> str:
        """Get the current time in a timezone."""
        return f"14:30 in {timezone}"

    adapter = PydanticAIAdapter(agent)
    output = adapter.run_sync("What's the weather in Istanbul?")

    print(f"\n[tool_select] tool_calls={output.tool_calls}")

    assert output.tool_calls, "Expected a tool call"
    weather_calls = [tc for tc in output.tool_calls if tc["name"] == "get_weather"]
    assert weather_calls, f"Expected get_weather, got: {[tc['name'] for tc in output.tool_calls]}"
    assert "istanbul" in weather_calls[0]["arguments"].get("city", "").lower()


# ─────────────────────────────────────────────
# TEST 5: Cost and token tracking
# ─────────────────────────────────────────────


def test_cost_and_tokens():
    agent = Agent(MODEL, system_prompt="Be brief.")
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("Say hello in 3 languages.")
    total = output.input_tokens + output.output_tokens
    print(f"\n[cost] tokens={output.input_tokens}+{output.output_tokens}={total}")

    assert TokenCountUnder(500).check(output).passed
    assert output.input_tokens > 0
    assert output.output_tokens > 0
    assert output.raw.get("requests", 0) >= 1


# ─────────────────────────────────────────────
# TEST 6: Latency and length
# ─────────────────────────────────────────────


def test_latency_and_length():
    agent = Agent(MODEL, system_prompt="Write exactly 3 sentences.")
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("Tell me about Python programming.")
    word_count = len(output.content.split())
    print(f"\n[length] words={word_count}, latency={output.latency_ms:.0f}ms")

    assert LatencyUnder(15_000).check(output).passed
    assert LengthBetween(5, 200).check(output).passed


# ─────────────────────────────────────────────
# TEST 7: Boolean output type
# ─────────────────────────────────────────────


def test_boolean_output():
    agent = Agent(MODEL, output_type=bool)
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("Is the sky blue?")
    print(f"\n[bool] content={output.content!r}")

    assert output.content in ("true", "false")
    assert output.raw["output_type"] == "bool"


# ─────────────────────────────────────────────
# TEST 8: Dependencies
# ─────────────────────────────────────────────


@dataclass
class UserCtx:
    name: str


def test_agent_with_deps():
    agent = Agent(
        MODEL,
        deps_type=UserCtx,
        system_prompt="Greet the user by name.",
    )

    @agent.instructions
    def add_name(ctx: RunContext[UserCtx]) -> str:
        return f"The user's name is {ctx.deps.name}."

    adapter = PydanticAIAdapter(agent, deps=UserCtx("Alice"))
    output = adapter.run_sync("Hello!")

    print(f"\n[deps] content={output.content!r}")
    assert "alice" in output.content.lower(), f"Expected 'Alice' in response: {output.content}"


# ─────────────────────────────────────────────
# TEST 9: Retry with adapter
# ─────────────────────────────────────────────


def test_retry_with_adapter():
    agent = Agent(MODEL, system_prompt="Be creative and random.")
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync(
        "Flip a coin. Reply with ONLY 'heads' or 'tails'.",
        retries=3,
        retry_delay=0.5,
        retry_if=lambda out: out.content.strip().lower() not in ("heads", "tails"),
    )
    print(f"\n[retry] content={output.content!r}")
    assert output.content.strip().lower() in ("heads", "tails")


# ─────────────────────────────────────────────
# TEST 10: System prompt controls behavior
# ─────────────────────────────────────────────


def test_system_prompt_behavior():
    agent = Agent(
        MODEL,
        system_prompt="You are a pirate. Use 'arr', 'matey', 'ahoy' etc.",
    )
    adapter = PydanticAIAdapter(agent)

    output = adapter.run_sync("What is the capital of France?")
    print(f"\n[pirate] content={output.content!r}")

    assert Contains("paris", case_sensitive=False).check(output).passed
    pirate_result = ContainsAny(
        ["arr", "matey", "ahoy", "aye", "ye", "pirate", "seas"],
        case_sensitive=False,
    ).check(output)
    assert pirate_result.passed, f"Expected pirate speak: {output.content}"


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_basic_text_agent,
        test_structured_output,
        test_agent_with_tools,
        test_tool_selection,
        test_cost_and_tokens,
        test_latency_and_length,
        test_boolean_output,
        test_agent_with_deps,
        test_retry_with_adapter,
        test_system_prompt_behavior,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASSED  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAILED  {name}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
