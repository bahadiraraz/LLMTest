"""
Pytest plugin integration tests.

Tests the @llm_test decorator, llm fixture, retry logic,
and all assertion patterns — run with: pytest tests/test_pytest_plugin.py -v
"""

import json

import pytest
from pydantic import BaseModel

from llmtest import LLMInput, expect, llm_test

MODEL = "claude-haiku-4-5-20251001"
PROVIDER = "anthropic"


# ─────────────────────────────────────────────
# 1. Decorator: basic response
# ─────────────────────────────────────────────


@llm_test(
    expect.is_not_empty(),
    expect.contains("4"),
    provider=PROVIDER,
    model=MODEL,
    temperature=0.0,
)
def test_decorator_basic(llm):
    """Decorator pattern — assertions auto-checked after function returns."""
    output = llm("What is 2 + 2? Answer with just the number.")
    assert output.content.strip()  # manual assert also works


# ─────────────────────────────────────────────
# 2. Fixture: basic response
# ─────────────────────────────────────────────


@pytest.mark.llm_provider("anthropic")
def test_fixture_basic(llm):
    """Fixture pattern — no decorator, manual asserts only."""
    output = llm("What is 2 + 2? Answer with just the number.", model=MODEL, temperature=0.0)
    assert "4" in output.content
    assert output.latency_ms > 0
    assert output.input_tokens > 0


# ─────────────────────────────────────────────
# 3. Decorator: structured JSON output
# ─────────────────────────────────────────────


@llm_test(
    expect.valid_json(),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="Respond only with valid JSON. No markdown, no explanation.",
    temperature=0.0,
)
def test_decorator_json(llm):
    output = llm(
        "Extract person info: 'Alice is 30 and lives in Tokyo'. "
        'Return JSON: {"name": "...", "age": ..., "city": "..."}'
    )
    # Strip markdown fences if present
    raw = (
        output.content.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    data = json.loads(raw)
    assert data["name"] == "Alice"
    assert data["age"] == 30


# ─────────────────────────────────────────────
# 4. Decorator: Pydantic structured output
# ─────────────────────────────────────────────


class CityInfo(BaseModel):
    name: str
    country: str
    population: int


@llm_test(
    expect.structured_output(CityInfo),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="Respond only with valid JSON matching the exact schema. No markdown.",
    temperature=0.0,
)
def test_decorator_pydantic(llm):
    llm(
        'Extract city info as JSON: {"name": "...", "country": "...", "population": ...} '
        "for Paris, France, population ~2100000"
    )


# ─────────────────────────────────────────────
# 5. Decorator: cost + token assertions
# ─────────────────────────────────────────────


@llm_test(
    expect.cost_under(0.01),
    expect.token_count_under(200),
    provider=PROVIDER,
    model=MODEL,
    temperature=0.0,
)
def test_decorator_cost_and_tokens(llm):
    output = llm("Say 'hello' in 3 languages.")
    assert output.cost_estimate_usd > 0, "Cost should be tracked"


# ─────────────────────────────────────────────
# 6. Decorator: latency + length assertions
# ─────────────────────────────────────────────


@llm_test(
    expect.latency_under(10_000),
    expect.length_between(min_words=3, max_words=200),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="Write exactly 3 sentences.",
    temperature=0.0,
)
def test_decorator_latency_and_length(llm):
    llm("Write 3 sentences about Python.")


# ─────────────────────────────────────────────
# 7. Decorator: tool calling
# ─────────────────────────────────────────────

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
}


@llm_test(
    expect.latency_under(10_000),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="Use the add tool for calculations.",
    temperature=0.0,
)
def test_decorator_tool_calling(llm):
    output = llm("What is 42 + 58? Use the add tool.", tools=[ADD_TOOL])
    assert output.tool_calls, "Expected tool call"
    call = output.tool_calls[0]
    assert call["name"] == "add"
    args = call["arguments"]
    assert sorted([args["a"], args["b"]]) == [42, 58]


# ─────────────────────────────────────────────
# 8. Fixture: tool calling with multiple tools
# ─────────────────────────────────────────────


@pytest.mark.llm_provider("anthropic")
def test_fixture_tool_choice(llm):
    """Fixture pattern — LLM picks the right tool from 3 options."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Do a math calculation",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]
    output = llm(
        "What's the weather in Istanbul?",
        model=MODEL,
        system_prompt="Use the appropriate tool.",
        temperature=0.0,
        tools=tools,
    )
    assert output.tool_calls
    assert output.tool_calls[0]["name"] == "get_weather"
    assert "istanbul" in output.tool_calls[0]["arguments"].get("city", "").lower()


# ─────────────────────────────────────────────
# 9. Decorator: system prompt controls behavior
# ─────────────────────────────────────────────


@llm_test(
    expect.contains("paris", case_sensitive=False),
    expect.contains_any(["arr", "matey", "ahoy", "aye", "ye", "pirate"]),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="You are a pirate. Answer everything in pirate speak. Use 'arr', 'matey', etc.",
    temperature=0.5,
)
def test_decorator_system_prompt(llm):
    llm("What is the capital of France?")


# ─────────────────────────────────────────────
# 10. Decorator: retry on flaky assertion
# ─────────────────────────────────────────────


@llm_test(
    expect.is_not_empty(),
    provider=PROVIDER,
    model=MODEL,
    temperature=1.0,
    retries=3,
    retry_delay=0.5,
)
def test_decorator_retry(llm):
    """Retry: ask for heads/tails, assert heads. Retries up to 3 times."""
    output = llm("Flip a coin. Reply with ONLY 'heads' or 'tails'.")
    assert "heads" in output.content.lower(), f"Wanted heads, got: {output.content}"


# ─────────────────────────────────────────────
# 11. Fixture: retry with retry_if
# ─────────────────────────────────────────────


@pytest.mark.llm_provider("anthropic")
def test_fixture_retry_if(llm):
    """Fixture retry_if: retry until response contains 'python'."""
    output = llm(
        "Name a popular programming language in one word.",
        model=MODEL,
        temperature=1.0,
        retries=3,
        retry_delay=0.5,
        retry_if=lambda out: "python" not in out.content.lower(),
    )
    # After retries, we might or might not have python — but should have something
    assert output.content.strip(), "Should have a non-empty response"


# ─────────────────────────────────────────────
# 12. Decorator: composable assertions (AND/OR)
# ─────────────────────────────────────────────


@llm_test(
    expect.contains("paris", case_sensitive=False) & expect.not_contains("London"),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="Answer concisely in one sentence.",
    temperature=0.0,
)
def test_decorator_composable(llm):
    llm("What is the capital of France?")


# ─────────────────────────────────────────────
# 13. Fixture: multi-turn conversation
# ─────────────────────────────────────────────


@pytest.mark.llm_provider("anthropic")
def test_fixture_multi_turn(llm):
    """Multi-turn conversation via LLMInput."""
    inp = LLMInput(
        messages=[
            {"role": "system", "content": "You are a helpful math tutor. Be concise."},
            {"role": "user", "content": "What is 10 * 5?"},
            {"role": "assistant", "content": "50"},
            {"role": "user", "content": "Now divide that by 2."},
        ],
        model=MODEL,
        temperature=0.0,
    )
    output = llm(inp)
    assert "25" in output.content, f"Expected 25 in response, got: {output.content}"


# ─────────────────────────────────────────────
# 14. Parametrized tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "country,capital",
    [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Japan", "Tokyo"),
    ],
)
@llm_test(
    expect.is_not_empty(),
    provider=PROVIDER,
    model=MODEL,
    system_prompt="Answer with only the city name. Nothing else.",
    temperature=0.0,
)
def test_parametrized_capitals(llm, country, capital):
    output = llm(f"What is the capital of {country}?")
    assert capital.lower() in output.content.lower()
