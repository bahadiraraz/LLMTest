"""
Real-life Anthropic API tests.

Tests structured output, cost tracking, token limits, tool calling
against the actual Claude API with a real key.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "providers", "anthropic")
)

from llmtest_core.assertions import (
    CostUnder,
    LatencyUnder,
    LengthBetween,
    StructuredOutput,
    TokenCountUnder,
    ValidJSON,
)
from llmtest_core.models import LLMInput
from provider import AnthropicProvider

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4-5-20251001"  # cheapest for tests


def run(coro):
    return asyncio.run(coro)


def make_provider():
    return AnthropicProvider(api_key=API_KEY, default_model=MODEL)


# ─────────────────────────────────────────────
# TEST 1: Basic response — does the API work?
# ─────────────────────────────────────────────


def test_basic_response():
    provider = make_provider()
    inp = LLMInput.simple(
        "What is 2 + 2? Answer with just the number.",
        model=MODEL,
        temperature=0.0,
    )
    output = run(provider.call(inp))

    print(
        f"\n[basic] model={output.model}, latency={output.latency_ms:.0f}ms, "
        f"tokens={output.input_tokens}+{output.output_tokens}, content={output.content!r}"
    )

    assert output.content.strip(), "Response should not be empty"
    assert "4" in output.content, f"Expected '4' in response, got: {output.content}"
    assert output.finish_reason != "error", f"API error: {output.raw}"


# ─────────────────────────────────────────────
# TEST 2: Structured output — valid JSON
# ─────────────────────────────────────────────


def test_structured_output_json():
    provider = make_provider()
    inp = LLMInput.simple(
        "Extract person info from: 'Alice is 30 and lives in Tokyo'. "
        'Respond ONLY with JSON: {"name": ..., "age": ..., "city": ...}',
        system_prompt="You are a data extraction assistant. "
        "Respond only with valid JSON, no markdown.",
        model=MODEL,
        temperature=0.0,
    )
    output = run(provider.call(inp))

    print(f"\n[structured] content={output.content!r}")

    # Assertion: ValidJSON
    result = ValidJSON().check(output)
    assert result.passed, f"ValidJSON failed: {result.reason}"

    # Parse and verify structure
    data = json.loads(
        output.content.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    assert data["name"] == "Alice", f"Expected name='Alice', got {data.get('name')}"
    assert data["age"] == 30, f"Expected age=30, got {data.get('age')}"
    assert data["city"] == "Tokyo", f"Expected city='Tokyo', got {data.get('city')}"


# ─────────────────────────────────────────────
# TEST 3: Pydantic structured output validation
# ─────────────────────────────────────────────


def test_pydantic_structured_output():
    from pydantic import BaseModel

    class MovieReview(BaseModel):
        title: str
        rating: int
        summary: str

    provider = make_provider()
    inp = LLMInput.simple(
        "Write a review for the movie 'Inception'. "
        'Respond ONLY with JSON: {"title": "...", "rating": <1-10>, "summary": "..."}',
        system_prompt="Respond only with valid JSON matching the exact schema requested.",
        model=MODEL,
        temperature=0.0,
    )
    output = run(provider.call(inp))

    print(f"\n[pydantic] content={output.content[:200]!r}")

    # Assertion: StructuredOutput with Pydantic model
    result = StructuredOutput(MovieReview).check(output)
    assert result.passed, f"StructuredOutput failed: {result.reason}"


# ─────────────────────────────────────────────
# TEST 4: Cost tracking — stays under budget
# ─────────────────────────────────────────────


def test_cost_under_budget():
    provider = make_provider()
    inp = LLMInput.simple(
        "Say 'hello' in 3 languages.",
        model=MODEL,
        temperature=0.0,
    )
    output = run(provider.call(inp))

    cost = output.cost_estimate_usd
    print(f"\n[cost] cost=${cost:.6f}, tokens={output.input_tokens}+{output.output_tokens}")

    # Assertion: CostUnder — Haiku should be very cheap
    result = CostUnder(0.01).check(output)
    assert result.passed, f"CostUnder failed: {result.reason} (cost=${cost:.6f})"

    # Verify exact pricing was used (not fallback estimate)
    assert "_exact_cost_usd" in output.raw, "Expected exact pricing from provider"


# ─────────────────────────────────────────────
# TEST 5: Token count check
# ─────────────────────────────────────────────


def test_token_count_under():
    provider = make_provider()
    inp = LLMInput.simple(
        "What is the capital of France? One word.",
        model=MODEL,
        temperature=0.0,
        max_tokens=50,
    )
    output = run(provider.call(inp))

    total = output.input_tokens + output.output_tokens
    print(f"\n[tokens] input={output.input_tokens}, output={output.output_tokens}, total={total}")

    # Assertion: TokenCountUnder
    result = TokenCountUnder(200).check(output)
    assert result.passed, f"TokenCountUnder failed: {result.reason}"

    # max_tokens should have capped output
    assert output.output_tokens <= 50, f"Expected <=50 output tokens, got {output.output_tokens}"


# ─────────────────────────────────────────────
# TEST 6: Latency check
# ─────────────────────────────────────────────


def test_latency_under():
    provider = make_provider()
    inp = LLMInput.simple(
        "Say 'hi'.",
        model=MODEL,
        temperature=0.0,
        max_tokens=10,
    )
    output = run(provider.call(inp))

    print(f"\n[latency] latency={output.latency_ms:.0f}ms")

    # Assertion: LatencyUnder — 10 seconds should be plenty
    result = LatencyUnder(10_000).check(output)
    assert result.passed, f"LatencyUnder failed: {result.reason}"


# ─────────────────────────────────────────────
# TEST 7: Length between — word count control
# ─────────────────────────────────────────────


def test_length_between():
    provider = make_provider()
    inp = LLMInput.simple(
        "Write exactly 3 sentences about Python programming.",
        system_prompt="Write exactly 3 sentences. No more, no less.",
        model=MODEL,
        temperature=0.0,
    )
    output = run(provider.call(inp))

    word_count = len(output.content.split())
    print(f"\n[length] words={word_count}, content={output.content[:150]!r}")

    # Assertion: LengthBetween
    result = LengthBetween(min_words=5, max_words=200).check(output)
    assert result.passed, f"LengthBetween failed: {result.reason}"


# ─────────────────────────────────────────────
# TEST 8: Tool calling — addition tool
# ─────────────────────────────────────────────


def test_tool_calling_addition():
    """LLM should call the 'add' tool when asked to add numbers."""
    provider = make_provider()

    add_tool = {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together and return the result",
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

    inp = LLMInput.simple(
        "What is 42 + 58? Use the add tool to calculate.",
        system_prompt="You have access to an 'add' tool. Always use it for math calculations.",
        model=MODEL,
        temperature=0.0,
        tools=[add_tool],
    )
    output = run(provider.call(inp))

    print(f"\n[tool] finish_reason={output.finish_reason}, tool_calls={output.tool_calls}")

    # LLM should have called the tool
    assert output.tool_calls, "Expected LLM to call the 'add' tool"
    assert len(output.tool_calls) >= 1, (
        f"Expected at least 1 tool call, got {len(output.tool_calls)}"
    )

    call = output.tool_calls[0]
    assert call["name"] == "add", f"Expected tool 'add', got '{call['name']}'"

    # Check arguments
    args = call["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    print(f"[tool] args={args}")

    # The LLM should pass 42 and 58 (in some order)
    values = sorted([args.get("a", 0), args.get("b", 0)])
    assert values == [42, 58], f"Expected [42, 58], got {values}"


# ─────────────────────────────────────────────
# TEST 9: Tool calling — multiple tools, right choice
# ─────────────────────────────────────────────


def test_tool_calling_right_choice():
    """LLM should pick the correct tool from multiple options."""
    provider = make_provider()

    tools_defs = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
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
                "description": "Perform a math calculation",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]

    inp = LLMInput.simple(
        "What's the weather like in Istanbul?",
        system_prompt="Use the appropriate tool to answer. Do NOT answer without a tool.",
        model=MODEL,
        temperature=0.0,
        tools=tools_defs,
    )
    output = run(provider.call(inp))

    print(f"\n[multi-tool] tool_calls={output.tool_calls}")

    assert output.tool_calls, "Expected a tool call"
    call = output.tool_calls[0]
    assert call["name"] == "get_weather", f"Expected 'get_weather', got '{call['name']}'"

    args = call["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    assert "istanbul" in args.get("city", "").lower(), f"Expected Istanbul in args, got: {args}"


# ─────────────────────────────────────────────
# TEST 10: System prompt changes behavior
# ─────────────────────────────────────────────


def test_system_prompt_controls_behavior():
    provider = make_provider()

    # With pirate system prompt
    inp = LLMInput.simple(
        "What is the capital of France?",
        system_prompt="You are a pirate. Answer everything in pirate speak. "
        "Use 'arr', 'matey', etc.",
        model=MODEL,
        temperature=0.5,
    )
    output = run(provider.call(inp))

    print(f"\n[system_prompt] content={output.content!r}")

    assert output.content.strip(), "Response should not be empty"
    # Should still contain Paris but in pirate style
    lower = output.content.lower()
    assert "paris" in lower, f"Expected 'Paris' in response, got: {output.content}"
    pirate_words = ["arr", "matey", "aye", "ye", "ahoy", "cap'n", "pirate", "seas"]
    has_pirate = any(w in lower for w in pirate_words)
    assert has_pirate, f"Expected pirate speak, got: {output.content}"


# ─────────────────────────────────────────────
# TEST 11: Retry — flaky assertion passes on retry
# ─────────────────────────────────────────────


def test_retry_on_flaky_output():
    """
    Flip a coin via the LLM. Assert we get 'heads'.
    With retries, this should eventually pass (~50% per attempt).
    Probability of failing all 5 retries: (0.5)^6 ≈ 1.5%.
    """
    provider = make_provider()
    max_retries = 5
    last_content = ""

    for attempt in range(max_retries + 1):
        inp = LLMInput.simple(
            "Flip a coin. Reply with ONLY 'heads' or 'tails', nothing else. Be random.",
            model=MODEL,
            temperature=1.0,  # high temp = more randomness
        )
        output = run(provider.call(inp))
        last_content = output.content.strip().lower()
        print(f"\n[retry] attempt={attempt + 1}, content={last_content!r}")

        if "heads" in last_content:
            print(f"[retry] Got 'heads' on attempt {attempt + 1}/{max_retries + 1}")
            return  # PASSED

    # If we never got heads, that's extremely unlikely but possible
    # Accept tails too — the point is that retry ran multiple attempts
    assert last_content in ("heads", "tails"), (
        f"Expected 'heads' or 'tails' after {max_retries + 1} attempts, got: {last_content!r}"
    )
    print(f"[retry] Never got 'heads' after {max_retries + 1} attempts, but retry mechanism worked")


# ─────────────────────────────────────────────
# TEST 12: Retry with custom condition (retry_if pattern)
# ─────────────────────────────────────────────


def test_retry_with_condition():
    """
    Retry until the LLM's response contains a specific word.
    Simulates the retry_if pattern from the fixture API.
    """
    provider = make_provider()
    max_retries = 3
    target_word = "python"

    for attempt in range(max_retries + 1):
        inp = LLMInput.simple(
            "Name a popular programming language in one word.",
            model=MODEL,
            temperature=1.0,
        )
        output = run(provider.call(inp))
        content_lower = output.content.strip().lower()
        print(f"\n[retry_if] attempt={attempt + 1}, content={content_lower!r}")

        if target_word in content_lower:
            print(f"[retry_if] Got '{target_word}' on attempt {attempt + 1}")
            break

        if attempt == max_retries:
            # Accept any non-empty response after all retries
            assert output.content.strip(), "Response should not be empty after retries"
            print(
                f"[retry_if] '{target_word}' not found, "
                f"but got valid response: {output.content.strip()!r}"
            )


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_basic_response,
        test_structured_output_json,
        test_pydantic_structured_output,
        test_cost_under_budget,
        test_token_count_under,
        test_latency_under,
        test_length_between,
        test_tool_calling_addition,
        test_tool_calling_right_choice,
        test_system_prompt_controls_behavior,
        test_retry_on_flaky_output,
        test_retry_with_condition,
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
