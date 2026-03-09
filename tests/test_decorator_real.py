"""
Real API test using the @llm_test decorator pattern.
Tests exactly what the landing page examples show.
"""

import os
import sys

# Add package paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "pytest-plugin"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "providers", "anthropic")
)

# Load .env
from pathlib import Path

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

from llmtest_core.assertions import Contains, CostUnder, StructuredOutput, ValidJSON
from llmtest_core.models import LLMInput

# Force load anthropic provider
from provider import AnthropicProvider

MODEL = "claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────
# TEST 1: Landing page "JSON" tab example
# valid_json with schema
# ─────────────────────────────────────────────


def test_json_with_schema():
    """
    Landing page example:
        person_schema = {"type": "object", "required": ["name", "age"]}
        @llm_test(expect.valid_json(schema=person_schema), ...)
    """
    import asyncio

    provider = AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_model=MODEL,
    )

    person_schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    inp = LLMInput.simple(
        "Return a person's name and age as JSON",
        system_prompt="Respond only with valid JSON, no markdown, no explanation.",
        model=MODEL,
        temperature=0.0,
    )

    output = asyncio.run(provider.call(inp))
    print(f"\n[json_schema] content={output.content!r}")
    tokens = f"{output.input_tokens}+{output.output_tokens}"
    print(f"[json_schema] tokens={tokens}, latency={output.latency_ms:.0f}ms")

    # Test ValidJSON without schema
    result1 = ValidJSON().check(output)
    print(f"[json_schema] ValidJSON (no schema): passed={result1.passed}")
    assert result1.passed, f"ValidJSON failed: {result1.reason}"

    # Test ValidJSON WITH schema (this is what the landing page shows)
    result2 = ValidJSON(schema=person_schema).check(output)
    print(
        f"[json_schema] ValidJSON (with schema): passed={result2.passed}, reason={result2.reason}"
    )
    assert result2.passed, f"ValidJSON with schema failed: {result2.reason}"

    print("  PASSED  test_json_with_schema")


# ─────────────────────────────────────────────
# TEST 2: Landing page "Structured" tab example
# structured_output with Pydantic
# ─────────────────────────────────────────────


def test_structured_pydantic():
    """
    Landing page example:
        class CityInfo(BaseModel):
            name: str
            population: int
        @llm_test(expect.structured_output(CityInfo), ...)
    """
    import asyncio

    from pydantic import BaseModel

    class CityInfo(BaseModel):
        name: str
        population: int

    provider = AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_model=MODEL,
    )

    inp = LLMInput.simple(
        "Return Tokyo info as JSON",
        system_prompt="Respond only with valid JSON. No markdown.",
        model=MODEL,
        temperature=0.0,
    )

    output = asyncio.run(provider.call(inp))
    print(f"\n[structured] content={output.content!r}")

    result = StructuredOutput(CityInfo).check(output)
    print(
        f"[structured] StructuredOutput(CityInfo): passed={result.passed}, reason={result.reason}"
    )
    assert result.passed, f"StructuredOutput failed: {result.reason}"

    print("  PASSED  test_structured_pydantic")


# ─────────────────────────────────────────────
# TEST 3: Landing page "Basic" tab example
# contains + cost_under
# ─────────────────────────────────────────────


def test_basic_contains():
    """
    Landing page example:
        @llm_test(expect.contains("Paris"), expect.cost_under(0.01), ...)
    """
    import asyncio

    provider = AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_model=MODEL,
    )

    inp = LLMInput.simple(
        "What is the capital of France?",
        model=MODEL,
        temperature=0.0,
    )

    output = asyncio.run(provider.call(inp))
    print(f"\n[basic] content={output.content!r}")

    result1 = Contains("Paris", case_sensitive=False).check(output)
    print(f"[basic] Contains('Paris'): passed={result1.passed}")
    assert result1.passed, f"Contains failed: {result1.reason}"

    result2 = CostUnder(0.01).check(output)
    print(f"[basic] CostUnder(0.01): passed={result2.passed}")
    assert result2.passed, f"CostUnder failed: {result2.reason}"

    print("  PASSED  test_basic_contains")


# ─────────────────────────────────────────────
# TEST 4: Full decorator test via pytest simulation
# This is the EXACT landing page pattern
# ─────────────────────────────────────────────


def test_decorator_pattern():
    """
    Simulates the decorator pattern without actually running pytest.
    Tests that the llm() callable + assertion checking works end-to-end.
    """
    import asyncio

    provider = AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_model=MODEL,
    )

    person_schema = {
        "type": "object",
        "required": ["name", "age"],
    }

    assertions = [ValidJSON(schema=person_schema)]
    captured_outputs = []

    def llm(prompt, *, model=None, system_prompt=None, temperature=None, **kw):
        inp = LLMInput.simple(
            prompt,
            system_prompt=system_prompt or "Respond only with valid JSON, no markdown.",
            model=model or MODEL,
            temperature=temperature or 0.0,
        )
        output = asyncio.run(provider.call(inp))
        captured_outputs.append(output)
        return output

    # This is exactly what the user writes:
    def test_json_response(llm):
        llm("Return a person's name and age as JSON")

    # Execute
    test_json_response(llm)

    # Check assertions (this is what the decorator does automatically)
    output = captured_outputs[-1]
    print(f"\n[decorator] content={output.content!r}")

    for assertion in assertions:
        result = assertion.check(output)
        print(f"[decorator] {assertion.name}: passed={result.passed}, reason={result.reason}")
        assert result.passed, f"Assertion {assertion.name} failed: {result.reason}"

    print("  PASSED  test_decorator_pattern")


if __name__ == "__main__":
    tests = [
        test_json_with_schema,
        test_structured_pydantic,
        test_basic_contains,
        test_decorator_pattern,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED  {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
