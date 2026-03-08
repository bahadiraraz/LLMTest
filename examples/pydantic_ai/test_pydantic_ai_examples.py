"""
Pydantic AI integration examples.

Test your Pydantic AI agents with llmtest assertions.
All the assertions you know (contains, cost_under, tool_called, etc.)
work seamlessly with Pydantic AI agents via PydanticAIAdapter.

Run: pytest examples/pydantic_ai/test_pydantic_ai_examples.py -v
"""

from dataclasses import dataclass

from llmtest_core.integrations.pydantic_ai import PydanticAIAdapter
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from llmtest import expect

# ─────────────────────────────────────────────
# EXAMPLE 1: Basic agent — text output
# ─────────────────────────────────────────────

basic_agent = Agent(
    "openai:gpt-5-mini",
    system_prompt="Answer concisely in one sentence.",
)


def test_basic_agent():
    adapter = PydanticAIAdapter(basic_agent)
    output = adapter.run_sync("What is the capital of France?")

    assert expect.contains("Paris").check(output).passed
    assert expect.is_not_empty().check(output).passed
    assert expect.latency_under(10_000).check(output).passed


# ─────────────────────────────────────────────
# EXAMPLE 2: Structured output — Pydantic model
# ─────────────────────────────────────────────


class CityInfo(BaseModel):
    name: str
    country: str
    population: int


structured_agent = Agent(
    "openai:gpt-5-mini",
    output_type=CityInfo,
    system_prompt="Extract city information accurately.",
)


def test_structured_output_agent():
    adapter = PydanticAIAdapter(structured_agent)
    output = adapter.run_sync("Tell me about Paris, France (population ~2.1 million)")

    # Content is JSON from Pydantic model
    assert expect.structured_output(CityInfo).check(output).passed
    assert expect.cost_under(0.01).check(output).passed


# ─────────────────────────────────────────────
# EXAMPLE 3: Agent with tools
# ─────────────────────────────────────────────

tool_agent = Agent(
    "openai:gpt-5-mini",
    system_prompt="Use the calculator tool for math. Always use the tool.",
)


@tool_agent.tool_plain
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # noqa: S307 — example only
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def test_agent_with_tools():
    adapter = PydanticAIAdapter(tool_agent)
    output = adapter.run_sync("What is 42 * 58?")

    assert "2436" in output.content
    # Tool should have been called
    assert output.tool_calls, "Expected calculator tool to be called"
    assert any(tc["name"] == "calculator" for tc in output.tool_calls)


# ─────────────────────────────────────────────
# EXAMPLE 4: Agent with dependencies
# ─────────────────────────────────────────────


@dataclass
class UserContext:
    user_name: str
    language: str


deps_agent = Agent(
    "openai:gpt-5-mini",
    deps_type=UserContext,
    system_prompt="Greet the user by name in their preferred language.",
)


@deps_agent.instructions
def personalize(ctx: RunContext[UserContext]) -> str:
    return f"The user's name is {ctx.deps.user_name}. Speak in {ctx.deps.language}."


def test_agent_with_deps():
    adapter = PydanticAIAdapter(deps_agent, deps=UserContext("Alice", "English"))
    output = adapter.run_sync("Hello!")

    assert expect.contains("Alice", case_sensitive=False).check(output).passed


# ─────────────────────────────────────────────
# EXAMPLE 5: Retry on flaky output
# ─────────────────────────────────────────────


def test_agent_with_retry():
    """Retry until the agent mentions a specific keyword."""
    adapter = PydanticAIAdapter(basic_agent)
    output = adapter.run_sync(
        "Name a European capital city.",
        retries=3,
        retry_if=lambda out: "paris" not in out.content.lower(),
    )
    # After retries, we should have some valid content
    assert output.content.strip(), "Should have a non-empty response"


# ─────────────────────────────────────────────
# EXAMPLE 6: Multi-turn conversation
# ─────────────────────────────────────────────


def test_multi_turn_conversation():
    adapter = PydanticAIAdapter(basic_agent)

    # First turn
    output1 = adapter.run_sync("My name is Bob.")

    # Second turn — pass message history
    adapter.run_sync(
        "What is my name?",
        message_history=output1.raw.get("_messages", []),
    )
    # Note: message_history from pydantic_ai needs the actual message objects
    # This is a simplified example; see pydantic_ai docs for full multi-turn


# ─────────────────────────────────────────────
# EXAMPLE 7: Model override for testing
# ─────────────────────────────────────────────


def test_model_override():
    """Use a cheaper model for testing, even if agent defaults to expensive one."""
    expensive_agent = Agent(
        "openai:gpt-5",
        system_prompt="Be concise.",
    )
    # Override to cheaper model for tests
    adapter = PydanticAIAdapter(expensive_agent, model="openai:gpt-5-mini")
    output = adapter.run_sync("Say hello")
    assert output.content.strip()


# ─────────────────────────────────────────────
# EXAMPLE 8: Boolean output agent
# ─────────────────────────────────────────────

bool_agent = Agent(
    "openai:gpt-5-mini",
    output_type=bool,
    system_prompt="Answer yes/no questions. Return true or false.",
)


def test_boolean_output():
    adapter = PydanticAIAdapter(bool_agent)
    output = adapter.run_sync("Is Paris the capital of France?")
    assert output.content in ("true", "false")
