"""
Agent testing examples.

Testing that your LLM calls the RIGHT tools in the RIGHT order
and doesn't get stuck in a loop.
"""

from assertllm import LLMInput, expect, llm_test

# Tool definitions
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

SUMMARIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "summarize",
        "description": "Summarize a given text",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


# ─────────────────────────────────────────────
# EXAMPLE 1: Tool calling — does the LLM use the right tool?
# ─────────────────────────────────────────────


@llm_test(
    expect.is_not_empty(),
    expect.cost_under(0.05),
    model="gpt-5-mini",
    system_prompt="You are a research assistant. Use the web_search tool to find information.",
)
def test_research_agent_uses_search(llm):
    output = llm(
        "Find the latest news about open source AI",
        tools=[SEARCH_TOOL, SUMMARIZE_TOOL],
    )
    # Check that the LLM chose to call a tool
    assert output.tool_calls, "Expected LLM to call a tool"
    tool_names = [tc["name"] for tc in output.tool_calls]
    assert "web_search" in tool_names, f"Expected web_search tool, got: {tool_names}"


# ─────────────────────────────────────────────
# EXAMPLE 2: Tool NOT called (direct answer expected)
# ─────────────────────────────────────────────


@llm_test(
    expect.contains("4"),
    model="gpt-5-mini",
    system_prompt="Answer directly. Only use tools if you truly need external data.",
)
def test_simple_question_no_tool(llm):
    output = llm("What is 2+2?", tools=[SEARCH_TOOL])
    # Simple math should NOT trigger a tool call
    assert not output.tool_calls, "LLM should answer directly without tools"


# ─────────────────────────────────────────────
# EXAMPLE 3: Multi-turn conversation
# ─────────────────────────────────────────────


@llm_test(
    expect.is_not_empty(),
    expect.latency_under(10_000),
)
def test_booking_flow(llm):
    """Multi-turn: agent should maintain context across messages."""
    inp = LLMInput(
        messages=[
            {"role": "system", "content": "You are a booking assistant. Confirm bookings clearly."},
            {"role": "user", "content": "I want to book a meeting"},
            {"role": "assistant", "content": "Sure! What date and time works for you?"},
            {"role": "user", "content": "Tomorrow at 3pm"},
        ],
        model="gpt-5-mini",
    )
    output = llm(inp)
    assert any(
        word in output.content.lower() for word in ["confirmed", "booked", "3pm", "tomorrow"]
    )


# ─────────────────────────────────────────────
# EXAMPLE 4: Budget guard — cost and token limits
# ─────────────────────────────────────────────


@llm_test(
    expect.cost_under(0.01),
    expect.token_count_under(1000),
    expect.latency_under(5000),
    model="gpt-5-nano",
    system_prompt="Be extremely brief.",
)
def test_agent_stays_within_budget(llm):
    output = llm("Summarize quantum computing in one sentence.")
    assert output.total_tokens < 1000


# ─────────────────────────────────────────────
# EXAMPLE 5: Fixture pattern for tool testing
# ─────────────────────────────────────────────


def test_tool_arguments_with_fixture(llm):
    """Use fixture pattern to inspect tool call arguments."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to someone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        }
    ]
    output = llm(
        "Send an email to bob@example.com saying the meeting is at 3pm",
        model="gpt-5-mini",
        system_prompt="Use the send_email tool to send emails.",
        tools=tools,
    )
    assert output.tool_calls, "Expected send_email tool call"
    call = output.tool_calls[0]
    assert call["name"] == "send_email"
    # Inspect the arguments the LLM chose
    import json

    args = (
        json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
    )
    assert "bob@example.com" in args.get("to", "")
