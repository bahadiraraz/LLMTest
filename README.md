# llmtest

**The pytest for LLMs.** Test your AI outputs like you test your code.
Built on **Pydantic** for type safety, validation, and JSON serialization.

[![CI](https://github.com/bahadiraraz/llmtest/actions/workflows/ci.yml/badge.svg)](https://github.com/bahadiraraz/llmtest/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/llmtest.svg)](https://badge.fury.io/py/llmtest)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Installation

```bash
pip install llmtest[openai]       # OpenAI (GPT-5.x, GPT-4.1, o3, o4-mini)
pip install llmtest[anthropic]    # Anthropic (Claude Opus 4.6, Sonnet 4.6, Haiku 4.5)
pip install llmtest[ollama]       # Local (Ollama — no API key)
pip install llmtest[pydantic-ai]  # Pydantic AI agents
pip install llmtest[all]          # Everything
```

<details>
<summary><strong>uv</strong></summary>

```bash
uv add llmtest[openai]
# or
uv add llmtest[anthropic]
uv add llmtest[all]
```
</details>

<details>
<summary><strong>PDM</strong></summary>

```bash
pdm add llmtest[openai]
# or
pdm add llmtest[anthropic]
pdm add llmtest[all]
```
</details>

<details>
<summary><strong>Poetry</strong></summary>

```bash
poetry add llmtest[openai]
# or
poetry add llmtest[anthropic]
poetry add llmtest[all]
```
</details>

<details>
<summary><strong>Rye</strong></summary>

```bash
rye add llmtest --features openai
# or
rye add llmtest --features anthropic
rye add llmtest --features all
```
</details>

<details>
<summary><strong>Conda + pip</strong></summary>

```bash
conda create -n llmtest python=3.12
conda activate llmtest
pip install llmtest[openai]
```
</details>

---

## The problem

Every team building AI products writes the same ad-hoc evaluation code.
No standards. No CI integration. No way to know if your last prompt change broke something.

```python
# What every AI team is doing today
response = openai.chat.completions.create(...)
assert "Paris" in response.choices[0].message.content  # works but...
# - No latency checks
# - No cost tracking
# - No agent trace validation
# - No structured output validation
# - Breaks silently in CI
```

## The solution

```python
from llmtest import expect, llm_test

@llm_test(
    expect.contains("Paris"),
    expect.latency_under(2000),       # ms
    expect.cost_under(0.001),         # USD
    provider="openai",                # or "anthropic", "ollama"
    model="gpt-5-mini",              # any model your provider supports
    system_prompt="Answer concisely.",
    temperature=0.0,
)
def test_capital_of_france(llm):
    output = llm("What is the capital of France?")
    # output is a real LLMOutput — inspect it freely
    assert "Paris" in output.content
```

```
$ pytest test_capitals.py -v

test_capitals.py::test_capital_of_france PASSED    [  0.8s]

────────── llmtest summary ──────────
  LLM tests: 1 passed
  Total cost: $0.000023
  Avg latency: 823ms
```

---

## Features

### Zero LLM calls for most assertions
Most assertions are **deterministic and instant** -- no paying an LLM to check if your output contains a word.

### Built on Pydantic
All models use `pydantic.BaseModel` -- auto-validation, JSON serialization, schema generation.

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

@llm_test(
    expect.structured_output(CityInfo),
    expect.cost_under(0.01),
    model="gpt-5-mini",
    system_prompt="Respond only with valid JSON matching the requested schema.",
)
def test_structured_extraction(llm):
    output = llm("Extract city info for Paris as JSON")
    import json
    data = json.loads(output.content)
    assert data["name"] == "Paris"
```

### Tool calling
```python
@llm_test(
    expect.is_not_empty(),
    model="gpt-5-mini",
    system_prompt="Use tools when needed.",
)
def test_tool_calling(llm):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]
    output = llm("Weather in Paris?", tools=tools)
    assert output.tool_calls
    assert output.tool_calls[0]["name"] == "get_weather"
```

### pytest fixture (no decorator)
```python
def test_with_fixture(llm):
    """Use the llm fixture — standard pytest style, no decorator needed."""
    output = llm("Say hello", model="gpt-5-mini")
    assert "hello" in output.content.lower()
    assert output.latency_ms < 5000
    assert output.cost_estimate_usd < 0.01
```

### Per-call model override
```python
@llm_test(
    expect.is_not_empty(),
    provider="anthropic",
    model="claude-sonnet-4-6-20260218",
)
def test_with_override(llm):
    output = llm("Explain gravity", model="claude-haiku-4-5-20251001", temperature=0.0)
    assert output.content
```

### 22+ built-in assertions
```python
# Text
expect.contains("Paris")              # Substring match
expect.not_contains("London")         # Forbidden text
expect.matches_regex(r"\d+")          # Regex pattern
expect.valid_json(schema={...})       # JSON + schema validation
expect.length_between(10, 100)        # Word count range
expect.starts_with("The")             # Prefix check
expect.ends_with(".")                 # Suffix check
expect.is_not_empty()                 # Non-empty check
expect.one_of(["yes", "no"])          # Exact match from options
expect.similar_to("reference", 0.7)   # Word overlap similarity
expect.contains_all(["A", "B"])       # Must contain all
expect.contains_any(["A", "B"])       # Must contain at least one
expect.structured_output(MyModel)     # Pydantic model validation

# Performance
expect.latency_under(2000)            # Max latency in ms
expect.cost_under(0.01)               # Max cost in USD
expect.token_count_under(500)         # Max total tokens

# Agent
expect.tool_called("search", times=1) # Tool usage
expect.no_loop()                       # No infinite loops
expect.tool_order(["a", "b"])          # Tool call sequence

# Composable
expect.all_of(a, b, c)               # AND
expect.any_of(a, b)                   # OR
expect.custom(lambda out: ...)        # Custom logic
```

### Composable with operators
```python
@llm_test(
    expect.contains("Paris") & expect.not_contains("London"),   # AND
    expect.contains("capital") | expect.contains("city"),       # OR
)
def test_short_capital_answer(llm): ...
```

### Retry -- handle non-deterministic LLM outputs
LLMs are non-deterministic. A test that fails once may pass on the next try.
llmtest has built-in retry support at both the decorator and fixture level.

```python
# Decorator: retry the whole test up to 3 times
@llm_test(
    expect.contains("Paris"),
    model="gpt-5-mini",
    retries=3,           # retry up to 3 times on failure
    retry_delay=0.5,     # wait 0.5s between retries
)
def test_capital_with_retry(llm):
    output = llm("What is the capital of France?")
    assert "Paris" in output.content

# Fixture: retry individual LLM calls with a condition
def test_with_retry_condition(llm):
    output = llm(
        "Name a European capital",
        model="gpt-5-mini",
        retries=3,
        retry_if=lambda out: "Paris" not in out.content,
    )
    assert output.content  # will have retried until "Paris" appeared (or 3 tries)
```

### Pydantic AI integration
Already using [Pydantic AI](https://ai.pydantic.dev)? Test your agents with llmtest assertions.

```bash
pip install llmtest[pydantic-ai]
```

```python
from pydantic_ai import Agent
from llmtest.integrations.pydantic_ai import PydanticAIAdapter
from llmtest import expect

# Your existing Pydantic AI agent
agent = Agent('openai:gpt-5-mini', system_prompt='Be concise.')

# Wrap it for testing
adapter = PydanticAIAdapter(agent)

def test_my_agent():
    output = adapter.run_sync("What is the capital of France?")

    # All llmtest assertions work
    assert expect.contains("Paris").check(output).passed
    assert expect.latency_under(5000).check(output).passed
    assert expect.cost_under(0.01).check(output).passed

def test_agent_tools():
    output = adapter.run_sync("Calculate 42 + 58")
    assert output.tool_calls  # inspect tool usage
    assert output.tool_calls[0]["name"] == "calculator"

def test_structured_output():
    from pydantic import BaseModel

    class CityInfo(BaseModel):
        name: str
        country: str

    agent = Agent('openai:gpt-5-mini', output_type=CityInfo)
    adapter = PydanticAIAdapter(agent)
    output = adapter.run_sync("Tell me about Paris")
    assert expect.structured_output(CityInfo).check(output).passed
```

Features: retry support, deps pass-through, model override, multi-turn, async (`adapter.run()`).

### CI/CD native
```yaml
# .github/workflows/llm-tests.yml
- name: Run LLM tests
  run: pytest -m llmtest --tb=short
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Reporters for CI
```python
from llmtest_core.reporters import JSONReporter, JUnitReporter

# JSON for dashboards
reporter = JSONReporter()
print(reporter.render(suite_result))

# JUnit XML for GitHub Actions / Jenkins
reporter = JUnitReporter()
with open("results.xml", "w") as f:
    f.write(reporter.render(suite_result))
```

---

## Failure messages that actually help

```
FAILED test_agent.py::test_research_agent

  x [HIGH] tool_called
    Reason: Tool 'web_search' called 3 times, expected exactly 1
    Suggestions:
      -> Add a max_iterations guard to your agent
      -> Check if tool output is being parsed correctly

  x [MEDIUM] cost_under
    Reason: Estimated cost $0.0312 exceeded limit of $0.0100

  i  Latency: 4823ms | Tokens: 1204 up 892 down | Est. cost: $0.0312
```

---

## Quick start

```bash
pip install llmtest[openai]
export OPENAI_API_KEY=sk-...
```

```python
# test_my_llm.py
from llmtest import expect, llm_test

@llm_test(
    expect.is_not_empty(),
    expect.contains("hello", case_sensitive=False),
    expect.latency_under(3000),
    model="gpt-5-mini",                      # pick your model
    system_prompt="You are a friendly bot.",  # optional system prompt
)
def test_greeting(llm):
    output = llm("Say hello")
    assert output.content  # real LLMOutput — inspect freely
```

```bash
pytest test_my_llm.py -v
```

### CLI

```bash
llmtest init            # Create a starter test file
llmtest run tests/      # Run LLM tests
llmtest providers       # List available providers
```

---

## Architecture

```
llmtest/
  packages/
    core/               # Pydantic models, assertions, runners (zero LLM dependency)
    pytest-plugin/      # @llm_test decorator, fixtures, hooks
    cli/                # CLI commands (llmtest run, init, providers)
    providers/
      openai/           # OpenAI provider (GPT-5.x, GPT-4.1, o3, o4-mini)
      anthropic/        # Anthropic provider (Claude Opus 4.6, Sonnet 4.6, Haiku 4.5)
      ollama/           # Ollama provider (local, free)
  tests/                # Unit tests (100% offline, no LLM calls)
  examples/
    basic/              # Getting started examples
    agent/              # Agent testing examples
    pydantic_ai/        # Pydantic AI integration examples
    ci/                 # CI/CD workflow examples
```

---

## Contributing

```bash
git clone https://github.com/bahadiraraz/llmtest
cd llmtest
```

<details>
<summary><strong>uv (recommended)</strong></summary>

```bash
uv sync --all-extras
uv run pytest tests/ -v
```
</details>

<details>
<summary><strong>pip</strong></summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -e packages/core -e packages/pytest-plugin
pip install -e packages/providers/openai -e packages/providers/anthropic
pytest tests/ -v
```
</details>

<details>
<summary><strong>PDM</strong></summary>

```bash
pdm install -G dev
pdm run pytest tests/ -v
```
</details>

<details>
<summary><strong>Poetry</strong></summary>

```bash
poetry install --all-extras
poetry run pytest tests/ -v
```
</details>

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

---

## Author

**Bahadır Araz**

- [bahadiraraz.com](https://bahadiraraz.com)
- [github.com/bahadiraraz](https://github.com/bahadiraraz)
- [bahadiraraz@protonmail.com](mailto:bahadiraraz@protonmail.com)

---

## License

MIT -- see [LICENSE](LICENSE) for details.
