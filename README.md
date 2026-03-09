<p align="center">
  <img src=".github/icon.svg" alt="assertllm" width="80" height="80" />
</p>

<h1 align="center">assertllm</h1>

<p align="center">
  <strong>The pytest for LLMs.</strong> Test your AI outputs like you test your code.
</p>

<p align="center">
  <a href="https://github.com/bahadiraraz/llmtest/actions/workflows/ci.yml"><img src="https://github.com/bahadiraraz/llmtest/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/assertllm/"><img src="https://img.shields.io/pypi/v/assertllm.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/assertllm/"><img src="https://img.shields.io/pypi/pyversions/assertllm.svg" alt="Python versions" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT" /></a>
</p>

## Quick Start

```bash
pip install assertllm[anthropic]
```

```python
from assertllm import expect, llm_test

@llm_test(
    expect.contains("Paris"),
    expect.latency_under(2000),
    expect.cost_under(0.001),
    model="claude-sonnet-4-6",
)
def test_capital(llm):
    llm("What is the capital of France?")
```

```bash
pytest test_capitals.py -v
```

```
test_capitals.py::test_capital
  AI response: "The capital of France is Paris."
  ✓ contains("Paris")
  ✓ latency_under(2000) — 823ms
  ✓ cost_under(0.001) — $0.000023
  PASSED

────────── assertllm summary ──────────
  LLM tests: 1 passed
  Assertions: 3/3 passed
  Total cost: $0.000023
  Avg latency: 823ms
```

## Features

- **22+ assertions** — text, performance, agent, composable
- **Zero LLM calls** for most checks — deterministic and instant
- **Built on Pydantic** — auto-validation, JSON serialization, schema generation
- **Multi-provider** — OpenAI, Anthropic, Ollama out of the box
- **Agent testing** — tool calls, loop detection, call ordering
- **Retry support** — handle non-deterministic outputs
- **CI/CD native** — pytest markers, JUnit XML, JSON reporters
- **Pydantic AI integration** — test your existing agents

## Installation

```bash
pip install assertllm[anthropic]    # Anthropic
pip install assertllm[openai]       # OpenAI
pip install assertllm[ollama]       # Ollama (local)
pip install assertllm[all]          # Everything
```

## Documentation

Full docs at **[docs.assertllm.dev](https://docs.assertllm.dev)**

## Contributing

```bash
git clone https://github.com/bahadiraraz/llmtest
cd llmtest
uv sync --all-extras
uv run pytest tests/ -v
```

## Roadmap

> **This library is under active development.** More providers (Gemini, DeepSeek, Mistral, etc.) and additional assertion types are coming soon.

## License

MIT
