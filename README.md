<p align="center">
  <img src=".github/icon.svg" alt="LLMTest" width="80" height="80" />
</p>

<h1 align="center">llmtest</h1>

<p align="center">
  <strong>The pytest for LLMs.</strong> Test your AI outputs like you test your code.
</p>

<p align="center">
  <a href="https://badge.fury.io/py/llmtest"><img src="https://badge.fury.io/py/llmtest.svg" alt="PyPI version" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</p>

## Quick Start

```bash
pip install llmtest[anthropic]
```

```python
from llmtest import expect, llm_test

@llm_test(
    expect.contains("Paris"),
    expect.latency_under(2000),
    expect.cost_under(0.001),
    model="claude-sonnet-4-20250514",
)
def test_capital(llm):
    output = llm("What is the capital of France?")
    assert "Paris" in output.content
```

```bash
pytest test_capitals.py -v
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
pip install llmtest[anthropic]    # Anthropic
pip install llmtest[openai]       # OpenAI
pip install llmtest[ollama]       # Ollama (local)
pip install llmtest[all]          # Everything
```

## Documentation

Full docs at **[docs.llmtest.dev](https://docs.llmtest.dev)**

## Contributing

```bash
git clone https://github.com/bahadiraraz/llmtest
cd llmtest
uv sync --all-extras
uv run pytest tests/ -v
```

## License

MIT
