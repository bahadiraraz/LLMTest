# llmtest

**The pytest for LLMs.** Test your AI outputs like you test your code.

[![PyPI version](https://badge.fury.io/py/llmtest.svg)](https://badge.fury.io/py/llmtest)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
