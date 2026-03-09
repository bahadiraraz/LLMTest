"""Test that verifies the terminal output format works correctly."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "pytest-plugin"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "packages", "providers", "anthropic"),
)

# Load .env
from pathlib import Path

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

# Force load anthropic provider
from llmtest_core import expect  # noqa: F401
from llmtest_pytest.plugin import llm_test  # noqa: F401
from provider import AnthropicProvider  # noqa: F401


@llm_test(
    expect.contains("Paris"),
    expect.latency_under(5000),
    expect.cost_under(0.01),
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
)
def test_capital(llm):
    llm("What is the capital of France?")
