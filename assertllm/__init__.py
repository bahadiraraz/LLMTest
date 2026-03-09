"""
assertllm — public API.

Single import surface:
    from assertllm import expect, llm_test, LLMInput, LLMOutput
"""

# Re-export everything from llmtest_core
from llmtest_core import (  # noqa: F401
    AgentStep,
    AgentTrace,
    # Providers
    BaseProvider,
    # Models
    LLMInput,
    LLMOutput,
    MockProvider,
    ProviderRegistry,
    RunConfig,
    Severity,
    # Runners
    TestCase,
    TestDataCase,
    # Dataset
    TestDataset,
    TestResult,
    TestRunner,
    TestStatus,
    TestSuiteResult,
    # Assertions namespace
    expect,
)


# pytest decorator — lazy import
def llm_test(*args, **kwargs):
    """Lazy wrapper for pytest decorator."""
    from llmtest_pytest.plugin import llm_test as _llm_test

    return _llm_test(*args, **kwargs)


__version__ = "0.3.0"
