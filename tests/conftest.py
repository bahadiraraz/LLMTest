"""
conftest.py — register llmtest pytest plugin and providers.
"""

import importlib.util
import os
import sys
from importlib.metadata import entry_points

# Skip collection of test files whose dependencies aren't installed
collect_ignore = []
if not importlib.util.find_spec("pydantic_ai"):
    collect_ignore.append("test_pydantic_ai_integration.py")

# Add package paths so pytest can find them
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)  # for `import llmtest`
sys.path.insert(0, os.path.join(ROOT, "packages", "core"))
sys.path.insert(0, os.path.join(ROOT, "packages", "pytest-plugin"))

# Only register via conftest if the entry point isn't already installed.
# When installed with `pip install -e`, pytest11 entry point auto-registers
# the plugin. Registering again via pytest_plugins causes a conflict.
_ep = entry_points()
_pytest11 = _ep.select(group="pytest11") if hasattr(_ep, "select") else _ep.get("pytest11", [])
_already_registered = any(ep.name == "llmtest" for ep in _pytest11)

if not _already_registered:
    pytest_plugins = ["llmtest_pytest.plugin"]


def _import_provider(name: str, path: str):
    """Import a provider module by absolute path to avoid name collisions."""
    spec = importlib.util.spec_from_file_location(f"provider_{name}", path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


# Import providers — triggers auto-register with ProviderRegistry
_providers = {
    "anthropic": os.path.join(ROOT, "packages", "providers", "anthropic", "provider.py"),
    "openai": os.path.join(ROOT, "packages", "providers", "openai", "provider.py"),
}

for name, path in _providers.items():
    if os.path.exists(path):
        try:
            _import_provider(name, path)
        except ImportError:
            pass  # SDK not installed, skip
