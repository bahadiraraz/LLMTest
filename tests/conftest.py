"""
conftest.py — register llmtest pytest plugin and providers.
"""

import importlib.util
import os
import sys

# Add package paths so pytest can find them
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)  # for `import llmtest`
sys.path.insert(0, os.path.join(ROOT, "packages", "core"))
sys.path.insert(0, os.path.join(ROOT, "packages", "pytest-plugin"))

# Register the pytest plugin
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
