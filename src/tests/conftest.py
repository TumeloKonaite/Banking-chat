import importlib
import sys
import types

import pytest


try:
    import langchain_chroma  # noqa: F401
except ModuleNotFoundError:
    # Test-only fallback to keep imports deterministic in environments where
    # langchain-chroma is not installed.
    shim = types.ModuleType("langchain_chroma")

    class _Chroma:  # pragma: no cover - only used when dependency is absent
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_documents(cls, *args, **kwargs):
            return cls()

        def as_retriever(self, *args, **kwargs):
            return self

        def invoke(self, *args, **kwargs):
            return []

        def delete_collection(self):
            return None

    shim.Chroma = _Chroma
    sys.modules["langchain_chroma"] = shim


@pytest.fixture
def app_module(monkeypatch):
    # Prevent demo-mode auth/startup behavior in tests.
    monkeypatch.setenv("DEMO_MODE", "0")
    monkeypatch.setenv("SHOW_GRADIO", "0")
    monkeypatch.delenv("DEMO_API_KEY", raising=False)

    module_name = "src.server.app"
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)
    return module
