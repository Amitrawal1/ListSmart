"""Make `app` importable and force every service into mock mode for tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    # Blank keys so tests never call real services, even if a .env exists
    for name in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "SERPAPI_KEY", "TINKER_API_KEY", "TINKER_MODEL_PATH"]:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("WRITER_BACKEND", "mock")
