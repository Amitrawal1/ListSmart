"""Make `app` importable and force every service into mock mode for tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    # Blank keys so tests never call real services, even if a .env exists
    for name in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "SERPAPI_KEY", "TINKER_API_KEY", "TINKER_MODEL_PATH"]:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("WRITER_BACKEND", "mock")
    monkeypatch.setenv("VISION_BACKEND", "mock")
    monkeypatch.setenv("OFFLINE_MODE", "false")
    # Port 9 is never Ollama, so a real local Ollama is never touched by accident
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    # Feedback goes to a throwaway file, not the real data/ folder
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))

    from app import ollama_client
    ollama_client.clear_status_cache()
