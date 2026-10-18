"""Settings read from environment variables (and the project-root .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Folder paths used across the app
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
EVAL_SUMMARY_PATH = PROJECT_ROOT / "finetune" / "results" / "eval_summary.json"

# Load .env from the project root; real env vars win over the file
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    gemma_model: str
    google_api_key: str
    serpapi_key: str
    writer_backend: str
    tinker_api_key: str
    tinker_model_path: str
    tinker_base_model: str
    tinker_renderer: str

    @property
    def has_gemma(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_serpapi(self) -> bool:
        return bool(self.serpapi_key)

    @property
    def has_tinker(self) -> bool:
        return bool(self.tinker_api_key and self.tinker_model_path)


def get_settings() -> Settings:
    """Read settings fresh each call so tests can change env vars."""
    backend = _env("WRITER_BACKEND", "mock").lower()
    if backend not in ("tinker", "gemma", "mock"):
        backend = "mock"
    return Settings(
        gemma_model=_env("GEMMA_MODEL", "gemma-3-27b-it"),
        # Either name works; GOOGLE_API_KEY is checked first
        google_api_key=_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY"),
        serpapi_key=_env("SERPAPI_KEY"),
        writer_backend=backend,
        tinker_api_key=_env("TINKER_API_KEY"),
        tinker_model_path=_env("TINKER_MODEL_PATH"),
        tinker_base_model=_env("TINKER_BASE_MODEL", "Qwen/Qwen3-8B"),
        # Must match the chat template used in finetune/train.py
        tinker_renderer=_env("TINKER_RENDERER", "qwen3_disable_thinking"),
    )
