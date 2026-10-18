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
# Seller feedback is saved here (the data/ folder is gitignored)
DEFAULT_FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.jsonl"

WRITER_BACKENDS = ("ollama", "tinker", "gemma", "mock")
VISION_BACKENDS = ("ollama", "gemma_api", "mock")

# Load .env from the project root; real env vars win over the file
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read "true"/"1"/"yes" style env vars as a bool."""
    value = _env(name)
    if not value:
        return default
    return value.lower() in ("1", "true", "yes", "on")


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
    vision_backend: str
    ollama_host: str
    ollama_vision_model: str
    ollama_writer_model: str
    offline_mode: bool
    feedback_path: Path

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
    # Local Ollama is the default; unknown values fall back to safe mock mode
    backend = _env("WRITER_BACKEND", "ollama").lower()
    if backend not in WRITER_BACKENDS:
        backend = "mock"
    vision = _env("VISION_BACKEND", "ollama").lower()
    if vision not in VISION_BACKENDS:
        vision = "mock"
    return Settings(
        gemma_model=_env("GEMMA_MODEL", "gemma-4-26b-a4b-it"),
        # Either name works; GOOGLE_API_KEY is checked first
        google_api_key=_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY"),
        serpapi_key=_env("SERPAPI_KEY"),
        writer_backend=backend,
        tinker_api_key=_env("TINKER_API_KEY"),
        tinker_model_path=_env("TINKER_MODEL_PATH"),
        tinker_base_model=_env("TINKER_BASE_MODEL", "Qwen/Qwen3-8B"),
        # Must match the chat template used in finetune/train.py
        tinker_renderer=_env("TINKER_RENDERER", "qwen3_disable_thinking"),
        vision_backend=vision,
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_vision_model=_env("OLLAMA_VISION_MODEL", "gemma3:4b"),
        ollama_writer_model=_env("OLLAMA_WRITER_MODEL", "listsmart-writer"),
        # true = never send anything off the laptop (no SerpApi, no hosted models)
        offline_mode=_env_bool("OFFLINE_MODE", False),
        feedback_path=Path(_env("FEEDBACK_PATH") or DEFAULT_FEEDBACK_PATH),
    )
