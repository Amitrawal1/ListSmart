"""Small helpers shared by prepare_data.py, train.py and evaluate.py."""

import json
import os
import sys
from pathlib import Path

# Folder paths used by every script
FINETUNE_DIR = Path(__file__).resolve().parent
DATA_DIR = FINETUNE_DIR / "data"
RESULTS_DIR = FINETUNE_DIR / "results"
BACKEND_DIR = FINETUNE_DIR.parent / "backend"

# Make backend/app/prompts.py importable so training, eval and the app share ONE prompt format
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from app import prompts  # noqa: E402  (import must come after the sys.path change)
except ImportError:
    # If backend/app/__init__.py pulls in web-only packages, load prompts.py on its own
    import importlib.util

    _spec = importlib.util.spec_from_file_location("prompts", BACKEND_DIR / "app" / "prompts.py")
    prompts = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(prompts)

# Default base model to fine-tune (see README for why we picked it)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def load_env() -> None:
    """Load keys from a .env file (finetune/.env or the project root .env) if one exists."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for env_file in (FINETUNE_DIR / ".env", FINETUNE_DIR.parent / ".env"):
        if env_file.exists():
            load_dotenv(env_file, override=False)


def read_jsonl(path) -> list[dict]:
    """Read a .jsonl file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    """Write a list of dicts to a .jsonl file (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_messages(row: dict) -> list[dict]:
    """Build the system+user chat messages for one dataset row (same as the web app)."""
    return prompts.build_writer_messages(row["attributes"], row["platform"], row.get("notes", ""))


def training_conversation(row: dict) -> list[dict]:
    """Prompt messages plus the reference listing as the assistant answer to learn."""
    target = prompts.listing_to_target(row["reference"])
    return prompt_messages(row) + [{"role": "assistant", "content": target}]


def choose_renderer_name(model_name: str) -> str:
    """Pick the cookbook renderer for a model, preferring 'thinking disabled' so we get JSON directly."""
    from tinker_cookbook import model_info

    names = model_info.get_recommended_renderer_names(model_name)
    # Short JSON answers do not need chain-of-thought, so use a no-thinking variant when one exists
    for name in names:
        if "disable_thinking" in name:
            return name
    return names[0]


def require_env(name: str) -> str:
    """Return an env var or exit with a clear message."""
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"ERROR: environment variable {name} is not set (put it in .env or export it).")
    return value
