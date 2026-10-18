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

# The raw Qwen3 prompt builder used when the writer runs locally in Ollama (same file as the app)
try:
    from app import chat_template  # noqa: E402
except ImportError:
    import importlib.util

    _spec = importlib.util.spec_from_file_location("chat_template", BACKEND_DIR / "app" / "chat_template.py")
    chat_template = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(chat_template)

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


def data_file(split: str, variant: str):
    """Path of one dataset split: 'raw' = data/train.jsonl, 'grounded' = data/train_grounded.jsonl."""
    suffix = "_grounded" if variant == "grounded" else ""
    return DATA_DIR / f"{split}{suffix}.jsonl"


def grounded_missing(split: str) -> int:
    """How many rows of the raw split are neither in the grounded file nor in the dropped log."""
    raw_ids = {r["id"] for r in read_jsonl(data_file(split, "raw"))}
    done = {r["id"] for r in read_jsonl(data_file(split, "grounded"))}
    dropped_file = DATA_DIR / "grounded_dropped.jsonl"
    if dropped_file.exists():
        done |= {r["id"] for r in read_jsonl(dropped_file) if r.get("split") == split}
    return len(raw_ids - done)


def pick_data_variant(requested: str | None, splits: list[str]) -> str:
    """Use the asked-for data variant, else 'grounded' when its files exist, else 'raw'.

    Prints which one is used, so it is always clear what a run trained or evaluated on.
    """
    if requested:
        variant = requested
        reason = "--data-variant"
    elif not all(data_file(s, "grounded").exists() for s in splits):
        variant, reason = "raw", "no grounded files yet"
    else:
        missing = {s: grounded_missing(s) for s in splits}
        if any(missing.values()):
            # e.g. after `build_grounded_data.py --limit 20`: do not silently train/eval on 20 rows
            variant = "raw"
            reason = "grounded files are not finished, rows still to do: " + ", ".join(
                f"{s}={n}" for s, n in missing.items() if n
            ) + " (finish build_grounded_data.py, or force with --data-variant grounded)"
        else:
            variant, reason = "grounded", "grounded files found (build_grounded_data.py)"
    files = ", ".join(data_file(s, variant).name for s in splits)
    print(f"data variant: {variant} ({reason}) -> {files}")
    return variant


def require_env(name: str) -> str:
    """Return an env var or exit with a clear message."""
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"ERROR: environment variable {name} is not set (put it in .env or export it).")
    return value
