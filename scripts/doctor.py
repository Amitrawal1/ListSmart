#!/usr/bin/env python3
"""ListSmart setup checker ("doctor").

Run it from anywhere:
    python3 scripts/doctor.py          # check everything, change nothing
    python3 scripts/doctor.py --init   # also copy .env.example to .env if .env is missing

It prints a checklist:
    ✓  = fine
    !  = optional / warning (the app still runs, maybe in a fallback or mock mode)
    ✗  = needs fixing (a fix command is printed under it)
and ends with "What to do next", in order.

Only the Python standard library is used, so it works before any venv exists.
It never prints the VALUE of a key, only whether it is set.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FINETUNE_DIR = PROJECT_ROOT / "finetune"
RESULTS_DIR = FINETUNE_DIR / "results"

MIN_PYTHON = (3, 11)
# Ollama release that dropped LoRA ADAPTER support (same value as finetune/export_to_ollama.py)
OLLAMA_ADAPTERS_REMOVED = (0, 34, 1)

# Rough download size (GB) of each model the app uses
MODEL_SIZES_GB = {"gemma3:4b": 3.3, "qwen3:8b": 5.2}
# listsmart-writer: you bring a ~5 GB .gguf to the Mac, and `ollama create` copies it (~5 GB more)
# before you can delete the .gguf. So you need ~10 GB free for a moment.
WRITER_IMPORT_GB = 10.0
# Building listsmart-writer ON this Mac (export_to_ollama.py --cleanup) needs about this much
LOCAL_EXPORT_GB = 27.0
# Keep a little space free so macOS does not struggle
DISK_MARGIN_GB = 2.0

# Keys we report on: name -> what it is used for. All are optional.
# name -> (what it is used for, place in "What to do next")
KEYS = {
    "TINKER_API_KEY": ("Tinker: train.py, evaluate.py, the Colab export, hosted writer", 10),
    "GOOGLE_API_KEY": ("hosted Gemma: photo reader fallback, base writer, LLM judge", 91),
    "SERPAPI_KEY": ("live competitor prices (else sample prices)", 92),
    "TINKER_MODEL_PATH": ("hosted writer on Tinker: paste finetune/results/finetuned_model.txt", 93),
}

OK, WARN, FAIL = "ok", "warn", "fail"
SYMBOLS = {OK: "✓", WARN: "!", FAIL: "✗"}


@dataclass
class Check:
    """One line of the checklist."""

    name: str
    status: str  # OK, WARN or FAIL
    detail: str = ""
    fix: str = ""  # command or advice shown under a ! or ✗
    order: int = 50  # lower = earlier in "What to do next"


# ---------------------------------------------------------------- pure helpers (tested)
def parse_version(text):
    """First 'X.Y.Z' in some text -> (X, Y, Z) as ints, or None."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in match.groups()) if match else None


def version_at_least(version, minimum):
    """True if version tuple >= minimum tuple (None counts as False)."""
    return version is not None and tuple(version) >= tuple(minimum)


def parse_env_file(text):
    """Read KEY=value lines like python-dotenv does (simple version).

    Skips blank lines and # comments, removes 'export ', quotes, and inline ' # comments'.
    """
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            # An unquoted value ends at " #" (an inline comment)
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            if value.startswith("#"):
                value = ""
        values[key.strip()] = value
    return values


def merged_env(file_values, environ):
    """Real environment variables win over the .env file (same rule as backend/app/config.py)."""
    merged = dict(file_values)
    for key, value in environ.items():
        if value.strip():
            merged[key] = value
    return merged


def key_is_set(env, name):
    """True if a key has a non-empty value. GEMINI_API_KEY counts for GOOGLE_API_KEY (like config.py)."""
    if name == "GOOGLE_API_KEY":
        return bool(env.get("GOOGLE_API_KEY", "").strip() or env.get("GEMINI_API_KEY", "").strip())
    return bool(env.get(name, "").strip())


def normalize_model(name):
    """'gemma3:4b' stays, 'listsmart-writer' becomes 'listsmart-writer:latest'."""
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def model_present(wanted, names):
    """Is model `wanted` in the list of model names Ollama reported?"""
    target = normalize_model(wanted)
    return any(normalize_model(n) == target for n in names)


def models_from_manifest_dir(manifests_root):
    """Model names found in ~/.ollama/models/manifests (works even when Ollama is not running).

    Layout: manifests/<host>/<namespace>/<model>/<tag>. Official models live under
    registry.ollama.ai/library and are shown as just '<model>:<tag>'.
    """
    root = Path(manifests_root)
    names = []
    if not root.is_dir():
        return names
    for tag_file in root.glob("*/*/*/*"):
        if not tag_file.is_file():
            continue
        tag, model, namespace, host = tag_file.name, tag_file.parent.name, tag_file.parent.parent.name, \
            tag_file.parent.parent.parent.name
        if host == "registry.ollama.ai" and namespace == "library":
            names.append(f"{model}:{tag}")
        else:
            names.append(f"{host}/{namespace}/{model}:{tag}")
    return sorted(names)


def manifest_size_gb(manifest_text):
    """Total size (GB) of the layers listed in one Ollama manifest JSON."""
    try:
        data = json.loads(manifest_text)
    except (ValueError, TypeError):
        return 0.0
    layers = data.get("layers", []) + ([data["config"]] if isinstance(data.get("config"), dict) else [])
    return round(sum(int(layer.get("size", 0)) for layer in layers) / 1e9, 1)


def disk_needed_gb(present_models, vision_model="gemma3:4b", writer_model="listsmart-writer"):
    """List of (what, GB) still to download/import, given the models already in Ollama."""
    needs = []
    if not model_present(vision_model, present_models):
        needs.append((f"ollama pull {vision_model}", MODEL_SIZES_GB.get(vision_model, 3.3)))
    if not model_present(writer_model, present_models):
        needs.append((f"import {writer_model} (.gguf from Colab + Ollama's copy)", WRITER_IMPORT_GB))
    return needs


def eval_summary_problem(text):
    """None if eval_summary.json looks like a real result, else a short reason."""
    try:
        data = json.loads(text)
    except ValueError:
        return "is not valid JSON"
    if not isinstance(data, dict):
        return "is not a JSON object"
    if "placeholder" in str(data.get("note", "")).lower():
        return "is the EXAMPLE placeholder, not a real result"
    if data.get("available") is False:
        return 'says "available": false'
    return None


def next_steps(checks):
    """Fix commands of every ! and ✗ check, in setup order (`order`), without repeats."""
    todo = [c for c in checks if c.status != OK and c.fix]
    # sorted() keeps the original order for checks with the same `order`
    todo = sorted(todo, key=lambda c: c.order)
    steps, seen = [], set()
    for c in todo:
        if c.fix not in seen:
            seen.add(c.fix)
            steps.append(c.fix)
    return steps


def format_check(check):
    """Text for one checklist line (+ an indented fix line)."""
    line = f"  {SYMBOLS[check.status]} {check.name}"
    if check.detail:
        line += f": {check.detail}"
    if check.fix and check.status != OK:
        line += "\n      fix: " + check.fix.replace("\n", "\n           ")
    return line


# ---------------------------------------------------------------- things that touch the machine
def run_quiet(cmd, timeout=20):
    """Run a command, return (exit code, stdout+stderr). Never raises."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def http_json(url, timeout=2.0):
    """GET a URL and parse JSON, or None if it fails (e.g. Ollama not running)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - any failure just means "not reachable"
        return None


def check_python():
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return Check("Python version", OK if ok else FAIL, f"{v.major}.{v.minor}.{v.micro} (need 3.11+)",
                 "install Python 3.11+ (e.g. `brew install python@3.12`) and use it to make the venvs", order=1)


def check_backend_venv():
    # README uses ListSmart/.venv; backend/.venv also works. Use whichever exists.
    for venv in (BACKEND_DIR / ".venv", PROJECT_ROOT / ".venv"):
        py = venv / "bin" / "python"
        if py.exists():
            code, out = run_quiet([str(py), "-c", "import fastapi, uvicorn, dotenv; print(fastapi.__version__)"])
            rel = venv.relative_to(PROJECT_ROOT)
            if code == 0:
                return Check("Backend venv", OK, f"{rel} (fastapi {out.strip().splitlines()[-1]})")
            return Check("Backend venv", FAIL, f"{rel} exists but cannot import fastapi/uvicorn",
                         f"{rel}/bin/pip install -r backend/requirements.txt", order=2)
    return Check("Backend venv", FAIL, "no backend/.venv (or .venv) found",
                 "python3 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements.txt",
                 order=2)


def check_env_file(init):
    env_path, example = PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"
    if env_path.exists():
        return Check(".env file", OK, "found")
    if init and example.exists():
        shutil.copyfile(example, env_path)
        return Check(".env file", OK, "copied from .env.example (fill in any keys you have)")
    return Check(".env file", WARN, "missing (the app uses defaults; all keys empty)",
                 "python3 scripts/doctor.py --init   (copies .env.example to .env)", order=3)


def load_env_values():
    env_path = PROJECT_ROOT / ".env"
    file_values = parse_env_file(env_path.read_text()) if env_path.exists() else {}
    return merged_env(file_values, os.environ)


def check_keys(env):
    checks = []
    for name, (use, order) in KEYS.items():
        if key_is_set(env, name):
            checks.append(Check(f"key {name}", OK, f"set - {use}"))
        else:
            prefix = "" if order < 90 else "optional: "
            checks.append(Check(f"key {name}", WARN, f"not set - {use}",
                                f"{prefix}add {name}=... to .env", order=order))
    return checks


def check_ollama(env):
    """Ollama installed / version / running. Returns (checks, model names, running?)."""
    checks = []
    host = env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not shutil.which("ollama"):
        checks.append(Check("Ollama installed", FAIL, "`ollama` not found on PATH",
                            "install Ollama from https://ollama.com/download (or `brew install ollama`)", order=4))
    else:
        checks.append(Check("Ollama installed", OK, shutil.which("ollama")))

    server = http_json(f"{host}/api/version")
    running = server is not None
    version = parse_version(server.get("version", "")) if running else None
    if version is None and shutil.which("ollama"):
        version = parse_version(run_quiet(["ollama", "--version"])[1])
    if running:
        checks.append(Check("Ollama running", OK, host))
    else:
        checks.append(Check("Ollama running", FAIL, f"no answer from {host}",
                            "open the Ollama app, or run `ollama serve` in another terminal", order=5))

    if version:
        vtext = ".".join(map(str, version))
        if version_at_least(version, OLLAMA_ADAPTERS_REMOVED):
            checks.append(Check("Ollama version", OK,
                                f"{vtext} - note: 0.34.1+ refuses LoRA ADAPTERs, so listsmart-writer must be the "
                                "merged .gguf from export_to_ollama.py / the Colab notebook (see finetune/README.md)"))
        else:
            checks.append(Check("Ollama version", OK, f"{vtext} (old enough for ADAPTER, but the merged .gguf is "
                                                      "still the recommended way)"))

    # Model list: from the server if it runs, else from the files on disk
    names = []
    if running:
        tags = http_json(f"{host}/api/tags") or {}
        names = [m.get("name", "") for m in tags.get("models", [])]
    if not names:
        names = models_from_manifest_dir(Path.home() / ".ollama" / "models" / "manifests")
    return checks, names, running


def check_models(env, names, running):
    vision = env.get("OLLAMA_VISION_MODEL") or "gemma3:4b"
    writer = env.get("OLLAMA_WRITER_MODEL") or "listsmart-writer"
    source = "" if running else " (read from ~/.ollama because the server is off)"
    checks = []
    if model_present(vision, names):
        checks.append(Check(f"model {vision}", OK, "present (photo reader)" + source))
    else:
        checks.append(Check(f"model {vision}", FAIL, "missing (photo reader)" + source,
                            f"ollama pull {vision}      (~3.3 GB)", order=7))
    if model_present(writer, names):
        checks.append(Check(f"model {writer}", OK, "present (fine-tuned writer)" + source))
    else:
        checks.append(Check(f"model {writer}", FAIL, "missing (fine-tuned writer)" + source,
                            f"build {writer} on Google Colab with finetune/colab_export.ipynb (after train.py), then run "
                            "its last cell's Mac commands: ollama create listsmart-writer -f finetune/export/Modelfile",
                            order=14))
    if model_present("qwen3:8b", names):
        checks.append(Check("model qwen3:8b", OK, "present (base model for the local eval)" + source))
    else:
        checks.append(Check("model qwen3:8b", WARN, "missing - optional: base model for "
                            "`evaluate.py --backend ollama`, or as the writer until listsmart-writer exists",
                            "optional: ollama pull qwen3:8b   (~5.2 GB, only if disk allows; `ollama rm qwen3:8b` after the eval)",
                            order=95))
    # Other models just take disk space; tell the user how much
    others = unused_models(env, names)
    if others:
        sizes = []
        for n in others:
            model, _, tag = n.partition(":")
            mf = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / model / tag
            gb = manifest_size_gb(mf.read_text()) if mf.is_file() else 0.0
            sizes.append(f"{n} ({gb} GB)" if gb else n)
        checks.append(Check("other Ollama models", WARN, "not used by ListSmart: " + ", ".join(sizes)))
    return checks


def unused_models(env, names):
    """Ollama models that ListSmart does not use (candidates for `ollama rm`)."""
    used = (env.get("OLLAMA_VISION_MODEL") or "gemma3:4b", env.get("OLLAMA_WRITER_MODEL") or "listsmart-writer",
            "qwen3:8b")
    return [n for n in names if not any(model_present(m, [n]) for m in used)]


def check_disk(env, names):
    free = shutil.disk_usage(Path.home()).free / 1e9
    needs = disk_needed_gb(names, env.get("OLLAMA_VISION_MODEL") or "gemma3:4b",
                           env.get("OLLAMA_WRITER_MODEL") or "listsmart-writer")
    total = sum(gb for _, gb in needs)
    checks = []
    if not needs:
        checks.append(Check("free disk", OK, f"{free:.1f} GB free; all needed models are present"))
    else:
        what = " + ".join(f"{label} ~{gb:g} GB" for label, gb in needs)
        status = OK if free >= total + DISK_MARGIN_GB else FAIL
        rm = " && ".join(f"ollama rm {n}" for n in unused_models(env, names)) or "`ollama rm <model>` for unused models"
        checks.append(Check("free disk", status, f"{free:.1f} GB free, need ~{total:g} GB (+{DISK_MARGIN_GB:g} spare) "
                            f"for: {what}",
                            f"free some disk: {rm}; empty the Trash; delete big files in ~/Downloads", order=6))
    writer = env.get("OLLAMA_WRITER_MODEL") or "listsmart-writer"
    if not model_present(writer, names) and free < LOCAL_EXPORT_GB:
        checks.append(Check("local export", WARN, f"building {writer} on this Mac needs ~{LOCAL_EXPORT_GB:g} GB "
                            f"free, you have {free:.1f} GB - use the Colab notebook (finetune/colab_export.ipynb)"))
    return checks


def check_finetune():
    checks = []
    test = FINETUNE_DIR / "data" / "test.jsonl"
    if test.exists():
        rows = sum(1 for line in test.read_text(encoding="utf-8").splitlines() if line.strip())
        checks.append(Check("finetune/data/test.jsonl", OK, f"{rows} test rows"))
    else:
        checks.append(Check("finetune/data/test.jsonl", FAIL, "missing (the fixed eval set)",
                            "cd finetune && .venv/bin/python prepare_data.py   (or `git checkout finetune/data/test.jsonl`)",
                            order=12))

    venv_py = FINETUNE_DIR / ".venv" / "bin" / "python"
    if venv_py.exists():
        checks.append(Check("finetune venv", OK, "finetune/.venv"))
    else:
        checks.append(Check("finetune venv", WARN, "missing - needed for training / eval",
                            "cd finetune && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
                            order=11))

    model_txt = RESULTS_DIR / "finetuned_model.txt"
    if model_txt.exists() and model_txt.read_text().strip().startswith("tinker://"):
        checks.append(Check("finetune/results/finetuned_model.txt", OK, "has a tinker:// path"))
    else:
        checks.append(Check("finetune/results/finetuned_model.txt", WARN,
                            "missing - training has not run yet (needs TINKER_API_KEY)",
                            "cd finetune && .venv/bin/python train.py", order=13))
    ext = RESULTS_DIR / "external_weights.txt"
    if ext.exists() and ext.read_text().strip():
        checks.append(Check("finetune/results/external_weights.txt", OK, "downloadable weights saved (for export)"))
    else:
        checks.append(Check("finetune/results/external_weights.txt", WARN,
                            "missing - the export will use finetuned_model.txt instead (older download API)"))

    summary = RESULTS_DIR / "eval_summary.json"
    if summary.exists():
        problem = eval_summary_problem(summary.read_text(encoding="utf-8"))
        if problem:
            checks.append(Check("finetune/results/eval_summary.json", WARN, problem,
                                "cd finetune && .venv/bin/python evaluate.py", order=15))
        else:
            checks.append(Check("finetune/results/eval_summary.json", OK, "real eval results (shown in the app)"))
    else:
        checks.append(Check("finetune/results/eval_summary.json", WARN,
                            'missing - the app\'s "Base vs Fine-tuned" panel will say "not available"',
                            "cd finetune && .venv/bin/python evaluate.py   (after train.py; add --backend ollama --limit 20 "
                                "--no-judge for laptop speed numbers)", order=15))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check that ListSmart is set up on this machine.")
    parser.add_argument("--init", action="store_true", help="copy .env.example to .env if .env is missing")
    args = parser.parse_args(argv)

    # Make sure ✓/✗ print even on an odd terminal encoding
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    sections = []
    sections.append(("Python", [check_python(), check_backend_venv()]))
    env_check = check_env_file(args.init)
    env = load_env_values()
    sections.append(("Settings (.env)", [env_check] + check_keys(env)))
    ollama_checks, names, running = check_ollama(env)
    sections.append(("Ollama (local models)", ollama_checks + check_models(env, names, running)))
    sections.append(("Disk", check_disk(env, names)))
    sections.append(("Fine-tune results", check_finetune()))

    print(f"ListSmart doctor  ({PROJECT_ROOT})")
    all_checks = []
    for title, checks in sections:
        print(f"\n{title}")
        for c in checks:
            print(format_check(c))
        all_checks.extend(checks)

    fails = sum(c.status == FAIL for c in all_checks)
    warns = sum(c.status == WARN for c in all_checks)
    print(f"\nSummary: {fails} to fix (✗), {warns} optional/warnings (!)")
    steps = next_steps(all_checks)
    if steps:
        print("\nWhat to do next (in this order):")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    else:
        print("\nAll set. Start the app (inside the backend venv):  cd backend && uvicorn app.main:app --reload")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
