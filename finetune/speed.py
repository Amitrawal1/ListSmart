"""Small pure helpers for evaluate.py: Ollama requests, speed numbers, hardware name, cost.

Kept separate (and free of network calls) so they are easy to unit test.
"""

import platform
import statistics
import subprocess

# Same assumed prices as backend/app/costs.py (keep them in sync)
USD_TO_INR = 84.0
TINKER_USD_PER_1M_TOKENS = 0.60


# ---------------------------------------------------------------- Ollama
def ollama_generate_payload(model, raw_prompt, stop, temperature, max_tokens, seed, num_ctx=4096):
    """Body for Ollama POST /api/generate in raw mode (we send the exact training prompt)."""
    return {
        "model": model,
        "prompt": raw_prompt,
        # raw=True: Ollama must not wrap our prompt in its own chat template
        "raw": True,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": list(stop),
            "seed": seed,
            "num_ctx": num_ctx,
        },
    }


def ollama_call_stats(reply: dict, wall_s: float) -> dict:
    """Turn one /api/generate reply into our per-call record (text + speed + token counts).

    Ollama reports durations in nanoseconds; eval_* is the generated part only.
    """
    out_tokens = int(reply.get("eval_count") or 0)
    eval_s = (reply.get("eval_duration") or 0) / 1e9
    return {
        "text": reply.get("response", ""),
        "latency_s": round(wall_s, 3),
        "prompt_tokens": int(reply.get("prompt_eval_count") or 0),
        "output_tokens": out_tokens,
        "tokens_per_s": round(out_tokens / eval_s, 2) if eval_s > 0 else None,
        "load_s": round((reply.get("load_duration") or 0) / 1e9, 3),
    }


def tokens_call_stats(text, prompt_tokens, output_tokens, wall_s) -> dict:
    """Per-call record for backends that only give token counts (Tinker, closed APIs)."""
    return {
        "text": text,
        "latency_s": round(wall_s, 3),
        "prompt_tokens": int(prompt_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        # Wall-clock speed: includes network and queue time, so it is lower than pure decode speed
        "tokens_per_s": round(output_tokens / wall_s, 2) if wall_s and wall_s > 0 and output_tokens else None,
    }


# ---------------------------------------------------------------- summaries
def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def speed_summary(calls: list[dict]) -> dict:
    """Median latency and tokens/s over many calls (median, so the first slow model load does not dominate)."""
    calls = [c for c in calls if c]
    return {
        "median_latency_s": _median([c.get("latency_s") for c in calls]),
        "tokens_per_s": _median([c.get("tokens_per_s") for c in calls]),
        "mean_prompt_tokens": round(statistics.mean([c.get("prompt_tokens", 0) for c in calls]), 1) if calls else None,
        "mean_output_tokens": round(statistics.mean([c.get("output_tokens", 0) for c in calls]), 1) if calls else None,
        "n_calls": len(calls),
    }


def cost_per_1000_listings_inr(mean_tokens_per_listing, usd_per_1m_tokens, usd_to_inr=USD_TO_INR):
    """INR for 1000 listings at a flat price per 1M (prompt + output) tokens. None if unknown."""
    if mean_tokens_per_listing is None or usd_per_1m_tokens is None:
        return None
    usd = mean_tokens_per_listing * 1000 * usd_per_1m_tokens / 1_000_000
    return round(usd * usd_to_inr, 2)


def cost_split_per_1000_inr(mean_in, mean_out, usd_per_1m_in, usd_per_1m_out, usd_to_inr=USD_TO_INR):
    """INR for 1000 listings when input and output tokens have different prices (closed APIs)."""
    if None in (mean_in, mean_out, usd_per_1m_in, usd_per_1m_out):
        return None
    usd = 1000 * (mean_in * usd_per_1m_in + mean_out * usd_per_1m_out) / 1_000_000
    return round(usd * usd_to_inr, 2)


# ---------------------------------------------------------------- hardware
def format_hardware(system: str, chip: str, ram_bytes) -> str:
    """e.g. 'macOS arm64, Apple M1, 8 GB RAM'."""
    parts = [system.strip()] if system and system.strip() else []
    if chip and chip.strip():
        parts.append(chip.strip())
    if ram_bytes:
        parts.append(f"{round(int(ram_bytes) / 2**30)} GB RAM")
    return ", ".join(parts) or "unknown"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 - tool missing on this OS
        return ""


def detect_hardware() -> str:
    """Best-effort description of THIS machine (used for the local Ollama numbers)."""
    system = platform.system()
    chip, ram = "", None
    if system == "Darwin":
        system = f"macOS {platform.machine()}"
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        ram = _run(["sysctl", "-n", "hw.memsize"]) or None
    elif system == "Linux":
        system = f"Linux {platform.machine()}"
        try:
            with open("/proc/cpuinfo") as f:
                chip = next((ln.split(":", 1)[1] for ln in f if ln.startswith("model name")), "")
            with open("/proc/meminfo") as f:
                kb = next((ln.split()[1] for ln in f if ln.startswith("MemTotal")), None)
                ram = int(kb) * 1024 if kb else None
        except OSError:
            pass
    else:
        chip = platform.processor()
    return format_hardware(system, chip, ram)
