"""Tiny helpers for talking to a local Ollama server over plain HTTP (no extra package)."""

import threading
import time

import requests

# Connect fast (is Ollama even running?), but give a slow 8GB laptop time to answer
CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 180
# /api/tags is only a quick "are you there?" check
TAGS_TIMEOUT_S = 1.5
# Health checks reuse the last answer for this long so the page can poll cheaply
STATUS_CACHE_S = 10

_status_lock = threading.Lock()
_status_cache: dict[str, tuple[float, bool, list[str]]] = {}


def base_url(host: str) -> str:
    """Accept "localhost:11434" or "http://localhost:11434" and return a full URL."""
    host = (host or "http://localhost:11434").strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
    return host


def post(host: str, path: str, payload: dict) -> dict:
    """POST JSON to Ollama and return the JSON reply. Raises on any HTTP/network error."""
    resp = requests.post(
        base_url(host) + path,
        json=payload,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")
    return data


def status(host: str) -> tuple[bool, list[str]]:
    """Return (reachable, model names). Cached for a few seconds per host."""
    url = base_url(host)
    now = time.monotonic()
    with _status_lock:
        cached = _status_cache.get(url)
        if cached and now - cached[0] < STATUS_CACHE_S:
            return cached[1], list(cached[2])

    try:
        resp = requests.get(url + "/api/tags", timeout=TAGS_TIMEOUT_S)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", []) or []]
        result = (True, [m for m in models if m])
    except Exception:
        result = (False, [])

    with _status_lock:
        _status_cache[url] = (now, result[0], result[1])
    return result[0], list(result[1])


def clear_status_cache() -> None:
    """Forget cached health answers (used by tests)."""
    with _status_lock:
        _status_cache.clear()
