"""Small helpers shared by vision.py and writer.py for calling Gemma."""

import json
import re

from .config import Settings

# One shared client per API key, so we do not reconnect on every request
_clients: dict[str, object] = {}


def get_genai_client(settings: Settings):
    """Return a google-genai Client (imported lazily so tests run without it)."""
    from google import genai

    key = settings.google_api_key
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key)
    return _clients[key]


def extract_json_object(text: str) -> dict:
    """Find and parse the first {...} JSON object in model text."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    chunk = text[start : end + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        # Common model mistake: trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", chunk)
        return json.loads(cleaned)
