"""Step 1: Gemma looks at product photos and returns ProductAttributes JSON.

Backends (env VISION_BACKEND): ollama (gemma3:4b on this laptop) -> gemma_api (hosted) -> mock.
"""

import base64
import io
import logging

from . import ollama_client
from .config import Settings
from .llm_utils import extract_json_object, get_genai_client

log = logging.getLogger(__name__)

# Fields every ProductAttributes dict must have, with their empty defaults
ATTRIBUTE_DEFAULTS = {
    "product_type": "",
    "category": "",
    "colors": [],
    "material": "",
    "pattern": "",
    "style": "",
    "gender": "",
    "key_features": [],
    "visible_text": "",
    "condition": "new",
}

# Gemma on the Gemini API has no system-instruction or JSON mode, so all rules live in the user turn
VISION_PROMPT = (
    "You are a product cataloguing assistant for an Indian e-commerce marketplace. "
    "Look at the product photo(s) and describe ONLY what you can actually see.\n"
    "Reply with ONE JSON object and nothing else, with exactly these keys:\n"
    '- "product_type": short noun, e.g. "kurti", "ceramic mug", "tote bag"\n'
    '- "category": marketplace category, e.g. "Women Ethnic Wear", "Home & Kitchen"\n'
    '- "colors": list of main colours\n'
    '- "material": likely material, or "" if unsure\n'
    '- "pattern": e.g. "solid", "floral", "block print", or ""\n'
    '- "style": e.g. "straight", "boho", "minimal", or ""\n'
    '- "gender": "women", "men", "unisex", "kids" or "" if not relevant\n'
    '- "key_features": list of 2-6 visible features\n'
    '- "visible_text": any brand/label text printed on the product, or ""\n'
    '- "condition": "new" or "used"\n'
    "Do not guess sizes, brands or certifications."
)

MOCK_ATTRIBUTES = {
    "product_type": "kurti",
    "category": "Women Ethnic Wear",
    "colors": ["maroon"],
    "material": "cotton",
    "pattern": "block print",
    "style": "straight",
    "gender": "women",
    "key_features": ["3/4 sleeves", "mandarin collar"],
    "visible_text": "",
    "condition": "new",
}


def normalize_attributes(data: dict) -> dict:
    """Make sure every key exists and has the right type (str or list of str)."""
    out = {}
    for key, default in ATTRIBUTE_DEFAULTS.items():
        value = data.get(key, default)
        if isinstance(default, list):
            # Accept a comma string or a list, always return a list of strings
            if isinstance(value, str):
                value = [v for v in value.split(",")]
            elif not isinstance(value, list):
                value = [value] if value else []
            out[key] = [str(v).strip() for v in value if str(v).strip()]
        else:
            out[key] = "" if value is None else str(value).strip()
    if not out["condition"]:
        out["condition"] = "new"
    return out


def mock_attributes() -> dict:
    return dict(MOCK_ATTRIBUTES, colors=list(MOCK_ATTRIBUTES["colors"]),
                key_features=list(MOCK_ATTRIBUTES["key_features"]))


def _to_ollama_image(data: bytes, mime: str) -> str:
    """Base64 text for Ollama. WEBP is turned into PNG because not every Ollama build reads it."""
    if mime == "image/webp":
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
        except Exception as exc:  # Pillow missing or bad file -> send it as it is
            log.warning("could not convert webp to png: %s", exc)
    return base64.b64encode(data).decode("ascii")


def ollama_attributes(images: list[tuple[bytes, str]], settings: Settings) -> dict:
    """Local Gemma 3 (via Ollama) reads the photos. Nothing leaves the laptop."""
    payload = {
        "model": settings.ollama_vision_model,
        "messages": [{
            "role": "user",
            "content": VISION_PROMPT,
            "images": [_to_ollama_image(data, mime) for data, mime in images],
        }],
        # Ollama can force valid JSON output; low temperature keeps it factual
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 800},
    }
    reply = ollama_client.post(settings.ollama_host, "/api/chat", payload)
    text = (reply.get("message") or {}).get("content", "")
    return normalize_attributes(extract_json_object(text))


def gemma_api_attributes(images: list[tuple[bytes, str]], settings: Settings) -> dict:
    """Hosted Gemma (Google AI Studio) reads the photos."""
    from google.genai import types

    client = get_genai_client(settings)
    # Images first, then the instructions, all in one user turn
    parts = [types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images]
    parts.append(types.Part.from_text(text=VISION_PROMPT))
    response = client.models.generate_content(
        model=settings.gemma_model,
        contents=[types.Content(role="user", parts=parts)],
        # Extra room because Gemma 4 thinks before answering
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048),
    )
    return normalize_attributes(extract_json_object(response.text or ""))


def vision_chain(settings: Settings) -> list[str]:
    """Backends to try in order. Offline mode drops the hosted one."""
    chain = {"ollama": ["ollama", "gemma_api"], "gemma_api": ["gemma_api"], "mock": []}
    steps = chain[settings.vision_backend]
    if settings.offline_mode:
        steps = [b for b in steps if b == "ollama"]
    return steps


def extract_attributes(images: list[tuple[bytes, str]], settings: Settings) -> tuple[dict, str]:
    """Return (attributes, source) where source is "ollama", "gemma_api" or "mock".

    images is a list of (raw_bytes, mime_type). Blocking call: run it in a thread.
    """
    for backend in vision_chain(settings):
        try:
            if backend == "ollama":
                return ollama_attributes(images, settings), "ollama"
            if backend == "gemma_api":
                if not settings.has_gemma:
                    raise RuntimeError("GOOGLE_API_KEY not set")
                return gemma_api_attributes(images, settings), "gemma_api"
        except Exception as exc:  # any network / parse error -> try the next one
            log.warning("vision backend %s failed, falling back: %s", backend, exc)
    return mock_attributes(), "mock"
