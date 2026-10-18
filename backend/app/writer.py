"""Step 2: turn product attributes into a marketplace Listing.

Backends (env WRITER_BACKEND): ollama (fine-tuned, on this laptop) -> tinker (fine-tuned,
hosted) -> gemma (base, hosted) -> mock. If the chosen backend fails we fall back down
that chain and report what was used.
"""

import logging
import threading

from . import ollama_client
from .chat_template import STOP_STRINGS, build_raw_prompt
from .config import Settings
from .llm_utils import get_genai_client
from .prompts import PLATFORM_RULES, build_writer_messages, parse_listing

log = logging.getLogger(__name__)

MAX_TOKENS = 700


# ---------- mock backend ----------

def _join(*words: str) -> str:
    return " ".join(w for w in words if w).strip()


def mock_listing(attributes: dict, platform: str = "meesho", notes: str = "") -> dict:
    """Deterministic listing built from the attributes (no model needed)."""
    rules = PLATFORM_RULES.get(platform, PLATFORM_RULES["meesho"])
    product = attributes.get("product_type") or "product"
    color = ", ".join(attributes.get("colors") or [])
    material = attributes.get("material", "")
    pattern = attributes.get("pattern", "")
    style = attributes.get("style", "")
    gender = attributes.get("gender", "")
    features = attributes.get("key_features") or []

    title = _join(color.title(), material.title(), pattern.title(), style.title(),
                  product.title(), f"for {gender.title()}" if gender else "")
    if features:
        title = f"{title} with {', '.join(features[:2])}"
    title = title[: rules["title_max"]].rstrip(" ,")

    para1 = f"Meet your new favourite {_join(color, material, product)}."
    para2 = (f"It features {', '.join(features)}." if features else
             "Simple, versatile and easy to pair with your everyday look.")
    para3 = f"Seller notes: {notes.strip()}" if notes.strip() else "Order now and enjoy quick delivery."
    description = "\n\n".join([para1, para2, para3])

    bullets = [
        f"Material: {material or 'quality fabric'}",
        f"Colour: {color or 'as shown'}",
        f"Design: {_join(pattern, style) or 'classic'}",
        f"Highlights: {', '.join(features) if features else 'comfortable everyday use'}",
        f"Condition: {attributes.get('condition') or 'new'}",
    ]

    raw_tags = [product, *(attributes.get("colors") or []), material, pattern, style,
                _join(gender, product), attributes.get("category", ""), *features]
    tags, seen = [], set()
    for t in raw_tags:
        t = (t or "").strip().lower()[: rules["tag_len_max"]]
        if t and t not in seen:
            seen.add(t)
            tags.append(t)

    search_query = _join(color.split(",")[0] if color else "", material, product, "for " + gender if gender else "")
    return {
        "title": title,
        "description": description,
        "bullets": bullets,
        "tags": tags[: rules["tags_max"]],
        "search_query": search_query or product,
    }


# ---------- ollama backend ----------

def ollama_listing(attributes: dict, platform: str, notes: str, settings: Settings) -> dict:
    """Local fine-tuned writer in Ollama, fed the exact prompt text used in training."""
    messages = build_writer_messages(attributes, platform, notes)
    payload = {
        "model": settings.ollama_writer_model,
        # raw=True: Ollama must not add its own chat template on top of ours
        "prompt": build_raw_prompt(messages),
        "raw": True,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": MAX_TOKENS, "stop": STOP_STRINGS},
    }
    reply = ollama_client.post(settings.ollama_host, "/api/generate", payload)
    return parse_listing(reply.get("response", ""), platform)


# ---------- gemma backend ----------

def gemma_listing(attributes: dict, platform: str, notes: str, settings: Settings) -> dict:
    """Base Gemma writes the listing with the same prompt as the fine-tuned model."""
    from google.genai import types

    messages = build_writer_messages(attributes, platform, notes)
    # Gemma on the Gemini API has no system role, so put the system text at the top of the user turn
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    client = get_genai_client(settings)
    response = client.models.generate_content(
        model=settings.gemma_model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=f"{system}\n\n{user}")])],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=MAX_TOKENS),
    )
    return parse_listing(response.text or "", platform)


# ---------- tinker backend ----------

# The sampling client is slow to create, so we build it once and reuse it
_tinker_lock = threading.Lock()
_tinker_cache: dict[str, tuple] = {}


def _get_tinker(settings: Settings):
    """Return (sampling_client, renderer) for the fine-tuned checkpoint, cached."""
    key = settings.tinker_model_path
    with _tinker_lock:
        if key in _tinker_cache:
            return _tinker_cache[key]

        # Lazy imports so the app runs without tinker installed
        import os

        import tinker
        from tinker_cookbook import renderers

        # The tinker SDK reads the key from the environment
        os.environ.setdefault("TINKER_API_KEY", settings.tinker_api_key)
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=settings.tinker_model_path)
        tokenizer = sampling_client.get_tokenizer()

        # Use the same chat template as training (thinking turned off for Qwen3)
        renderer_name = settings.tinker_renderer
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        _tinker_cache[key] = (sampling_client, renderer)
        return _tinker_cache[key]


def tinker_listing(attributes: dict, platform: str, notes: str, settings: Settings) -> dict:
    """Sample the fine-tuned LoRA model on Tinker."""
    from tinker import types

    sampling_client, renderer = _get_tinker(settings)
    messages = build_writer_messages(attributes, platform, notes)
    prompt = renderer.build_generation_prompt(messages)
    params = types.SamplingParams(
        max_tokens=MAX_TOKENS, temperature=0.7, stop=renderer.get_stop_sequences()
    )
    # sample() returns a Future; .result() waits for the answer
    result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
    message, _termination = renderer.parse_response(result.sequences[0].tokens)
    content = message["content"] if isinstance(message, dict) else getattr(message, "content", message)
    # Content can be a list of parts on newer renderers; keep only the text
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return parse_listing(str(content), platform)


# ---------- dispatcher ----------

def writer_chain(settings: Settings) -> list[str]:
    """Backends to try in order. Offline mode keeps only the on-device ones."""
    chain = {
        "ollama": ["ollama", "tinker", "gemma", "mock"],
        "tinker": ["tinker", "gemma", "mock"],
        "gemma": ["gemma", "mock"],
        "mock": ["mock"],
    }
    steps = chain[settings.writer_backend]
    if settings.offline_mode:
        steps = [b for b in steps if b in ("ollama", "mock")]
    return steps


def write_listing(attributes: dict, platform: str, notes: str, settings: Settings) -> tuple[dict, str]:
    """Return (listing, backend_used). Blocking call: run it in a thread."""
    for backend in writer_chain(settings):
        try:
            if backend == "ollama":
                return ollama_listing(attributes, platform, notes, settings), "ollama"
            if backend == "tinker":
                if not settings.has_tinker:
                    raise RuntimeError("TINKER_API_KEY / TINKER_MODEL_PATH not set")
                return tinker_listing(attributes, platform, notes, settings), "tinker"
            if backend == "gemma":
                if not settings.has_gemma:
                    raise RuntimeError("GOOGLE_API_KEY not set")
                return gemma_listing(attributes, platform, notes, settings), "gemma"
        except Exception as exc:
            log.warning("writer backend %s failed, falling back: %s", backend, exc)
            continue
    return mock_listing(attributes, platform, notes), "mock"
