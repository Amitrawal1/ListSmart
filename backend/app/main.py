"""ListSmart FastAPI app: photos -> attributes -> listing -> competitor pricing."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import replace

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import ollama_client
from .config import EVAL_SUMMARY_PATH, FRONTEND_DIR, get_settings
from .costs import estimate_cost_inr
from .feedback import Feedback, feedback_summary, save_feedback
from .pricing import get_pricing
from .prompts import build_writer_messages
from .vision import VISION_PROMPT, extract_attributes
from .writer import write_listing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("listsmart")

MAX_IMAGES = 5
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
PLATFORMS = {"meesho", "etsy"}

app = FastAPI(title="ListSmart", version="1.0.0")

# Open CORS so the frontend works from any origin during the hackathon
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def sniff_image_type(data: bytes) -> str | None:
    """Detect the real image type from the first bytes (don't trust the filename)."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def read_images(images: list[UploadFile]) -> list[tuple[bytes, str]]:
    """Check count, type and size of uploads; return (bytes, mime) pairs."""
    images = [f for f in images if f and f.filename]
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(400, f"Upload between 1 and {MAX_IMAGES} images.")
    out = []
    for f in images:
        # Read one byte past the limit so we can tell if the file is too big
        data = await f.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than 8 MB.")
        if not data:
            raise HTTPException(400, f"{f.filename} is empty.")
        mime = sniff_image_type(data)
        if mime not in ALLOWED_TYPES:
            raise HTTPException(415, f"{f.filename} must be a JPG, PNG or WEBP image.")
        out.append((data, mime))
    return out


def parse_cost(cost_price: str | None) -> float | None:
    """Empty string -> None, otherwise a positive float."""
    if cost_price is None or not str(cost_price).strip():
        return None
    try:
        value = float(cost_price)
    except ValueError:
        raise HTTPException(400, "cost_price must be a number.")
    if value <= 0:
        raise HTTPException(400, "cost_price must be greater than 0.")
    return value


def parse_offline(offline: str | None) -> bool | None:
    """"true"/"false" form value -> bool; empty means "use the OFFLINE_MODE setting"."""
    if offline is None or not str(offline).strip():
        return None
    value = str(offline).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise HTTPException(400, "offline must be 'true' or 'false'.")


# Which outside service each hosted backend sends data to
HOSTED_SERVICE = {"gemma_api": "gemma_api", "gemma": "gemma_api", "tinker": "tinker", "serpapi": "serpapi"}
LOCAL_BACKENDS = {"ollama", "mock"}


@app.get("/api/health")
def health():
    s = get_settings()
    ollama_up, ollama_models = ollama_client.status(s.ollama_host)
    return {
        "ok": True,
        "writer_backend": s.writer_backend,
        "vision_backend": s.vision_backend,
        "offline_mode": s.offline_mode,
        "gemma": s.has_gemma,
        "serpapi": s.has_serpapi,
        "tinker": s.has_tinker,
        "ollama": ollama_up,
        "ollama_models": ollama_models,
    }


@app.post("/api/generate")
async def generate(
    images: list[UploadFile] = File(...),
    platform: str = Form("meesho"),
    notes: str = Form(""),
    cost_price: str | None = Form(None),
    offline: str | None = Form(None),
):
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    platform = (platform or "meesho").strip().lower()
    if platform not in PLATFORMS:
        raise HTTPException(400, "platform must be 'meesho' or 'etsy'.")
    # Keep seller notes short so nobody can send a giant prompt
    if len(notes or "") > 1000:
        raise HTTPException(400, "notes must be 1000 characters or less.")
    cost = parse_cost(cost_price)
    image_data = await read_images(images)
    settings = get_settings()
    # The form field (if sent) wins over the OFFLINE_MODE setting for this request
    offline_flag = parse_offline(offline)
    if offline_flag is not None:
        settings = replace(settings, offline_mode=offline_flag)

    # Each step needs the previous one's output, so they run in order.
    # Blocking calls go to a worker thread so the server stays responsive.
    timings = {}
    t = time.perf_counter()
    attributes, vision_source = await asyncio.to_thread(extract_attributes, image_data, settings)
    timings["vision"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    listing, writer_used = await asyncio.to_thread(write_listing, attributes, platform, notes or "", settings)
    timings["writer"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    pricing, pricing_source = await asyncio.to_thread(get_pricing, listing["search_query"], settings, cost)
    timings["pricing"] = int((time.perf_counter() - t) * 1000)

    # List every outside service that actually received data, without repeats
    data_left = []
    for used in (vision_source, writer_used, pricing_source):
        service = HOSTED_SERVICE.get(used)
        if service and service not in data_left:
            data_left.append(service)

    writer_prompt_chars = sum(len(m["content"]) for m in build_writer_messages(attributes, platform, notes or ""))
    cost_inr = estimate_cost_inr(
        vision_backend=vision_source,
        writer_backend=writer_used,
        pricing_source=pricing_source,
        n_images=len(image_data),
        vision_prompt_chars=len(VISION_PROMPT),
        vision_output_chars=len(json.dumps(attributes)),
        writer_prompt_chars=writer_prompt_chars,
        writer_output_chars=len(json.dumps(listing)),
    )

    return {
        "request_id": request_id,
        "attributes": attributes,
        "listing": listing,
        "pricing": pricing,
        "meta": {
            "writer_backend": writer_used,
            "vision_backend": vision_source,
            "pricing_source": pricing_source,
            "offline": settings.offline_mode,
            "ran_locally": vision_source in LOCAL_BACKENDS and writer_used in LOCAL_BACKENDS,
            "data_left_device": data_left,
            "cost_inr": cost_inr,
            "timings_ms": timings,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        },
    }


@app.post("/api/feedback")
async def post_feedback(item: Feedback):
    # File writes are blocking, so do them in a worker thread
    await asyncio.to_thread(save_feedback, item, get_settings().feedback_path)
    return {"ok": True}


@app.get("/api/feedback/summary")
async def get_feedback_summary():
    return await asyncio.to_thread(feedback_summary, get_settings().feedback_path)


@app.get("/api/eval")
def eval_summary():
    if not EVAL_SUMMARY_PATH.exists():
        return {"available": False}
    try:
        return JSONResponse(json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read eval summary: %s", exc)
        return {"available": False}


# Serve the static frontend (only if the folder exists, so the API works on its own)
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    page = FRONTEND_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"ok": True, "message": "ListSmart API. Frontend not found; see /docs."}
