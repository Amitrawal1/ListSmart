"""ListSmart FastAPI app: photos -> attributes -> listing -> competitor pricing."""

import asyncio
import json
import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import EVAL_SUMMARY_PATH, FRONTEND_DIR, get_settings
from .pricing import get_pricing
from .vision import extract_attributes
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


@app.get("/api/health")
def health():
    s = get_settings()
    return {
        "ok": True,
        "writer_backend": s.writer_backend,
        "gemma": s.has_gemma,
        "serpapi": s.has_serpapi,
        "tinker": s.has_tinker,
    }


@app.post("/api/generate")
async def generate(
    images: list[UploadFile] = File(...),
    platform: str = Form("meesho"),
    notes: str = Form(""),
    cost_price: str | None = Form(None),
):
    started = time.perf_counter()
    platform = (platform or "meesho").strip().lower()
    if platform not in PLATFORMS:
        raise HTTPException(400, "platform must be 'meesho' or 'etsy'.")
    cost = parse_cost(cost_price)
    image_data = await read_images(images)
    settings = get_settings()

    # Each step needs the previous one's output, so they run in order.
    # Blocking SDK calls go to a worker thread so the server stays responsive.
    attributes, vision_source = await asyncio.to_thread(extract_attributes, image_data, settings)
    listing, writer_used = await asyncio.to_thread(write_listing, attributes, platform, notes or "", settings)
    pricing, pricing_source = await asyncio.to_thread(get_pricing, listing["search_query"], settings, cost)

    return {
        "attributes": attributes,
        "listing": listing,
        "pricing": pricing,
        "meta": {
            "writer_backend": writer_used,
            "vision_backend": vision_source,
            "pricing_source": pricing_source,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        },
    }


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
