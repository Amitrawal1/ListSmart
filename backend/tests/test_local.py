"""Part 2 tests: Ollama vision/writer (faked), fallbacks, offline mode, feedback, health."""

import io
import json

import requests
from fastapi.testclient import TestClient
from PIL import Image

from app import ollama_client, pricing
from app.chat_template import STOP_STRINGS
from app.config import get_settings
from app.main import app
from app.vision import extract_attributes
from app.writer import write_listing

client = TestClient(app)

GOOD_LISTING = {
    "title": "Maroon Cotton Kurti", "description": "Nice kurti.",
    "bullets": ["a", "b", "c", "d", "e"], "tags": ["kurti", "cotton"],
    "search_query": "maroon cotton kurti",
}
OLLAMA_ATTRS = {"product_type": "ceramic mug", "category": "Home & Kitchen", "colors": ["blue"],
                "material": "ceramic", "key_features": ["handle"], "condition": "new"}


def tiny_image(fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 128)).save(buf, format=fmt)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, data: dict, status: int = 200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def fake_ollama(monkeypatch, calls: list, models=("gemma3:4b", "listsmart-writer:latest")):
    """Pretend Ollama is running: answer /api/chat, /api/generate and /api/tags."""
    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        if url.endswith("/api/chat"):
            return FakeResponse({"message": {"role": "assistant", "content": _dumps(OLLAMA_ATTRS)}})
        if url.endswith("/api/generate"):
            return FakeResponse({"response": _dumps(GOOD_LISTING), "done": True})
        return FakeResponse({"error": "not found"}, 404)

    def fake_get(url, timeout=None):
        calls.append((url, None))
        return FakeResponse({"models": [{"name": m} for m in models]})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.requests, "get", fake_get)


def ollama_down(monkeypatch):
    """Pretend nothing is listening on the Ollama port."""
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(ollama_client.requests, "post", refuse)
    monkeypatch.setattr(ollama_client.requests, "get", refuse)


def _dumps(obj) -> str:
    return json.dumps(obj)


def use_ollama(monkeypatch):
    monkeypatch.setenv("VISION_BACKEND", "ollama")
    monkeypatch.setenv("WRITER_BACKEND", "ollama")


# ---------- vision ----------

def test_ollama_vision_sends_base64_images(monkeypatch):
    calls = []
    fake_ollama(monkeypatch, calls)
    use_ollama(monkeypatch)
    attrs, source = extract_attributes([(tiny_image(), "image/png")], get_settings())
    assert source == "ollama"
    assert attrs["product_type"] == "ceramic mug" and attrs["pattern"] == ""
    url, payload = calls[0]
    assert url == "http://127.0.0.1:9/api/chat"
    assert payload["model"] == "gemma3:4b" and payload["format"] == "json"
    assert payload["stream"] is False
    assert isinstance(payload["messages"][0]["images"][0], str)


def test_ollama_vision_converts_webp(monkeypatch):
    calls = []
    fake_ollama(monkeypatch, calls)
    use_ollama(monkeypatch)
    extract_attributes([(tiny_image("WEBP"), "image/webp")], get_settings())
    import base64
    sent = base64.b64decode(calls[0][1]["messages"][0]["images"][0])
    assert sent.startswith(b"\x89PNG")


def test_vision_falls_back_to_mock_when_ollama_down(monkeypatch):
    ollama_down(monkeypatch)
    use_ollama(monkeypatch)
    attrs, source = extract_attributes([(tiny_image(), "image/png")], get_settings())
    assert source == "mock" and attrs["product_type"] == "kurti"


# ---------- writer ----------

def test_ollama_writer_uses_raw_prompt(monkeypatch):
    calls = []
    fake_ollama(monkeypatch, calls)
    use_ollama(monkeypatch)
    listing, used = write_listing(OLLAMA_ATTRS, "etsy", "", get_settings())
    assert used == "ollama" and listing["title"] == GOOD_LISTING["title"]
    url, payload = calls[0]
    assert url.endswith("/api/generate")
    assert payload["model"] == "listsmart-writer" and payload["raw"] is True
    assert payload["prompt"].startswith("<|im_start|>system\n")
    assert payload["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert payload["options"]["stop"] == STOP_STRINGS


def test_writer_falls_back_to_mock_when_ollama_down(monkeypatch):
    ollama_down(monkeypatch)
    use_ollama(monkeypatch)
    listing, used = write_listing(OLLAMA_ATTRS, "meesho", "", get_settings())
    assert used == "mock" and len(listing["bullets"]) == 5


def test_writer_falls_back_on_bad_json(monkeypatch):
    def bad_post(url, json=None, timeout=None):
        return FakeResponse({"response": "sorry, no json"})

    monkeypatch.setattr(ollama_client.requests, "post", bad_post)
    use_ollama(monkeypatch)
    _, used = write_listing(OLLAMA_ATTRS, "meesho", "", get_settings())
    assert used == "mock"


# ---------- /api/generate ----------

def post_generate(data=None):
    return client.post("/api/generate", files=[("images", ("p.png", tiny_image(), "image/png"))], data=data or {})


def test_generate_with_ollama_is_local_and_free(monkeypatch):
    calls = []
    fake_ollama(monkeypatch, calls)
    use_ollama(monkeypatch)
    r = post_generate({"platform": "meesho"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["request_id"]) == 32
    meta = body["meta"]
    assert meta["vision_backend"] == "ollama" and meta["writer_backend"] == "ollama"
    assert meta["ran_locally"] is True and meta["data_left_device"] == []
    assert meta["cost_inr"] == 0
    assert set(meta["timings_ms"]) == {"vision", "writer", "pricing"}


def test_offline_skips_serpapi(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "fake-key")

    def must_not_call(*args, **kwargs):
        raise AssertionError("SerpApi was called in offline mode")

    monkeypatch.setattr(pricing.requests, "get", must_not_call)
    r = post_generate({"offline": "true", "cost_price": "200"})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["pricing_source"] == "skipped"
    assert body["meta"]["data_left_device"] == [] and body["meta"]["cost_inr"] == 0
    assert body["pricing"]["competitors"] == []
    assert body["pricing"]["suggested_price"] == 259  # 200 * 1.25 = 250 -> 259


def test_offline_mode_env_skips_hosted_writer(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    monkeypatch.setenv("WRITER_BACKEND", "gemma")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    body = post_generate().json()
    assert body["meta"]["writer_backend"] == "mock" and body["meta"]["pricing_source"] == "skipped"


def test_hosted_cost_estimate(monkeypatch):
    from app.costs import estimate_cost_inr
    cost = estimate_cost_inr("ollama", "tinker", "serpapi", 1, 0, 0, 4000, 4000)
    # 2000 tokens * $0.60/1M + $0.015 search, times 84
    assert cost == round((2000 * 0.60 / 1_000_000 + 0.015) * 84, 4)
    assert estimate_cost_inr("ollama", "ollama", "skipped", 3, 100, 100, 100, 100) == 0


def test_bad_offline_value():
    assert post_generate({"offline": "maybe"}).status_code == 400


# ---------- health ----------

def test_health_reports_ollama(monkeypatch):
    calls = []
    fake_ollama(monkeypatch, calls)
    body = client.get("/api/health").json()
    assert body["ollama"] is True
    assert body["ollama_models"] == ["gemma3:4b", "listsmart-writer:latest"]
    assert body["vision_backend"] == "mock" and body["offline_mode"] is False
    # Second call within 10s is served from the cache
    client.get("/api/health")
    assert len([c for c in calls if c[0].endswith("/api/tags")]) == 1


def test_health_when_ollama_down(monkeypatch):
    ollama_down(monkeypatch)
    body = client.get("/api/health").json()
    assert body["ollama"] is False and body["ollama_models"] == []


# ---------- feedback ----------

def test_feedback_round_trip(tmp_path):
    assert client.get("/api/feedback/summary").json()["count"] == 0
    items = [
        {"request_id": "a1", "rating": "up", "used": True, "would_pay": "yes", "price_ok_inr": 99,
         "comment": "great", "listing": GOOD_LISTING},
        {"request_id": "a2", "rating": "down", "used": False, "would_pay": "maybe"},
        {"request_id": "a3", "rating": None, "would_pay": None, "comment": ""},
    ]
    for item in items:
        r = client.post("/api/feedback", json=item)
        assert r.status_code == 200 and r.json() == {"ok": True}
    assert client.get("/api/feedback/summary").json() == {
        "count": 3, "up": 1, "down": 1, "used": 1, "would_pay": {"yes": 1, "maybe": 1, "no": 0},
    }
    lines = (tmp_path / "feedback.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["listing"]["title"] == GOOD_LISTING["title"]


def test_feedback_rejects_bad_values():
    assert client.post("/api/feedback", json={"rating": "meh"}).status_code == 422
    assert client.post("/api/feedback", json={"would_pay": "sure"}).status_code == 422
    assert client.post("/api/feedback", json={"price_ok_inr": -5}).status_code == 422
