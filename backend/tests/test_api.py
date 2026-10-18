import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["writer_backend"] == "mock"
    assert body["gemma"] is False and body["serpapi"] is False and body["tinker"] is False


def test_generate_mock():
    r = client.post(
        "/api/generate",
        files=[("images", ("p.png", tiny_png(), "image/png"))],
        data={"platform": "meesho", "notes": "free shipping", "cost_price": "300"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"attributes", "listing", "pricing", "meta"}
    assert body["meta"]["writer_backend"] == "mock"
    assert len(body["listing"]["bullets"]) == 5
    p = body["pricing"]
    assert p["currency"] == "INR" and p["suggested_price"] >= 375
    assert p["stats"]["count"] == len(p["competitors"])
    assert p["margin_pct"] is not None


def test_generate_rejects_bad_input():
    png = tiny_png()
    # Too many images
    files = [("images", (f"{i}.png", png, "image/png")) for i in range(6)]
    assert client.post("/api/generate", files=files).status_code == 400
    # Not an image
    r = client.post("/api/generate", files=[("images", ("a.txt", b"hello", "text/plain"))])
    assert r.status_code == 415
    # Bad platform
    r = client.post("/api/generate", files=[("images", ("p.png", png, "image/png"))], data={"platform": "ebay"})
    assert r.status_code == 400


def test_eval_endpoint():
    r = client.get("/api/eval")
    assert r.status_code == 200
    assert "available" in r.json()
