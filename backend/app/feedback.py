"""Seller feedback: one JSON line per answer in data/feedback.jsonl."""

import json
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Only one thread may write to the file at a time, so lines never get mixed up
_file_lock = threading.Lock()


class Feedback(BaseModel):
    request_id: str = Field("", max_length=64)
    rating: Literal["up", "down"] | None = None
    used: bool = False
    would_pay: Literal["yes", "maybe", "no"] | None = None
    price_ok_inr: float | None = Field(None, ge=0, le=100_000)
    comment: str = Field("", max_length=2000)
    listing: dict | None = None

    # Stop very large listings from filling up the feedback file
    @field_validator("listing")
    @classmethod
    def listing_not_too_big(cls, value):
        if value is not None and len(json.dumps(value, ensure_ascii=False)) > 20_000:
            raise ValueError("listing is too large")
        return value


def save_feedback(item: Feedback, path: Path) -> None:
    """Append one feedback line (with a timestamp), creating the folder if needed."""
    record = {"ts": int(time.time()), **item.model_dump()}
    line = json.dumps(record, ensure_ascii=False)
    with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def feedback_summary(path: Path) -> dict:
    """Count thumbs up/down, "used it" and "would pay" answers."""
    summary = {"count": 0, "up": 0, "down": 0, "used": 0,
               "would_pay": {"yes": 0, "maybe": 0, "no": 0}}
    if not path.exists():
        return summary
    with _file_lock:
        lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip a broken line instead of failing the whole summary
        summary["count"] += 1
        if row.get("rating") in ("up", "down"):
            summary[row["rating"]] += 1
        if row.get("used"):
            summary["used"] += 1
        if row.get("would_pay") in summary["would_pay"]:
            summary["would_pay"][row["would_pay"]] += 1
    return summary
