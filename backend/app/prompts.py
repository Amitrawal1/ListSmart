"""Shared prompt format for the listing writer.

This file is used by BOTH the web app (backend/app/writer.py) and the
fine-tuning scripts (finetune/). Keep them in sync by only editing here,
otherwise the fine-tuned model sees a different prompt at serving time.
"""

import json
import re

# Limits for each marketplace
PLATFORM_RULES = {
    "meesho": {"title_max": 100, "tags_max": 13, "tag_len_max": 20},
    "etsy": {"title_max": 140, "tags_max": 13, "tag_len_max": 20},
}

SYSTEM_PROMPT = (
    "You are an expert e-commerce copywriter for Indian and global marketplaces. "
    "You write clear, honest, search-friendly product listings. "
    "Only describe what is given in the product attributes, never invent sizes, "
    "brands or certifications. Reply with ONE JSON object and nothing else."
)


def build_writer_messages(attributes: dict, platform: str = "meesho", notes: str = "") -> list[dict]:
    """Build the chat messages sent to the writer model."""
    platform = platform if platform in PLATFORM_RULES else "meesho"
    rules = PLATFORM_RULES[platform]

    user_text = (
        f"Platform: {platform}\n"
        f"Product attributes (JSON):\n{json.dumps(attributes, ensure_ascii=False, sort_keys=True)}\n"
        f"Seller notes: {notes.strip() or 'none'}\n\n"
        "Write the listing as JSON with exactly these keys:\n"
        f'- "title": string, at most {rules["title_max"]} characters, main keywords first\n'
        '- "description": 2 to 4 short paragraphs separated by blank lines\n'
        '- "bullets": list of exactly 5 short benefit-focused bullet points\n'
        f'- "tags": list of up to {rules["tags_max"]} search tags, each at most {rules["tag_len_max"]} characters\n'
        '- "search_query": a short query (3-6 words) to find similar products online'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def listing_to_target(listing: dict) -> str:
    """Turn a listing dict into the exact text the model should learn to output."""
    keys = ["title", "description", "bullets", "tags", "search_query"]
    return json.dumps({k: listing.get(k) for k in keys}, ensure_ascii=False)


def _extract_json_block(text: str) -> str:
    """Pull the first {...} block out of model text (models sometimes add ``` fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


def parse_listing(text: str, platform: str = "meesho") -> dict:
    """Parse model output into a clean Listing dict.

    Raises ValueError if the output is not valid JSON or a field is missing,
    so callers (and the eval) can count it as a format failure.
    """
    data = json.loads(_extract_json_block(text))
    required = ["title", "description", "bullets", "tags", "search_query"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"missing fields: {missing}")

    rules = PLATFORM_RULES.get(platform, PLATFORM_RULES["meesho"])

    # Normalise types so the frontend always gets the same shape
    bullets = data["bullets"] if isinstance(data["bullets"], list) else [str(data["bullets"])]
    tags = data["tags"] if isinstance(data["tags"], list) else str(data["tags"]).split(",")
    return {
        "title": str(data["title"]).strip(),
        "description": str(data["description"]).strip(),
        "bullets": [str(b).strip() for b in bullets if str(b).strip()],
        "tags": [str(t).strip() for t in tags if str(t).strip()][: rules["tags_max"]],
        "search_query": str(data["search_query"]).strip(),
    }
