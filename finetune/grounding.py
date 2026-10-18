"""Pure helpers for build_grounded_data.py (no network, tested in tests/test_grounding.py).

The idea: a big TEACHER model reads the original Amazon listing and
  1. EXTRACTS product attributes (the same JSON Gemma vision gives the app), then
  2. REWRITES the listing so it states ONLY facts found in those attributes / notes.
This file builds the teacher prompts, parses the teacher's answers and checks them.
"""

import json
import re

import metrics
from common import prompts

ATTRIBUTE_KEYS = [
    "product_type", "category", "colors", "material", "pattern", "style",
    "gender", "key_features", "visible_text", "condition",
]
GENDERS = {"women", "men", "unisex", "kids", "girls", "boys", ""}
MAX_KEY_FEATURES = 8
MAX_REPEAT_RATIO = 0.05   # max share of repeated 4-word phrases inside one field
MIN_COVERAGE = 0.8        # share of attribute values the title + description must mention


# ---------------------------------------------------------------- original listing
def original_text(row: dict) -> tuple[str, bool]:
    """The original listing as plain text for the teacher, plus True if it is the real original.

    Rows made by an older prepare_data.py have no source.original; then we fall back to the
    (lossy, brand-stripped) reference listing and return False so the caller can warn.
    """
    orig = (row.get("source") or {}).get("original")
    if orig:
        lines = [f"Title: {orig.get('title', '')}"]
        if orig.get("categories"):
            lines.append("Categories: " + " > ".join(orig["categories"]))
        if orig.get("features"):
            lines.append("Features:\n" + "\n".join(f"- {f}" for f in orig["features"]))
        if orig.get("description"):
            lines.append(f"Description: {orig['description']}")
        if orig.get("details"):
            lines.append("Details:\n" + "\n".join(f"- {k}: {v}" for k, v in orig["details"].items()))
        return "\n".join(lines), True
    ref = row["reference"]
    text = (f"Title: {ref['title']}\nFeatures:\n" + "\n".join(f"- {b}" for b in ref["bullets"])
            + f"\nDescription: {ref['description']}")
    return text, False


def brand_names(row: dict) -> list[str]:
    """Shop / brand names that must never appear in attributes or listings."""
    orig = (row.get("source") or {}).get("original") or {}
    names = []
    for name in (orig.get("store", ""), orig.get("brand", "")):
        name = str(name or "").strip()
        if len(name) >= 3 and name.lower() not in ("generic", "unknown", "none") and name.lower() not in names:
            names.append(name.lower())
    return names


def mentions_brand(text: str, brands: list[str]) -> str:
    """Return the first brand name found as a whole word in the text, else ''."""
    low = str(text).lower()
    for brand in brands:
        if re.search(r"(?<![a-z0-9])" + re.escape(brand) + r"(?![a-z0-9])", low):
            return brand
    return ""


# ---------------------------------------------------------------- step 1: EXTRACT
EXTRACT_SYSTEM = (
    "You are a careful product data annotator. You read a real marketplace listing and write down "
    "the product facts it states, as JSON. You never guess and never add facts that are not in the listing. "
    "Reply with ONE JSON object and nothing else."
)


def extract_messages(row: dict) -> list[dict]:
    """Chat messages asking the teacher to turn the original listing into ProductAttributes JSON."""
    listing, _ = original_text(row)
    brands = brand_names(row)
    avoid = ", ".join(brands) if brands else "none given"
    user = (
        f"Original listing:\n{listing}\n\n"
        "Write the product attributes as JSON with exactly these keys:\n"
        '- "product_type": short product noun, e.g. "kurti", "paper napkins", "door stopper"\n'
        '- "category": short shop category, e.g. "Women Ethnic Wear", "Home Kitchen"\n'
        '- "colors": list of the product colours (lowercase), [] if not stated\n'
        '- "material": the main material actually stated, "" if not stated (do not guess from the theme or name)\n'
        '- "pattern": e.g. "floral", "solid", "striped", "" if not stated\n'
        '- "style": e.g. "casual", "boho", "modern", "" if not stated\n'
        '- "gender": one of "women", "men", "unisex", "kids", "girls", "boys", or ""\n'
        f'- "key_features": up to {MAX_KEY_FEATURES} short phrases (max 10 words each) with concrete TRUE facts '
        "from the listing: count / pack size, size or dimensions with units, capacity, main uses, notable parts. "
        "Copy numbers exactly. No marketing words, no brand names, no care labels\n"
        '- "visible_text": ""\n'
        '- "condition": "new"\n\n'
        f"Never include these brand / shop names anywhere: {avoid}."
    )
    return [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": user}]


def _json_object(text: str) -> dict:
    """First {...} block in the text as a dict (raises ValueError if there is none)."""
    text = re.sub(r"```(?:json)?", "", str(text))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def parse_attributes(text: str) -> dict:
    """Parse the teacher's answer into a clean ProductAttributes dict (SPEC shape)."""
    data = _json_object(text)

    def as_text(value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def as_list(value) -> list[str]:
        items = value if isinstance(value, list) else str(value or "").split(",")
        return [as_text(v) for v in items if as_text(v)]

    gender = as_text(data.get("gender")).lower()
    return {
        "product_type": as_text(data.get("product_type")).lower(),
        "category": as_text(data.get("category")),
        "colors": [c.lower() for c in as_list(data.get("colors"))][:4],
        "material": as_text(data.get("material")).lower(),
        "pattern": as_text(data.get("pattern")).lower(),
        "style": as_text(data.get("style")).lower(),
        "gender": gender if gender in GENDERS else "",
        "key_features": as_list(data.get("key_features"))[:MAX_KEY_FEATURES],
        "visible_text": "",
        "condition": as_text(data.get("condition")).lower() or "new",
    }


def attribute_problems(attrs: dict, brands: list[str]) -> list[str]:
    """Plain-English problems with extracted attributes ([] = good)."""
    problems = []
    if not attrs.get("product_type"):
        problems.append("product_type is empty")
    if len(attrs.get("key_features") or []) < 2:
        problems.append("give at least 2 key_features with concrete facts")
    long = [f for f in attrs.get("key_features") or [] if len(f.split()) > 12]
    if long:
        problems.append(f"key_features must be short phrases (max 10 words): {long[0]!r}")
    brand = mentions_brand(json.dumps(attrs, ensure_ascii=False), brands)
    if brand:
        problems.append(f"remove the brand name {brand!r}")
    return problems


# ---------------------------------------------------------------- step 2: REWRITE
REWRITE_SYSTEM = (
    "You are an expert e-commerce copywriter who writes the perfect TRAINING EXAMPLE for a small listing-writer "
    "model. The small model will only ever see the product attributes and seller notes, so your listing must "
    "state ONLY facts found there. Reply with ONE JSON object and nothing else."
)


def rewrite_messages(row: dict, attrs: dict) -> list[dict]:
    """Chat messages asking the teacher to write the grounded target listing for this row."""
    platform = row["platform"]
    rules = prompts.PLATFORM_RULES[platform]
    writer_prompt = prompts.build_writer_messages(attrs, platform, row.get("notes", ""))[1]["content"]
    listing, _ = original_text(row)
    brands = brand_names(row)
    user = (
        "This is EXACTLY what the small model sees:\n"
        f"<<<\n{writer_prompt}\n>>>\n\n"
        "The real listing this product came from (use it ONLY for tone, useful wording and which facts matter; "
        f"do NOT copy any fact that is not in the attributes or notes above):\n<<<\n{listing}\n>>>\n\n"
        "Write the ideal answer. Rules:\n"
        "- Every fact (numbers, sizes, counts, materials, uses) must come from the attributes or seller notes. "
        "Never add sizes, dimensions, counts, certifications or claims that are not there.\n"
        f"- Mention the product_type, every colour, the material, pattern and style (when not empty) "
        "in the title or description.\n"
        f'- "title": at most {rules["title_max"]} characters, main keywords first, no repeated words or phrases.\n'
        '- "description": 2 to 4 short paragraphs separated by a blank line ("\\n\\n"), friendly and specific.\n'
        '- "bullets": exactly 5 different benefit-focused bullets, each one sentence.\n'
        f'- "tags": 8 to {rules["tags_max"]} different search tags, each at most {rules["tag_len_max"]} characters.\n'
        '- "search_query": 3 to 6 words.\n'
        "- No keyword stuffing: never repeat the same phrase.\n"
        "- No brand or shop names" + (f" (never write: {', '.join(brands)})" if brands else "")
        + ", no prices, shipping, warranty or guarantee talk.\n"
        'Reply with ONE JSON object with keys "title", "description", "bullets", "tags", "search_query".'
    )
    return [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}]


def feedback_messages(messages: list[dict], answer: str, problems: list[str]) -> list[dict]:
    """The same chat plus the teacher's last answer and what was wrong with it (for one retry)."""
    fix = "Your answer has these problems:\n" + "\n".join(f"- {p}" for p in problems)
    fix += "\nWrite the corrected JSON object only."
    return messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": fix}]


# ---------------------------------------------------------------- step 3: VALIDATE
def _numbers(text: str) -> set[str]:
    """Numbers in a text, e.g. '10.5 x 10.5 inches' -> {'10.5'}."""
    return set(re.findall(r"\d+(?:\.\d+)?", str(text)))


def unsupported_numbers(listing: dict, attrs: dict, notes: str) -> list[str]:
    """Numbers in the listing that are not in the attributes or notes (= invented facts)."""
    allowed = _numbers(json.dumps(attrs, ensure_ascii=False) + " " + str(notes or ""))
    text = json.dumps({k: listing.get(k) for k in metrics.LISTING_KEYS}, ensure_ascii=False)
    return sorted(_numbers(text) - allowed)


CHECK_MESSAGES = {
    "title_length_ok": "title is too long for the platform (or empty)",
    "bullets_exactly_5": "need exactly 5 bullets",
    "tags_ok": "need 1-13 tags, each at most 20 characters",
    "description_paragraphs_ok": 'description needs 2-4 paragraphs separated by a blank line ("\\n\\n")',
    "search_query_ok": "search_query must be 3-6 words",
}


def listing_problems(text: str, attrs: dict, notes: str, platform: str, brands: list[str]) -> tuple[dict | None, list[str]]:
    """Check one teacher listing. Returns (listing or None, plain-English problems; [] = good)."""
    listing, raw, error = metrics.parse_output(text, platform)
    if not metrics.is_format_valid(listing):
        return None, [f"not a valid listing JSON with all 5 non-empty fields ({error or 'empty field'})"]
    problems = [CHECK_MESSAGES[name] for name, ok in metrics.constraint_checks(raw, platform).items() if not ok]
    ratio = metrics.max_field_repeat_ratio(listing)
    if ratio > MAX_REPEAT_RATIO:
        problems.append(f"it repeats the same phrases ({ratio:.0%} repeated 4-word phrases); say each thing once")
    bullets = [b.lower() for b in listing["bullets"]]
    if len(set(bullets)) < len(bullets):
        problems.append("bullets must all be different")
    tags = [str(t).strip().lower() for t in raw.get("tags") or []]
    if len(set(tags)) < len(tags):
        problems.append("tags must all be different")
    coverage = metrics.attribute_coverage(attrs, listing)
    if coverage is not None and coverage < MIN_COVERAGE:
        missing = [v for v in metrics.attribute_values(attrs)
                   if not metrics._mentioned(v, f"{listing['title']} {listing['description']}")]
        problems.append(f"title/description must mention these attribute values: {missing}")
    invented = unsupported_numbers(listing, attrs, notes)
    if invented:
        problems.append(f"these numbers are not in the attributes or notes, remove them: {invented}")
    brand = mentions_brand(json.dumps(listing, ensure_ascii=False), brands)
    if brand:
        problems.append(f"remove the brand name {brand!r}")
    return listing, problems


# ---------------------------------------------------------------- cost estimate
def rough_tokens(text: str) -> int:
    """Rough token count for English text (about 3.5 characters per token)."""
    return max(1, round(len(text) / 3.5))


def estimate_cost(prefill_tokens: float, sample_tokens: float, usd_prefill_1m: float, usd_sample_1m: float) -> float:
    """USD cost of a run from token counts and per-1M-token prices."""
    return prefill_tokens / 1e6 * usd_prefill_1m + sample_tokens / 1e6 * usd_sample_1m


# ---------------------------------------------------------------- mock teacher (dry run only)
def mock_attributes(row: dict) -> dict:
    """Offline stand-in for the teacher's EXTRACT answer: the old rule-based attributes, cleaned up."""
    a = dict(row["attributes"])
    feats = [" ".join(f.split()[:8]) for f in a.get("key_features") or []]
    a["key_features"] = [f for f in feats if f][:MAX_KEY_FEATURES] or [f"everyday {a.get('product_type') or 'item'}"]
    if len(a["key_features"]) < 2:
        a["key_features"].append("easy to use")
    a["visible_text"] = ""
    return a


def mock_listing(row: dict, attrs: dict, sloppy: bool = False) -> dict:
    """Offline stand-in for the teacher's REWRITE answer. sloppy=True makes a looping one (to test retries)."""
    rules = prompts.PLATFORM_RULES[row["platform"]]
    ptype = attrs.get("product_type") or "product"
    words = [*(attrs.get("colors") or []), attrs.get("material", ""), attrs.get("pattern", ""), attrs.get("style", "")]
    words = [w for w in words if w]
    title = " ".join(w.title() for w in words + [ptype])
    if attrs.get("gender"):
        title += f" for {attrs['gender'].title()}"
    feats = list(attrs.get("key_features") or [])
    para1 = f"Meet this {' '.join(words + [ptype])}, made for everyday use."
    para2 = "Key details: " + "; ".join(feats[:4]) + "." if feats else "Simple, useful and easy to care for."
    description = para1 + "\n\n" + para2
    if sloppy:
        description = " ".join([f"{ptype} {ptype} best {ptype}"] * 6) + "\n\n" + para2
    bullets = [f"{f[:1].upper()}{f[1:]}" for f in feats[:5]]
    fillers = ["Looks neat and well made", "Easy to use every day", "Simple to care for",
               "Nice choice for a gift", "Fits many settings"]
    bullets += [f for f in fillers if f not in bullets][: 5 - len(bullets)]
    tags = [ptype] + [f"{w} {ptype}" for w in words] + words
    tags = [t for i, t in enumerate(tags) if len(t) <= rules["tag_len_max"] and t not in tags[:i]][: rules["tags_max"]]
    query = " ".join((words[:2] + ptype.split() + ["online"])[:6])
    if len(query.split()) < 3:
        query += " buy online"
    return {
        "title": title[: rules["title_max"]].strip(),
        "description": description,
        "bullets": bullets,
        "tags": tags or [ptype],
        "search_query": query,
    }
