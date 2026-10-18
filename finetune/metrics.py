"""Pure metric functions for the base-vs-fine-tuned evaluation (no network, easy to test).

Every function takes plain Python dicts/strings, so pytest can check them offline.
"""

import json
import re

from common import prompts

LISTING_KEYS = ["title", "description", "bullets", "tags", "search_query"]


# ---------------------------------------------------------------- 1. format validity
def parse_output(text: str, platform: str) -> tuple[dict | None, dict | None, str]:
    """Parse raw model text.

    Returns (clean_listing, raw_json, error). clean_listing comes from the shared
    prompts.parse_listing (exactly what the app shows). raw_json is the untouched JSON
    object, needed for constraint checks because parse_listing trims extra tags.
    """
    try:
        listing = prompts.parse_listing(text, platform)
    except (ValueError, TypeError, json.JSONDecodeError) as err:
        return None, None, str(err) or err.__class__.__name__
    raw = _raw_json(text)
    return listing, raw, ""


def _raw_json(text: str) -> dict:
    """Grab the first {...} block as a dict (same idea as prompts.py, kept local on purpose)."""
    text = re.sub(r"```(?:json)?", "", text)
    data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    return data if isinstance(data, dict) else {}


def is_format_valid(listing: dict | None) -> bool:
    """Valid = parsed OK and every field is non-empty."""
    if not listing:
        return False
    return all(bool(listing.get(k)) for k in LISTING_KEYS)


# ---------------------------------------------------------------- 2. constraint compliance
def count_paragraphs(text: str) -> int:
    """Number of paragraphs separated by blank lines."""
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def constraint_checks(raw: dict | None, platform: str) -> dict[str, bool]:
    """Check the platform rules from the prompt. All False if the output did not parse."""
    names = ["title_length_ok", "bullets_exactly_5", "tags_ok", "description_paragraphs_ok", "search_query_ok"]
    if not raw:
        return {n: False for n in names}
    rules = prompts.PLATFORM_RULES.get(platform, prompts.PLATFORM_RULES["meesho"])

    title = str(raw.get("title") or "").strip()
    bullets = raw.get("bullets") if isinstance(raw.get("bullets"), list) else []
    tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
    query_words = len(str(raw.get("search_query") or "").split())
    paragraphs = count_paragraphs(str(raw.get("description") or ""))
    return {
        "title_length_ok": 0 < len(title) <= rules["title_max"],
        "bullets_exactly_5": len([b for b in bullets if str(b).strip()]) == 5,
        "tags_ok": 1 <= len(tags) <= rules["tags_max"]
        and all(0 < len(str(t).strip()) <= rules["tag_len_max"] for t in tags),
        "description_paragraphs_ok": 2 <= paragraphs <= 4,
        "search_query_ok": 3 <= query_words <= 6,
    }


def is_compliant(checks: dict[str, bool]) -> bool:
    """True only when every single rule passes."""
    return bool(checks) and all(checks.values())


# ---------------------------------------------------------------- 3. attribute coverage
def attribute_values(attributes: dict) -> list[str]:
    """The attribute values we expect the listing to mention."""
    values = []
    for key in ("product_type", "material", "pattern", "style"):
        if attributes.get(key):
            values.append(str(attributes[key]))
    values += [str(c) for c in attributes.get("colors") or [] if c]
    return values


def _mentioned(value: str, text: str) -> bool:
    """A value counts as mentioned if each of its words (ignoring plural 's') appears in the text."""
    words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) > 1]
    if not words:
        return False
    text_words = {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower())}
    return all(w.rstrip("s") in text_words for w in words)


def attribute_coverage(attributes: dict, listing: dict | None) -> float | None:
    """Share of attribute values found in the title + description (None if nothing to check)."""
    values = attribute_values(attributes)
    if not values:
        return None
    if not listing:
        return 0.0
    text = f"{listing.get('title', '')} {listing.get('description', '')}"
    return sum(_mentioned(v, text) for v in values) / len(values)


# ---------------------------------------------------------------- repetition (keyword stuffing / loops)
def ngram_repeat_ratio(text: str, n: int = 4) -> float:
    """Share of word n-grams that are repeats of an earlier n-gram (0 = no repetition).

    "10 Pack Door Stopper, 10 Pack Door Stopper, ..." scores close to 1.
    """
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def max_field_repeat_ratio(listing: dict | None, n: int = 4) -> float:
    """Worst repeat ratio over title, description and bullets, each checked on its own.

    Fields are checked separately because a title phrase showing up again in the
    description is normal; the same phrase looping inside one field is not.
    """
    if not listing:
        return 0.0
    bullets = " ".join(str(b) for b in listing.get("bullets") or [])
    fields = [listing.get("title", ""), listing.get("description", ""), bullets]
    return max(ngram_repeat_ratio(f, n) for f in fields)


# ---------------------------------------------------------------- 4. ROUGE-L
def listing_text(listing: dict | None) -> str:
    """Flatten a listing into one string for text overlap scores."""
    if not listing:
        return ""
    bullets = listing.get("bullets") or []
    return "\n".join([str(listing.get("title", "")), str(listing.get("description", ""))] + [str(b) for b in bullets])


_scorer = None


def rouge_l(prediction: dict | None, reference: dict) -> float:
    """ROUGE-L F1 between the generated and the reference listing text (0 if invalid)."""
    global _scorer
    if not prediction:
        return 0.0
    if _scorer is None:
        from rouge_score import rouge_scorer

        _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _scorer.score(listing_text(reference), listing_text(prediction))["rougeL"].fmeasure


# ---------------------------------------------------------------- 5. judge helpers
def parse_judge_verdict(text: str) -> str:
    """Read 'A', 'B' or 'TIE' out of the judge's reply (defaults to TIE if unclear)."""
    # First try the JSON reply we asked for: {"winner": "A"}
    try:
        verdict = str(_raw_json(text).get("winner", ""))
    except ValueError:
        verdict = ""
    if not verdict:
        m = re.search(r"\b(TIE|A|B)\b", text.upper())
        verdict = m.group(1) if m else "TIE"
    verdict = verdict.strip().upper()
    return verdict if verdict in ("A", "B", "TIE") else "TIE"


def verdict_to_winner(verdict: str, finetuned_is_a: bool) -> str:
    """Map the blind A/B verdict back to 'finetuned', 'base' or 'tie'."""
    if verdict == "TIE":
        return "tie"
    a_won = verdict == "A"
    return "finetuned" if a_won == finetuned_is_a else "base"


def judge_tally(winners: list[str]) -> dict:
    """Win rates from a list of 'finetuned' / 'base' / 'tie'."""
    n = len(winners)
    if n == 0:
        return {"finetuned_win": None, "base_win": None, "tie": None, "n": 0}
    return {
        "finetuned_win": round(winners.count("finetuned") / n, 4),
        "base_win": round(winners.count("base") / n, 4),
        "tie": round(winners.count("tie") / n, 4),
        "n": n,
    }


# ---------------------------------------------------------------- per-example + summary
def score_example(text: str, row: dict) -> dict:
    """All automatic metrics for one model output."""
    platform = row["platform"]
    listing, raw, error = parse_output(text, platform)
    checks = constraint_checks(raw, platform)
    valid = is_format_valid(listing)
    return {
        "listing": listing,
        "error": error,
        "format_valid": valid,
        "constraint_compliance": valid and is_compliant(checks),
        **checks,
        "attribute_coverage": attribute_coverage(row["attributes"], listing),
        "rouge_l": rouge_l(listing, row["reference"]),
    }


METRIC_NAMES = [
    "format_valid",
    "constraint_compliance",
    "title_length_ok",
    "bullets_exactly_5",
    "tags_ok",
    "description_paragraphs_ok",
    "search_query_ok",
    "attribute_coverage",
    "rouge_l",
]


def mean(values) -> float | None:
    """Average that skips None values (None if nothing left)."""
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(scores: list[dict]) -> dict:
    """Average every metric over the test set."""
    return {name: mean(s[name] for s in scores) for name in METRIC_NAMES}
