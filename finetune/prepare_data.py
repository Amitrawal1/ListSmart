"""Build the ListSmart fine-tuning dataset from real, well-rated Amazon listings.

Source: HuggingFace dataset "McAuley-Lab/Amazon-Reviews-2023", files
raw/meta_categories/meta_<Category>.jsonl (product metadata, public, no login).
The files are huge (GBs), so we STREAM them line by line over HTTP and stop as
soon as we have enough good listings.

Each good listing becomes one row:
  {"id", "platform", "notes", "attributes": ProductAttributes, "reference": Listing, "source": {...}}
Attributes are extracted from the listing text with simple rules (no vision needed),
and the reference Listing is cleaned up to obey the platform limits in prompts.py.

source.original keeps the cleaned ORIGINAL listing text (title, description, features,
details, categories, brand) so build_grounded_data.py can show it to a teacher model.

Usage:
  python prepare_data.py --dry-run            # quick look at a few rows, writes nothing
  python prepare_data.py --limit 300          # small dataset for testing
  python prepare_data.py                      # full dataset (~2100 rows)
  python prepare_data.py --fill-source        # only ADD source.original to the existing data/*.jsonl
                                              # (same rows, same ids, nothing else changes)
"""

import argparse
import html
import json
import math
import os
import random
import re
import sys

import requests

from common import DATA_DIR, prompts, read_jsonl, write_jsonl

HF_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_{category}.jsonl"
)
DEFAULT_CATEGORIES = ["Clothing_Shoes_and_Jewelry", "Home_and_Kitchen", "Handmade_Products"]

# ---------------------------------------------------------------- vocabularies
COLORS = [
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple", "orange", "brown",
    "grey", "gray", "beige", "maroon", "navy", "gold", "silver", "cream", "ivory", "teal",
    "turquoise", "olive", "khaki", "peach", "coral", "lavender", "mustard", "multicolor",
    "burgundy", "tan", "charcoal", "mint", "magenta", "wine", "copper", "bronze", "rose gold",
]
MATERIALS = [
    "stainless steel", "sterling silver", "cotton", "silk", "polyester", "wool", "leather",
    "linen", "denim", "rayon", "viscose", "nylon", "satin", "velvet", "chiffon", "georgette",
    "crepe", "jute", "bamboo", "wooden", "wood", "ceramic", "glass", "brass", "copper",
    "plastic", "acrylic", "resin", "metal", "iron", "aluminum", "clay", "marble", "stone",
    "cashmere", "fleece", "spandex", "lace", "canvas", "suede", "porcelain", "terracotta",
    "soy wax", "beeswax", "felt", "macrame", "microfiber", "gold plated", "silver plated",
]
PATTERNS = [
    "polka dot", "tie dye", "block print", "animal print", "floral", "striped", "stripe",
    "checked", "plaid", "solid", "printed", "embroidered", "geometric", "paisley", "abstract",
    "leopard", "camo", "chevron", "damask", "ikat", "graphic", "quilted", "knit",
]
STYLES = [
    "casual", "formal", "vintage", "boho", "bohemian", "modern", "rustic", "classic",
    "minimalist", "ethnic", "traditional", "farmhouse", "sporty", "elegant", "retro",
]
# Longer phrases first so "tote bag" wins over "bag"
PRODUCT_TYPES = sorted(
    [
        "kurti", "kurta", "saree", "dress", "t-shirt", "shirt", "top", "blouse", "jeans",
        "pants", "leggings", "skirt", "shorts", "jacket", "hoodie", "sweater", "cardigan",
        "coat", "scarf", "hat", "beanie", "cap", "socks", "shoes", "sneakers", "sandals",
        "boots", "slippers", "tote bag", "handbag", "backpack", "bag", "wallet", "purse",
        "belt", "watch", "necklace", "earrings", "bracelet", "ring", "pendant", "anklet",
        "keychain", "brooch", "hair clip", "headband", "mug", "cup", "bowl", "plate", "vase",
        "candle", "pillow cover", "cushion cover", "pillow", "blanket", "throw", "quilt",
        "bed sheet", "towel", "curtain", "rug", "doormat", "placemat", "lamp", "wall art",
        "wall decor", "clock", "mirror", "basket", "planter", "coaster", "tray", "jar",
        "bottle", "soap", "ornament", "picture frame", "napkin", "apron", "tablecloth",
        "cutting board", "knife", "spoon", "pan", "pot", "storage box", "organizer",
        "journal", "notebook", "bookmark", "doll", "pajamas", "swimsuit", "bra", "panties",
        "gloves", "tie", "sunglasses", "jumpsuit", "romper", "vest", "tunic", "kimono",
        "cosmetic bag", "pouch", "wind chime", "dream catcher", "garland", "sign", "tumbler",
    ],
    key=len,
    reverse=True,
)
# Phrases that hint the text is talking about the shop, not the product
BAD_SENTENCE_WORDS = [
    "amazon", "warranty", "click", "add to cart", "customer service", "refund", "seller",
    "contact us", "money back", "http", "www.", "@", "guarantee", "satisfaction",
    "shipped", "shipping", "storefront", "visit our", "our store", "our shop", "we manufacture", "our company", "restocking", "prices", "order now", "buy now",
]
FILLER_WORDS = set("a an the and or for with of to in on at by from is are your our & -".split())
STOPWORDS = set(
    "a an the and or for with of to in on at by from is are this that your you our it as be "
    "set pack pcs piece pieces new size sizes inch inches cm mm x".split()
)


# ---------------------------------------------------------------- text helpers
def clean_text(text: str) -> str:
    """Remove HTML, trademark symbols and extra spaces."""
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[®™©]", "", text)
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def remove_brand(text: str, brand: str) -> str:
    """Drop the shop/brand name, because the prompt says never invent brands."""
    if brand and brand.lower() not in ("generic", "unknown", "none") and len(brand) >= 3:
        text = re.sub(re.escape(brand) + r"(['’]s)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" -|,")


def is_english(text: str) -> bool:
    """Rough English check: almost all ASCII and contains common English words."""
    if not text:
        return False
    ascii_share = sum(c.isascii() for c in text) / len(text)
    words = set(re.findall(r"[a-z]+", text.lower()))
    return ascii_share > 0.97 and len(words & {"the", "and", "with", "for", "is", "this"}) >= 2


def cut_at_word(text: str, max_len: int) -> str:
    """Shorten text to max_len characters without cutting a word in half."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    words = text[:max_len].rsplit(" ", 1)[0].split()
    # Do not end on a dangling filler word like "for" or "with"
    while len(words) > 1 and words[-1].lower().strip(",;:") in FILLER_WORDS:
        words.pop()
    return " ".join(words).rstrip(" ,;:-|&/(")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (simple regex, good enough for product copy)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if 15 < len(p.strip()) <= 300]


def find_terms(text: str, vocab: list[str]) -> list[str]:
    """Return vocabulary words that appear as whole words in the text (in vocab order)."""
    low = " " + text.lower() + " "
    found = []
    for term in vocab:
        # Allow simple plurals too ("curtains" matches "curtain")
        if re.search(r"(?<![a-z])" + re.escape(term) + r"(?:s|es)?(?![a-z])", low):
            # Skip "wood" if we already found "wooden", "stripe" if "striped" etc.
            if not any(term in f or f in term for f in found):
                found.append(term)
    return found


def is_useful_feature(feature: str) -> bool:
    """Keep real selling points, drop Amazon boilerplate like 'Pull On closure' or care labels."""
    low = feature.lower()
    if not (25 <= len(feature) <= 400) or len(feature.split()) < 4:
        return False
    boring = ["closure", "imported", "machine wash", "hand wash", "do not bleach", "dry clean", "asin", "model number"]
    return not any(b in low for b in boring + BAD_SENTENCE_WORDS)


def cut_bullet(text: str, max_len: int = 180) -> str:
    """Shorten a bullet to whole sentences when possible, else at a word boundary."""
    if len(text) <= max_len:
        return text
    kept = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(kept) + len(sent) + 1 > max_len:
            break
        kept = (kept + " " + sent).strip()
    return kept if len(kept) >= 40 else cut_at_word(text, max_len)


def sentence_case_prefix(feature: str) -> str:
    """Turn 'SOFT FABRIC: lovely cotton' into 'Soft fabric: lovely cotton'."""
    m = re.match(r"^([A-Z0-9 ,&'/\-]{3,50})\s*(?::|\u2013| - )\s*(.*)$", feature)
    if m:
        return m.group(1).strip().capitalize() + ": " + m.group(2).strip()
    if feature.isupper():
        return feature.capitalize()
    return feature


# ---------------------------------------------------------------- attribute extraction
def detect_product_type(title: str, categories: list[str]) -> str:
    """Find the product noun in the title, else fall back to the last category name."""
    found = find_terms(title, PRODUCT_TYPES)
    if found:
        # Pick the one that appears first in the title (usually the main noun phrase)
        low = title.lower()
        return min(found, key=lambda t: low.find(t))
    generic = {"women", "men", "girls", "boys", "baby", "clothing", "accessories", "novelty", "jewelry", "home decor"}
    leaf = categories[-1].split("&")[0].split(",")[0].strip().lower() if categories else ""
    if leaf and leaf not in generic and len(leaf.split()) <= 3:
        return leaf[:-1] if leaf.endswith("s") and not leaf.endswith("ss") else leaf
    # Otherwise use the last word of the first part of the title ("... Water-Resistant Pullover")
    head = re.split(r"\s[-|,(]|,|\(", title)[0]
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", head)
    return words[-1].lower() if words else "product"


def detect_gender(title: str, details: dict, categories: list[str]) -> str:
    """Read gender from the Department field, title or categories."""
    text = " ".join([str(details.get("Department", "")), title, " ".join(categories)]).lower()
    if re.search(r"\bunisex\b", text):
        return "unisex"
    if re.search(r"\b(kids?|child|children|toddlers?|baby|babies)\b", text) or (
        re.search(r"\bgirls?\b", text) and re.search(r"\bboys?\b", text)
    ):
        return "kids"
    if re.search(r"\b(women|womens|women's|woman|ladies)\b", text):
        return "women"
    if re.search(r"\b(men|mens|men's|man)\b", text):
        return "men"
    if re.search(r"\bgirls?\b", text):
        return "girls"
    if re.search(r"\bboys?\b", text):
        return "boys"
    return ""


def extract_attributes(meta: dict, title: str, features: list[str], description: str) -> dict:
    """Rule-based ProductAttributes (same keys Gemma vision returns in the app)."""
    details = meta.get("details") or {}
    categories = [c for c in (meta.get("categories") or []) if c]
    text = " ".join([title] + features + [description])

    colors = find_terms(str(details.get("Color", "")) + " " + title, COLORS)
    material_field = str(details.get("Material", "") or details.get("Fabric Type", ""))
    materials = find_terms(material_field + " " + title, MATERIALS) or find_terms(text, MATERIALS)
    patterns = find_terms(title + " " + " ".join(features), PATTERNS)
    styles = find_terms(str(details.get("Style", "")) + " " + text, STYLES)

    # Short key features: the caps prefix ("Soft fabric") or the first few words
    key_features = []
    for feat in features:
        m = re.match(r"^(\S+ [^:]{2,48}):", feat)
        short = m.group(1) if m else cut_at_word(re.split(r"[.;!,\u2013]| - ", feat)[0], 60)
        short = short.strip().lower() if short.isupper() else short.strip()
        if len(short) >= 3 and short.lower() not in [k.lower() for k in key_features]:
            key_features.append(short)
        if len(key_features) == 5:
            break

    category = " ".join(categories[1:3]) if len(categories) > 1 else meta.get("main_category", "")
    return {
        "product_type": detect_product_type(title, categories),
        "category": category,
        "colors": colors[:3],
        "material": materials[0] if materials else "",
        "pattern": patterns[0] if patterns else "",
        "style": styles[0] if styles else "",
        "gender": detect_gender(title, details, categories),
        "key_features": key_features,
        "visible_text": "",
        "condition": "new",
    }


# ---------------------------------------------------------------- reference listing
def build_description(sentences: list[str]) -> str:
    """Group sentences into 2-4 short paragraphs separated by blank lines."""
    sentences = sentences[:8]
    n_paras = max(2, min(4, math.ceil(len(sentences) / 2)))
    size = math.ceil(len(sentences) / n_paras)
    paras = [" ".join(sentences[i : i + size]) for i in range(0, len(sentences), size)]
    return "\n\n".join(paras[:4])


def build_tags(attrs: dict, title: str, is_handmade: bool, max_tags: int, max_len: int) -> list[str]:
    """Search tags from attributes + title keywords, each short enough for the platform."""
    ptype = attrs["product_type"]
    candidates = [ptype]
    for color in attrs["colors"]:
        candidates.append(f"{color} {ptype}")
    for key in ("material", "pattern", "style"):
        if attrs[key]:
            candidates.append(f"{attrs[key]} {ptype}")
    if attrs["gender"] in ("women", "men", "kids", "girls", "boys"):
        candidates.append(f"{ptype} for {attrs['gender']}")
    if is_handmade:
        candidates.append(f"handmade {ptype}")
    candidates += [attrs[k] for k in ("material", "pattern", "style") if attrs[k]]
    # Two-word phrases from the title, skipping filler words and numbers
    skip = STOPWORDS | {"women", "womens", "women's", "men", "mens", "men's", "girls", "boys", "kids"}
    words = [w for w in re.findall(r"[a-z][a-z\-']+", title.lower()) if w not in skip and len(w) >= 4]
    # Only keep two-word phrases that end in the product noun, e.g. "engraved ring"
    noun = ptype.split()[-1]
    candidates += [f"{a} {b}" for a, b in zip(words, words[1:]) if b.rstrip("s") == noun.rstrip("s")]
    candidates += [w for w in words if len(w) >= 5]

    tags, seen = [], set()
    for tag in candidates:
        tag = re.sub(r"\s+", " ", tag).strip()
        if 2 < len(tag) <= max_len and tag not in seen:
            tags.append(tag)
            seen.add(tag)
        if len(tags) == max_tags:
            break
    return tags


def build_search_query(attrs: dict) -> str:
    """3-6 word query used to find competitor prices."""
    words = []
    for part in (attrs["colors"][0] if attrs["colors"] else "", attrs["material"], attrs["product_type"]):
        words += part.split()
    if attrs["gender"] in ("women", "men", "kids", "girls", "boys"):
        words += ["for", attrs["gender"]]
    if len(words) < 3 and attrs["pattern"]:
        words = attrs["pattern"].split() + words
    if len(words) < 3:
        words += [w.lower() for w in attrs["category"].split()[:2]]
    if len(words) < 3:
        words += ["online"]
    return " ".join(words[:6])


def convert(meta: dict, rng: random.Random, min_rating: float, min_reviews: int) -> dict | None:
    """Turn one raw Amazon metadata record into a training row, or None if it is not good enough."""
    # Basic quality filters: well rated, enough reviews, real features and description
    if (meta.get("average_rating") or 0) < min_rating or (meta.get("rating_number") or 0) < min_reviews:
        return None
    brand = clean_text(meta.get("store") or "")
    detail_brand = clean_text((meta.get("details") or {}).get("Brand", ""))
    title = remove_brand(remove_brand(clean_text(meta.get("title")), brand), detail_brand)
    if not (20 <= len(title) <= 250):
        return None
    features = [remove_brand(remove_brand(clean_text(f), brand), detail_brand) for f in (meta.get("features") or [])]
    # Strip decorative symbols like "※" or "★" from the start of each feature
    features = [re.sub(r"^[^A-Za-z0-9]+", "", f) for f in features]
    features = [f for f in features if is_useful_feature(f)]
    if len(features) < 3:
        return None
    description = remove_brand(remove_brand(clean_text(" ".join(meta.get("description") or [])), brand), detail_brand)
    if len(description) < 120 or not is_english(title + " " + description):
        return None

    # Keep only sentences about the product (no shop talk, links or guarantees)
    sentences, seen = [], set()
    for sent in split_sentences(description):
        key = sent.lower()
        if key not in seen and not any(bad in key for bad in BAD_SENTENCE_WORDS):
            sentences.append(sent)
            seen.add(key)
    if len(sentences) < 2:
        return None

    platform = rng.choice(["meesho", "etsy"])
    rules = prompts.PLATFORM_RULES[platform]
    is_handmade = "handmade" in (meta.get("main_category") or "").lower() or "handmade" in title.lower()
    attrs = extract_attributes(meta, title, features, description)

    # Exactly 5 bullets: features first, then extra description sentences if needed
    bullets = [cut_bullet(sentence_case_prefix(f)) for f in features[:5]]
    for sentence in sentences[8:] + sentences[::-1]:
        if len(bullets) == 5:
            break
        candidate = cut_bullet(sentence)
        if candidate not in bullets:
            bullets.append(candidate)
    if len(bullets) < 5:
        return None

    reference = {
        "title": cut_at_word(title, rules["title_max"]),
        "description": build_description(sentences),
        "bullets": bullets,
        "tags": build_tags(attrs, title, is_handmade, rules["tags_max"], rules["tag_len_max"]),
        "search_query": build_search_query(attrs),
    }
    if len(reference["title"]) < 15 or len(reference["tags"]) < 3:
        return None

    # Some rows get seller notes so the model learns to use them (only notes the text supports)
    full_text = (title + " " + description).lower()
    notes = ""
    if is_handmade and rng.random() < 0.5:
        notes = "Handmade item."
    elif "gift" in full_text and rng.random() < 0.4:
        notes = "Good as a gift."

    return {
        "id": meta.get("parent_asin"),
        "platform": platform,
        "notes": notes,
        "attributes": attrs,
        "reference": reference,
        "source": {
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "main_category": meta.get("main_category"),
            "parent_asin": meta.get("parent_asin"),
            "average_rating": meta.get("average_rating"),
            "rating_number": meta.get("rating_number"),
            "original": original_listing(meta),
        },
    }


def original_listing(meta: dict) -> dict:
    """The original listing text, lightly cleaned (HTML removed) but NOT brand-stripped.

    The teacher model in build_grounded_data.py reads this to pull out true facts.
    We keep the brand names too, so the teacher can be told exactly which names to leave out.
    """
    details = meta.get("details") or {}
    return {
        "title": clean_text(meta.get("title")),
        "description": cut_at_word(clean_text(" ".join(meta.get("description") or [])), 3000),
        "features": [clean_text(f)[:400] for f in (meta.get("features") or [])][:12],
        # Only short detail fields ("Material", "Color", "Item Dimensions" ...), long ones are noise
        "details": {str(k): clean_text(v)[:150] for k, v in details.items() if isinstance(v, (str, int, float))},
        "categories": [c for c in (meta.get("categories") or []) if c],
        "store": clean_text(meta.get("store") or ""),
        "brand": clean_text(details.get("Brand", "")),
    }


# ---------------------------------------------------------------- download + split
def stream_category(category: str, max_scan: int):
    """Yield raw metadata dicts from one category file, streamed over HTTP."""
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    url = HF_URL.format(category=category)
    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        for i, line in enumerate(resp.iter_lines()):
            if i >= max_scan:
                break
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def title_key(row: dict) -> str:
    """Normalised title start, used to drop near-duplicate variants of the same product."""
    return re.sub(r"[^a-z]", "", row["reference"]["title"].lower())[:40]


def collect_rows(categories, per_category, max_scan, seed, min_rating, min_reviews) -> list[dict]:
    """Download and convert listings until each category has enough good rows."""
    rng = random.Random(seed)
    rows, seen_ids, seen_titles = [], set(), set()
    for category in categories:
        kept = scanned = 0
        print(f"[{category}] streaming ...", flush=True)
        for meta in stream_category(category, max_scan):
            scanned += 1
            row = convert(meta, rng, min_rating, min_reviews)
            # Split by product: one row per product id and no near-duplicate titles
            if not row or row["id"] in seen_ids or title_key(row) in seen_titles:
                continue
            seen_ids.add(row["id"])
            seen_titles.add(title_key(row))
            rows.append(row)
            kept += 1
            if kept >= per_category:
                break
        print(f"[{category}] kept {kept} good listings out of {scanned} scanned", flush=True)
    return rows


def split_rows(rows: list[dict], seed: int, min_test: int = 120):
    """Shuffle (seeded) and split into train/val/test by product (each product in one split only)."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test = min_test if n >= 5 * min_test else max(1, int(n * 0.2))
    n_val = max(1, min(100, int(n * 0.05)))
    test, val, train = rows[:n_test], rows[n_test : n_test + n_val], rows[n_test + n_val :]
    return train, val, test


def fill_source(rows: list[dict], out_dir: str) -> None:
    """Add source.original to the rows already in out_dir, matched by product id.

    Nothing else in the files changes, so the test set (and its ids) stays exactly the same.
    `rows` must come from collect_rows() with the same settings that built the files.
    """
    by_id = {r["id"]: r for r in rows}
    files = ["train.jsonl", "val.jsonl", "test.jsonl", "sample_train.jsonl", "sample_test.jsonl"]
    for name in files:
        path = f"{out_dir}/{name}"
        if not os.path.exists(path):
            continue
        existing = read_jsonl(path)
        found = changed_ref = 0
        for row in existing:
            new = by_id.get(row["id"])
            if not new:
                continue
            found += 1
            row.setdefault("source", {})["original"] = new["source"]["original"]
            # Sanity check: the rebuilt row should be the same listing we already had
            if new["reference"] != row["reference"]:
                changed_ref += 1
        write_jsonl(path, existing)
        note = f", {changed_ref} rebuilt references differ (data upstream changed?)" if changed_ref else ""
        print(f"{name}: added source.original to {found}/{len(existing)} rows{note}")
        if found < len(existing):
            print(f"  WARNING: {len(existing) - found} rows not found - run with the same --limit/--categories/--seed as before")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=2100, help="total number of good listings to collect")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--max-scan", type=int, default=300_000, help="max raw lines to read per category")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-reviews", type=int, default=5)
    parser.add_argument("--min-test", type=int, default=120, help="test set size when there is enough data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    parser.add_argument("--sample-size", type=int, default=20, help="rows copied to data/sample_*.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="fetch a handful of rows, print them, write nothing")
    parser.add_argument("--fill-source", action="store_true",
                        help="only add source.original (original listing text) to the existing data files")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting data/test.jsonl even if the new test ids differ from the old ones")
    args = parser.parse_args()

    if args.dry_run:
        rows = collect_rows(args.categories, 2, 3000, args.seed, args.min_rating, args.min_reviews)
        for row in rows[:2]:
            print(json.dumps(row, indent=2, ensure_ascii=False))
        print(f"dry run OK: built {len(rows)} rows, nothing written")
        return

    per_category = math.ceil(args.limit / len(args.categories))
    rows = collect_rows(args.categories, per_category, args.max_scan, args.seed, args.min_rating, args.min_reviews)
    rows = rows[: args.limit]
    if not rows:
        sys.exit("No rows collected - check your internet connection or the filters.")

    if args.fill_source:
        fill_source(rows, args.out_dir)
        return

    train, val, test = split_rows(rows, args.seed, args.min_test)
    if len(test) < args.min_test:
        print(f"WARNING: test set has only {len(test)} rows (SPEC wants >= 100). Use a bigger --limit for the real run.")

    out = args.out_dir
    # The test set must never change silently (results are compared across runs on it)
    old_test = f"{out}/test.jsonl"
    if os.path.exists(old_test) and not args.force:
        old_ids = [r["id"] for r in read_jsonl(old_test)]
        if old_ids != [r["id"] for r in test]:
            sys.exit("ERROR: the rebuilt test set has different product ids than data/test.jsonl. "
                     "Nothing was written. Use --fill-source to only add the original text, "
                     "or --force to really replace the test set.")
    write_jsonl(f"{out}/train.jsonl", train)
    write_jsonl(f"{out}/val.jsonl", val)
    write_jsonl(f"{out}/test.jsonl", test)
    # Small samples that are kept in git so people can inspect the format
    write_jsonl(f"{out}/sample_train.jsonl", train[: args.sample_size])
    write_jsonl(f"{out}/sample_test.jsonl", test[: max(1, args.sample_size // 2)])

    platforms = {p: sum(r["platform"] == p for r in rows) for p in ("meesho", "etsy")}
    print(f"wrote train={len(train)} val={len(val)} test={len(test)} to {out}  platforms={platforms}")


if __name__ == "__main__":
    main()
