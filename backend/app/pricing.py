"""Step 3: competitor prices from SerpApi Google Shopping + a simple suggested price.

The math functions at the top are pure (no network) so they are easy to test.
"""

import hashlib
import logging
import math
import re

import requests

from .config import Settings

log = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"
UNDERCUT = 0.95        # suggest 5% under the median
MIN_MARKUP = 1.25      # never go below cost + 25%


# ---------- pure math helpers ----------

def parse_price(value) -> float | None:
    """Turn 1299, "₹1,299.00" or "Rs. 1,299" into 1299.0. Returns None if no number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 and math.isfinite(value) else None
    # Keep digits and dots only, after dropping thousands commas
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    number = float(match.group(0).replace(",", ""))
    return number if number > 0 else None


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default). q is 0..100."""
    if not sorted_values:
        raise ValueError("no values")
    pos = (len(sorted_values) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def remove_outliers_iqr(prices: list[float]) -> list[float]:
    """Drop prices outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Needs 4+ prices to bother."""
    if len(prices) < 4:
        return list(prices)
    s = sorted(prices)
    q1, q3 = percentile(s, 25), percentile(s, 75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [p for p in prices if low <= p <= high]


def compute_stats(prices: list[float]) -> dict:
    """count/min/median/max/p25/p75 of the given prices (rounded to 2 decimals)."""
    if not prices:
        return {"count": 0, "min": None, "median": None, "max": None, "p25": None, "p75": None}
    s = sorted(prices)
    r = lambda x: round(x, 2)  # noqa: E731
    return {
        "count": len(s),
        "min": r(s[0]),
        "median": r(percentile(s, 50)),
        "max": r(s[-1]),
        "p25": r(percentile(s, 25)),
        "p75": r(percentile(s, 75)),
    }


def round_to_ending_9(x: float) -> int:
    """Nearest whole number ending in 9 (474 -> 479, 463 -> 459). Ties go down."""
    lower = math.floor((x - 9) / 10) * 10 + 9   # largest ...9 <= x
    upper = lower + 10
    result = lower if (x - lower) <= (upper - x) else upper
    return max(9, int(result))


def round_up_to_ending_9(x: float) -> int:
    """Smallest whole number ending in 9 that is >= x (used for the cost floor)."""
    return max(9, int(math.ceil((x - 9) / 10) * 10 + 9))


def suggest_price(stats: dict, cost_price: float | None = None) -> tuple[int | None, list[str]]:
    """Apply the SPEC rule. Returns (suggested_price, list of plain-English reasons)."""
    reasons: list[str] = []
    median = stats.get("median")

    if median is None:
        if cost_price:
            price = round_up_to_ending_9(cost_price * MIN_MARKUP)
            return price, [f"No competitor prices found, so we priced at cost + 25% (₹{price})."]
        return None, ["No competitor prices found and no cost price given, so no suggestion."]

    price = round_to_ending_9(median * UNDERCUT)
    reasons.append(f"Started 5% under the competitor median of ₹{median:g} and rounded to ₹{price}.")

    # Keep inside the middle half of the market when the rounding pushed it out
    p25, p75 = stats.get("p25"), stats.get("p75")
    if p25 is not None and p75 is not None and p25 <= p75:
        if price < p25:
            price = round_up_to_ending_9(p25) if round_up_to_ending_9(p25) <= p75 else math.ceil(p25)
            reasons.append(f"Raised to ₹{price} to stay inside the typical range (₹{p25:g}-₹{p75:g}).")
        elif price > p75:
            price = round_to_ending_9(p75) if round_to_ending_9(p75) >= p25 else math.floor(p75)
            reasons.append(f"Lowered to ₹{price} to stay inside the typical range (₹{p25:g}-₹{p75:g}).")

    # Cost floor always wins so the seller never loses money
    if cost_price:
        floor = cost_price * MIN_MARKUP
        if price < floor:
            price = round_up_to_ending_9(floor)
            reasons.append(f"Raised to ₹{price} so you keep at least a 25% markup on your cost of ₹{cost_price:g}.")
    return int(price), reasons


def margin_pct(price: float | None, cost_price: float | None) -> float | None:
    """Profit as a % of the selling price, or None without a cost price."""
    if not price or not cost_price:
        return None
    return round((price - cost_price) / price * 100, 1)


# ---------- data sources ----------

def mock_competitors(query: str) -> list[dict]:
    """Deterministic fake competitor list (same query -> same prices)."""
    seed = int(hashlib.md5(query.encode()).hexdigest(), 16)
    base = 300 + seed % 400  # somewhere between ₹300 and ₹699
    offsets = [-0.4, -0.25, -0.15, -0.05, 0.0, 0.05, 0.1, 0.2, 0.35, 2.5]  # last one is an outlier
    stores = ["Meesho", "Flipkart", "Amazon.in", "Myntra", "Ajio", "Snapdeal"]
    items = []
    for i, off in enumerate(offsets):
        items.append({
            "title": f"{query.title()} - option {i + 1}",
            "price": float(round(base * (1 + off))),
            "source": stores[i % len(stores)],
            "link": "",
            "thumbnail": "",
        })
    return items


def fetch_serpapi(query: str, api_key: str, timeout: float = 15) -> list[dict]:
    """Call SerpApi google_shopping for India and return competitor dicts."""
    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": "in",
        "hl": "en",
        "location": "India",
        "api_key": api_key,
    }
    resp = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])

    items = []
    for r in data.get("shopping_results", []) or []:
        # extracted_price is a number when present; otherwise parse the price string
        price = parse_price(r.get("extracted_price")) or parse_price(r.get("price"))
        if price is None:
            continue
        items.append({
            "title": r.get("title", ""),
            "price": price,
            "source": r.get("source", ""),
            "link": r.get("product_link") or r.get("link") or "",
            "thumbnail": r.get("thumbnail", ""),
        })
    return items


def get_pricing(query: str, settings: Settings, cost_price: float | None = None) -> tuple[dict, str]:
    """Return (pricing dict per SPEC, source) where source is "serpapi" or "mock"."""
    query = (query or "").strip() or "product"
    source = "mock"
    competitors: list[dict] = []
    if settings.has_serpapi:
        try:
            competitors = fetch_serpapi(query, settings.serpapi_key)
            source = "serpapi"
        except Exception as exc:
            log.warning("SerpApi failed, using mock competitors: %s", exc)
    if source == "mock":
        competitors = mock_competitors(query)

    # Remove outliers, then show only the competitors that survived
    kept = remove_outliers_iqr([c["price"] for c in competitors])
    lo, hi = (min(kept), max(kept)) if kept else (0, 0)
    shown = [c for c in competitors if lo <= c["price"] <= hi] if kept else []
    dropped = len(competitors) - len(shown)

    stats = compute_stats(kept)
    suggested, reasons = suggest_price(stats, cost_price)
    if dropped:
        reasons.insert(0, f"Ignored {dropped} unusual price(s) far from the rest.")
    if stats["count"]:
        reasons.insert(0, f"Compared {stats['count']} similar products ({'live Google Shopping' if source == 'serpapi' else 'sample data'}).")

    return {
        "currency": "INR",
        "competitors": shown[:20],
        "stats": stats,
        "suggested_price": suggested,
        "strategy": " ".join(reasons),
        "margin_pct": margin_pct(suggested, cost_price),
    }, source
