"""Rough cost of one /api/generate request in INR.

Local (ollama) and mock steps cost 0. Hosted steps are estimated from rough token counts.
All assumed prices live here so they are easy to check and update.
"""

# Assumed exchange rate (Oct 2026, rounded)
USD_TO_INR = 84.0

# Tinker sampling price for Qwen3-8B, input + output tokens (assumed ~$0.60 per 1M tokens)
TINKER_USD_PER_1M_TOKENS = 0.60
# Gemma on Google AI Studio is free today; change this if you move to a paid tier
GEMMA_API_USD_PER_1M_TOKENS = 0.0
# SerpApi Developer plan: $75 for 5,000 searches -> $0.015 per search (assumed)
SERPAPI_USD_PER_SEARCH = 0.015

# Rough rule of thumb: about 4 characters of English per token
CHARS_PER_TOKEN = 4
# Gemma 3 turns each image into 256 tokens
TOKENS_PER_IMAGE = 256


def tokens_from_chars(chars: int) -> int:
    """Very rough token count from a character count."""
    return max(0, int(chars)) // CHARS_PER_TOKEN


def estimate_cost_inr(
    vision_backend: str,
    writer_backend: str,
    pricing_source: str,
    n_images: int,
    vision_prompt_chars: int,
    vision_output_chars: int,
    writer_prompt_chars: int,
    writer_output_chars: int,
) -> float:
    """Add up the hosted steps of one request. Returns INR rounded to 4 decimals."""
    usd = 0.0
    if vision_backend == "gemma_api":
        tokens = n_images * TOKENS_PER_IMAGE + tokens_from_chars(vision_prompt_chars + vision_output_chars)
        usd += tokens * GEMMA_API_USD_PER_1M_TOKENS / 1_000_000

    writer_tokens = tokens_from_chars(writer_prompt_chars + writer_output_chars)
    if writer_backend == "tinker":
        usd += writer_tokens * TINKER_USD_PER_1M_TOKENS / 1_000_000
    elif writer_backend == "gemma":
        usd += writer_tokens * GEMMA_API_USD_PER_1M_TOKENS / 1_000_000

    if pricing_source == "serpapi":
        usd += SERPAPI_USD_PER_SEARCH
    return round(usd * USD_TO_INR, 4)
