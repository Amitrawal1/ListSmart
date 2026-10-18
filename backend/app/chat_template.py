"""Turn chat messages into the exact raw text Qwen3 saw during fine-tuning.

Training used the tinker-cookbook renderer "qwen3_disable_thinking". When we run the
model locally in Ollama we send this raw text (raw mode), so the model gets the same
format it learned and does not start "thinking" before the JSON.
"""

# Ollama stops generating when it sees one of these
STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]


def build_raw_prompt(messages: list[dict]) -> str:
    """Build a ChatML prompt that ends where the assistant answer should start."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    # Open the assistant turn with an empty think block (thinking switched off)
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)
