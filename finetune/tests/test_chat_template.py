"""Prove backend/app/chat_template.py builds EXACTLY the prompt the model saw in training.

Training used the tinker-cookbook renderer "qwen3_disable_thinking". Here we build the
generation prompt with that renderer, then compare it with build_raw_prompt():
  - as text (decode the renderer's tokens with the Qwen3 tokenizer), and
  - as tokens (encode our raw text the way Ollama does, special tokens included).
Needs the Qwen3 tokenizer (a few MB, cached by HuggingFace after the first run).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA_DIR, chat_template, prompt_messages, read_jsonl  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-8B"
RENDERER = "qwen3_disable_thinking"


@pytest.fixture(scope="module")
def tokenizer_and_renderer():
    """Load the Qwen3 tokenizer + cookbook renderer, or skip if we are offline with no cache."""
    renderers = pytest.importorskip("tinker_cookbook.renderers")
    tokenizer_utils = pytest.importorskip("tinker_cookbook.tokenizer_utils")
    try:
        tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    except Exception as err:  # noqa: BLE001 - no network and no cached tokenizer
        pytest.skip(f"Qwen3 tokenizer not available: {err}")
    return tokenizer, renderers.get_renderer(RENDERER, tokenizer)


def sample_message_lists():
    """Real prompts from the test sample (both platforms, with and without notes) + an edge case."""
    rows = read_jsonl(DATA_DIR / "sample_test.jsonl")
    lists = [prompt_messages(r) for r in rows]
    lists.append([
        {"role": "system", "content": "Line one\n\nLine two with unicode: ₹499, café"},
        {"role": "user", "content": "{\"a\": 1}\n"},
    ])
    return lists


def test_samples_cover_both_platforms():
    platforms = {r["platform"] for r in read_jsonl(DATA_DIR / "sample_test.jsonl")}
    assert platforms == {"meesho", "etsy"}


@pytest.mark.parametrize("index", range(len(sample_message_lists())))
def test_raw_prompt_matches_renderer_text_and_tokens(tokenizer_and_renderer, index):
    tokenizer, renderer = tokenizer_and_renderer
    messages = sample_message_lists()[index]

    renderer_tokens = renderer.build_generation_prompt(messages).to_ints()
    ours = chat_template.build_raw_prompt(messages)

    # Same text...
    assert tokenizer.decode(renderer_tokens) == ours
    # ...and the same tokens when the text is tokenised with special tokens parsed (like Ollama raw mode)
    assert tokenizer.encode(ours, add_special_tokens=False) == renderer_tokens


def test_matches_hf_chat_template_with_thinking_off(tokenizer_and_renderer):
    """Extra check: Qwen's own chat template with enable_thinking=False gives the same prompt."""
    tokenizer, _ = tokenizer_and_renderer
    messages = sample_message_lists()[0]
    hf_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert hf_text == chat_template.build_raw_prompt(messages)


def test_stop_strings_cover_renderer_stop_tokens(tokenizer_and_renderer):
    tokenizer, renderer = tokenizer_and_renderer
    for token_id in renderer.get_stop_sequences():
        assert tokenizer.decode([token_id]) in chat_template.STOP_STRINGS


def test_prompt_ends_with_open_assistant_turn():
    text = chat_template.build_raw_prompt([{"role": "user", "content": "hi"}])
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
