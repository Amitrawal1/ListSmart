"""Offline tests for speed.py and the new speed/cost/Ollama helpers in evaluate.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
import speed  # noqa: E402


def test_ollama_payload_is_raw_with_stops():
    body = speed.ollama_generate_payload("listsmart-writer", "PROMPT", ["<|im_end|>"], 0.0, 512, 7, 4096)
    assert body["raw"] is True and body["stream"] is False
    assert body["prompt"] == "PROMPT"
    assert body["options"] == {"temperature": 0.0, "num_predict": 512, "stop": ["<|im_end|>"], "seed": 7, "num_ctx": 4096}


def test_ollama_call_stats_uses_eval_duration_in_nanoseconds():
    reply = {"response": "{}", "eval_count": 200, "eval_duration": 20_000_000_000,
             "prompt_eval_count": 700, "load_duration": 3_000_000_000}
    stats = speed.ollama_call_stats(reply, 31.2)
    assert stats["tokens_per_s"] == 10.0
    assert stats["latency_s"] == 31.2
    assert stats["prompt_tokens"] == 700 and stats["output_tokens"] == 200
    assert stats["load_s"] == 3.0


def test_ollama_call_stats_handles_error_reply():
    stats = speed.ollama_call_stats({"response": ""}, 1.0)
    assert stats["text"] == "" and stats["tokens_per_s"] is None and stats["output_tokens"] == 0


def test_tokens_call_stats_wall_clock_speed():
    assert speed.tokens_call_stats("x", 100, 50, 2.0)["tokens_per_s"] == 25.0
    assert speed.tokens_call_stats("x", 100, 0, 2.0)["tokens_per_s"] is None


def test_speed_summary_uses_median():
    calls = [{"latency_s": s, "tokens_per_s": t, "prompt_tokens": 700, "output_tokens": 300}
             for s, t in [(60.0, 2.0), (20.0, 8.0), (22.0, 9.0)]]  # first call includes model load
    out = speed.speed_summary(calls)
    assert out["median_latency_s"] == 22.0
    assert out["tokens_per_s"] == 8.0
    assert out["mean_prompt_tokens"] == 700 and out["mean_output_tokens"] == 300 and out["n_calls"] == 3


def test_speed_summary_empty():
    assert speed.speed_summary([])["median_latency_s"] is None


def test_cost_per_1000_listings():
    # 1000 tokens per listing at $0.60 / 1M -> $0.60 per 1000 listings -> 50.4 INR at 84
    assert speed.cost_per_1000_listings_inr(1000, 0.60, 84.0) == 50.4
    assert speed.cost_per_1000_listings_inr(None, 0.60) is None
    # 700 in at $1, 300 out at $5 -> $0.7 + $1.5 = $2.2 per 1000 -> 184.8 INR
    assert speed.cost_split_per_1000_inr(700, 300, 1.0, 5.0, 84.0) == 184.8


def test_format_hardware():
    assert speed.format_hardware("macOS arm64", "Apple M1", 8 * 2**30) == "macOS arm64, Apple M1, 8 GB RAM"
    assert speed.format_hardware("", "", None) == "unknown"
    assert isinstance(speed.detect_hardware(), str)


def test_pick_closed_provider():
    assert evaluate.pick_closed_provider("auto", {}) is None
    assert evaluate.pick_closed_provider("auto", {"ANTHROPIC_API_KEY": "k"}) == "anthropic"
    assert evaluate.pick_closed_provider("auto", {"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}) == "openai"
    assert evaluate.pick_closed_provider("openai", {"ANTHROPIC_API_KEY": "k"}) is None


def test_cost_block_shape():
    calls = [{"latency_s": 1, "tokens_per_s": 5, "prompt_tokens": 700, "output_tokens": 300}]
    block = evaluate.cost_block(calls, 0.60, 84.0)
    assert block["local"] == 0
    assert block["tinker_hosted"] == 50.4
    assert block["closed_api"] is None
    closed = {"calls": calls, "usd_in": 1.0, "usd_out": 5.0}
    assert evaluate.cost_block(calls, 0.60, 84.0, closed)["closed_api"] == 184.8


def test_saved_call_reads_old_and_new_files():
    assert evaluate.saved_call({"raw": "t"}) == {"text": "t"}
    assert evaluate.saved_call({"raw": "t", "call": {"latency_s": 2}})["latency_s"] == 2


def test_update_main_summary(tmp_path):
    local = {"speed": {"hardware": "Apple M1"}, "cost_per_1000_listings_inr": {"local": 0},
             "base_model": "qwen3:8b", "finetuned_model": "listsmart-writer", "n_test": 20,
             "metrics": {}, "judge": None, "generated_at": "now"}
    main_file = tmp_path / "eval_summary.json"
    assert evaluate.update_main_summary(main_file, local) is False  # nothing to update yet

    main_file.write_text(json.dumps({"available": True, "note": "example placeholder"}))
    assert evaluate.update_main_summary(main_file, local) is False  # never touch the placeholder

    main_file.write_text(json.dumps({"available": True, "metrics": {"x": 1}, "speed": {"hardware": "Tinker (hosted)"}}))
    assert evaluate.update_main_summary(main_file, local) is True
    data = json.loads(main_file.read_text())
    assert data["metrics"] == {"x": 1}  # Tinker results kept
    assert data["speed"]["hardware"] == "Apple M1"
    assert data["speed_tinker"]["hardware"] == "Tinker (hosted)"
    assert data["local_eval"]["n_test"] == 20
