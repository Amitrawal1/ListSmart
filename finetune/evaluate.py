"""Evaluate the base model vs the fine-tuned model on the SAME held-out test prompts.

Both models get identical prompts (prompts.build_writer_messages), the same chat format and
low temperature. Two ways to sample them (--backend):
  tinker  (default) base Qwen/Qwen3-8B vs the tinker:// LoRA, hosted on Tinker
  ollama  base `qwen3:8b` vs our `listsmart-writer`, running LOCALLY in Ollama (raw prompts built by
          backend/app/chat_template.py, the same text the model saw in training) Metrics (see SPEC.md and metrics.py):
  1. format validity       2. constraint compliance   3. attribute coverage
  4. ROUGE-L vs reference  5. Gemma LLM-judge, pairwise + blind (random A/B order)

Writes:
  results/eval_outputs.jsonl  every test example with both outputs + scores
  results/eval_summary.json   summary in the SPEC shape (served by GET /api/eval)
  results/eval_report.md      markdown table for the README / demo
  (--backend ollama writes to results/ollama/ instead, and copies its local "speed" numbers
   into results/eval_summary.json so the app shows laptop speed; see --no-update-main-summary)
The summary also has "speed" (median latency, tokens/s, hardware) and
"cost_per_1000_listings_inr" (local / tinker_hosted / closed_api).

Usage:
  python evaluate.py --dry-run           # offline pipeline check with mock outputs (writes to results/dry_run/)
  python evaluate.py                     # real run (needs TINKER_API_KEY, GOOGLE_API_KEY for the judge)
  python evaluate.py --no-judge          # skip the Gemma judge
  python evaluate.py --reuse-outputs     # re-score / re-judge saved outputs without sampling again
  python evaluate.py --backend ollama --limit 20 --no-judge   # local run on the laptop (Ollama running)
  python evaluate.py --backend ollama --dry-run               # offline check of the Ollama path
  python evaluate.py --closed-baseline   # optional: also score a closed API model (needs OPENAI_API_KEY
                                         # or ANTHROPIC_API_KEY; skipped when neither is set)
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import metrics
import speed
from common import (
    DATA_DIR,
    DEFAULT_BASE_MODEL,
    RESULTS_DIR,
    chat_template,
    choose_renderer_name,
    load_env,
    prompt_messages,
    prompts,
    read_jsonl,
    require_env,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-file", default=str(DATA_DIR / "test.jsonl"))
    p.add_argument("--base-model", default=None, help="default: from results/train_info.json, else Qwen/Qwen3-8B")
    p.add_argument("--finetuned-model", default=None, help="tinker://... path (default: results/finetuned_model.txt)")
    p.add_argument("--renderer", default=None, help="default: same renderer train.py used")
    p.add_argument("--limit", type=int, default=None, help="only use the first N test rows")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=32, help="sampling requests in flight at once")
    p.add_argument("--judge-model", default=os.environ.get("GEMMA_MODEL", "gemma-3-27b-it"))
    p.add_argument("--judge-sleep", type=float, default=2.1, help="seconds between judge calls (free-tier rate limit)")
    p.add_argument("--no-judge", action="store_true", help="skip the Gemma LLM judge")
    p.add_argument("--n-examples", type=int, default=3, help="examples copied into the summary for the frontend")
    p.add_argument("--seed", type=int, default=1234, help="seed for the random A/B order")
    p.add_argument("--reuse-outputs", action="store_true", help="load results/eval_outputs.jsonl instead of sampling")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--dry-run", action="store_true", help="no API calls: mock outputs, written to results/dry_run/")
    # ---- local Ollama backend
    p.add_argument("--backend", choices=["tinker", "ollama"], default="tinker",
                   help="where to run the two models (default: tinker)")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--ollama-base-model", default="qwen3:8b", help="base model name in Ollama")
    p.add_argument("--ollama-finetuned-model", default=os.environ.get("OLLAMA_WRITER_MODEL", "listsmart-writer"),
                   help="fine-tuned model name in Ollama (made by export_to_ollama.py)")
    p.add_argument("--num-ctx", type=int, default=4096, help="Ollama context window (prompt ~700 + answer)")
    p.add_argument("--ollama-timeout", type=float, default=600, help="seconds to wait for one Ollama answer")
    p.add_argument("--no-update-main-summary", action="store_true",
                   help="ollama backend: do not copy local speed numbers into results/eval_summary.json")
    # ---- cost assumptions (same as backend/app/costs.py)
    p.add_argument("--tinker-usd-per-1m", type=float, default=speed.TINKER_USD_PER_1M_TOKENS,
                   help="assumed Tinker sampling price for Qwen3-8B, USD per 1M tokens")
    p.add_argument("--usd-to-inr", type=float, default=speed.USD_TO_INR)
    # ---- optional closed-model baseline (never required)
    p.add_argument("--closed-baseline", action="store_true",
                   help="also run a closed API model on the same prompts (only if OPENAI_API_KEY or ANTHROPIC_API_KEY is set)")
    p.add_argument("--closed-provider", choices=["auto", "openai", "anthropic"], default="auto")
    p.add_argument("--closed-model", default=None, help="default: gpt-4o-mini (openai) / claude-haiku-4-5 (anthropic)")
    p.add_argument("--closed-usd-per-1m-in", type=float, default=None, help="input price, USD per 1M tokens (assumed default per provider)")
    p.add_argument("--closed-usd-per-1m-out", type=float, default=None, help="output price, USD per 1M tokens")
    return p.parse_args()


# ---------------------------------------------------------------- sampling
def sample_all(sampling_client, renderer, rows, args, label):
    """Generate one listing per test row on Tinker; returns one call record per row (text, latency, tokens)."""
    import tinker
    from tinker_cookbook.renderers import get_text_content

    params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
        seed=args.seed,
    )
    outputs = []
    for start in range(0, len(rows), args.concurrency):
        chunk = rows[start : start + args.concurrency]
        # Fire off a chunk of requests, then wait for all of them
        prompts_in = [renderer.build_generation_prompt(prompt_messages(row)) for row in chunk]
        sent_at = time.time()
        futures = [
            sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params)
            for prompt in prompts_in
        ]
        for prompt, fut in zip(prompts_in, futures):
            tokens = fut.result().sequences[0].tokens
            # Latency here = time since the chunk was sent (requests run in parallel on Tinker)
            wall_s = time.time() - sent_at
            message, _ = renderer.parse_response(tokens)
            outputs.append(speed.tokens_call_stats(get_text_content(message), prompt.length, len(tokens), wall_s))
        print(f"[{label}] sampled {len(outputs)}/{len(rows)}", flush=True)
    return outputs


def sample_ollama(rows, args, model, label):
    """Generate one listing per row with a LOCAL Ollama model, one request at a time.

    We send the exact training prompt in raw mode, so the fine-tuned model sees the format it learned.
    """
    import requests

    url = args.ollama_host.rstrip("/") + "/api/generate"
    calls = []
    for i, row in enumerate(rows):
        raw_prompt = chat_template.build_raw_prompt(prompt_messages(row))
        payload = speed.ollama_generate_payload(
            model, raw_prompt, chat_template.STOP_STRINGS, args.temperature, args.max_tokens, args.seed, args.num_ctx
        )
        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=(5, args.ollama_timeout))
            resp.raise_for_status()
            reply = resp.json()
        except Exception as err:  # noqa: BLE001 - keep going, this row just scores as invalid
            print(f"  [{label}] row {row['id']}: Ollama error {err.__class__.__name__}: {str(err)[:150]}")
            reply = {"response": ""}
        calls.append(speed.ollama_call_stats(reply, time.time() - start))
        last = calls[-1]
        print(f"[{label}] {i + 1}/{len(rows)}  {last['latency_s']:.1f}s  {last['tokens_per_s']} tok/s", flush=True)
    # Free the RAM before the next model loads (two 8B models do not fit in 8GB together)
    try:
        requests.post(url, json={"model": model, "keep_alive": 0}, timeout=(5, 60))
    except Exception:  # noqa: BLE001 - unloading is only a nicety
        pass
    return calls


def check_ollama_models(args, models):
    """Exit early with a clear message if Ollama is not running or a model is missing."""
    import requests

    try:
        tags = requests.get(args.ollama_host.rstrip("/") + "/api/tags", timeout=5).json()
    except Exception as err:  # noqa: BLE001
        sys.exit(f"ERROR: cannot reach Ollama at {args.ollama_host} ({err.__class__.__name__}). Start it with: ollama serve")
    names = {m.get("name", "") for m in tags.get("models", [])}
    # "qwen3:8b" may be listed as "qwen3:8b", and "listsmart-writer" as "listsmart-writer:latest"
    missing = [m for m in models if m not in names and f"{m}:latest" not in names]
    if missing:
        hint = "ollama pull qwen3:8b" if "qwen3:8b" in missing else "python export_to_ollama.py"
        sys.exit(f"ERROR: Ollama models not found: {missing}. Have: {sorted(names)}. Try: {hint}")


# ---------------------------------------------------------------- closed baseline (optional)
CLOSED_DEFAULTS = {
    # Assumed list prices (USD per 1M tokens, input/output). Check the provider's pricing page.
    "openai": {"model": "gpt-4o-mini", "in": 0.15, "out": 0.60},
    "anthropic": {"model": "claude-haiku-4-5", "in": 1.00, "out": 5.00},
}


def pick_closed_provider(choice, env):
    """Which closed API to use: the one asked for if its key is set, else the first with a key, else None."""
    keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    if choice != "auto":
        return choice if env.get(keys[choice], "").strip() else None
    for name, key in keys.items():
        if env.get(key, "").strip():
            return name
    return None


def sample_closed(rows, args, provider, model):
    """Plain HTTPS calls to a closed chat API with the same messages. Returns one call record per row."""
    import requests

    calls = []
    for i, row in enumerate(rows):
        messages = prompt_messages(row)
        start = time.time()
        text, tok_in, tok_out = "", 0, 0
        try:
            if provider == "openai":
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    json={"model": model, "messages": messages, "temperature": args.temperature,
                          "max_completion_tokens": args.max_tokens},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                tok_in, tok_out = data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"]
            else:
                system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                    json={"model": model, "max_tokens": args.max_tokens, "temperature": args.temperature,
                          "system": system, "messages": [m for m in messages if m["role"] != "system"]},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                text = "".join(b.get("text", "") for b in data.get("content", []))
                tok_in, tok_out = data["usage"]["input_tokens"], data["usage"]["output_tokens"]
        except Exception as err:  # noqa: BLE001 - one failed row scores as invalid
            print(f"  [closed] row {row['id']}: {err.__class__.__name__}: {str(err)[:150]}")
        calls.append(speed.tokens_call_stats(text, tok_in, tok_out, time.time() - start))
        if (i + 1) % 10 == 0:
            print(f"[closed] {i + 1}/{len(rows)}", flush=True)
    return calls


def mock_output(row, flavour):
    """Offline stand-in for a model (dry run only). 'base' is deliberately sloppier."""
    a = row["attributes"]
    words = " ".join([*(a.get("colors") or [])[:1], a.get("material", ""), a.get("product_type", "")]).split()
    listing = {
        "title": " ".join(w.capitalize() for w in words) or "Product",
        "description": f"A {' '.join(words)}.\n\nMade for everyday use.",
        "bullets": [f"Feature {i + 1}" for i in range(5 if flavour == "finetuned" else 4)],
        "tags": [w for w in words][:13] or ["product"],
        "search_query": " ".join(words[:5] + ["online"] * max(0, 3 - len(words))),
    }
    text = json.dumps(listing)
    return text if flavour == "finetuned" else "Here is your listing:\n" + text


# ---------------------------------------------------------------- judge
JUDGE_TEMPLATE = """You are judging two product listings written for the same product.

Platform: {platform} (title at most {title_max} characters, exactly 5 bullets, up to {tags_max} tags of at most {tag_len_max} characters)
Product attributes (the ONLY facts the writer was given):
{attributes}
Seller notes: {notes}

Listing A:
{a}

Listing B:
{b}

Which listing is better for a real seller? Judge: (1) accuracy - uses the attributes and does not invent
brands, sizes or claims, (2) follows the platform rules above, (3) clear, persuasive, search-friendly writing.
Reply with ONE JSON object only: {{"winner": "A" or "B" or "TIE", "reason": "one short sentence"}}"""


def make_judge(model_name):
    """Return a function prompt -> reply text using Gemma through google-genai."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=200)

    def call(prompt):
        # Retry a few times because free-tier keys hit rate limits now and then
        for attempt in range(4):
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt, config=config)
                return resp.text or ""
            except Exception as err:  # noqa: BLE001 - any API error: wait and retry
                wait = 5 * (attempt + 1)
                print(f"  judge error ({err.__class__.__name__}: {str(err)[:120]}), retrying in {wait}s")
                time.sleep(wait)
        return ""

    return call


def judge_example(row, base_score, ft_score, rng, judge_call):
    """Blind pairwise comparison. Invalid outputs lose automatically (no judge call needed)."""
    base_ok, ft_ok = base_score["format_valid"], ft_score["format_valid"]
    if not base_ok and not ft_ok:
        return {"winner": "tie", "mode": "auto_both_invalid"}
    if base_ok != ft_ok:
        return {"winner": "finetuned" if ft_ok else "base", "mode": "auto_one_invalid"}

    # Shuffle which model is shown as A so the judge cannot favour a position
    finetuned_is_a = rng.random() < 0.5
    a, b = (ft_score, base_score) if finetuned_is_a else (base_score, ft_score)
    rules = prompts.PLATFORM_RULES[row["platform"]]
    prompt = JUDGE_TEMPLATE.format(
        platform=row["platform"],
        attributes=json.dumps(row["attributes"], ensure_ascii=False),
        notes=row.get("notes") or "none",
        a=json.dumps(a["listing"], ensure_ascii=False, indent=1),
        b=json.dumps(b["listing"], ensure_ascii=False, indent=1),
        **rules,
    )
    reply = judge_call(prompt)
    verdict = metrics.parse_judge_verdict(reply)
    return {
        "winner": metrics.verdict_to_winner(verdict, finetuned_is_a),
        "verdict": verdict,
        "finetuned_is_a": finetuned_is_a,
        "mode": "judged",
        "reply": reply[:300],
    }


# ---------------------------------------------------------------- reporting
def example_view(score, raw_text):
    """What the frontend shows for one model output."""
    if score["listing"]:
        return score["listing"]
    return {"invalid_output": raw_text[:600], "error": score["error"]}


def write_report(path, summary):
    """Markdown table: metric | base | fine-tuned | change."""
    lines = [
        "# ListSmart writer: base vs fine-tuned",
        "",
        f"- Base model: `{summary['base_model']}`",
        f"- Fine-tuned: `{summary['finetuned_model']}`",
        f"- Test examples: {summary['n_test']} (held out, never seen in training)",
        f"- Generated: {summary['generated_at']}",
        "",
        "| Metric | Base | Fine-tuned | Change |",
        "|---|---|---|---|",
    ]
    for name, vals in summary["metrics"].items():
        b, f = vals["base"], vals["finetuned"]
        change = f"{f - b:+.3f}" if b is not None and f is not None else "n/a"
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {name} | {fmt(b)} | {fmt(f)} | {change} |")
    judge = summary.get("judge")
    lines.append("")
    if judge:
        lines += [
            f"**Gemma judge ({summary['settings']['judge_model']}, blind pairwise, random A/B order, n={judge['n']}):** "
            f"fine-tuned wins {judge['finetuned_win']:.0%}, base wins {judge['base_win']:.0%}, ties {judge['tie']:.0%}.",
            f"({judge.get('auto_decided', 0)} pairs were decided automatically because one or both outputs were not valid JSON.)",
        ]
    else:
        lines.append("Judge skipped (--no-judge).")
    # Speed + cost (older summaries may not have them)
    sp, cost = summary.get("speed"), summary.get("cost_per_1000_listings_inr")
    if sp:
        lines += [
            "",
            f"**Speed** ({sp['hardware']}): base median {sp['base']['median_latency_s']} s per listing, "
            f"{sp['base']['tokens_per_s']} tokens/s; fine-tuned median {sp['finetuned']['median_latency_s']} s, "
            f"{sp['finetuned']['tokens_per_s']} tokens/s.",
        ]
    if cost:
        closed = "n/a" if cost.get("closed_api") is None else f"₹{cost['closed_api']}"
        lines.append(
            f"**Cost per 1000 listings:** local ₹{cost['local']}, Tinker hosted ₹{cost['tinker_hosted']}, closed API {closed}."
        )
    lines += [
        "",
        "Metric notes: format_valid = parses into a Listing with all fields; constraint_compliance = passes every",
        "platform rule (title length, exactly 5 bullets, tag count/length, 2-4 paragraphs, 3-6 word query);",
        "attribute_coverage = share of attribute values (type, colours, material, pattern, style) mentioned in",
        "title+description; rouge_l = ROUGE-L F1 vs the real reference listing.",
    ]
    path.write_text("\n".join(lines) + "\n")


def mock_call(row, flavour):
    """Dry-run call record: mock text plus made-up (but plausible) speed numbers."""
    text = mock_output(row, flavour)
    out_tokens = len(text) // 4
    return speed.tokens_call_stats(text, 700, out_tokens, 20.0 if flavour == "base" else 18.0)


def saved_call(saved_side):
    """Rebuild a call record from a saved eval_outputs.jsonl entry (old files have no speed info)."""
    return {"text": saved_side["raw"], **(saved_side.get("call") or {})}


def speed_block(base_calls, ft_calls, hardware):
    """SPEC 'speed' block: median latency + tokens/s per model, and where it ran."""
    return {
        "base": speed.speed_summary(base_calls),
        "finetuned": speed.speed_summary(ft_calls),
        "hardware": hardware,
    }


def cost_block(ft_calls, tinker_usd_per_1m, usd_to_inr, closed=None):
    """SPEC 'cost_per_1000_listings_inr': local is free, hosted costs come from measured token counts."""
    ft_speed = speed.speed_summary(ft_calls)
    tokens = None
    if ft_speed["mean_prompt_tokens"] is not None and ft_speed["mean_output_tokens"]:
        tokens = ft_speed["mean_prompt_tokens"] + ft_speed["mean_output_tokens"]
    closed_inr = None
    if closed:
        cs = speed.speed_summary(closed["calls"])
        closed_inr = speed.cost_split_per_1000_inr(
            cs["mean_prompt_tokens"], cs["mean_output_tokens"], closed["usd_in"], closed["usd_out"], usd_to_inr
        )
    return {
        "local": 0,
        "tinker_hosted": speed.cost_per_1000_listings_inr(tokens, tinker_usd_per_1m, usd_to_inr),
        "closed_api": closed_inr,
        "assumptions": {
            "usd_to_inr": usd_to_inr,
            "tinker_usd_per_1m_tokens": tinker_usd_per_1m,
            "tokens_per_listing": tokens,
            "local_note": "runs on the seller's own laptop: no API fees (electricity not counted)",
        },
    }


def update_main_summary(main_file, local_summary):
    """Copy the local Ollama speed/cost into the main results/eval_summary.json (only if it is a real run)."""
    if not main_file.exists():
        print(f"(no {main_file.name} yet: run the Tinker eval first to fill the main summary)")
        return False
    main = json.loads(main_file.read_text())
    if not main.get("available") or main.get("note"):
        print(f"({main_file.name} is a placeholder or dry run: not updated)")
        return False
    # Keep the hosted numbers under another name; the app shows the laptop numbers as "speed"
    if main.get("speed") and "Tinker" in str(main["speed"].get("hardware", "")):
        main["speed_tinker"] = main["speed"]
    main["speed"] = local_summary["speed"]
    main["cost_per_1000_listings_inr"] = local_summary["cost_per_1000_listings_inr"]
    main["local_eval"] = {
        "base_model": local_summary["base_model"],
        "finetuned_model": local_summary["finetuned_model"],
        "n_test": local_summary["n_test"],
        "metrics": local_summary["metrics"],
        "judge": local_summary["judge"],
        "generated_at": local_summary["generated_at"],
    }
    main_file.write_text(json.dumps(main, indent=2, ensure_ascii=False))
    print(f"updated {main_file} with local speed + cost")
    return True


def main():
    args = parse_args()
    load_env()
    from pathlib import Path

    out_dir = Path(args.out_dir)
    if args.backend == "ollama" and out_dir == RESULTS_DIR:
        # Local runs are usually a --limit subset: keep them apart from the full Tinker eval
        out_dir = RESULTS_DIR / "ollama"
    if args.dry_run:
        # Never let mock numbers land where the app reads real results
        out_dir = RESULTS_DIR / "dry_run" / args.backend if args.backend == "ollama" else RESULTS_DIR / "dry_run"
        args.no_judge = True
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_file = out_dir / "eval_outputs.jsonl"

    rows = read_jsonl(args.test_file)[: args.limit]
    train_info_file = RESULTS_DIR / "train_info.json"
    train_info = json.loads(train_info_file.read_text()) if train_info_file.exists() else {}
    base_model = args.base_model or train_info.get("base_model") or DEFAULT_BASE_MODEL
    finetuned_model = args.finetuned_model
    if not finetuned_model and (RESULTS_DIR / "finetuned_model.txt").exists():
        finetuned_model = (RESULTS_DIR / "finetuned_model.txt").read_text().strip()
    hardware = "Tinker (hosted GPUs; latency includes network + queue)"
    if args.backend == "ollama":
        # Names as they appear in Ollama; remember which Tinker run the local model came from
        source = finetuned_model
        base_model, finetuned_model = args.ollama_base_model, args.ollama_finetuned_model
        hardware = speed.detect_hardware()

    # ---- get raw outputs (+ speed per call) for both models
    if args.dry_run:
        finetuned_model = finetuned_model or "tinker://dry-run/mock"
        renderer_name = "mock"
        if args.backend == "ollama":
            renderer_name = "raw chat_template (qwen3_disable_thinking)"
            example = chat_template.build_raw_prompt(prompt_messages(rows[0]))
            payload = speed.ollama_generate_payload(
                finetuned_model, example, chat_template.STOP_STRINGS, args.temperature, args.max_tokens, args.seed, args.num_ctx
            )
            print(f"[dry run] would POST {args.ollama_host.rstrip('/')}/api/generate for {len(rows)} rows x 2 models")
            print(f"[dry run] models: base={base_model} finetuned={finetuned_model}  hardware: {hardware}")
            print("[dry run] payload options:", json.dumps(payload["options"]), " raw:", payload["raw"])
            print("[dry run] prompt tail:", repr(example[-120:]))
        base_calls = [mock_call(r, "base") for r in rows]
        ft_calls = [mock_call(r, "finetuned") for r in rows]
    elif args.reuse_outputs:
        saved = {o["id"]: o for o in read_jsonl(outputs_file)}
        rows = [r for r in rows if r["id"] in saved]
        base_calls = [saved_call(saved[r["id"]]["base"]) for r in rows]
        ft_calls = [saved_call(saved[r["id"]]["finetuned"]) for r in rows]
        renderer_name = args.renderer or train_info.get("renderer") or "unknown"
    elif args.backend == "ollama":
        renderer_name = "raw chat_template (qwen3_disable_thinking)"
        check_ollama_models(args, [base_model, finetuned_model])
        print(f"ollama base={base_model} finetuned={finetuned_model} n={len(rows)} on {hardware}")
        base_calls = sample_ollama(rows, args, base_model, "base")
        ft_calls = sample_ollama(rows, args, finetuned_model, "finetuned")
    else:
        if not finetuned_model:
            sys.exit("No fine-tuned model: run train.py first or pass --finetuned-model tinker://...")
        require_env("TINKER_API_KEY")
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        renderer_name = args.renderer or train_info.get("renderer") or choose_renderer_name(base_model)
        renderer = renderers.get_renderer(renderer_name, get_tokenizer(base_model))
        service_client = tinker.ServiceClient()
        print(f"base={base_model} finetuned={finetuned_model} renderer={renderer_name} n={len(rows)}")
        base_client = service_client.create_sampling_client(base_model=base_model)
        ft_client = service_client.create_sampling_client(model_path=finetuned_model)
        base_calls = sample_all(base_client, renderer, rows, args, "base")
        ft_calls = sample_all(ft_client, renderer, rows, args, "finetuned")
    base_texts = [c["text"] for c in base_calls]
    ft_texts = [c["text"] for c in ft_calls]

    # ---- optional closed-model baseline (skipped unless asked AND a key is set)
    closed = None
    if args.closed_baseline:
        provider = pick_closed_provider(args.closed_provider, os.environ)
        if args.dry_run or not provider:
            reason = "dry run" if args.dry_run else "no OPENAI_API_KEY / ANTHROPIC_API_KEY set"
            print(f"closed baseline skipped ({reason})")
        else:
            d = CLOSED_DEFAULTS[provider]
            model = args.closed_model or d["model"]
            print(f"closed baseline: {provider} {model}")
            calls = sample_closed(rows, args, provider, model)
            closed = {
                "provider": provider, "model": model, "calls": calls,
                "usd_in": args.closed_usd_per_1m_in if args.closed_usd_per_1m_in is not None else d["in"],
                "usd_out": args.closed_usd_per_1m_out if args.closed_usd_per_1m_out is not None else d["out"],
            }
            closed["scores"] = [metrics.score_example(c["text"], r) for c, r in zip(calls, rows)]

    # ---- automatic metrics
    base_scores = [metrics.score_example(t, r) for t, r in zip(base_texts, rows)]
    ft_scores = [metrics.score_example(t, r) for t, r in zip(ft_texts, rows)]

    # ---- LLM judge
    judgements = [None] * len(rows)
    if not args.no_judge:
        judge_call = make_judge(args.judge_model)
        rng = random.Random(args.seed)
        for i, row in enumerate(rows):
            judgements[i] = judge_example(row, base_scores[i], ft_scores[i], rng, judge_call)
            if judgements[i]["mode"] == "judged":
                time.sleep(args.judge_sleep)
            if (i + 1) % 10 == 0:
                print(f"[judge] {i + 1}/{len(rows)}", flush=True)

    # ---- per-example file ("call" holds latency + token counts for that one generation)
    def call_info(c):
        return {k: v for k, v in c.items() if k != "text"}

    per_example = []
    for i, (row, bs, fs, jd) in enumerate(zip(rows, base_scores, ft_scores, judgements)):
        entry = {
            "id": row["id"],
            "platform": row["platform"],
            "notes": row.get("notes", ""),
            "attributes": row["attributes"],
            "reference": row["reference"],
            "base": {"raw": base_texts[i], **bs, "call": call_info(base_calls[i])},
            "finetuned": {"raw": ft_texts[i], **fs, "call": call_info(ft_calls[i])},
            "judge": jd,
        }
        if closed:
            entry["closed"] = {"raw": closed["calls"][i]["text"], **closed["scores"][i], "call": call_info(closed["calls"][i])}
        per_example.append(entry)
    write_jsonl(outputs_file, per_example)

    # ---- summary in the SPEC shape
    base_summary, ft_summary = metrics.summarize(base_scores), metrics.summarize(ft_scores)
    judge = None
    if not args.no_judge:
        judge = metrics.judge_tally([j["winner"] for j in judgements])
        judge["auto_decided"] = sum(j["mode"] != "judged" for j in judgements)
    summary = {
        # available=false for dry runs so the app never shows mock numbers
        "available": not args.dry_run,
        "base_model": base_model,
        "finetuned_model": finetuned_model,
        "n_test": len(rows),
        "metrics": {name: {"base": base_summary[name], "finetuned": ft_summary[name]} for name in metrics.METRIC_NAMES},
        "judge": judge,
        "examples": [
            {
                "platform": rows[i]["platform"],
                "attributes": rows[i]["attributes"],
                "base": example_view(base_scores[i], base_texts[i]),
                "finetuned": example_view(ft_scores[i], ft_texts[i]),
                "reference": rows[i]["reference"],
            }
            for i in range(min(args.n_examples, len(rows)))
        ],
        "speed": speed_block(base_calls, ft_calls, hardware),
        "cost_per_1000_listings_inr": cost_block(ft_calls, args.tinker_usd_per_1m, args.usd_to_inr, closed),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "backend": args.backend,
            "renderer": renderer_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "judge_model": None if args.no_judge else args.judge_model,
            "test_file": Path(args.test_file).name,
        },
    }
    if args.backend == "ollama":
        summary["settings"]["ollama_host"] = args.ollama_host
        summary["settings"]["finetuned_source"] = source
    if closed:
        closed_summary = metrics.summarize(closed["scores"])
        summary["closed_baseline"] = {
            "provider": closed["provider"],
            "model": closed["model"],
            "metrics": {name: closed_summary[name] for name in metrics.METRIC_NAMES},
            "speed": speed.speed_summary(closed["calls"]),
        }
    if args.dry_run:
        summary["note"] = "DRY RUN with mock outputs - not real results"
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_report(out_dir / "eval_report.md", summary)

    print("\nmetric                      base   finetuned")
    for name, vals in summary["metrics"].items():
        fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{name:26s} {fmt(vals['base'])}   {fmt(vals['finetuned'])}")
    if judge:
        print(f"judge: finetuned_win={judge['finetuned_win']} base_win={judge['base_win']} tie={judge['tie']}")
    sp = summary["speed"]
    print(f"speed ({sp['hardware']}): base {sp['base']['median_latency_s']}s / {sp['base']['tokens_per_s']} tok/s, "
          f"finetuned {sp['finetuned']['median_latency_s']}s / {sp['finetuned']['tokens_per_s']} tok/s")
    print(f"cost per 1000 listings (INR): {json.dumps({k: v for k, v in summary['cost_per_1000_listings_inr'].items() if k != 'assumptions'})}")
    print(f"\nwrote {outputs_file}, {out_dir / 'eval_summary.json'}, {out_dir / 'eval_report.md'}")

    if args.backend == "ollama" and not args.dry_run and not args.no_update_main_summary:
        update_main_summary(RESULTS_DIR / "eval_summary.json", summary)


if __name__ == "__main__":
    main()
