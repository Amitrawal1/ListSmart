"""Evaluate the base model vs the fine-tuned model on the SAME held-out test prompts.

Both models are sampled through Tinker with identical prompts (prompts.build_writer_messages),
the same renderer and low temperature. Metrics (see SPEC.md and metrics.py):
  1. format validity       2. constraint compliance   3. attribute coverage
  4. ROUGE-L vs reference  5. Gemma LLM-judge, pairwise + blind (random A/B order)

Writes:
  results/eval_outputs.jsonl  every test example with both outputs + scores
  results/eval_summary.json   summary in the SPEC shape (served by GET /api/eval)
  results/eval_report.md      markdown table for the README / demo

Usage:
  python evaluate.py --dry-run           # offline pipeline check with mock outputs (writes to results/dry_run/)
  python evaluate.py                     # real run (needs TINKER_API_KEY, GOOGLE_API_KEY for the judge)
  python evaluate.py --no-judge          # skip the Gemma judge
  python evaluate.py --reuse-outputs     # re-score / re-judge saved outputs without sampling again
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import metrics
from common import (
    DATA_DIR,
    DEFAULT_BASE_MODEL,
    RESULTS_DIR,
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
    return p.parse_args()


# ---------------------------------------------------------------- sampling
def sample_all(sampling_client, renderer, rows, args, label):
    """Generate one listing per test row; returns raw text outputs in the same order."""
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
        futures = [
            sampling_client.sample(
                prompt=renderer.build_generation_prompt(prompt_messages(row)),
                num_samples=1,
                sampling_params=params,
            )
            for row in chunk
        ]
        for fut in futures:
            tokens = fut.result().sequences[0].tokens
            message, _ = renderer.parse_response(tokens)
            outputs.append(get_text_content(message))
        print(f"[{label}] sampled {len(outputs)}/{len(rows)}", flush=True)
    return outputs


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
    lines += [
        "",
        "Metric notes: format_valid = parses into a Listing with all fields; constraint_compliance = passes every",
        "platform rule (title length, exactly 5 bullets, tag count/length, 2-4 paragraphs, 3-6 word query);",
        "attribute_coverage = share of attribute values (type, colours, material, pattern, style) mentioned in",
        "title+description; rouge_l = ROUGE-L F1 vs the real reference listing.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    load_env()
    from pathlib import Path

    out_dir = Path(args.out_dir)
    if args.dry_run:
        # Never let mock numbers land where the app reads real results
        out_dir = RESULTS_DIR / "dry_run"
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

    # ---- get raw outputs for both models
    if args.dry_run:
        finetuned_model = finetuned_model or "tinker://dry-run/mock"
        renderer_name = "mock"
        base_texts = [mock_output(r, "base") for r in rows]
        ft_texts = [mock_output(r, "finetuned") for r in rows]
    elif args.reuse_outputs:
        saved = {o["id"]: o for o in read_jsonl(outputs_file)}
        rows = [r for r in rows if r["id"] in saved]
        base_texts = [saved[r["id"]]["base"]["raw"] for r in rows]
        ft_texts = [saved[r["id"]]["finetuned"]["raw"] for r in rows]
        renderer_name = args.renderer or train_info.get("renderer") or "unknown"
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
        base_texts = sample_all(base_client, renderer, rows, args, "base")
        ft_texts = sample_all(ft_client, renderer, rows, args, "finetuned")

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

    # ---- per-example file
    per_example = []
    for row, bt, ft, bs, fs, jd in zip(rows, base_texts, ft_texts, base_scores, ft_scores, judgements):
        per_example.append({
            "id": row["id"],
            "platform": row["platform"],
            "notes": row.get("notes", ""),
            "attributes": row["attributes"],
            "reference": row["reference"],
            "base": {"raw": bt, **bs},
            "finetuned": {"raw": ft, **fs},
            "judge": jd,
        })
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
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "renderer": renderer_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "judge_model": None if args.no_judge else args.judge_model,
            "test_file": Path(args.test_file).name,
        },
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
    print(f"\nwrote {outputs_file}, {out_dir / 'eval_summary.json'}, {out_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
