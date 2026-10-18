"""Build GROUNDED training data with a big teacher model on Tinker (v2 data, see README "v2").

Why: in v1 the targets were real Amazon listings. They state facts that are NOT in our
rule-based attributes, repeat keywords, and skip attributes, so the fine-tuned model learned
to invent facts, loop and skip attributes. Here a big teacher model fixes the data, per row:
  1. EXTRACT  read the original listing -> ProductAttributes JSON (like Gemma vision in the app)
  2. REWRITE  write the target listing using ONLY facts in those attributes / seller notes
  3. VALIDATE parse + platform rules + no repetition + attribute coverage + no invented numbers
              + no brand names; one retry with feedback, else the row is dropped
              (train/val only: test rows are never dropped, they fall back to the old data)

Reads  data/{train,val,test}.jsonl  (from prepare_data.py, with source.original)
Writes data/{train,val,test}_grounded.jsonl  (same row shape; old values kept as
       attributes_original / reference_original) and data/grounded_dropped.jsonl.
The test set keeps exactly the same product ids; its new reference is only used for ROUGE.

Resumable: rows already written (or dropped) are skipped, so just run it again after a stop.

Usage:
  python build_grounded_data.py --dry-run --limit 5   # offline, mock teacher, writes data/dry_run/
  python build_grounded_data.py --limit 20            # first 20 rows of each split (check quality)
  python build_grounded_data.py                       # everything (asks before spending money)
  python build_grounded_data.py --yes                 # same, no question
"""

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import grounding
from common import DATA_DIR, choose_renderer_name, data_file, load_env, read_jsonl, require_env

# Tinker prices, USD per 1M tokens (prefill, sample), from the "Models & Pricing" page in Oct 2026.
# Check https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/ before a big run.
TEACHER_PRICES = {
    "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50),
    "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335),
    "Qwen/Qwen3.8-27B": (0.6975, 5.595),
}
DEFAULT_TEACHER = "Qwen/Qwen3.5-397B-A17B"

# Rough answer sizes used for the cost estimate (tokens)
EXTRACT_OUT_TOKENS = 250
REWRITE_OUT_TOKENS = 700
RETRY_SHARE = 0.3  # assume up to 30% of calls need one retry (with the longer feedback chat)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER,
                   help=f"Tinker model used as teacher (priced: {', '.join(TEACHER_PRICES)})")
    p.add_argument("--renderer", default=None, help="cookbook renderer (default: picked from the model, thinking off)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N rows of each split (rows already done count too)")
    p.add_argument("--concurrency", type=int, default=64, help="rows handled at once (sample requests in flight)")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--usd-per-1m-prefill", type=float, default=None, help="override the teacher prefill price")
    p.add_argument("--usd-per-1m-sample", type=float, default=None, help="override the teacher sample price")
    p.add_argument("--retry-dropped", action="store_true", help="try rows listed in grounded_dropped.jsonl again")
    p.add_argument("--allow-missing-original", action="store_true",
                   help="run even if rows lack source.original (the teacher then sees the lossy reference)")
    p.add_argument("--yes", action="store_true", help="do not ask before spending money")
    p.add_argument("--dry-run", action="store_true", help="offline: mock teacher, writes to data/dry_run/, no Tinker")
    return p.parse_args()


# ---------------------------------------------------------------- teachers
class TinkerTeacher:
    """Samples the teacher model on Tinker. run() sends every request first, then waits for all."""

    def __init__(self, model, renderer_name, temperature, seed):
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.renderer = renderers.get_renderer(renderer_name, get_tokenizer(model))
        self.client = tinker.ServiceClient().create_sampling_client(base_model=model)
        stop = self.renderer.get_stop_sequences()
        self.params = {
            "extract": tinker.SamplingParams(max_tokens=600, temperature=temperature, stop=stop, seed=seed),
            "rewrite": tinker.SamplingParams(max_tokens=1500, temperature=temperature, stop=stop, seed=seed),
        }
        self.prefill_tokens = self.sample_tokens = 0

    def run(self, jobs):
        """jobs = list of (kind, messages, row, attrs, attempt). Returns one answer text per job."""
        from tinker_cookbook.renderers import get_text_content

        prompts_in = [self.renderer.build_generation_prompt(job[1]) for job in jobs]
        # Fire off every request before waiting, so Tinker works on them in parallel
        futures = [
            self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params[job[0]])
            for prompt, job in zip(prompts_in, jobs)
        ]
        texts = []
        for prompt, fut, job in zip(prompts_in, futures, jobs):
            try:
                tokens = fut.result().sequences[0].tokens
                message, _ = self.renderer.parse_response(tokens)
                texts.append(get_text_content(message))
                self.prefill_tokens += prompt.length
                self.sample_tokens += len(tokens)
            except Exception as err:  # noqa: BLE001 - one failed call just counts as a bad answer
                print(f"  teacher error on {job[2]['id']} ({job[0]}): {err.__class__.__name__}: {str(err)[:150]}")
                texts.append("")
        return texts


class MockTeacher:
    """Offline stand-in for --dry-run. Some rows answer badly on purpose so the retry/drop paths run."""

    def __init__(self):
        self.prefill_tokens = self.sample_tokens = 0

    def run(self, jobs):
        texts = []
        for kind, messages, row, attrs, attempt in jobs:
            bucket = zlib.crc32(row["id"].encode()) % 12
            if kind == "extract":
                # 1 in 12 rows: first answer is not JSON (tests the retry)
                text = "Sorry, here you go" if bucket == 1 and attempt == 0 else json.dumps(grounding.mock_attributes(row))
            else:
                # 1 in 12 rows: first answer loops; 1 in 12 rows: always loops (tests drop / fallback)
                sloppy = bucket == 2 or (bucket == 3 and attempt == 0)
                text = json.dumps(grounding.mock_listing(row, attrs, sloppy=sloppy))
            self.prefill_tokens += sum(grounding.rough_tokens(m["content"]) for m in messages)
            self.sample_tokens += grounding.rough_tokens(text)
            texts.append(text)
        return texts


# ---------------------------------------------------------------- one chunk of rows
def ask_with_retry(teacher, kind, rows, messages, attrs_list, check):
    """Ask the teacher once per row, check each answer, and retry the bad ones once with feedback.

    check(text, row, attrs) -> (value or None, problems). Returns (values, problems, tries) per row.
    """
    jobs = [(kind, m, r, a, 0) for r, m, a in zip(rows, messages, attrs_list)]
    answers = teacher.run(jobs)
    results = [check(t, r, a) for t, r, a in zip(answers, rows, attrs_list)]
    tries = [1] * len(rows)

    bad = [i for i, (_, problems) in enumerate(results) if problems]
    if bad:
        retry_jobs = [
            (kind, grounding.feedback_messages(messages[i], answers[i], results[i][1]), rows[i], attrs_list[i], 1)
            for i in bad
        ]
        for i, text in zip(bad, teacher.run(retry_jobs)):
            results[i] = check(text, rows[i], attrs_list[i])
            tries[i] = 2
    return [r[0] for r in results], [r[1] for r in results], tries


def check_attributes(text, row, _attrs):
    """Parse + check an EXTRACT answer."""
    try:
        attrs = grounding.parse_attributes(text)
    except (ValueError, TypeError) as err:
        return None, [f"not valid JSON ({err})"]
    return attrs, grounding.attribute_problems(attrs, grounding.brand_names(row))


def check_listing(text, row, attrs):
    """Parse + check a REWRITE answer."""
    return grounding.listing_problems(text, attrs, row.get("notes", ""), row["platform"], grounding.brand_names(row))


def process_chunk(rows, split, teacher, teacher_name):
    """Extract -> rewrite -> validate for a chunk of rows. Returns (good rows, dropped log entries)."""
    is_test = split == "test"
    out, dropped = [], []

    # 1. EXTRACT attributes from the original listing
    attrs, attr_problems, extract_tries = ask_with_retry(
        teacher, "extract", rows, [grounding.extract_messages(r) for r in rows], [None] * len(rows), check_attributes
    )
    keep_rows, keep_attrs, fallbacks = [], [], {}
    for row, a, problems in zip(rows, attrs, attr_problems):
        if problems and not is_test:
            dropped.append({"id": row["id"], "split": split, "stage": "extract", "problems": problems})
            continue
        if problems:
            # Test rows are never dropped: keep the old rule-based attributes for this one
            print(f"  [test] {row['id']}: extract failed {problems} -> keeping old attributes")
            a = row["attributes"]
            fallbacks[row["id"]] = ["attributes"]
        keep_rows.append(row)
        keep_attrs.append(a)
    extract_tries = {r["id"]: t for r, t in zip(rows, extract_tries)}

    # 2. REWRITE the listing from those attributes, 3. VALIDATE it
    listings, list_problems, list_tries = ask_with_retry(
        teacher, "rewrite", keep_rows,
        [grounding.rewrite_messages(r, a) for r, a in zip(keep_rows, keep_attrs)], keep_attrs, check_listing,
    )
    for row, a, listing, problems, tries in zip(keep_rows, keep_attrs, listings, list_problems, list_tries):
        if problems and not is_test:
            dropped.append({"id": row["id"], "split": split, "stage": "rewrite", "problems": problems})
            continue
        fallback = fallbacks.get(row["id"], [])
        if problems:
            print(f"  [test] {row['id']}: rewrite failed {problems} -> keeping old reference")
            listing = row["reference"]
            fallback = fallback + ["reference"]
        out.append({
            **row,
            "attributes": a,
            "reference": listing,
            "attributes_original": row["attributes"],
            "reference_original": row["reference"],
            "grounding": {
                "teacher": teacher_name,
                "extract_tries": extract_tries[row["id"]],
                "rewrite_tries": tries,
                "fallback": fallback,
                "problems": problems,
            },
        })
    return out, dropped


# ---------------------------------------------------------------- helpers
def append_jsonl(path, rows):
    """Append rows to a .jsonl file right away (so a stopped run keeps its work)."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path):
    """Ids already in an output file (empty set if it does not exist yet)."""
    return {r["id"] for r in read_jsonl(path)} if Path(path).exists() else set()


def estimate(pending, prices):
    """Rough token + dollar estimate for the rows still to do."""
    prefill = sample = 0.0
    for split, rows in pending.items():
        for row in rows:
            ext_in = sum(grounding.rough_tokens(m["content"]) for m in grounding.extract_messages(row))
            rew_in = sum(grounding.rough_tokens(m["content"]) for m in grounding.rewrite_messages(row, row["attributes"]))
            prefill += ext_in + rew_in
            sample += EXTRACT_OUT_TOKENS + REWRITE_OUT_TOKENS
    # Retries resend the chat plus the first answer
    prefill_total = prefill * (1 + RETRY_SHARE) + sample * RETRY_SHARE
    sample_total = sample * (1 + RETRY_SHARE)
    cost = grounding.estimate_cost(prefill_total, sample_total, *prices) if prices else None
    return prefill_total, sample_total, cost


def show_examples(rows, n=2):
    """Print a few finished rows so you can eyeball the quality."""
    for row in rows[:n]:
        orig = (row.get("source") or {}).get("original") or {}
        print("\n" + "=" * 80)
        print(f"id={row['id']} platform={row['platform']} notes={row.get('notes') or 'none'}")
        print(f"ORIGINAL title: {orig.get('title') or row['reference_original']['title']}")
        print(f"OLD attributes: {json.dumps(row['attributes_original'], ensure_ascii=False)}")
        print(f"NEW attributes: {json.dumps(row['attributes'], ensure_ascii=False)}")
        print("NEW listing:\n" + json.dumps(row["reference"], ensure_ascii=False, indent=1))
        print(f"grounding: {row['grounding']}")


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    load_env()

    out_dir = DATA_DIR / "dry_run" if args.dry_run else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped_file = out_dir / "grounded_dropped.jsonl"
    if args.dry_run:
        # Start fresh each dry run so it is repeatable
        for split in args.splits:
            (out_dir / f"{split}_grounded.jsonl").unlink(missing_ok=True)
        dropped_file.unlink(missing_ok=True)

    renderer_name = args.renderer or choose_renderer_name(args.teacher_model)
    prices = TEACHER_PRICES.get(args.teacher_model)
    if args.usd_per_1m_prefill is not None and args.usd_per_1m_sample is not None:
        prices = (args.usd_per_1m_prefill, args.usd_per_1m_sample)
    print(f"teacher={args.teacher_model} renderer={renderer_name} "
          f"price/1M prefill,sample={prices or 'unknown (pass --usd-per-1m-prefill/--usd-per-1m-sample)'}")

    # Work out which rows still need doing in each split
    skip_dropped = set() if args.retry_dropped else done_ids(dropped_file)
    pending, missing_original = {}, 0
    for split in args.splits:
        rows = read_jsonl(data_file(split, "raw"))[: args.limit]
        done = done_ids(out_dir / f"{split}_grounded.jsonl")
        pending[split] = [r for r in rows if r["id"] not in done and r["id"] not in skip_dropped]
        missing_original += sum(not (r.get("source") or {}).get("original") for r in pending[split])
        print(f"[{split}] {len(rows)} rows, {len(done)} already done, {len(pending[split])} to do")

    if missing_original:
        print(f"WARNING: {missing_original} rows have no source.original (made by an older prepare_data.py). "
              "Run `python prepare_data.py --fill-source` first (same ids, adds the original listing text).")
        if not (args.dry_run or args.allow_missing_original):
            sys.exit("Stopped. (Or pass --allow-missing-original to use the lossy reference instead.)")

    n_rows = sum(len(r) for r in pending.values())
    if n_rows == 0:
        print("Nothing to do (all rows already done).")
        return
    prefill, sample, cost = estimate(pending, prices)
    rounds = sum(math.ceil(len(r) / args.concurrency) for r in pending.values()) * 4
    cost_text = f"~${cost:.2f}" if cost is not None else "unknown"
    print(f"estimate: {n_rows} rows, ~{prefill / 1e6:.2f}M prefill + ~{sample / 1e6:.2f}M sampled tokens "
          f"-> {cost_text} (rough upper estimate, assumes {RETRY_SHARE:.0%} retries); "
          f"{rounds} rounds of parallel calls, roughly {rounds * 0.5:.0f}-{rounds * 1.5:.0f} min")

    if args.dry_run:
        teacher = MockTeacher()
    else:
        if not args.yes and input("Spend this on Tinker? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Cancelled, nothing was sent.")
        require_env("TINKER_API_KEY")
        teacher = TinkerTeacher(args.teacher_model, renderer_name, args.temperature, args.seed)

    examples, totals = [], {}
    for split, rows in pending.items():
        out_file = out_dir / f"{split}_grounded.jsonl"
        kept = dropped_n = fallback_n = 0
        for start in range(0, len(rows), args.concurrency):
            good, dropped = process_chunk(rows[start : start + args.concurrency], split, teacher, args.teacher_model)
            append_jsonl(out_file, good)
            append_jsonl(dropped_file, dropped)
            kept += len(good)
            dropped_n += len(dropped)
            fallback_n += sum(bool(r["grounding"]["fallback"]) for r in good)
            examples += good[: max(0, 2 - len(examples))]
            print(f"[{split}] {start + len(good) + len(dropped)}/{len(rows)} done: kept {kept}, dropped {dropped_n}"
                  + (f", test fallbacks {fallback_n}" if split == "test" else ""), flush=True)
        totals[split] = {"kept": kept, "dropped": dropped_n, "fallbacks": fallback_n}

    show_examples(examples)
    spent = grounding.estimate_cost(teacher.prefill_tokens, teacher.sample_tokens, *prices) if prices else None
    print("\n" + "=" * 80)
    print(f"summary: {json.dumps(totals)}")
    print(f"tokens used: {teacher.prefill_tokens:,} prefill + {teacher.sample_tokens:,} sampled"
          + (f" -> ~${spent:.2f}" if spent is not None and not args.dry_run else ""))
    print(f"wrote {', '.join(str(out_dir / f'{s}_grounded.jsonl') for s in pending)} (+ {dropped_file.name})")
    if args.dry_run:
        print("dry run OK (mock teacher, no Tinker calls made)")
    else:
        print("Next: python train.py   (it picks the grounded files automatically)")


if __name__ == "__main__":
    main()
