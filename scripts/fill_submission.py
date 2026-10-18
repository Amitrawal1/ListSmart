"""Fill the number placeholders in docs/SUBMISSION.md from the real eval results.

Run from the project root after evaluate.py has finished:
    python3 scripts/fill_submission.py

It writes docs/SUBMISSION.filled.md and leaves docs/SUBMISSION.md untouched.
Only placeholders we can fill from real files are filled; the rest (first users,
links, your own sentences) stay as {{...}} so you can see what is still missing.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "finetune" / "results"
DRAFT = ROOT / "docs" / "SUBMISSION.md"
OUT = ROOT / "docs" / "SUBMISSION.filled.md"


def pct(value):
    """0.8417 -> '84.2%'. Missing values stay as 'n/a'."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def load(path: Path) -> dict | None:
    """Read a JSON file, or return None if it is missing."""
    return json.loads(path.read_text()) if path.exists() else None


def values_from_summary(summary: dict) -> dict:
    """Turn eval_summary.json into {placeholder_name: text}."""
    values = {"n_test": str(summary.get("n_test", "n/a"))}
    metrics = summary.get("metrics", {})
    for name in ["format_valid", "constraint_compliance", "attribute_coverage"]:
        for side in ["base", "finetuned", "closed"]:
            if side in metrics.get(name, {}):
                values[f"{name}_{side}"] = pct(metrics[name][side])
    for side in ["base", "finetuned", "closed"]:
        if side in metrics.get("rouge_l", {}):
            values[f"rouge_l_{side}"] = f"{metrics['rouge_l'][side]:.2f}"

    judge = summary.get("judge") or {}
    for key in ["finetuned_win", "base_win", "tie"]:
        if key in judge:
            values[f"judge_{key}"] = pct(judge[key])

    speed = summary.get("speed") or {}
    for side in ["base", "finetuned"]:
        latency = (speed.get(side) or {}).get("median_latency_s")
        if latency is not None:
            values[f"speed_{side}_median_latency_s"] = f"{latency:.1f}"
    if speed.get("hardware"):
        values["hardware"] = str(speed["hardware"])

    cost = summary.get("cost_per_1000_listings_inr") or {}
    for key in ["local", "tinker_hosted", "closed_api"]:
        if key in cost:
            values[f"cost_{key}"] = "not measured" if cost[key] is None else f"{cost[key]:.2f}"
    return values


def main() -> int:
    summary = load(RESULTS / "eval_summary.json")
    if summary is None or not summary.get("available"):
        print("No real finetune/results/eval_summary.json yet. Run: cd finetune && python evaluate.py")
        return 1

    values = values_from_summary(summary)

    # Version 1 numbers (the run that lost), kept in results/v1/
    v1 = load(RESULTS / "v1" / "eval_summary.json")
    if v1:
        for key, text in values_from_summary(v1).items():
            values[f"v1_{key}"] = text

    # Training set size = rows the model actually trained on
    train_info = load(RESULTS / "train_info.json") or {}
    if train_info.get("n_train"):
        values["n_train"] = str(train_info["n_train"])

    text = DRAFT.read_text()
    filled = re.sub(r"{{(\w+)}}", lambda m: values.get(m.group(1), m.group(0)), text)
    OUT.write_text(filled)

    left = sorted(set(re.findall(r"{{(\w+)}}", filled)) - {"double_braces"})
    print(f"wrote {OUT.relative_to(ROOT)}  (filled {len(values)} values)")
    print(f"still to fill by hand ({len(left)}): {', '.join(left)}")

    # Plain-words verdict so nobody posts a claim the numbers do not back up
    judge = summary.get("judge") or {}
    if judge:
        ft, base = judge.get("finetuned_win", 0), judge.get("base_win", 0)
        verdict = "fine-tune WINS" if ft > base else "fine-tune does NOT beat base yet"
        print(f"judge: finetuned {pct(ft)} vs base {pct(base)} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
