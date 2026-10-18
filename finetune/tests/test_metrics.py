"""Offline tests for metrics.py and the data files (run: pytest finetune/tests)."""

import json
import sys
from pathlib import Path

import pytest

# Let tests import finetune/*.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics  # noqa: E402
from common import DATA_DIR, prompts, read_jsonl, training_conversation  # noqa: E402

ATTRS = {
    "product_type": "kurti",
    "category": "Women Ethnic Wear",
    "colors": ["maroon"],
    "material": "cotton",
    "pattern": "block print",
    "style": "",
    "gender": "women",
    "key_features": ["3/4 sleeves"],
    "visible_text": "",
    "condition": "new",
}

GOOD = {
    "title": "Maroon Cotton Block Print Kurti for Women",
    "description": "A soft cotton kurti with a maroon block print.\n\nGreat for daily wear.",
    "bullets": ["Soft cotton", "Block print", "3/4 sleeves", "Easy care", "Daily wear"],
    "tags": ["cotton kurti", "maroon kurti", "block print kurti"],
    "search_query": "maroon cotton kurti women",
}
ROW = {"platform": "meesho", "notes": "", "attributes": ATTRS, "reference": GOOD}


def test_good_output_scores_perfectly():
    s = metrics.score_example("```json\n" + json.dumps(GOOD) + "\n```", ROW)
    assert s["format_valid"] and s["constraint_compliance"]
    assert s["attribute_coverage"] == 1.0
    assert s["rouge_l"] == pytest.approx(1.0)


def test_garbage_output_is_invalid():
    s = metrics.score_example("Sure! Here is your listing: Title - nice kurti", ROW)
    assert not s["format_valid"] and not s["constraint_compliance"]
    assert s["attribute_coverage"] == 0.0 and s["rouge_l"] == 0.0
    assert s["error"]


def test_missing_field_is_invalid():
    bad = {k: v for k, v in GOOD.items() if k != "tags"}
    assert not metrics.score_example(json.dumps(bad), ROW)["format_valid"]


def test_constraint_checks_catch_each_rule():
    bad = dict(GOOD)
    bad["title"] = "x" * 101  # too long for meesho, fine for etsy
    bad["bullets"] = GOOD["bullets"][:4]
    bad["tags"] = [f"tag{i}" for i in range(14)]  # more than 13
    bad["description"] = "only one paragraph"
    bad["search_query"] = "kurti"
    checks = metrics.constraint_checks(bad, "meesho")
    assert not any(checks.values())
    assert metrics.constraint_checks(bad, "etsy")["title_length_ok"]


def test_tag_too_long_fails():
    bad = dict(GOOD, tags=["this tag is far too long for meesho"])
    assert not metrics.constraint_checks(bad, "meesho")["tags_ok"]


def test_attribute_coverage_partial_and_plural():
    listing = {"title": "Cotton Kurtis", "description": "Lovely and soft."}
    # product_type (kurti, via plural), material yes; colour and pattern missing -> 2/4
    assert metrics.attribute_coverage(ATTRS, listing) == 0.5
    assert metrics.attribute_coverage({"colors": []}, listing) is None


def test_judge_parsing_and_mapping():
    assert metrics.parse_judge_verdict('{"winner": "B", "reason": "x"}') == "B"
    assert metrics.parse_judge_verdict("I think TIE") == "TIE"
    assert metrics.parse_judge_verdict("no idea") == "TIE"
    assert metrics.verdict_to_winner("A", finetuned_is_a=True) == "finetuned"
    assert metrics.verdict_to_winner("A", finetuned_is_a=False) == "base"
    assert metrics.verdict_to_winner("TIE", finetuned_is_a=False) == "tie"
    tally = metrics.judge_tally(["finetuned", "finetuned", "base", "tie"])
    assert tally == {"finetuned_win": 0.5, "base_win": 0.25, "tie": 0.25, "n": 4}


def test_summarize_skips_none():
    scores = [metrics.score_example(json.dumps(GOOD), ROW), metrics.score_example("junk", ROW)]
    summary = metrics.summarize(scores)
    assert summary["format_valid"] == 0.5
    assert set(summary) == set(metrics.METRIC_NAMES)


def test_training_target_round_trips_through_parse_listing():
    convo = training_conversation(ROW)
    assert [m["role"] for m in convo] == ["system", "user", "assistant"]
    assert prompts.parse_listing(convo[-1]["content"], "meesho") == GOOD


# ---- data checks (only run when prepare_data.py has produced the files)
SAMPLE = DATA_DIR / "sample_train.jsonl"


@pytest.mark.skipif(not SAMPLE.exists(), reason="run prepare_data.py first")
def test_sample_references_obey_platform_rules():
    for row in read_jsonl(SAMPLE):
        target = prompts.listing_to_target(row["reference"])
        s = metrics.score_example(target, row)
        assert s["format_valid"], row["id"]
        assert s["constraint_compliance"], (row["id"], {k: s[k] for k in metrics.METRIC_NAMES})


@pytest.mark.skipif(not (DATA_DIR / "test.jsonl").exists(), reason="run prepare_data.py first")
def test_splits_do_not_share_products():
    ids = {name: {r["id"] for r in read_jsonl(DATA_DIR / f"{name}.jsonl")} for name in ("train", "val", "test")}
    assert not ids["train"] & ids["test"]
    assert not ids["train"] & ids["val"]
    assert not ids["val"] & ids["test"]
