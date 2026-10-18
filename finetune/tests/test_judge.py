"""Offline test of the blind A/B judge logic in evaluate.py using a fake judge."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
import metrics  # noqa: E402
from test_metrics import GOOD, ROW  # noqa: E402

WORSE = dict(GOOD, title="Item")


def test_judge_maps_blind_order_back_to_the_right_model():
    base = metrics.score_example(json.dumps(WORSE), ROW)
    ft = metrics.score_example(json.dumps(GOOD), ROW)

    # Fake judge that always picks whichever listing has the longer title
    def fake_judge(prompt):
        a = prompt.split("Listing A:")[1].split("Listing B:")[0]
        return '{"winner": "%s"}' % ("A" if GOOD["title"] in a else "B")

    rng = random.Random(0)
    results = [evaluate.judge_example(ROW, base, ft, rng, fake_judge) for _ in range(20)]
    assert all(r["winner"] == "finetuned" for r in results)
    # Both A/B orders were actually used
    assert {r["finetuned_is_a"] for r in results} == {True, False}


def test_invalid_outputs_are_decided_without_the_judge():
    good = metrics.score_example(json.dumps(GOOD), ROW)
    bad = metrics.score_example("not json", ROW)
    never = lambda prompt: (_ for _ in ()).throw(AssertionError("judge should not be called"))  # noqa: E731
    rng = random.Random(0)
    assert evaluate.judge_example(ROW, bad, good, rng, never)["winner"] == "finetuned"
    assert evaluate.judge_example(ROW, good, bad, rng, never)["winner"] == "base"
    assert evaluate.judge_example(ROW, bad, bad, rng, never)["winner"] == "tie"
