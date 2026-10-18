"""Offline tests for the grounded-data helpers (grounding.py, metrics repetition, build_grounded_data.py)."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_grounded_data as bgd  # noqa: E402
import common  # noqa: E402
import grounding  # noqa: E402
import metrics  # noqa: E402
import prepare_data  # noqa: E402
from test_metrics import ATTRS, GOOD, ROW  # noqa: E402

# A row as prepare_data.py now writes it (with the original listing text)
ORIG_ROW = dict(
    copy.deepcopy(ROW),
    id="TEST1",
    source={"original": {
        "title": "Rangrez Maroon Cotton Block Print Kurti for Women",
        "description": "Soft cotton kurti with hand block print.",
        "features": ["3/4 sleeves", "Mandarin collar"],
        "details": {"Material": "Cotton"},
        "categories": ["Clothing", "Women"],
        "store": "Rangrez",
        "brand": "Generic",
    }},
)


# ---------------------------------------------------------------- repetition
def test_repeat_ratio_is_zero_for_normal_text_and_high_for_loops():
    assert metrics.ngram_repeat_ratio("A soft cotton kurti with a maroon block print.") == 0.0
    loop = ", ".join(["10 Pack Door Stopper"] * 10)
    assert metrics.ngram_repeat_ratio(loop) > 0.5
    assert metrics.ngram_repeat_ratio("too short") == 0.0


def test_field_repeat_ratio_checks_fields_separately():
    # The title phrase showing up again in the description is fine
    listing = dict(GOOD, description="Maroon Cotton Block Print Kurti for Women. Great for daily wear.\n\nSoft.")
    assert metrics.max_field_repeat_ratio(listing) == 0.0
    looping = dict(GOOD, bullets=["best cotton kurti ever"] * 5)
    assert metrics.max_field_repeat_ratio(looping) > grounding.MAX_REPEAT_RATIO
    assert metrics.max_field_repeat_ratio(None) == 0.0


# ---------------------------------------------------------------- original text + brands
def test_original_text_uses_source_original_and_falls_back_to_reference():
    text, real = grounding.original_text(ORIG_ROW)
    assert real and "Rangrez" in text and "Mandarin collar" in text and "Material: Cotton" in text
    text, real = grounding.original_text(ROW)
    assert not real and GOOD["title"] in text


def test_brand_names_skip_generic():
    assert grounding.brand_names(ORIG_ROW) == ["rangrez"]
    assert grounding.brand_names(ROW) == []
    assert grounding.mentions_brand("by RANGREZ studio", ["rangrez"]) == "rangrez"
    assert grounding.mentions_brand("rangrezzz", ["rangrez"]) == ""


def test_extract_prompt_names_the_brand_to_avoid():
    user = grounding.extract_messages(ORIG_ROW)[1]["content"]
    assert "Original listing:" in user and "rangrez" in user and '"key_features"' in user


# ---------------------------------------------------------------- attributes
def test_parse_attributes_cleans_types_and_fills_spec_keys():
    text = '```json\n{"product_type": " Kurti ", "colors": "Maroon, Gold", "gender": "female", ' \
           '"key_features": ["3/4 sleeves", ""], "visible_text": "RANGREZ"}\n```'
    a = grounding.parse_attributes(text)
    assert list(a) == grounding.ATTRIBUTE_KEYS
    assert a["product_type"] == "kurti" and a["colors"] == ["maroon", "gold"]
    assert a["gender"] == "" and a["key_features"] == ["3/4 sleeves"]
    assert a["visible_text"] == "" and a["condition"] == "new"


def test_parse_attributes_rejects_non_json():
    try:
        grounding.parse_attributes("Sorry, I cannot help")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_attribute_problems():
    assert grounding.attribute_problems(dict(ATTRS, key_features=["3/4 sleeves", "mandarin collar"]), ["rangrez"]) == []
    problems = grounding.attribute_problems(dict(ATTRS, product_type="", key_features=["Rangrez print"]), ["rangrez"])
    assert any("product_type" in p for p in problems)
    assert any("2 key_features" in p for p in problems)
    assert any("rangrez" in p for p in problems)


# ---------------------------------------------------------------- listing validation
def check(listing, attrs=ATTRS, notes="", brands=()):
    return grounding.listing_problems(json.dumps(listing), attrs, notes, "meesho", list(brands))


def test_good_listing_has_no_problems():
    listing, problems = check(GOOD)
    assert listing and problems == []


def test_listing_problems_catch_each_failure():
    assert check({"title": "x"})[1][0].startswith("not a valid listing")
    assert any("5 bullets" in p for p in check(dict(GOOD, bullets=GOOD["bullets"][:4]))[1])
    looping = dict(GOOD, description=" ".join(["maroon cotton kurti best"] * 8) + "\n\nNice.")
    assert any("repeats" in p for p in check(looping)[1])
    assert any("different" in p for p in check(dict(GOOD, tags=["kurti", "Kurti"]))[1])
    no_material = dict(GOOD, title="Maroon Kurti", description="A maroon kurti.\n\nNice.")
    assert any("mention" in p for p in check(no_material)[1])
    sized = dict(GOOD, description=GOOD["description"] + " Size 10.5 x 10.5 inches.")
    assert any("10.5" in p for p in check(sized)[1])
    branded = dict(GOOD, title="Rangrez " + GOOD["title"])
    assert any("brand" in p for p in check(branded, brands=["rangrez"])[1])


def test_numbers_from_attributes_or_notes_are_allowed():
    attrs = dict(ATTRS, key_features=["3/4 sleeves", "pack of 20 napkins"])
    listing = dict(GOOD, bullets=GOOD["bullets"][:4] + ["Pack of 20, 2 ply"])
    assert grounding.unsupported_numbers(listing, attrs, "2 ply paper") == []
    assert grounding.unsupported_numbers(listing, attrs, "") == ["2"]


def test_feedback_messages_add_answer_and_problems():
    msgs = grounding.rewrite_messages(ORIG_ROW, ATTRS)
    out = grounding.feedback_messages(msgs, "{bad}", ["need exactly 5 bullets"])
    assert out[:2] == msgs and out[2] == {"role": "assistant", "content": "{bad}"}
    assert "need exactly 5 bullets" in out[3]["content"] and out[3]["role"] == "user"


def test_rewrite_prompt_contains_the_exact_writer_prompt():
    writer_prompt = common.prompts.build_writer_messages(ATTRS, "meesho", "")[1]["content"]
    user = grounding.rewrite_messages(ORIG_ROW, ATTRS)[1]["content"]
    assert writer_prompt in user and "rangrez" in user


def test_estimate_cost():
    assert grounding.estimate_cost(2e6, 1e6, 1.5, 7.5) == 10.5
    assert grounding.rough_tokens("x" * 35) == 10


# ---------------------------------------------------------------- pipeline with the mock teacher
def make_rows(n):
    rows = []
    for i in range(n):
        row = copy.deepcopy(ORIG_ROW)
        row["id"] = f"ROW{i}"
        rows.append(row)
    return rows


def test_mock_listing_passes_validation():
    attrs = grounding.mock_attributes(ORIG_ROW)
    text = json.dumps(grounding.mock_listing(ORIG_ROW, attrs))
    assert grounding.listing_problems(text, attrs, "", "meesho", ["rangrez"])[1] == []
    sloppy = json.dumps(grounding.mock_listing(ORIG_ROW, attrs, sloppy=True))
    assert grounding.listing_problems(sloppy, attrs, "", "meesho", [])[1]


def test_train_rows_can_be_dropped_but_test_rows_never():
    rows = make_rows(40)
    good, dropped = bgd.process_chunk(rows, "train", bgd.MockTeacher(), "mock")
    assert len(good) + len(dropped) == 40 and dropped  # the mock always-loops on some ids
    assert all(r["grounding"]["rewrite_tries"] in (1, 2) for r in good)

    good, dropped = bgd.process_chunk(rows, "test", bgd.MockTeacher(), "mock")
    assert dropped == [] and [r["id"] for r in good] == [r["id"] for r in rows]
    fallbacks = [r for r in good if r["grounding"]["fallback"]]
    assert fallbacks and all(r["reference"] == r["reference_original"] for r in fallbacks)
    # Old values are kept next to the new ones
    assert all("attributes_original" in r and "reference_original" in r for r in good)


# ---------------------------------------------------------------- data variant + prepare_data
def test_pick_data_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    common.write_jsonl(tmp_path / "test.jsonl", [{"id": "A"}, {"id": "B"}])
    assert common.pick_data_variant(None, ["test"]) == "raw"
    # Unfinished grounded file (like after --limit): still raw, so nothing runs on a partial test set
    common.write_jsonl(tmp_path / "test_grounded.jsonl", [{"id": "A"}])
    assert common.grounded_missing("test") == 1
    assert common.pick_data_variant(None, ["test"]) == "raw"
    common.write_jsonl(tmp_path / "test_grounded.jsonl", [{"id": "A"}, {"id": "B"}])
    assert common.pick_data_variant(None, ["test"]) == "grounded"
    # Dropped train rows count as done
    common.write_jsonl(tmp_path / "train.jsonl", [{"id": "C"}, {"id": "D"}])
    common.write_jsonl(tmp_path / "train_grounded.jsonl", [{"id": "C"}])
    common.write_jsonl(tmp_path / "grounded_dropped.jsonl", [{"id": "D", "split": "train"}])
    assert common.pick_data_variant(None, ["train", "test"]) == "grounded"
    (tmp_path / "train_grounded.jsonl").unlink()
    assert common.pick_data_variant(None, ["train", "test"]) == "raw"
    assert common.pick_data_variant("raw", ["test"]) == "raw"
    assert common.data_file("val", "grounded").name == "val_grounded.jsonl"


def test_fill_source_only_adds_original(tmp_path):
    old = [dict(copy.deepcopy(ROW), id="A", source={"parent_asin": "A"}),
           dict(copy.deepcopy(ROW), id="B", source={"parent_asin": "B"})]
    common.write_jsonl(tmp_path / "test.jsonl", old)
    rebuilt = [dict(copy.deepcopy(ROW), id="B", source={"original": {"title": "orig B"}})]
    prepare_data.fill_source(rebuilt, str(tmp_path))
    new = common.read_jsonl(tmp_path / "test.jsonl")
    assert [r["id"] for r in new] == ["A", "B"]
    assert "original" not in new[0]["source"] and new[1]["source"]["original"] == {"title": "orig B"}
    assert new[1]["reference"] == old[1]["reference"] and new[1]["source"]["parent_asin"] == "B"


def test_original_listing_keeps_brand_and_text():
    meta = {"title": "Rangrez <b>Kurti</b>", "description": ["Soft &amp; light."], "features": ["3/4 sleeves"],
            "details": {"Brand": "Rangrez", "Material": "Cotton", "Nested": {"a": 1}},
            "categories": ["Clothing"], "store": "Rangrez"}
    orig = prepare_data.original_listing(meta)
    assert orig["title"] == "Rangrez Kurti" and orig["description"] == "Soft & light."
    assert orig["details"] == {"Brand": "Rangrez", "Material": "Cotton"} and orig["brand"] == "Rangrez"
