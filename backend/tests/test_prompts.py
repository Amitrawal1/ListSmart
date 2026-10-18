import pytest

from app.prompts import build_writer_messages, parse_listing
from app.writer import mock_listing

GOOD = '{"title":"T","description":"D","bullets":["a","b","c","d","e"],"tags":["x","y"],"search_query":"q"}'


def test_parse_plain_json():
    out = parse_listing(GOOD)
    assert out["title"] == "T" and len(out["bullets"]) == 5


def test_parse_fenced_with_chatter():
    out = parse_listing(f"Sure! Here it is:\n```json\n{GOOD}\n```\nHope this helps.")
    assert out["search_query"] == "q"


def test_parse_string_tags_and_limit():
    text = '{"title":"T","description":"D","bullets":"one","tags":"' + ",".join(f"t{i}" for i in range(20)) + '","search_query":"q"}'
    out = parse_listing(text)
    assert out["bullets"] == ["one"]
    assert len(out["tags"]) == 13


def test_parse_missing_field_raises():
    with pytest.raises(ValueError):
        parse_listing('{"title":"T"}')
    with pytest.raises(ValueError):
        parse_listing("no json here")


def test_messages_shape():
    msgs = build_writer_messages({"product_type": "mug"}, "etsy", "")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "140" in msgs[1]["content"]


def test_mock_listing_respects_rules():
    attrs = {"product_type": "kurti", "colors": ["maroon"], "material": "cotton",
             "key_features": ["3/4 sleeves"], "gender": "women"}
    out = mock_listing(attrs, "meesho")
    assert len(out["title"]) <= 100
    assert len(out["bullets"]) == 5
    assert len(out["tags"]) <= 13 and all(len(t) <= 20 for t in out["tags"])
    assert out["search_query"]
