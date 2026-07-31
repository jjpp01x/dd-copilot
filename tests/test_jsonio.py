import json

import pytest

from dd_copilot.jsonio import parse_json_response


def test_plain_json_object():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_markdown_fences_are_tolerated():
    fenced = '```json\n{"mentioned": true, "value": "x"}\n```'
    assert parse_json_response(fenced) == {"mentioned": True, "value": "x"}


def test_leading_prose_is_tolerated():
    assert parse_json_response('Here is the answer:\n{"a": 1}') == {"a": 1}


def test_trailing_prose_after_the_object_is_ignored():
    """The failure that broke a real Ollama run: the model emitted a valid
    object and then kept talking. json.loads raised 'Extra data'."""
    assert parse_json_response('{"a": 1}\n\nLet me know if you need more.') == {"a": 1}


def test_a_second_object_after_the_first_is_ignored():
    """Small local models sometimes emit one object per line. The first
    complete object is the answer; the rest is noise."""
    assert parse_json_response('{"a": 1}\n{"a": 2}\n{"a": 3}') == {"a": 1}


def test_nested_braces_do_not_truncate_the_object():
    payload = {"claims": [{"text": "x", "figure": "8 ms"}], "note": "ok"}
    assert parse_json_response(json.dumps(payload)) == payload


def test_prose_before_and_a_second_object_after():
    text = 'Sure!\n```json\n{"claims": []}\n```\nAnd here is another: {"claims": [1]}'
    assert parse_json_response(text) == {"claims": []}


def test_text_with_no_json_raises_a_readable_error():
    with pytest.raises(ValueError) as excinfo:
        parse_json_response("I could not answer that.")

    assert "no JSON object" in str(excinfo.value)


def test_a_top_level_array_is_rejected():
    """Every call site expects a mapping; returning a list would fail later
    with a confusing AttributeError instead of here."""
    with pytest.raises(ValueError):
        parse_json_response("[1, 2, 3]")
