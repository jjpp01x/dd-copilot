import json
from unittest.mock import MagicMock

from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index
from dd_copilot.extract import run_extraction, _parse_json_response

SOURCE_TEXT = (
    "Isomorphic Labs combines artificial intelligence and biology to accelerate "
    "drug discovery. The company is a DeepMind spin-off."
)


def test_run_extraction_marks_field_as_not_mentioned_when_citation_is_fabricated(monkeypatch):
    doc = Document(source_name="isomorphic-labs", text=SOURCE_TEXT)
    nodes = chunk_document(doc, chunk_size=60, chunk_overlap=10)
    index = build_index(nodes)

    fake_provider = MagicMock()

    haiku_payload = {
        "value": "Solves drug discovery.",
        "citation": "cures rare diseases in 24 hours",
        "mentioned": True,
    }
    sonnet_payload = {
        "confidence_score": 3,
        "confidence_justification": "Public material is limited.",
    }

    fake_provider.complete.side_effect = [
        json.dumps(haiku_payload),  # problem
        json.dumps({"value": "", "citation": "", "mentioned": False}),  # differentiation
        json.dumps({"value": "", "citation": "", "mentioned": False}),  # performance
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),  # risk 1
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),  # risk 2
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),  # risk 3
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),  # risk 4
        json.dumps(sonnet_payload),  # final synthesis
    ]

    result = run_extraction(fake_provider, index, SOURCE_TEXT, "isomorphic-labs")

    assert result.extraction.problem.mentioned is False
    assert result.extraction.problem.citations == []
    assert result.confidence_score == 3


def test_parse_json_response_strips_markdown_fences():
    fenced = '```json\n{"mentioned": true, "value": "x"}\n```'
    assert _parse_json_response(fenced) == {"mentioned": True, "value": "x"}


def test_parse_json_response_extracts_json_from_surrounding_text():
    noisy = 'Sure, here is the answer:\n{"mentioned": false}\nHope that helps!'
    assert _parse_json_response(noisy) == {"mentioned": False}
