import json
from unittest.mock import MagicMock

from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index
from dd_copilot.extract import run_extraction

SOURCE_TEXT = (
    "Isomorphic Labs combines artificial intelligence and biology to accelerate "
    "drug discovery. The company is a DeepMind spin-off."
)


def _fake_haiku_response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def _fake_sonnet_response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def test_run_extraction_marks_field_as_not_mentioned_when_citation_is_fabricated(monkeypatch):
    doc = Document(source_name="isomorphic-labs", text=SOURCE_TEXT)
    nodes = chunk_document(doc, chunk_size=60, chunk_overlap=10)
    index = build_index(nodes)

    fake_client = MagicMock()

    haiku_payload = {
        "value": "Solves drug discovery.",
        "citation": "cures rare diseases in 24 hours",
        "mentioned": True,
    }
    sonnet_payload = {
        "confidence_score": 3,
        "confidence_justification": "Public material is limited.",
    }

    fake_client.messages.create.side_effect = [
        _fake_haiku_response(haiku_payload),  # problem
        _fake_haiku_response({"value": "", "citation": "", "mentioned": False}),  # differentiation
        _fake_haiku_response({"value": "", "citation": "", "mentioned": False}),  # performance
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 1
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 2
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 3
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 4
        _fake_sonnet_response(sonnet_payload),  # final synthesis
    ]

    result = run_extraction(fake_client, index, SOURCE_TEXT, "isomorphic-labs")

    assert result.extraction.problem.mentioned is False
    assert result.extraction.problem.citations == []
    assert result.confidence_score == 3
