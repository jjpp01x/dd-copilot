import json
from unittest.mock import MagicMock

from dd_copilot.pipeline import analyze

SOURCE_TEXT = (
    "Isomorphic Labs combines artificial intelligence and biology to accelerate "
    "drug discovery. The company is a DeepMind spin-off founded in 2021."
)


def _response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def test_analyze_returns_markdown_with_fixed_sections():
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _response({"value": "Accelerates drug discovery.", "citation": "accelerate drug discovery", "mentioned": True}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"confidence_score": 3, "confidence_justification": "Public material is limited."}),
    ]

    markdown = analyze(SOURCE_TEXT, fake_client)

    assert "# Technical Due Diligence Report" in markdown
    assert "accelerate drug discovery" in markdown
