import json
from unittest.mock import MagicMock

from dd_copilot.pipeline import analyze

SOURCE_TEXT = (
    "Isomorphic Labs combines artificial intelligence and biology to accelerate "
    "drug discovery. The company is a DeepMind spin-off founded in 2021."
)


def test_analyze_returns_markdown_with_fixed_sections():
    fake_provider = MagicMock()
    fake_provider.complete.side_effect = [
        json.dumps({"value": "Accelerates drug discovery.", "citation": "accelerate drug discovery", "mentioned": True}),
        json.dumps({"value": "", "citation": "", "mentioned": False}),
        json.dumps({"value": "", "citation": "", "mentioned": False}),
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),
        json.dumps({"mentioned": False, "detail": None, "citation": ""}),
        json.dumps({"confidence_score": 3, "confidence_justification": "Public material is limited."}),
    ]

    markdown = analyze(SOURCE_TEXT, fake_provider)

    assert "# Technical Due Diligence Report" in markdown
    assert "accelerate drug discovery" in markdown
