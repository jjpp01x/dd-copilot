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
        *[json.dumps({"mentioned": False, "detail": None, "citation": ""})] * 6,  # six checklist risks
        json.dumps({"claims": []}),
        json.dumps({"confidence_score": 3, "confidence_justification": "Public material is limited."}),
    ]

    markdown, _report_input = analyze(SOURCE_TEXT, fake_provider)

    assert "# Technical Due Diligence Report" in markdown
    assert "accelerate drug discovery" in markdown
    assert "## 4. Claims Assessed" in markdown


def test_analyze_returns_the_structured_input_beside_the_markdown(monkeypatch):
    """El Markdown es un derivado; el ReportInput es la fuente. Descartarlo
    obligaba a quien quisiera los datos a parsear la presentación."""
    from dd_copilot import pipeline
    from dd_copilot.models import ChecklistField, ExtractionResult, ReportInput

    report_input = ReportInput(
        source_name="acme",
        extraction=ExtractionResult(
            problem=ChecklistField(value="p", mentioned=True),
            differentiation=ChecklistField(value="d", mentioned=True),
            performance=ChecklistField(value="perf", mentioned=True),
            risks=[],
        ),
        confidence_score=3,
        confidence_justification="j",
    )
    monkeypatch.setattr(pipeline, "ingest", lambda source: _FakeDocument())
    monkeypatch.setattr(pipeline, "chunk_document", lambda document: [])
    monkeypatch.setattr(pipeline, "build_index", lambda nodes: None)
    monkeypatch.setattr(
        pipeline, "run_extraction", lambda *args, **kwargs: report_input
    )

    markdown, returned = pipeline.analyze("acme", provider=None)

    assert returned is report_input
    assert markdown.startswith("#")


class _FakeDocument:
    text = "cuerpo"
    source_name = "acme"
