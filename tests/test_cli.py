import json
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from dd_copilot.cli import app

runner = CliRunner()


def _fake_provider(payloads: list[dict]) -> MagicMock:
    provider = MagicMock()
    provider.complete.side_effect = [json.dumps(payload) for payload in payloads]
    return provider


PAYLOADS = [
    {"value": "Solves X.", "citation": "Solves X.", "mentioned": True},
    {"value": "", "citation": "", "mentioned": False},
    {"value": "", "citation": "", "mentioned": False},
    *[{"mentioned": False, "detail": None, "citation": ""}] * 6,  # six checklist risks
    {"claims": []},
    {"confidence_score": 2, "confidence_justification": "Little material."},
]


def test_analyze_command_writes_markdown_file(tmp_path):
    output_path = tmp_path / "report.md"
    fake_provider = _fake_provider(PAYLOADS)

    with patch("dd_copilot.cli.build_provider", return_value=fake_provider) as mock_build_provider:
        result = runner.invoke(app, ["analyze", "Test text about a startup that solves X.", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Technical Due Diligence Report" in output_path.read_text()
    assert mock_build_provider.call_args.args == ("claude",)
    assert mock_build_provider.call_args.kwargs["tracker"] is not None


def test_analyze_command_selects_ollama_provider(tmp_path):
    output_path = tmp_path / "report.md"
    fake_provider = _fake_provider(PAYLOADS)

    with patch("dd_copilot.cli.build_provider", return_value=fake_provider) as mock_build_provider:
        result = runner.invoke(
            app,
            [
                "analyze",
                "Test text about a startup that solves X.",
                "--output",
                str(output_path),
                "--provider",
                "ollama",
            ],
        )

    assert result.exit_code == 0
    assert output_path.exists()
    # The tracker is threaded through so spend is measured, not estimated.
    assert mock_build_provider.call_args.args == ("ollama",)
    assert mock_build_provider.call_args.kwargs["tracker"] is not None


def test_confidential_mode_with_remote_provider_exits_without_analyzing(tmp_path):
    output_path = tmp_path / "report.md"

    with patch("dd_copilot.cli.build_provider") as mock_build_provider:
        result = runner.invoke(
            app,
            ["analyze", "Client material.", "--output", str(output_path), "--mode", "confidential"],
        )

    assert result.exit_code == 2
    assert not output_path.exists()
    # It must refuse *before* constructing a provider, not after sending anything.
    mock_build_provider.assert_not_called()


def test_analyze_command_appends_an_audit_record(tmp_path):
    output_path = tmp_path / "report.md"
    audit_path = tmp_path / "audit.jsonl"
    fake_provider = _fake_provider(PAYLOADS)

    with patch("dd_copilot.cli.build_provider", return_value=fake_provider):
        result = runner.invoke(
            app,
            ["analyze", "Test text about a startup that solves X.", "--output", str(output_path),
             "--audit-log", str(audit_path)],
        )

    assert result.exit_code == 0
    record = json.loads(audit_path.read_text().strip())
    assert record["mode"] == "public"
    assert len(record["report_sha256"]) == 64


def test_budget_cap_aborts_the_run_with_a_distinct_exit_code(tmp_path):
    """Exit code 3 distinguishes 'ran out of budget' from a confidential-mode
    refusal (2) or a normal failure — a caller scripting this needs to tell
    them apart."""
    from dd_copilot.costs import BudgetExceeded

    output_path = tmp_path / "report.md"

    with patch("dd_copilot.cli.build_provider"), \
         patch("dd_copilot.cli.analyze", side_effect=BudgetExceeded("spend reached $9.99")):
        result = runner.invoke(
            app,
            ["analyze", "Some startup text.", "--output", str(output_path),
             "--max-cost-usd", "0.50"],
        )

    assert result.exit_code == 3
    assert not output_path.exists()


def test_docx_flag_writes_a_word_document(tmp_path):
    import pytest

    pytest.importorskip("docx")
    output_path = tmp_path / "report.md"
    docx_path = tmp_path / "report.docx"
    fake_provider = _fake_provider(PAYLOADS)

    with patch("dd_copilot.cli.build_provider", return_value=fake_provider):
        result = runner.invoke(
            app,
            ["analyze", "Test text about a startup that solves X.",
             "--output", str(output_path), "--docx", str(docx_path)],
        )

    assert result.exit_code == 0, result.output
    assert docx_path.exists()


def test_without_the_docx_flag_no_word_document_is_written(tmp_path):
    output_path = tmp_path / "report.md"
    fake_provider = _fake_provider(PAYLOADS)

    with patch("dd_copilot.cli.build_provider", return_value=fake_provider):
        result = runner.invoke(
            app, ["analyze", "Test text.", "--output", str(output_path)]
        )

    assert result.exit_code == 0
    assert not list(tmp_path.glob("*.docx"))
