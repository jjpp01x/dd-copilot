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
    mock_build_provider.assert_called_once_with("claude")


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
    mock_build_provider.assert_called_once_with("ollama")
