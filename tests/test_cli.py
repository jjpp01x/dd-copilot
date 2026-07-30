import json
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from dd_copilot.cli import app

runner = CliRunner()


def _response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def test_analyze_command_writes_markdown_file(tmp_path):
    output_path = tmp_path / "report.md"
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _response({"value": "Solves X.", "citation": "Solves X.", "mentioned": True}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"confidence_score": 2, "confidence_justification": "Little material."}),
    ]

    with patch("dd_copilot.cli.build_anthropic_client", return_value=fake_client):
        result = runner.invoke(app, ["analyze", "Test text about a startup that solves X.", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Technical Due Diligence Report" in output_path.read_text()
