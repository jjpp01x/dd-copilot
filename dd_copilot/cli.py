import os

import typer
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console

from dd_copilot.confidentiality import (
    ConfidentialModeViolation,
    Mode,
    assert_provider_allowed,
    write_audit_record,
)
from dd_copilot.pipeline import analyze
from dd_copilot.providers import ClaudeProvider, LLMProvider, OllamaProvider

app = typer.Typer()
console = Console()


def build_provider(name: str) -> LLMProvider:
    if name == "ollama":
        return OllamaProvider()
    load_dotenv()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    return ClaudeProvider(Anthropic(api_key=api_key))


@app.command()
def analyze_command(
    source: str = typer.Argument(..., help="URL, path to a PDF, or raw pasted text."),
    output: str = typer.Option("report.md", "--output", "-o", help="Path to the output Markdown file."),
    provider: str = typer.Option("claude", "--provider", help="LLM provider to use: claude or ollama."),
    mode: Mode = typer.Option(
        Mode.PUBLIC,
        "--mode",
        help="public: published material, remote models allowed. "
        "confidential: client material, local models only.",
    ),
    audit_log: str = typer.Option("audit.jsonl", "--audit-log", help="Path to the JSONL audit trail."),
) -> None:
    """Analyzes `source` and generates a technical due-diligence report in Markdown."""
    try:
        assert_provider_allowed(mode, provider)
    except ConfidentialModeViolation as exc:
        console.print(f"[bold red]Refusing to run:[/bold red] {exc}")
        raise typer.Exit(code=2)

    if mode is Mode.CONFIDENTIAL:
        console.print("[bold yellow]CONFIDENTIAL MODE[/bold yellow] — analysis stays on this machine.")

    llm_provider = build_provider(provider)
    console.print(f"[bold]Analyzing:[/bold] {source[:80]}...")
    markdown = analyze(source, llm_provider)
    with open(output, "w", encoding="utf-8") as f:
        f.write(markdown)
    write_audit_record(
        audit_log,
        source_name=source[:80],
        mode=mode,
        provider=provider,
        report_markdown=markdown,
    )
    console.print(f"[bold green]Report generated:[/bold green] {output}")


# Typer collapses a Typer() app with a single @app.command() into a bare
# root command (no subcommand name required/accepted). Registering the same
# function again under an explicit name keeps `ddcopilot analyze <source>`
# working as a named subcommand instead of `ddcopilot <source>`.
app.command(name="analyze")(analyze_command)
