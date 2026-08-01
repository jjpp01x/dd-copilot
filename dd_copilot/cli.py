import os
from pathlib import Path

import typer
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console

from dd_copilot.costs import BudgetExceeded, CostTracker
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


def build_provider(name: str, tracker: CostTracker | None = None) -> LLMProvider:
    if name == "ollama":
        # Local inference has no API charge, so there is nothing to track.
        return OllamaProvider()
    load_dotenv()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    return ClaudeProvider(Anthropic(api_key=api_key), tracker=tracker)


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
    docx: str = typer.Option(
        None,
        "--docx",
        help="Also write the report as a Word document — the format a client receives.",
    ),
    json_out: str = typer.Option(
        None,
        "--json",
        help="También escribe los datos estructurados en JSON — es lo que "
        "consume expert-probe. El Markdown es un derivado de este fichero.",
    ),
    max_cost_usd: float = typer.Option(
        None,
        "--max-cost-usd",
        help="Abort the run once spend crosses this ceiling. Omit for no cap.",
    ),
) -> None:
    """Analyzes `source` and generates a technical due-diligence report in Markdown."""
    try:
        assert_provider_allowed(mode, provider)
    except ConfidentialModeViolation as exc:
        console.print(f"[bold red]Refusing to run:[/bold red] {exc}")
        raise typer.Exit(code=2)

    if mode is Mode.CONFIDENTIAL:
        console.print("[bold yellow]CONFIDENTIAL MODE[/bold yellow] — analysis stays on this machine.")

    tracker = CostTracker(max_usd=max_cost_usd)
    llm_provider = build_provider(provider, tracker=tracker)
    console.print(f"[bold]Analyzing:[/bold] {source[:80]}...")
    try:
        markdown, report_input = analyze(source, llm_provider)
    except BudgetExceeded as exc:
        console.print(f"[bold red]Stopped:[/bold red] {exc}")
        raise typer.Exit(code=3)
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
    if json_out:
        Path(json_out).write_text(
            report_input.model_dump_json(indent=2), encoding="utf-8"
        )
        console.print(f"[bold green]JSON:[/bold green] {json_out}")
    if docx:
        try:
            from dd_copilot.docx_export import write_docx
        except ImportError:
            console.print(
                "[bold red]--docx needs python-docx:[/bold red] pip install -e \".[docx]\""
            )
            raise typer.Exit(code=4)
        write_docx(markdown, docx)
        console.print(f"[bold green]Word document:[/bold green] {docx}")
    if tracker.calls:
        console.print(
            f"Cost: [bold]${tracker.total_usd:.4f}[/bold] over {tracker.calls} model calls "
            f"(measured, not estimated)."
        )


# Typer collapses a Typer() app with a single @app.command() into a bare
# root command (no subcommand name required/accepted). Registering the same
# function again under an explicit name keeps `ddcopilot analyze <source>`
# working as a named subcommand instead of `ddcopilot <source>`.
app.command(name="analyze")(analyze_command)
