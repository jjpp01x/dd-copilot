import os

import typer
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console

from dd_copilot.pipeline import analyze

app = typer.Typer()
console = Console()


def build_anthropic_client() -> Anthropic:
    load_dotenv()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    return Anthropic(api_key=api_key)


@app.command()
def analyze_command(
    source: str = typer.Argument(..., help="URL, path to a PDF, or raw pasted text."),
    output: str = typer.Option("report.md", "--output", "-o", help="Path to the output Markdown file."),
) -> None:
    """Analyzes `source` and generates a technical due-diligence report in Markdown."""
    client = build_anthropic_client()
    console.print(f"[bold]Analyzing:[/bold] {source[:80]}...")
    markdown = analyze(source, client)
    with open(output, "w", encoding="utf-8") as f:
        f.write(markdown)
    console.print(f"[bold green]Report generated:[/bold green] {output}")


# Typer collapses a Typer() app with a single @app.command() into a bare
# root command (no subcommand name required/accepted). Registering the same
# function again under an explicit name keeps `ddcopilot analyze <source>`
# working as a named subcommand instead of `ddcopilot <source>`.
app.command(name="analyze")(analyze_command)
