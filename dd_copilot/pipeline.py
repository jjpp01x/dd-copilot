from dd_copilot.ingest import ingest
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index
from dd_copilot.extract import run_extraction
from dd_copilot.models import ReportInput
from dd_copilot.providers import LLMProvider
from dd_copilot.report import render_report


def analyze(source: str, provider: LLMProvider) -> tuple[str, ReportInput]:
    """Runs ingest -> chunk -> index -> extract -> report.

    Returns the rendered Markdown and the structured input it was rendered
    from. Downstream tools need the structure — the citations with their
    source_chunk_id never survive the Markdown rendering — so discarding it
    here forced consumers to parse the presentation layer.
    """
    document = ingest(source)
    nodes = chunk_document(document)
    index = build_index(nodes)
    report_input = run_extraction(provider, index, document.text, document.source_name)
    return render_report(report_input), report_input
