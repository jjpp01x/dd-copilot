from dd_copilot.ingest import ingest
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index
from dd_copilot.extract import run_extraction
from dd_copilot.report import render_report


def analyze(source: str, client) -> str:
    """Ejecuta ingest -> chunk -> index -> extract -> report y devuelve el Markdown final."""
    document = ingest(source)
    nodes = chunk_document(document)
    index = build_index(nodes)
    report_input = run_extraction(client, index, document.text, document.source_name)
    return render_report(report_input)
