import os
from dataclasses import dataclass

import trafilatura
from pypdf import PdfReader


@dataclass
class Document:
    source_name: str
    text: str


def ingest_text(text: str, source_name: str = "texto pegado") -> Document:
    return Document(source_name=source_name, text=text)


def ingest_pdf(path: str) -> Document:
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    source_name = os.path.basename(path)
    return Document(source_name=source_name, text=text)


def ingest_url(url: str) -> Document:
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) or ""
    return Document(source_name=url, text=text)


def ingest(source: str) -> Document:
    """Detects whether `source` is a URL, a local file path (PDF or text), or raw text, and dispatches accordingly."""
    if source.startswith("http://") or source.startswith("https://"):
        return ingest_url(source)
    if os.path.exists(source):
        if source.lower().endswith(".pdf"):
            return ingest_pdf(source)
        with open(source, encoding="utf-8") as f:
            return ingest_text(f.read(), source_name=os.path.basename(source))
    return ingest_text(source)
