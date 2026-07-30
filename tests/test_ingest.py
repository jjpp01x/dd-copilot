import textwrap
from dd_copilot.ingest import ingest, ingest_text, Document

def test_ingest_text_returns_document_with_default_source_name():
    doc = ingest_text("Este es el texto pegado por el usuario.")
    assert isinstance(doc, Document)
    assert doc.source_name == "texto pegado"
    assert "texto pegado por el usuario" in doc.text

def test_ingest_dispatches_plain_text_when_not_url_or_file(tmp_path):
    doc = ingest("Isomorphic Labs combina IA y biología para acelerar el descubrimiento de fármacos.")
    assert "Isomorphic Labs" in doc.text
    assert doc.source_name == "texto pegado"

def test_ingest_dispatches_to_pdf_when_path_exists_and_ends_in_pdf(tmp_path, monkeypatch):
    fake_pdf = tmp_path / "whitepaper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    def fake_ingest_pdf(path):
        return Document(source_name="whitepaper.pdf", text="contenido extraído del PDF")

    monkeypatch.setattr("dd_copilot.ingest.ingest_pdf", fake_ingest_pdf)
    doc = ingest(str(fake_pdf))
    assert doc.source_name == "whitepaper.pdf"
    assert doc.text == "contenido extraído del PDF"

def test_ingest_dispatches_to_url_when_source_starts_with_http(monkeypatch):
    def fake_ingest_url(url):
        return Document(source_name=url, text="contenido de la web")

    monkeypatch.setattr("dd_copilot.ingest.ingest_url", fake_ingest_url)
    doc = ingest("https://isomorphiclabs.com")
    assert doc.source_name == "https://isomorphiclabs.com"
    assert doc.text == "contenido de la web"
