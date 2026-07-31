import pytest

from dd_copilot.docx_export import write_docx

docx = pytest.importorskip("docx")

REPORT = """# Technical Due Diligence Report — robotics-startup

## 1. Executive Summary

A controller for embedded robotics.

## 4. Claims Assessed

| Claim | Figure | Verdict | Why |
| --- | --- | --- | --- |
| Inference runs in 8 ms. | 8 ms | Verifiable | Benchmark stated. |
| Scales to 1,000 robots. | 1,000 | Plausible | No method stated. |

## 5. Questions for the Next Founder Call

- On "Scales to 1,000 robots.": under what conditions was this measured?
"""


def _read(path):
    document = docx.Document(str(path))
    return document, [p.text for p in document.paragraphs]


def test_writes_a_readable_docx(tmp_path):
    out = tmp_path / "report.docx"

    write_docx(REPORT, out)

    assert out.exists()
    _document, paragraphs = _read(out)
    assert any("Technical Due Diligence Report" in p for p in paragraphs)


def test_headings_become_word_headings_not_literal_hashes(tmp_path):
    """A client opening this in Word should see a navigable document, not a
    text file with '##' in it."""
    out = tmp_path / "report.docx"

    write_docx(REPORT, out)

    document, paragraphs = _read(out)
    assert not any(p.startswith("#") for p in paragraphs)
    styles = {p.style.name for p in document.paragraphs if p.text.strip()}
    assert any(s.startswith("Heading") or s == "Title" for s in styles)


def test_the_claims_table_becomes_a_real_table(tmp_path):
    """The claims table is the part of the report worth reading; rendering it
    as pipe-separated text would bury it."""
    out = tmp_path / "report.docx"

    write_docx(REPORT, out)

    document, _paragraphs = _read(out)
    assert len(document.tables) == 1
    table = document.tables[0]
    assert [c.text for c in table.rows[0].cells] == ["Claim", "Figure", "Verdict", "Why"]
    assert table.rows[1].cells[2].text == "Verifiable"
    # The markdown separator row must not survive as data.
    assert all("---" not in c.text for r in table.rows for c in r.cells)


def test_bullets_become_list_paragraphs(tmp_path):
    out = tmp_path / "report.docx"

    write_docx(REPORT, out)

    _document, paragraphs = _read(out)
    assert not any(p.startswith("- ") for p in paragraphs)
    assert any("under what conditions" in p for p in paragraphs)


def test_escaped_pipes_are_unescaped_in_table_cells(tmp_path):
    """report.py escapes pipes so the markdown table survives; Word needs the
    original text back."""
    out = tmp_path / "report.docx"

    write_docx(
        "## 4. Claims Assessed\n\n| Claim | Figure | Verdict | Why |\n"
        "| --- | --- | --- | --- |\n"
        "| 8 ms \\| measured on-device | 8 ms | Verifiable | Stated. |\n",
        out,
    )

    document, _paragraphs = _read(out)
    assert document.tables[0].rows[1].cells[0].text == "8 ms | measured on-device"
