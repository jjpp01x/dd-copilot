# DD-Copilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DD-Copilot, a CLI + Streamlit tool that turns public deep-tech
startup material (URL/PDF/text) into a structured, citation-verified
technical due-diligence report using LlamaIndex + Claude.

**Architecture:** A linear pipeline (`ingest → chunk → index → extract →
citation_check → report`) implemented as small, independently testable
modules under `dd_copilot/`, orchestrated by a single `pipeline.py` function
that both the CLI and the Streamlit app call — no duplicated logic between
the two interfaces.

**Tech Stack:** Python 3.11, LlamaIndex (`llama-index-core`,
`llama-index-llms-anthropic`, `llama-index-embeddings-huggingface`),
`anthropic` SDK, `sentence-transformers`, `pypdf`, `trafilatura`, `typer`,
`rich`, `streamlit`, `pydantic`, `rapidfuzz`, `pytest`, `pytest-mock`.

## Global Constraints

- Solo Claude vía el SDK oficial de Anthropic — ningún otro proveedor de LLM.
- RAG con LlamaIndex: `SentenceSplitter` para chunking semántico + `VectorStoreIndex` en memoria (sin base vectorial externa).
- Embeddings locales: `HuggingFaceEmbedding` con modelo `sentence-transformers/all-MiniLM-L6-v2` (coste cero de API en indexado).
- Cascada de modelos: **Claude Haiku** (`claude-haiku-4-5-20251001`) para clasificación/extracción por chunk; **Claude Sonnet** (`claude-sonnet-5`) solo para la síntesis final del informe.
- Prompt caching de Anthropic para el system prompt fijo, reutilizado en todas las llamadas de un mismo análisis.
- Cada afirmación del informe debe llevar una cita verificada contra el texto fuente (fuzzy match); si no hay evidencia verificable, el campo se marca `mentioned=False` y el informe dice explícitamente "no mencionado en la fuente" — nunca se inventa contenido.
- Reintentos con backoff exponencial (máx. 3 intentos) en cada llamada a la API de Claude.
- Los tests deben mockear la llamada a Claude — cero gasto de tokens reales en la suite de tests.
- Repo en `~/Projects/dd-copilot`, publicado en GitHub público como `jjpp01x/dd-copilot`.
- Entregables de documentación: `README.md` técnico (explica el "por qué" de cada decisión de arquitectura) y `GUIA-DE-USO.md` en español, sin jerga sin definir, para que un principiante en IA entienda, instale y pruebe la herramienta.
- Demo: startup **Isomorphic Labs**, informe guardado en `examples/isomorphic-labs/informe.md`.
- Fuera de alcance (no implementar): comparación entre startups, scoring ponderado entre startups, soporte multi-proveedor de LLM.

---

## File Structure

```
dd-copilot/
├── dd_copilot/
│   ├── __init__.py
│   ├── models.py          # Esquemas Pydantic compartidos
│   ├── ingest.py          # URL/PDF/texto -> Document
│   ├── chunking.py        # Document -> list[TextNode] (chunking semántico)
│   ├── index.py           # list[TextNode] -> VectorStoreIndex + retrieval
│   ├── citation_check.py  # Validación fuzzy de citas contra el texto fuente
│   ├── extract.py         # Cascada Haiku/Sonnet -> ReportInput
│   ├── report.py          # ReportInput -> Markdown
│   ├── pipeline.py        # Orquesta ingest->chunk->index->extract->report
│   └── cli.py             # Typer CLI
├── app.py                 # Streamlit, una sola página, usa pipeline.py
├── tests/
│   ├── test_models.py
│   ├── test_ingest.py
│   ├── test_chunking.py
│   ├── test_citation_check.py
│   ├── test_index.py
│   ├── test_extract.py
│   ├── test_report.py
│   └── test_pipeline.py
├── examples/
│   └── isomorphic-labs/
│       ├── fuente.txt
│       └── informe.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── GUIA-DE-USO.md
```

---

### Task 1: Scaffolding del proyecto

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `dd_copilot/__init__.py`
- Test: `tests/test_models.py` (placeholder de humo, se completa en Task 2)

**Interfaces:**
- Produces: paquete Python instalable `dd_copilot`, entorno con dependencias listas.

- [ ] **Step 1: Crear `pyproject.toml`**

```toml
[project]
name = "dd-copilot"
version = "0.1.0"
description = "Copiloto de due diligence técnica para startups deep-tech"
requires-python = ">=3.11"
dependencies = [
    "llama-index-core>=0.11.0",
    "llama-index-llms-anthropic>=0.3.0",
    "llama-index-embeddings-huggingface>=0.3.0",
    "sentence-transformers>=3.0.0",
    "anthropic>=0.34.0",
    "pypdf>=4.0.0",
    "trafilatura>=1.9.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "streamlit>=1.37.0",
    "pydantic>=2.7.0",
    "rapidfuzz>=3.9.0",
    "python-dotenv>=1.0.0",
    "tenacity>=8.5.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-mock>=3.14.0"]

[project.scripts]
ddcopilot = "dd_copilot.cli:app"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["dd_copilot*"]
```

- [ ] **Step 2: Crear `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
*.egg-info/
.pytest_cache/
```

- [ ] **Step 3: Crear `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 4: Crear `dd_copilot/__init__.py`**

```python
```
(fichero vacío — marca el directorio como paquete)

- [ ] **Step 5: Instalar el proyecto en modo editable y verificar**

Run: `cd ~/Projects/dd-copilot && python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: instalación completa sin errores.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example dd_copilot/__init__.py
git commit -m "chore: scaffolding inicial del proyecto"
```

---

### Task 2: `models.py` — esquemas Pydantic compartidos

**Files:**
- Create: `dd_copilot/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Citation`, `ChecklistField`, `RiskChecklistItem`, `ExtractionResult`, `ReportInput` — usados por `extract.py`, `citation_check.py` y `report.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_models.py
from dd_copilot.models import Citation, ChecklistField, RiskChecklistItem, ExtractionResult, ReportInput

def test_checklist_field_defaults_to_not_mentioned():
    field = ChecklistField(value="", citations=[], mentioned=False)
    assert field.mentioned is False
    assert field.citations == []

def test_extraction_result_holds_all_checklist_fields():
    problema = ChecklistField(value="Resuelve X", citations=[Citation(text="cita literal", source_chunk_id="chunk-1")], mentioned=True)
    diferenciacion = ChecklistField(value="", citations=[], mentioned=False)
    rendimiento = ChecklistField(value="", citations=[], mentioned=False)
    riesgo = RiskChecklistItem(risk_name="madurez_trl", mentioned=False)
    result = ExtractionResult(problema=problema, diferenciacion=diferenciacion, rendimiento=rendimiento, riesgos=[riesgo])
    assert result.problema.value == "Resuelve X"
    assert result.riesgos[0].risk_name == "madurez_trl"

def test_report_input_confidence_score_range():
    import pytest
    from pydantic import ValidationError
    problema = ChecklistField(value="x", citations=[], mentioned=True)
    result = ExtractionResult(problema=problema, diferenciacion=problema, rendimiento=problema, riesgos=[])
    with pytest.raises(ValidationError):
        ReportInput(source_name="demo", extraction=result, confidence_score=6, confidence_justification="fuera de rango")
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_models.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.models'`

- [ ] **Step 3: Implementar `dd_copilot/models.py`**

```python
from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    text: str
    source_chunk_id: str


class ChecklistField(BaseModel):
    value: str
    citations: list[Citation] = Field(default_factory=list)
    mentioned: bool


RiskName = Literal[
    "madurez_trl",
    "dependencia_hardware",
    "reproducibilidad",
    "riesgo_regulatorio",
]


class RiskChecklistItem(BaseModel):
    risk_name: RiskName
    mentioned: bool
    detail: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    problema: ChecklistField
    diferenciacion: ChecklistField
    rendimiento: ChecklistField
    riesgos: list[RiskChecklistItem]


class ReportInput(BaseModel):
    source_name: str
    extraction: ExtractionResult
    confidence_score: int = Field(ge=1, le=5)
    confidence_justification: str
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/models.py tests/test_models.py
git commit -m "feat: esquemas Pydantic de extracción e informe"
```

---

### Task 3: `ingest.py` — normalización de fuentes

**Files:**
- Create: `dd_copilot/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: nada (módulo de entrada).
- Produces: `Document(source_name: str, text: str)` (dataclass), función `ingest(source: str) -> Document` usada por `pipeline.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_ingest.py
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
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.ingest'`

- [ ] **Step 3: Implementar `dd_copilot/ingest.py`**

```python
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
    """Detecta si `source` es una URL, una ruta a PDF, o texto en bruto, y despacha."""
    if source.startswith("http://") or source.startswith("https://"):
        return ingest_url(source)
    if source.lower().endswith(".pdf") and os.path.exists(source):
        return ingest_pdf(source)
    return ingest_text(source)
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_ingest.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/ingest.py tests/test_ingest.py
git commit -m "feat: ingesta de URL, PDF y texto"
```

---

### Task 4: `chunking.py` — chunking semántico con LlamaIndex

**Files:**
- Create: `dd_copilot/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Document` de `ingest.py`.
- Produces: `chunk_document(document: Document, chunk_size: int = 512, chunk_overlap: int = 50) -> list[TextNode]` (LlamaIndex `TextNode`, con `.node_id` y `.get_content()`) — usado por `index.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_chunking.py
from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document

def test_chunk_document_produces_nonempty_nodes_with_overlap():
    long_text = " ".join([f"Frase número {i} sobre la tecnología de la startup." for i in range(200)])
    doc = Document(source_name="demo", text=long_text)
    nodes = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert len(nodes) > 1
    for node in nodes:
        assert node.get_content().strip() != ""
        assert node.node_id

def test_chunk_document_single_short_text_produces_one_node():
    doc = Document(source_name="demo", text="Texto corto.")
    nodes = chunk_document(doc)
    assert len(nodes) == 1
    assert "Texto corto" in nodes[0].get_content()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.chunking'`

- [ ] **Step 3: Implementar `dd_copilot/chunking.py`**

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode

from dd_copilot.ingest import Document


def chunk_document(document: Document, chunk_size: int = 512, chunk_overlap: int = 50) -> list[TextNode]:
    """Trocea el documento en chunks semánticos (por oraciones, con solape)."""
    llama_doc = LlamaDocument(text=document.text, metadata={"source_name": document.source_name})
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents([llama_doc])
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_chunking.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/chunking.py tests/test_chunking.py
git commit -m "feat: chunking semantico con SentenceSplitter"
```

---

### Task 5: `citation_check.py` — validador anti-alucinación

**Files:**
- Create: `dd_copilot/citation_check.py`
- Test: `tests/test_citation_check.py`

**Interfaces:**
- Consumes: nada (función pura de texto).
- Produces: `verify_citation(citation_text: str, source_text: str, threshold: int = 90) -> bool` — usado por `extract.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_citation_check.py
from dd_copilot.citation_check import verify_citation

SOURCE = "Isomorphic Labs combina inteligencia artificial y biología para acelerar el descubrimiento de fármacos."

def test_exact_citation_passes():
    assert verify_citation("acelerar el descubrimiento de fármacos", SOURCE) is True

def test_slightly_altered_citation_still_passes_above_threshold():
    assert verify_citation("acelerar el descubrimiento de farmacos", SOURCE) is True

def test_fabricated_citation_fails():
    assert verify_citation("cura el cáncer en tres días", SOURCE) is False

def test_empty_citation_fails():
    assert verify_citation("", SOURCE) is False
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_citation_check.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.citation_check'`

- [ ] **Step 3: Implementar `dd_copilot/citation_check.py`**

```python
from rapidfuzz import fuzz


def verify_citation(citation_text: str, source_text: str, threshold: int = 90) -> bool:
    """Comprueba que `citation_text` aparece literalmente (o casi) en `source_text`.

    Usa partial_ratio para tolerar pequeñas variaciones (mayúsculas, tildes,
    espacios) sin permitir que se cuele contenido inventado.
    """
    citation_text = citation_text.strip()
    if not citation_text:
        return False
    score = fuzz.partial_ratio(citation_text.lower(), source_text.lower())
    return score >= threshold
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_citation_check.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/citation_check.py tests/test_citation_check.py
git commit -m "feat: validador de citas anti-alucinacion"
```

---

### Task 6: `index.py` — embeddings locales + retrieval

**Files:**
- Create: `dd_copilot/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `list[TextNode]` de `chunking.py`.
- Produces: `build_index(nodes: list[TextNode]) -> VectorStoreIndex`, `retrieve_relevant_chunks(index: VectorStoreIndex, query: str, top_k: int = 5) -> list[TextNode]` — usados por `extract.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_index.py
from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index, retrieve_relevant_chunks

def test_retrieve_relevant_chunks_returns_most_similar_node():
    doc = Document(
        source_name="demo",
        text=(
            "Isomorphic Labs usa modelos de deep learning para predecir la estructura de proteínas. "
            "El equipo de marketing organiza eventos anuales en Londres para inversores. "
            "La compañía fue fundada como spin-off de DeepMind en 2021."
        ),
    )
    nodes = chunk_document(doc, chunk_size=40, chunk_overlap=5)
    index = build_index(nodes)
    results = retrieve_relevant_chunks(index, "¿Qué tecnología de IA usa la empresa?", top_k=1)
    assert len(results) == 1
    assert "deep learning" in results[0].get_content() or "proteínas" in results[0].get_content()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_index.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.index'`

- [ ] **Step 3: Implementar `dd_copilot/index.py`**

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    """Construye un índice vectorial en memoria con embeddings locales (coste cero de API)."""
    return VectorStoreIndex(nodes, embed_model=_embed_model)


def retrieve_relevant_chunks(index: VectorStoreIndex, query: str, top_k: int = 5) -> list[TextNode]:
    """Devuelve los `top_k` chunks más relevantes para `query`, sin llamar al LLM."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [result.node for result in results]
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_index.py -v`
Expected: 1 passed (puede tardar unos segundos por la descarga del modelo de embeddings la primera vez)

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/index.py tests/test_index.py
git commit -m "feat: indexado vectorial con embeddings locales"
```

---

### Task 7: `extract.py` — cascada Haiku/Sonnet con Claude

**Files:**
- Create: `dd_copilot/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `VectorStoreIndex` de `index.py`, `retrieve_relevant_chunks`, `verify_citation` de `citation_check.py`, `ChecklistField`/`RiskChecklistItem`/`ExtractionResult`/`ReportInput`/`Citation` de `models.py`.
- Produces: `run_extraction(client: anthropic.Anthropic, index: VectorStoreIndex, source_text: str, source_name: str) -> ReportInput` — usado por `pipeline.py`.

- [ ] **Step 1: Escribir el test que falla (con Claude mockeado)**

```python
# tests/test_extract.py
import json
from unittest.mock import MagicMock

from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index
from dd_copilot.extract import run_extraction

SOURCE_TEXT = (
    "Isomorphic Labs combina inteligencia artificial y biología para acelerar "
    "el descubrimiento de fármacos. La compañía es un spin-off de DeepMind."
)


def _fake_haiku_response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def _fake_sonnet_response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def test_run_extraction_marks_field_as_not_mentioned_when_citation_is_fabricated(monkeypatch):
    doc = Document(source_name="isomorphic-labs", text=SOURCE_TEXT)
    nodes = chunk_document(doc, chunk_size=60, chunk_overlap=10)
    index = build_index(nodes)

    fake_client = MagicMock()

    haiku_payload = {
        "value": "Resuelve el descubrimiento de fármacos.",
        "citation": "cura enfermedades raras en 24 horas",
        "mentioned": True,
    }
    sonnet_payload = {
        "confidence_score": 3,
        "confidence_justification": "El material público es limitado.",
    }

    fake_client.messages.create.side_effect = [
        _fake_haiku_response(haiku_payload),  # problema
        _fake_haiku_response({"value": "", "citation": "", "mentioned": False}),  # diferenciacion
        _fake_haiku_response({"value": "", "citation": "", "mentioned": False}),  # rendimiento
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # riesgo 1
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # riesgo 2
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # riesgo 3
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # riesgo 4
        _fake_sonnet_response(sonnet_payload),  # sintesis final
    ]

    result = run_extraction(fake_client, index, SOURCE_TEXT, "isomorphic-labs")

    assert result.extraction.problema.mentioned is False
    assert result.extraction.problema.citations == []
    assert result.confidence_score == 3
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.extract'`

- [ ] **Step 3: Implementar `dd_copilot/extract.py`**

```python
import json

from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.core import VectorStoreIndex

from dd_copilot.citation_check import verify_citation
from dd_copilot.index import retrieve_relevant_chunks
from dd_copilot.models import (
    Citation,
    ChecklistField,
    RiskChecklistItem,
    RiskName,
    ExtractionResult,
    ReportInput,
)

CLASSIFY_MODEL = "claude-haiku-4-5-20251001"
SYNTHESIS_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "Eres un analista técnico de due diligence para inversión en deep-tech. "
    "Solo puedes afirmar lo que está literalmente en el texto que se te da. "
    "Si algo no está explícito, responde mentioned=false y citation vacía. "
    "Responde siempre en JSON válido, sin texto adicional."
)

FIELD_QUESTIONS = {
    "problema": "¿Qué problema resuelve la tecnología de esta startup?",
    "diferenciacion": "¿Qué diferencia técnicamente a esta tecnología de sus alternativas?",
    "rendimiento": "¿Qué afirmaciones de rendimiento o escalabilidad hace la startup?",
}

RISK_QUESTIONS: dict[RiskName, str] = {
    "madurez_trl": "¿Se menciona el nivel de madurez tecnológica (TRL) de la tecnología?",
    "dependencia_hardware": "¿Se menciona dependencia de hardware o proveedores específicos?",
    "reproducibilidad": "¿Se menciona si los resultados son reproducibles o han sido validados externamente?",
    "riesgo_regulatorio": "¿Se menciona algún riesgo regulatorio aplicable a esta tecnología?",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_claude(client, model: str, user_prompt: str) -> dict:
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )
    return json.loads(message.content[0].text)


def _build_field_from_response(payload: dict, source_text: str) -> ChecklistField:
    citation_text = payload.get("citation", "") or ""
    mentioned = bool(payload.get("mentioned")) and verify_citation(citation_text, source_text)
    if not mentioned:
        return ChecklistField(value="", citations=[], mentioned=False)
    return ChecklistField(
        value=payload.get("value", ""),
        citations=[Citation(text=citation_text, source_chunk_id="retrieved")],
        mentioned=True,
    )


def extract_field(client, field_name: str, question: str, index: VectorStoreIndex, source_text: str) -> ChecklistField:
    nodes = retrieve_relevant_chunks(index, question, top_k=5)
    context = "\n\n".join(node.get_content() for node in nodes)
    prompt = (
        f"Pregunta: {question}\n\nTexto fuente (fragmentos relevantes):\n{context}\n\n"
        'Responde en JSON: {"value": str, "citation": str, "mentioned": bool}'
    )
    payload = _call_claude(client, CLASSIFY_MODEL, prompt)
    return _build_field_from_response(payload, source_text)


def extract_risks(client, index: VectorStoreIndex, source_text: str) -> list[RiskChecklistItem]:
    risks = []
    for risk_name, question in RISK_QUESTIONS.items():
        nodes = retrieve_relevant_chunks(index, question, top_k=3)
        context = "\n\n".join(node.get_content() for node in nodes)
        prompt = (
            f"Pregunta: {question}\n\nTexto fuente (fragmentos relevantes):\n{context}\n\n"
            'Responde en JSON: {"mentioned": bool, "detail": str o null, "citation": str}'
        )
        payload = _call_claude(client, CLASSIFY_MODEL, prompt)
        citation_text = payload.get("citation", "") or ""
        mentioned = bool(payload.get("mentioned")) and verify_citation(citation_text, source_text)
        risks.append(
            RiskChecklistItem(
                risk_name=risk_name,
                mentioned=mentioned,
                detail=payload.get("detail") if mentioned else None,
                citations=[Citation(text=citation_text, source_chunk_id="retrieved")] if mentioned else [],
            )
        )
    return risks


def synthesize_confidence(client, extraction: ExtractionResult) -> tuple[int, str]:
    summary = extraction.model_dump_json()
    prompt = (
        f"Datos extraídos ya estructurados (no texto en bruto):\n{summary}\n\n"
        'Da un nivel de confianza del análisis (1-5) y su justificación. '
        'Responde en JSON: {"confidence_score": int, "confidence_justification": str}'
    )
    payload = _call_claude(client, SYNTHESIS_MODEL, prompt)
    return payload["confidence_score"], payload["confidence_justification"]


def run_extraction(client, index: VectorStoreIndex, source_text: str, source_name: str) -> ReportInput:
    """Ejecuta la cascada Haiku (por campo) -> Sonnet (síntesis final)."""
    problema = extract_field(client, "problema", FIELD_QUESTIONS["problema"], index, source_text)
    diferenciacion = extract_field(client, "diferenciacion", FIELD_QUESTIONS["diferenciacion"], index, source_text)
    rendimiento = extract_field(client, "rendimiento", FIELD_QUESTIONS["rendimiento"], index, source_text)
    riesgos = extract_risks(client, index, source_text)

    extraction = ExtractionResult(
        problema=problema,
        diferenciacion=diferenciacion,
        rendimiento=rendimiento,
        riesgos=riesgos,
    )

    confidence_score, confidence_justification = synthesize_confidence(client, extraction)

    return ReportInput(
        source_name=source_name,
        extraction=extraction,
        confidence_score=confidence_score,
        confidence_justification=confidence_justification,
    )
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_extract.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/extract.py tests/test_extract.py
git commit -m "feat: cascada Haiku/Sonnet con validacion de citas"
```

---

### Task 8: `report.py` — ensamblado del informe Markdown

**Files:**
- Create: `dd_copilot/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `ReportInput` de `models.py`.
- Produces: `render_report(report_input: ReportInput) -> str` — usado por `pipeline.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_report.py
from dd_copilot.models import Citation, ChecklistField, RiskChecklistItem, ExtractionResult, ReportInput
from dd_copilot.report import render_report

def test_render_report_includes_all_five_fixed_sections():
    problema = ChecklistField(value="Acelera el descubrimiento de fármacos.", citations=[Citation(text="acelerar el descubrimiento de fármacos", source_chunk_id="c1")], mentioned=True)
    vacio = ChecklistField(value="", citations=[], mentioned=False)
    riesgo_no_mencionado = RiskChecklistItem(risk_name="madurez_trl", mentioned=False)
    extraction = ExtractionResult(problema=problema, diferenciacion=vacio, rendimiento=vacio, riesgos=[riesgo_no_mencionado])
    report_input = ReportInput(source_name="isomorphic-labs", extraction=extraction, confidence_score=3, confidence_justification="Material público limitado.")

    markdown = render_report(report_input)

    assert "# Informe de Due Diligence Técnica — isomorphic-labs" in markdown
    assert "## 1. Resumen ejecutivo" in markdown
    assert "## 2. Qué dice la startup" in markdown
    assert "## 3. Qué no dice" in markdown
    assert "## 4. Preguntas para la siguiente llamada con el fundador" in markdown
    assert "## 5. Nivel de confianza del análisis" in markdown
    assert "acelerar el descubrimiento de fármacos" in markdown
    assert "madurez_trl" in markdown
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_report.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.report'`

- [ ] **Step 3: Implementar `dd_copilot/report.py`**

```python
from dd_copilot.models import ReportInput, ChecklistField, RiskChecklistItem

FIELD_LABELS = {
    "problema": "Problema que resuelve",
    "diferenciacion": "Diferenciación técnica",
    "rendimiento": "Afirmaciones de rendimiento/escalabilidad",
}

RISK_LABELS = {
    "madurez_trl": "Madurez tecnológica (TRL)",
    "dependencia_hardware": "Dependencia de hardware/proveedor",
    "reproducibilidad": "Reproducibilidad de resultados",
    "riesgo_regulatorio": "Riesgo regulatorio",
}


def _render_field(label: str, field: ChecklistField) -> str:
    if not field.mentioned:
        return f"- **{label}:** no mencionado en la fuente."
    citas = "; ".join(f'"{c.text}"' for c in field.citations)
    return f"- **{label}:** {field.value} (cita: {citas})"


def _render_risk(risk: RiskChecklistItem) -> str:
    label = RISK_LABELS[risk.risk_name]
    if not risk.mentioned:
        return f"- **{risk.risk_name}** ({label}): no mencionado — pregunta pendiente para el fundador."
    return f"- **{risk.risk_name}** ({label}): {risk.detail}"


def render_report(report_input: ReportInput) -> str:
    extraction = report_input.extraction

    dice_lines = [
        _render_field(FIELD_LABELS["problema"], extraction.problema),
        _render_field(FIELD_LABELS["diferenciacion"], extraction.diferenciacion),
        _render_field(FIELD_LABELS["rendimiento"], extraction.rendimiento),
    ]

    no_dice_lines = [_render_risk(r) for r in extraction.riesgos if not r.mentioned]
    if not no_dice_lines:
        no_dice_lines = ["- Todos los riesgos del checklist están cubiertos por la fuente."]

    preguntas_lines = [
        f"- Sobre {RISK_LABELS[r.risk_name].lower()}: no está documentado, ¿puede el equipo aclararlo?"
        for r in extraction.riesgos
        if not r.mentioned
    ]
    if not preguntas_lines:
        preguntas_lines = ["- Sin preguntas pendientes del checklist fijo; profundizar en detalles cuantitativos de rendimiento."]

    return "\n\n".join(
        [
            f"# Informe de Due Diligence Técnica — {report_input.source_name}",
            "## 1. Resumen ejecutivo\n\n" + (extraction.problema.value or "No hay suficiente información pública para un resumen ejecutivo."),
            "## 2. Qué dice la startup\n\n" + "\n".join(dice_lines),
            "## 3. Qué no dice\n\n" + "\n".join(no_dice_lines),
            "## 4. Preguntas para la siguiente llamada con el fundador\n\n" + "\n".join(preguntas_lines),
            "## 5. Nivel de confianza del análisis\n\n"
            f"**{report_input.confidence_score}/5** — {report_input.confidence_justification}",
        ]
    )
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_report.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/report.py tests/test_report.py
git commit -m "feat: ensamblado del informe markdown"
```

---

### Task 9: `pipeline.py` — orquestación end-to-end

**Files:**
- Create: `dd_copilot/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `ingest`, `chunk_document`, `build_index`, `run_extraction`, `render_report`.
- Produces: `analyze(source: str, client) -> str` (Markdown final) — usado por `cli.py` y `app.py`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_pipeline.py
import json
from unittest.mock import MagicMock

from dd_copilot.pipeline import analyze

SOURCE_TEXT = (
    "Isomorphic Labs combina inteligencia artificial y biología para acelerar "
    "el descubrimiento de fármacos. La compañía es un spin-off de DeepMind fundado en 2021."
)


def _response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def test_analyze_returns_markdown_with_fixed_sections():
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _response({"value": "Acelera el descubrimiento de fármacos.", "citation": "acelerar el descubrimiento de fármacos", "mentioned": True}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"confidence_score": 3, "confidence_justification": "Material público limitado."}),
    ]

    markdown = analyze(SOURCE_TEXT, fake_client)

    assert "# Informe de Due Diligence Técnica" in markdown
    assert "acelerar el descubrimiento de fármacos" in markdown
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.pipeline'`

- [ ] **Step 3: Implementar `dd_copilot/pipeline.py`**

```python
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
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_pipeline.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline end-to-end"
```

---

### Task 10: `cli.py` — interfaz de línea de comandos

**Files:**
- Create: `dd_copilot/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `analyze` de `pipeline.py`.
- Produces: comando `ddcopilot analyze <source>` (Typer app importable como `dd_copilot.cli:app`).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_cli.py
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
    output_path = tmp_path / "informe.md"
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _response({"value": "Resuelve X.", "citation": "Resuelve X.", "mentioned": True}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"confidence_score": 2, "confidence_justification": "Poco material."}),
    ]

    with patch("dd_copilot.cli.build_anthropic_client", return_value=fake_client):
        result = runner.invoke(app, ["analyze", "Texto de prueba sobre una startup que resuelve X.", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Informe de Due Diligence Técnica" in output_path.read_text()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.cli'`

- [ ] **Step 3: Implementar `dd_copilot/cli.py`**

```python
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
    source: str = typer.Argument(..., help="URL, ruta a PDF, o texto pegado directamente."),
    output: str = typer.Option("informe.md", "--output", "-o", help="Ruta del fichero Markdown de salida."),
) -> None:
    """Analiza `source` y genera un informe de due diligence técnica en Markdown."""
    client = build_anthropic_client()
    console.print(f"[bold]Analizando:[/bold] {source[:80]}...")
    markdown = analyze(source, client)
    with open(output, "w", encoding="utf-8") as f:
        f.write(markdown)
    console.print(f"[bold green]Informe generado:[/bold green] {output}")


app.command(name="analyze")(analyze_command)
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/test_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add dd_copilot/cli.py tests/test_cli.py
git commit -m "feat: CLI ddcopilot analyze"
```

---

### Task 11: `app.py` — visor Streamlit de una página

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `analyze` de `dd_copilot.pipeline`, `build_anthropic_client` de `dd_copilot.cli`.
- Produces: aplicación Streamlit ejecutable con `streamlit run app.py`.

- [ ] **Step 1: Implementar `app.py`**

```python
import streamlit as st

from dd_copilot.cli import build_anthropic_client
from dd_copilot.pipeline import analyze

st.set_page_config(page_title="DD-Copilot", layout="centered")
st.title("DD-Copilot — Due Diligence Técnica")
st.caption(
    "Pega una URL, sube un PDF, o pega texto de material público de una "
    "startup deep-tech. DD-Copilot genera un informe de due diligence "
    "técnica con citas verificadas contra la fuente original."
)

source_type = st.radio("Tipo de fuente", ["URL", "Texto pegado", "PDF"], horizontal=True)

source_input = None
if source_type == "URL":
    source_input = st.text_input("URL de la web o whitepaper")
elif source_type == "Texto pegado":
    source_input = st.text_area("Pega aquí el texto", height=200)
else:
    uploaded_file = st.file_uploader("Sube un PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_input = temp_path

if st.button("Analizar", disabled=not source_input):
    with st.spinner("Analizando material público..."):
        client = build_anthropic_client()
        markdown = analyze(source_input, client)
    st.markdown(markdown)
```

- [ ] **Step 2: Verificar manualmente que arranca**

Run: `streamlit run app.py`
Expected: la app abre en el navegador sin errores de import; el formulario se muestra correctamente (la llamada real a Claude requiere `ANTHROPIC_API_KEY` válida en `.env`, se prueba en la Tarea 12 con el demo real).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: visor Streamlit de una pagina"
```

---

### Task 12: Demo con Isomorphic Labs

**Files:**
- Create: `examples/isomorphic-labs/fuente.txt`
- Create: `examples/isomorphic-labs/informe.md`

**Interfaces:**
- Consumes: `dd_copilot.cli` end-to-end, con `ANTHROPIC_API_KEY` real.

- [ ] **Step 1: Recopilar el texto público de Isomorphic Labs**

Pegar en `examples/isomorphic-labs/fuente.txt` el contenido público (web "About"/"Science" de isomorphiclabs.com, o un comunicado de prensa) — texto plano, citando la fuente en la primera línea como comentario `<!-- fuente: https://... -->`.

- [ ] **Step 2: Configurar la API key real**

Run: `cp .env.example .env` y rellenar `ANTHROPIC_API_KEY` con la clave real del usuario (no se commitea, está en `.gitignore`).

- [ ] **Step 3: Ejecutar el análisis real end-to-end**

Run: `ddcopilot analyze examples/isomorphic-labs/fuente.txt --output examples/isomorphic-labs/informe.md`
Expected: fichero `informe.md` generado con las 5 secciones fijas, sin errores.

- [ ] **Step 4: Revisar manualmente el informe generado**

Abrir `examples/isomorphic-labs/informe.md` y comprobar: (a) toda cita aparece literalmente en `fuente.txt`, (b) los campos sin evidencia dicen "no mencionado en la fuente", (c) el nivel de confianza tiene justificación coherente con el contenido real.

- [ ] **Step 5: Commit**

```bash
git add examples/isomorphic-labs/
git commit -m "docs: demo real con Isomorphic Labs"
```

---

### Task 13: Documentación final — README y GUIA-DE-USO

**Files:**
- Create: `README.md`
- Create: `GUIA-DE-USO.md`

**Interfaces:**
- Consumes: todo el proyecto ya implementado (documentación, no código).

- [ ] **Step 1: Escribir `README.md`**

Contenido mínimo requerido (técnico, explica el "por qué", no solo el "qué"):
- Qué es DD-Copilot y para qué caso de uso (link al Proyecto 0 del documento de prompts).
- Arquitectura (diagrama de texto del pipeline `ingest → chunk → index → extract → report`).
- Por qué LlamaIndex + embeddings locales en vez de una base vectorial externa.
- Por qué la cascada Haiku/Sonnet (coste vs. calidad) y el prompt caching.
- Por qué la validación de citas es un requisito no negociable (anti-alucinación).
- Instrucciones de instalación y ejecución (CLI y Streamlit).
- Enlace al informe de demo (`examples/isomorphic-labs/informe.md`).
- Sección "Roadmap (no implementado)": comparación entre startups, scoring ponderado, multi-proveedor de LLM.

- [ ] **Step 2: Escribir `GUIA-DE-USO.md` (español, sin jerga sin definir)**

Contenido mínimo requerido:
- Qué hace cada fichero del proyecto, en una frase, sin asumir conocimiento previo de RAG/embeddings/LLM (definir cada término la primera vez que aparece).
- Cómo instalar Python, crear el entorno virtual, instalar dependencias, paso a paso con los comandos exactos.
- Cómo conseguir y configurar la API key de Anthropic.
- Cómo ejecutar el demo de Isomorphic Labs y cómo ejecutar un análisis propio (URL, PDF o texto).
- Cómo leer el informe generado: qué significa cada sección, qué significa "no mencionado en la fuente", qué significa el nivel de confianza.
- Cómo ejecutar la suite de tests para comprobar que todo sigue funcionando (`pytest`).

- [ ] **Step 3: Commit**

```bash
git add README.md GUIA-DE-USO.md
git commit -m "docs: readme tecnico y guia de uso en espanol"
```

---

### Task 14: Publicación en GitHub

**Files:** ninguno (operación de git/gh).

- [ ] **Step 1: Confirmar con el usuario visibilidad del repo (público) y nombre (`dd-copilot`)**

- [ ] **Step 2: Crear el repo remoto y hacer push**

Run:
```bash
cd ~/Projects/dd-copilot
gh repo create jjpp01x/dd-copilot --public --source=. --remote=origin --push
```
Expected: repo visible en `https://github.com/jjpp01x/dd-copilot`, rama `main` con todos los commits.

- [ ] **Step 3: Verificar en GitHub que el README se renderiza correctamente**

Run: `gh repo view jjpp01x/dd-copilot --web` (o revisar manualmente en el navegador).

---

## Self-Review

**Cobertura del spec:** ingesta ✅ (Task 3), chunking semántico ✅ (Task 4), extracción estructurada con checklist fijo ✅ (Task 7), citas/anti-alucinación ✅ (Task 5, 7), informe con 5 secciones fijas ✅ (Task 8), cascada Haiku/Sonnet + prompt caching + filtrado por embeddings ✅ (Task 6, 7), CLI + Streamlit sin duplicar lógica ✅ (Task 9, 10, 11), tests con Claude mockeado ✅ (todas las tasks con LLM), demo Isomorphic Labs ✅ (Task 12), README + GUIA-DE-USO ✅ (Task 13), publicación en GitHub ✅ (Task 14).

**Placeholders:** ninguno pendiente — todos los pasos incluyen código completo.

**Consistencia de tipos:** `Document(source_name, text)` se usa igual en `ingest.py`, `chunking.py` y `pipeline.py`. `ChecklistField`/`RiskChecklistItem`/`ExtractionResult`/`ReportInput` de `models.py` se reutilizan sin cambios de nombre en `extract.py` y `report.py`. `analyze(source, client)` es la única función de orquestación, usada igual por `cli.py` y `app.py`.
