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
- **Idioma del producto: inglés.** Todos los prompts enviados a Claude, las claves/etiquetas del checklist, y la plantilla del informe generado están en inglés (decisión post-Task 6: evita el problema de retrieval cross-lingual con `all-MiniLM-L6-v2`, que es un modelo solo-inglés). `README.md` también en inglés. Excepción: la documentación de uso para el usuario se entrega en dos ficheros — `GUIA-DE-USO.md` (español) y `USER-GUIDE.md` (inglés), mismo contenido en ambos.
- Checklist de riesgos con claves en inglés: `trl_maturity`, `hardware_dependency`, `reproducibility`, `regulatory_risk` (reemplaza los nombres en español usados provisionalmente en la Tarea 2).
- Cascada de modelos: **Claude Haiku** (`claude-haiku-4-5-20251001`) para clasificación/extracción por chunk; **Claude Sonnet** (`claude-sonnet-5`) solo para la síntesis final del informe.
- Prompt caching de Anthropic para el system prompt fijo, reutilizado en todas las llamadas de un mismo análisis.
- Cada afirmación del informe debe llevar una cita verificada contra el texto fuente (fuzzy match); si no hay evidencia verificable, el campo se marca `mentioned=False` y el informe dice explícitamente "Not mentioned in the source" — nunca se inventa contenido.
- Reintentos con backoff exponencial (máx. 3 intentos) en cada llamada a la API de Claude.
- Los tests deben mockear la llamada a Claude — cero gasto de tokens reales en la suite de tests.
- Repo en `~/Projects/dd-copilot`, publicado en GitHub público como `jjpp01x/dd-copilot`.
- Entregables de documentación: `README.md` (inglés) técnico (explica el "por qué" de cada decisión de arquitectura), `GUIA-DE-USO.md` (español) y `USER-GUIDE.md` (inglés) — mismo contenido, sin jerga sin definir, para que un principiante en IA entienda, instale y pruebe la herramienta.
- Demo: startup **Isomorphic Labs**, informe guardado en `examples/isomorphic-labs/report.md`.
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
│       ├── source.txt
│       └── report.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── GUIA-DE-USO.md
└── USER-GUIDE.md
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
    problem = ChecklistField(value="Solves X", citations=[Citation(text="literal citation", source_chunk_id="chunk-1")], mentioned=True)
    differentiation = ChecklistField(value="", citations=[], mentioned=False)
    performance = ChecklistField(value="", citations=[], mentioned=False)
    risk = RiskChecklistItem(risk_name="trl_maturity", mentioned=False)
    result = ExtractionResult(problem=problem, differentiation=differentiation, performance=performance, risks=[risk])
    assert result.problem.value == "Solves X"
    assert result.risks[0].risk_name == "trl_maturity"

def test_report_input_confidence_score_range():
    import pytest
    from pydantic import ValidationError
    problem = ChecklistField(value="x", citations=[], mentioned=True)
    result = ExtractionResult(problem=problem, differentiation=problem, performance=problem, risks=[])
    with pytest.raises(ValidationError):
        ReportInput(source_name="demo", extraction=result, confidence_score=6, confidence_justification="out of range")
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
    "trl_maturity",
    "hardware_dependency",
    "reproducibility",
    "regulatory_risk",
]


class RiskChecklistItem(BaseModel):
    risk_name: RiskName
    mentioned: bool
    detail: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    problem: ChecklistField
    differentiation: ChecklistField
    performance: ChecklistField
    risks: list[RiskChecklistItem]


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
            "Isomorphic Labs uses deep learning models to predict protein structure. "
            "The marketing team organizes annual events in London for investors. "
            "The company was founded as a DeepMind spin-off in 2021."
        ),
    )
    nodes = chunk_document(doc, chunk_size=20, chunk_overlap=5)
    index = build_index(nodes)
    results = retrieve_relevant_chunks(index, "How does the startup predict protein structures?", top_k=1)
    assert len(results) == 1
    assert "deep learning" in results[0].get_content() or "protein" in results[0].get_content()
```

**Nota (segunda corrección, tras hallazgo del revisor):** con `chunk_size=40`
el `SentenceSplitter` mezclaba dos frases no relacionadas en un mismo
chunk (no respeta límites de oración con ese tamaño), y la query genérica
"What AI technology..." no discriminaba bien con `all-MiniLM-L6-v2`. Se
corrigió a `chunk_size=20` (un chunk por frase, verificado) y a una query
más específica que sí referencia el contenido real del chunk correcto
("predict protein structures"), manteniendo la aserción original y
estricta de `top_k=1` — sin relajar el test a `top_k=2`.

**Nota (post-desviación en la primera ejecución de esta tarea):** la
versión original de este test usaba texto y query en español, lo que
expuso que `all-MiniLM-L6-v2` (modelo solo-inglés) falla el retrieval en
español. Con la decisión de idioma del producto en inglés, el test y el
corpus quedan en inglés y el modelo original vuelve a funcionar
correctamente — no se necesita un modelo multilingüe.

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
        "value": "Solves drug discovery.",
        "citation": "cures rare diseases in 24 hours",
        "mentioned": True,
    }
    sonnet_payload = {
        "confidence_score": 3,
        "confidence_justification": "Public material is limited.",
    }

    fake_client.messages.create.side_effect = [
        _fake_haiku_response(haiku_payload),  # problem
        _fake_haiku_response({"value": "", "citation": "", "mentioned": False}),  # differentiation
        _fake_haiku_response({"value": "", "citation": "", "mentioned": False}),  # performance
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 1
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 2
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 3
        _fake_haiku_response({"mentioned": False, "detail": None, "citation": ""}),  # risk 4
        _fake_sonnet_response(sonnet_payload),  # final synthesis
    ]

    result = run_extraction(fake_client, index, SOURCE_TEXT, "isomorphic-labs")

    assert result.extraction.problem.mentioned is False
    assert result.extraction.problem.citations == []
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
    "You are a technical due-diligence analyst for deep-tech investment. "
    "You may only state what is literally in the text you are given. "
    "If something is not explicit, respond mentioned=false and an empty citation. "
    "Always respond in valid JSON, with no additional text."
)

FIELD_QUESTIONS = {
    "problem": "What problem does this startup's technology solve?",
    "differentiation": "What technically differentiates this technology from alternatives?",
    "performance": "What performance or scalability claims does the startup make?",
}

RISK_QUESTIONS: dict[RiskName, str] = {
    "trl_maturity": "Is the technology's readiness level (TRL) mentioned?",
    "hardware_dependency": "Is dependency on specific hardware or vendors mentioned?",
    "reproducibility": "Is it mentioned whether results are reproducible or have been externally validated?",
    "regulatory_risk": "Is any applicable regulatory risk for this technology mentioned?",
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
        f"Question: {question}\n\nSource text (relevant excerpts):\n{context}\n\n"
        'Respond in JSON: {"value": str, "citation": str, "mentioned": bool}'
    )
    payload = _call_claude(client, CLASSIFY_MODEL, prompt)
    return _build_field_from_response(payload, source_text)


def extract_risks(client, index: VectorStoreIndex, source_text: str) -> list[RiskChecklistItem]:
    risks = []
    for risk_name, question in RISK_QUESTIONS.items():
        nodes = retrieve_relevant_chunks(index, question, top_k=3)
        context = "\n\n".join(node.get_content() for node in nodes)
        prompt = (
            f"Question: {question}\n\nSource text (relevant excerpts):\n{context}\n\n"
            'Respond in JSON: {"mentioned": bool, "detail": str or null, "citation": str}'
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
        f"Already-structured extracted data (not raw text):\n{summary}\n\n"
        'Give a confidence level for the analysis (1-5) and its justification. '
        'Respond in JSON: {"confidence_score": int, "confidence_justification": str}'
    )
    payload = _call_claude(client, SYNTHESIS_MODEL, prompt)
    return payload["confidence_score"], payload["confidence_justification"]


def run_extraction(client, index: VectorStoreIndex, source_text: str, source_name: str) -> ReportInput:
    """Runs the Haiku (per-field) -> Sonnet (final synthesis) cascade."""
    problem = extract_field(client, "problem", FIELD_QUESTIONS["problem"], index, source_text)
    differentiation = extract_field(client, "differentiation", FIELD_QUESTIONS["differentiation"], index, source_text)
    performance = extract_field(client, "performance", FIELD_QUESTIONS["performance"], index, source_text)
    risks = extract_risks(client, index, source_text)

    extraction = ExtractionResult(
        problem=problem,
        differentiation=differentiation,
        performance=performance,
        risks=risks,
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
    problem = ChecklistField(value="Accelerates drug discovery.", citations=[Citation(text="accelerates drug discovery", source_chunk_id="c1")], mentioned=True)
    empty = ChecklistField(value="", citations=[], mentioned=False)
    unmentioned_risk = RiskChecklistItem(risk_name="trl_maturity", mentioned=False)
    extraction = ExtractionResult(problem=problem, differentiation=empty, performance=empty, risks=[unmentioned_risk])
    report_input = ReportInput(source_name="isomorphic-labs", extraction=extraction, confidence_score=3, confidence_justification="Public material is limited.")

    markdown = render_report(report_input)

    assert "# Technical Due Diligence Report — isomorphic-labs" in markdown
    assert "## 1. Executive Summary" in markdown
    assert "## 2. What the Startup Says" in markdown
    assert "## 3. What It Doesn't Say" in markdown
    assert "## 4. Questions for the Next Founder Call" in markdown
    assert "## 5. Confidence Level" in markdown
    assert "accelerates drug discovery" in markdown
    assert "trl_maturity" in markdown
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/test_report.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'dd_copilot.report'`

- [ ] **Step 3: Implementar `dd_copilot/report.py`**

```python
from dd_copilot.models import ReportInput, ChecklistField, RiskChecklistItem

FIELD_LABELS = {
    "problem": "Problem it solves",
    "differentiation": "Technical differentiation",
    "performance": "Performance/scalability claims",
}

RISK_LABELS = {
    "trl_maturity": "Technology readiness level (TRL)",
    "hardware_dependency": "Hardware/vendor dependency",
    "reproducibility": "Reproducibility of results",
    "regulatory_risk": "Regulatory risk",
}


def _render_field(label: str, field: ChecklistField) -> str:
    if not field.mentioned:
        return f"- **{label}:** not mentioned in the source."
    citations = "; ".join(f'"{c.text}"' for c in field.citations)
    return f"- **{label}:** {field.value} (citation: {citations})"


def _render_risk(risk: RiskChecklistItem) -> str:
    label = RISK_LABELS[risk.risk_name]
    if not risk.mentioned:
        return f"- **{risk.risk_name}** ({label}): not mentioned — pending question for the founder."
    return f"- **{risk.risk_name}** ({label}): {risk.detail}"


def render_report(report_input: ReportInput) -> str:
    extraction = report_input.extraction

    says_lines = [
        _render_field(FIELD_LABELS["problem"], extraction.problem),
        _render_field(FIELD_LABELS["differentiation"], extraction.differentiation),
        _render_field(FIELD_LABELS["performance"], extraction.performance),
    ]

    doesnt_say_lines = [_render_risk(r) for r in extraction.risks if not r.mentioned]
    if not doesnt_say_lines:
        doesnt_say_lines = ["- All checklist risks are covered by the source."]

    question_lines = [
        f"- On {RISK_LABELS[r.risk_name].lower()}: not documented, can the team clarify?"
        for r in extraction.risks
        if not r.mentioned
    ]
    if not question_lines:
        question_lines = ["- No pending questions from the fixed checklist; dig deeper into quantitative performance details."]

    return "\n\n".join(
        [
            f"# Technical Due Diligence Report — {report_input.source_name}",
            "## 1. Executive Summary\n\n" + (extraction.problem.value or "Not enough public information for an executive summary."),
            "## 2. What the Startup Says\n\n" + "\n".join(says_lines),
            "## 3. What It Doesn't Say\n\n" + "\n".join(doesnt_say_lines),
            "## 4. Questions for the Next Founder Call\n\n" + "\n".join(question_lines),
            "## 5. Confidence Level\n\n"
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
    "Isomorphic Labs combines artificial intelligence and biology to accelerate "
    "drug discovery. The company is a DeepMind spin-off founded in 2021."
)


def _response(payload: dict):
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(payload))]
    return message


def test_analyze_returns_markdown_with_fixed_sections():
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _response({"value": "Accelerates drug discovery.", "citation": "accelerate drug discovery", "mentioned": True}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"value": "", "citation": "", "mentioned": False}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"mentioned": False, "detail": None, "citation": ""}),
        _response({"confidence_score": 3, "confidence_justification": "Public material is limited."}),
    ]

    markdown = analyze(SOURCE_TEXT, fake_client)

    assert "# Technical Due Diligence Report" in markdown
    assert "accelerate drug discovery" in markdown
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
st.title("DD-Copilot — Technical Due Diligence")
st.caption(
    "Paste a URL, upload a PDF, or paste text from public material of a "
    "deep-tech startup. DD-Copilot generates a technical due-diligence "
    "report with citations verified against the original source."
)

source_type = st.radio("Source type", ["URL", "Pasted text", "PDF"], horizontal=True)

source_input = None
if source_type == "URL":
    source_input = st.text_input("Website or whitepaper URL")
elif source_type == "Pasted text":
    source_input = st.text_area("Paste the text here", height=200)
else:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_input = temp_path

if st.button("Analyze", disabled=not source_input):
    with st.spinner("Analyzing public material..."):
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
- Create: `examples/isomorphic-labs/source.txt`
- Create: `examples/isomorphic-labs/report.md`

**Interfaces:**
- Consumes: `dd_copilot.cli` end-to-end, con `ANTHROPIC_API_KEY` real.

- [ ] **Step 1: Recopilar el texto público de Isomorphic Labs**

Pegar en `examples/isomorphic-labs/source.txt` el contenido público (web "About"/"Science" de isomorphiclabs.com, o un comunicado de prensa; si el original está en inglés, pegarlo tal cual — coherente con la decisión de idioma del producto) — texto plano, citando la fuente en la primera línea como comentario `<!-- source: https://... -->`.

- [ ] **Step 2: Configurar la API key real**

Run: `cp .env.example .env` y rellenar `ANTHROPIC_API_KEY` con la clave real del usuario (no se commitea, está en `.gitignore`).

- [ ] **Step 3: Ejecutar el análisis real end-to-end**

Run: `ddcopilot analyze examples/isomorphic-labs/source.txt --output examples/isomorphic-labs/report.md`
Expected: fichero `report.md` generado con las 5 secciones fijas, sin errores.

- [ ] **Step 4: Revisar manualmente el informe generado**

Abrir `examples/isomorphic-labs/report.md` y comprobar: (a) toda cita aparece literalmente en `source.txt`, (b) los campos sin evidencia dicen "Not mentioned in the source", (c) el nivel de confianza tiene justificación coherente con el contenido real.

- [ ] **Step 5: Commit**

```bash
git add examples/isomorphic-labs/
git commit -m "docs: demo real con Isomorphic Labs"
```

---

### Task 13: Documentación final — README (inglés) + guías de uso (ES/EN)

**Files:**
- Create: `README.md` (inglés)
- Create: `GUIA-DE-USO.md` (español)
- Create: `USER-GUIDE.md` (inglés — mismo contenido que `GUIA-DE-USO.md`)

**Interfaces:**
- Consumes: todo el proyecto ya implementado (documentación, no código).

- [ ] **Step 1: Escribir `README.md` (en inglés)**

Contenido mínimo requerido (técnico, explica el "por qué", no solo el "qué"):
- Qué es DD-Copilot y para qué caso de uso (link al Proyecto 0 del documento de prompts).
- Arquitectura (diagrama de texto del pipeline `ingest → chunk → index → extract → report`).
- Por qué LlamaIndex + embeddings locales en vez de una base vectorial externa.
- Por qué la cascada Haiku/Sonnet (coste vs. calidad) y el prompt caching.
- Por qué la validación de citas es un requisito no negociable (anti-alucinación).
- Nota sobre la decisión de idioma: producto en inglés, con enlace a las dos guías de uso.
- Instrucciones de instalación y ejecución (CLI y Streamlit).
- Enlace al informe de demo (`examples/isomorphic-labs/report.md`).
- Sección "Roadmap (not implemented)": comparación entre startups, scoring ponderado, multi-proveedor de LLM.

- [ ] **Step 2: Escribir `GUIA-DE-USO.md` (español, sin jerga sin definir) y `USER-GUIDE.md` (inglés, mismo contenido)**

Contenido mínimo requerido (idéntico en ambos ficheros, solo cambia el idioma):
- Qué hace cada fichero del proyecto, en una frase, sin asumir conocimiento previo de RAG/embeddings/LLM (definir cada término la primera vez que aparece).
- Cómo instalar Python, crear el entorno virtual, instalar dependencias, paso a paso con los comandos exactos.
- Cómo conseguir y configurar la API key de Anthropic.
- Cómo ejecutar el demo de Isomorphic Labs y cómo ejecutar un análisis propio (URL, PDF o texto).
- Cómo leer el informe generado (nota: el informe sale en inglés): qué significa cada sección, qué significa "Not mentioned in the source", qué significa el nivel de confianza.
- Cómo ejecutar la suite de tests para comprobar que todo sigue funcionando (`pytest`).

- [ ] **Step 3: Commit**

```bash
git add README.md GUIA-DE-USO.md USER-GUIDE.md
git commit -m "docs: readme en ingles y guias de uso en espanol e ingles"
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

**Cobertura del spec:** ingesta ✅ (Task 3), chunking semántico ✅ (Task 4), extracción estructurada con checklist fijo ✅ (Task 7), citas/anti-alucinación ✅ (Task 5, 7), informe con 5 secciones fijas ✅ (Task 8), cascada Haiku/Sonnet + prompt caching + filtrado por embeddings ✅ (Task 6, 7), CLI + Streamlit sin duplicar lógica ✅ (Task 9, 10, 11), tests con Claude mockeado ✅ (todas las tasks con LLM), demo Isomorphic Labs ✅ (Task 12), README (inglés) + GUIA-DE-USO/USER-GUIDE (ES/EN) ✅ (Task 13), publicación en GitHub ✅ (Task 14), decisión de idioma del producto en inglés y modelo de embeddings ✅ (actualizado tras revisión de Task 6, ver Global Constraints).

**Placeholders:** ninguno pendiente — todos los pasos incluyen código completo.

**Consistencia de tipos:** `Document(source_name, text)` se usa igual en `ingest.py`, `chunking.py` y `pipeline.py`. `ChecklistField`/`RiskChecklistItem`/`ExtractionResult`/`ReportInput` de `models.py` se reutilizan sin cambios de nombre en `extract.py` y `report.py`. `analyze(source, client)` es la única función de orquestación, usada igual por `cli.py` y `app.py`.
