# DD-Copilot

A minimal CLI + Streamlit tool that turns public material from a deep-tech
startup (a website, a whitepaper, a PDF) into a structured, citation-verified
technical due-diligence report.

This is Project 0 from a personal portfolio effort aimed at deep-tech
analyst roles: a small, working prototype of the "decision-ready brief" a
technology analyst produces during diligence — built end-to-end with
LlamaIndex (RAG) and Claude, with a deliberate focus on cost efficiency and
zero-hallucination guarantees.

## What it does

Given a URL, a PDF, or pasted text, DD-Copilot extracts:
- The problem the technology solves
- Its technical differentiation from alternatives
- Performance/scalability claims
- A fixed checklist of risks that are *not* addressed by the source
  (technology readiness level, hardware/vendor dependency, reproducibility,
  regulatory risk)

...and assembles a five-section Markdown report: Executive Summary, What the
Startup Says, What It Doesn't Say, Questions for the Next Founder Call, and a
justified Confidence Level.

See a real, unedited example: [`examples/isomorphic-labs/report.md`](examples/isomorphic-labs/report.md)
(generated from [`examples/isomorphic-labs/source.txt`](examples/isomorphic-labs/source.txt)).

## Architecture

```
ingest.py     URL / PDF / pasted text  ->  Document(source_name, text)
chunking.py   Document                 ->  semantic chunks (LlamaIndex SentenceSplitter)
index.py      chunks                   ->  in-memory vector index (local embeddings)
extract.py    index + LLM provider     ->  structured, citation-checked extraction
citation_check.py   fuzzy-verifies every citation against the source text
report.py     structured extraction    ->  final Markdown report
pipeline.py   orchestrates all of the above behind one function: analyze()
cli.py        Typer CLI:      ddcopilot analyze <source>
app.py        Streamlit UI:   streamlit run app.py
providers.py  pluggable LLM backend:   Claude (default) or local Ollama
```

`cli.py` and `app.py` are thin wrappers around the same `pipeline.analyze()`
function — no business logic is duplicated between the two interfaces.

## Why these decisions

**LlamaIndex + local embeddings, no external vector database.** The corpus
for a single due-diligence pass (one startup's public material) easily fits
in memory. `sentence-transformers/all-MiniLM-L6-v2` runs locally and costs
nothing per analysis — only the final extraction/synthesis calls hit a paid
API. A managed vector database would add operational surface with no
benefit at this scale.

**A cost-saving cascade, not one model for everything.** Each checklist
field and each risk question is a small, well-defined classification task —
that's Haiku's job. Only the final synthesis (turning already-structured
extraction into a confidence score and justification) goes to Sonnet, since
that's the one step where reasoning quality actually matters. Each Claude
call marks the system prompt with `cache_control` for Anthropic prompt
caching — the saving only actually materializes once the system prompt
grows past the model's minimum cacheable prefix (currently a few hundred
to a few thousand tokens depending on the model), so on this small a
prompt it's mostly future-proofing rather than a measured cost win today.
Retrieval-by-embeddings runs before any LLM call at all, so only the
top-k relevant chunks (not the whole document) are ever sent to a model —
that's the retrieval-side saving that's actually in effect now.

**Citation verification is a hard requirement, not a nice-to-have.** Every
claim in the report must carry a citation that is fuzzy-matched (via
`rapidfuzz`) against the literal source text. If a claim's citation doesn't
actually appear in the source, the field is discarded and the report says
"Not mentioned in the source" instead — the tool is designed to *never*
state something the source doesn't literally support, even under pressure
from a model that wants to be helpful and fill in a plausible-sounding
answer.

**A pluggable LLM provider, not a hardcoded Anthropic client.** `extract.py`
depends only on an `LLMProvider` protocol (`complete(system_prompt,
user_prompt, tier) -> str`). `ClaudeProvider` is the default, and is the
one that keeps the Haiku/Sonnet cascade described above — it's what's
recommended for production use. A second provider, `OllamaProvider`, runs
entirely against a local model (`llama3.1` by default) with zero API cost
and no API key at all. This exists for a very practical reason encountered
during development — see below — but it's also a legitimate architecture
choice: a due-diligence tool built for a resource-constrained analyst
should degrade gracefully to "free and local," not "unusable," when API
credit runs out.

## Language decision

All product-facing content — prompts sent to the LLM, the checklist keys,
and the report template — is in English, matching the language of most
public deep-tech material and the audience for this kind of report.

Two versions of the walkthrough guide are included:
[`GUIA-DE-USO.md`](GUIA-DE-USO.md) (Spanish) and [`USER-GUIDE.md`](USER-GUIDE.md)
(English) — same content, different language, since the tool's author
reads Spanish.

## Getting started

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY if using --provider claude
```

Run the CLI (Claude by default):

```bash
ddcopilot analyze "https://example-startup.com" --output report.md
```

Run it against a local model instead (no API key needed — requires
[Ollama](https://ollama.com) running locally with a model pulled, e.g.
`ollama pull llama3.1`):

```bash
ddcopilot analyze examples/isomorphic-labs/source.txt --output report.md --provider ollama
```

Or launch the Streamlit UI, which lets you pick the provider from a radio
button:

```bash
streamlit run app.py
```

Run the tests (all Claude/Ollama calls are mocked — no API key or network
access, and no tokens are spent, running the suite):

```bash
pytest
```

## A real bug found while building the demo

While generating the Isomorphic Labs demo, `ingest()` initially only
recognized a local file path ending in `.pdf`; any other existing file path
(like a `.txt` source file) silently fell through to being treated as raw
pasted text — meaning the *file path string itself*, not its contents, got
analyzed. The fix (`dd_copilot/ingest.py`) makes any existing local file
path read as text by default, with `.pdf` handled specially. A second,
related issue: embedding source URLs as an HTML comment inside the ingested
text caused a smaller local model to mistake a URL for a valid citation
(since it appeared verbatim in the source) — source attribution now lives
in a separate `SOURCES.md` file per example, never inside the text that
gets analyzed.

## Roadmap (not implemented)

- Automatic comparison across 2-3 startups in the same vertical.
- A weighted quantitative scoring model across startups.
- Additional LLM providers beyond Claude and Ollama (e.g. OpenAI).
