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

...and assembles a six-section Markdown report: Executive Summary, What the
Startup Says, What It Doesn't Say, Claims Assessed, Questions for the Next
Founder Call, and a justified Confidence Level.

## Claims assessed: the part that is actually judgement

Anyone can summarise a whitepaper. The section worth reading is
`## 4. Claims Assessed`, where every quantitative claim the source makes is
given one of three verdicts:

| Verdict | Means |
| --- | --- |
| **Verifiable** | The source states the figure *and* the method or conditions it was measured under. |
| **Plausible** | A figure is given with no method attached — consistent with the state of the art, but not evidence. |
| **Unsupported** | The claim contradicts the established state of the art, or rests on nothing at all. |

The verdict is about **the evidence the source offers**, never about whether
the technology works. "Unsupported" means the source gives no basis for the
number, not that the number is false — a distinction that matters when the
report ends up in front of an investment committee.

Two rules are enforced in `claims.py` rather than trusted to the model:

1. A claim whose citation cannot be verified against the source is **dropped
   entirely**. If it isn't in the text, it isn't a claim the startup made.
2. A claim **cannot be `verifiable` without a stated measurement method**,
   however confident the model sounds. A number with no conditions attached is
   at best plausible, and this is the single most common way a pitch overstates
   its evidence.

Every claim that is not `verifiable` becomes a question for the founder call.

**Discarded claims are declared, not hidden.** Rule 1 is a real filter: a model
that finds every claim in a source but leaves the citation field blank has all
of them dropped. Before this was surfaced, that produced a report stating *"the
source makes no quantitative claims"* — an assertion the evidence did not
support, from a tool built specifically not to make those. The claims still stay
out of the table; the report now says how many were lost and why, and says
plainly that this is a limit of the extraction rather than a finding about the
source. It was found by running the pipeline against a real paper, not by a
test.

## Handling modes

Client material and published material are not the same thing, so the tool
does not treat them the same way:

```bash
ddcopilot analyze <source>                          # --mode public (default)
ddcopilot analyze <source> --mode confidential --provider ollama
```

`--mode confidential` **refuses to run** against a remote provider rather than
warning about it — the promise that client material never leaves the machine
has to be enforced by the code, not by the operator remembering a flag. It
refuses before a provider is even constructed.

Every run appends one line to `audit.jsonl`: timestamp, mode, provider and a
SHA-256 of the report. A report handed over months ago can then be matched to
the run that produced it. In confidential mode the source name is redacted,
because *what* a client is looking at is itself sensitive.

## Worked examples

Two, chosen to contrast:

| Example | Source | What it demonstrates |
| --- | --- | --- |
| [`nano-swarm`](examples/nano-swarm/) | arXiv abstract on nano-drone swarm collision avoidance (ETH Zurich / Bologna) | Three figures graded `Verifiable`, each justified by the conditions the source actually states |
| [`isomorphic-labs`](examples/isomorphic-labs/) | A company mission page | A source with no assessable technical claims — a legitimate finding, and a poor demo |

Both are unedited output. Each directory's `SOURCES.md` records the provenance,
the exact command, and what the run got wrong — read those before treating the
reports as polished artefacts.

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

## Cost, measured rather than estimated

An analyst who doesn't know what their tooling costs can't defend using it. Every
Claude call records its real token usage from the API response, and the run
reports the total:

```bash
ddcopilot analyze <source> --max-cost-usd 0.50
```

Rates live in `costs.py` in US dollars per million tokens, taken from Anthropic's
published pricing: Haiku 4.5 at $1.00 / $5.00 and Sonnet 5 at $3.00 / $15.00.
Sonnet 5 also had an introductory rate of $2.00 / $10.00 through 2026-08-31 — the
standard rate is used deliberately, so the cap errs toward stopping early.

Two decisions worth stating:

- **An unpriced model raises rather than costing zero.** A silent zero would let
  an unrecognised model spend without ever tripping the cap, which is the exact
  failure the cap exists to prevent.
- **The cap is checked *after* each call, because a call's cost isn't knowable
  until it returns.** A run can overshoot by at most one call; what the cap
  prevents is the next one. Said plainly here rather than implied — a budget that
  silently overshoots is worse than no budget.

Writing this surfaced a real bug: the retry decorator wrapped `BudgetExceeded`
and retried it, spending *more* money at precisely the moment the budget ran out.
`BudgetExceeded` is now exempt from retries.

Exit codes: `2` = confidential mode refused a remote provider, `3` = budget cap
reached.

## Word export

Markdown is the working format; `.docx` is what a client receives.

```bash
pip install -e ".[docx]"
ddcopilot analyze <source> --docx report.docx
```

The conversion understands only what `report.py` emits — headings, bullets, one
pipe table, paragraphs — rather than pulling in a general Markdown engine for a
document whose shape we control. The claims table becomes a real Word table:
rendered as pipe-separated text it reads as noise, and it is the section worth
reading. `python-docx` is an optional extra, so the core pipeline installs
without it; `--docx` without it exits 4 with the install command.

## Known limitations

Stated plainly, because a diligence tool whose limits are undeclared is worse
than no tool at all.

- **It reads what it is given, and nothing else.** There is no cross-checking
  against papers, patents, funding records or the state of the art. A claim
  marked `plausible` is plausible *given no external evidence was consulted*.
- **`unsupported` depends on the model's world knowledge**, which has a
  training cutoff and no citation of its own. Of the three verdicts it is the
  one to trust least — treat it as a flag to investigate, never as a finding.
- **Citation verification proves provenance, not truth.** It guarantees the
  quoted text really appears in the source. Whether the source is right is a
  separate question the tool does not attempt.
- **The risk checklist is fixed and generic.** Six items cover common deep-tech
  failure modes; they are not tuned per vertical, and a domain expert will
  always have sharper questions.
- **Retrieval can miss.** Claims buried in material that doesn't match the
  retrieval query may never reach the model. Absence from the report is not
  evidence of absence from the source.
- **No human is replaced.** The output is a first pass that makes an analyst
  faster at reading; the judgement that matters still happens afterwards.

## Roadmap (not implemented)

- Automatic comparison across 2-3 startups in the same vertical.
- A weighted quantitative scoring model across startups.
- Additional LLM providers beyond Claude and Ollama (e.g. OpenAI).
