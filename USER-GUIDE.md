# User Guide — DD-Copilot

This guide explains, step by step and without assuming prior knowledge,
what each piece of the project does, how to install it, and how to use it.
The Spanish version (same content) is at [`GUIA-DE-USO.md`](GUIA-DE-USO.md).

## Before anything else: 3 key concepts

- **LLM (Large Language Model)**: an AI model like Claude, capable of
  reading text and responding with text (summarizing, extracting data,
  reasoning about it). Each time we "ask it" something, that's called a
  **call**.
- **RAG (Retrieval-Augmented Generation)**: instead of sending the LLM the
  entire document (expensive and imprecise), we first split the text into
  small pieces ("chunks"), convert them into numerical vectors
  ("embeddings") that capture their meaning, and only send the LLM the
  fragments that are actually relevant to the question we're asking.
- **Local embeddings**: the step of "converting text into vectors" is done
  here by a small model that runs on your own computer (not in the cloud),
  so it costs no money and uses no API tokens.

## What each file does

- `dd_copilot/ingest.py` — takes a URL, a path to a PDF, a path to a text
  file, or directly pasted text, and converts all of it into plain text.
- `dd_copilot/chunking.py` — splits that text into fragments ("chunks") by
  sentence, not by a fixed number of characters, so ideas aren't cut in
  half.
- `dd_copilot/index.py` — converts each fragment into an embedding (with a
  local model, free) and builds an in-memory index so it can find "which
  fragment is most relevant to this question" without calling the LLM.
- `dd_copilot/citation_check.py` — checks that a citation the LLM claims to
  have pulled from the text actually appears there (with some tolerance for
  minor variations). If it doesn't appear, it's discarded — an invented
  fact is never accepted.
- `dd_copilot/providers.py` — defines how to talk to the LLM. There are two
  options: **Claude** (default, recommended, requires an API key with
  credit) or **Ollama** (a model that runs on your computer, free, no API
  key needed, but more limited quality).
- `dd_copilot/extract.py` — asks the fixed checklist questions (problem
  solved, technical differentiation, performance, and 4 risks) using only
  the relevant fragments, and validates each answer with
  `citation_check.py`.
- `dd_copilot/report.py` — assembles everything into a Markdown report with
  5 fixed sections.
- `dd_copilot/pipeline.py` — chains all the steps above into a single
  function.
- `dd_copilot/cli.py` — the terminal command `ddcopilot analyze`.
- `app.py` — a simple web page (Streamlit) to use the tool without a
  terminal.

## Step-by-step installation

1. You need Python 3.12 or newer installed. Check your version:
   ```bash
   python3.12 --version
   ```
2. Create a virtual environment (an isolated space for this project's
   dependencies, so they don't mix with other projects on your computer):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
   You'll see your terminal prompt change to indicate the environment is
   active.
3. Install the project and its dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Setting up access to the LLM

You have two options — you don't need to set up both:

### Option A — Claude (recommended, requires credit in your Anthropic account)

1. Copy the example file: `cp .env.example .env`
2. Open `.env` and put your real Anthropic API key:
   `ANTHROPIC_API_KEY=sk-ant-...`
   (get it at [console.anthropic.com](https://console.anthropic.com),
   "API Keys" section; you need credit in "Plans & Billing").
3. The `.env` file is never uploaded to GitHub (it's in `.gitignore`).

### Option B — Local Ollama (free, no API key, more limited quality)

1. Install [Ollama](https://ollama.com) if you don't have it.
2. Download a model, for example:
   ```bash
   ollama pull llama3.1
   ```
3. You don't need to configure anything else — when you use
   `--provider ollama`, the tool talks directly to Ollama on your computer.

## Running the included demo

The demo with the real startup Isomorphic Labs is already generated in
`examples/isomorphic-labs/report.md`. To regenerate it yourself:

```bash
source .venv/bin/activate
ddcopilot analyze examples/isomorphic-labs/source.txt --output examples/isomorphic-labs/report.md --provider ollama
```

Or, if you have Claude credit, drop `--provider ollama` to use Claude (the
default, higher-quality provider).

## Analyzing your own startup

```bash
# With a URL:
ddcopilot analyze "https://a-deep-tech-startup.com" --output my-report.md

# With a PDF:
ddcopilot analyze path/to/whitepaper.pdf --output my-report.md

# With directly pasted text:
ddcopilot analyze "Text you copied from the startup's website..." --output my-report.md
```

Add `--provider ollama` to any of these commands if you don't have Claude
credit configured.

Or, if you prefer a visual interface in the browser:

```bash
streamlit run app.py
```

## Reading the generated report

The report has 5 fixed sections:

1. **Executive Summary** — a one-sentence summary of what problem the
   startup solves.
2. **What the Startup Says** — what the public material does claim about
   the problem, technical differentiation, and performance, each claim
   with its citation in quotes. If a field says **"Not mentioned in the
   source,"** it means the tool found no verifiable claim about that
   point — not that it doesn't exist, just that it isn't in the material
   you gave it.
3. **What It Doesn't Say** — the fixed checklist of 4 technical risks
   (technology readiness, hardware/vendor dependency, reproducibility of
   results, regulatory risk) that the public material does NOT cover. This
   is as useful as what it does say: it tells you what to ask.
4. **Questions for the Next Founder Call** — a ready-to-use list for a real
   call with the founding team, generated from the gaps found in the
   previous section.
5. **Confidence Level** — a number from 1 to 5 with its justification,
   indicating how much confidence the analysis itself has in itself (not in
   the startup) — for example, low if the public material was very scarce.

## Checking that everything still works

```bash
pytest
```

This runs the full test suite (24 tests total). No real call is made to
Claude or Ollama during the tests — all LLM responses are simulated
("mocked"), so running `pytest` costs no money and requires no configured
key.
