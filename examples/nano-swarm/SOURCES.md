# Source attribution

Attribution lives here, never inside `source.txt`. A URL embedded in the analysed
text was once mistaken by a local model for a verbatim citation — the text that
gets analysed contains only the material under analysis.

- **Title:** Multi-sensory Anti-collision Design for Autonomous Nano-swarm Exploration
- **Authors:** Mahyar Pourjabar, Manuele Rusci, Luca Bompani, Lorenzo Lamberti,
  Vlad Niculescu, Daniele Palossi, Luca Benini
- **Published:** 2023-12-20
- **Categories:** cs.RO, eess.SY
- **arXiv:** https://arxiv.org/abs/2312.13086
- **Retrieved:** 2026-07-31, via the arXiv API

`source.txt` is the paper's title and abstract, verbatim and unedited.

## Why this source

The previous example (a company mission page) was marketing copy: it contained no
quantitative technical claims at all, so the Claims Assessed table had nothing to
assess. That is a legitimate finding about the material, but a poor demonstration
of what the tool does.

This abstract was chosen because its claims differ in evidential quality *within a
single short text*, which is exactly the distinction the tool exists to make:

- a success-rate improvement reported from an in-field study against a named
  state-of-the-art baseline — figure **and** method;
- a collision-avoidance rate quoted with its fleet size and safety distance —
  figure **and** conditions;
- a throughput figure tied to a named processor;
- an obstacle-avoidance probability given as "about 40%" with no stated
  measurement method — a figure without conditions;
- a forward-looking statement about needing more capable processors, with no
  figure at all.

It is also a good fit for the target domain: nano-drone swarms are robotics, and
the work comes out of the PULP group at ETH Zurich and the University of Bologna.

## How this report was produced

```bash
ddcopilot analyze examples/nano-swarm/source.txt \
  --output examples/nano-swarm/report.md \
  --docx examples/nano-swarm/report.docx \
  --provider ollama
```

Generated with the local `llama3.1` provider, not Claude — the API key in use
has no credit balance, which is the same wall that put the Ollama provider in
this repo in the first place. This is the **first full run** after the citation
prompt was tightened, not a best-of-N pick.

## What this example shows, and what it does not

The Claims Assessed table is the part worth reading, and it is right: three
figures graded `Verifiable` with justifications that name the actual evidence
("the source specifies the number of agents and the safety distance").

Three weaknesses remain, all attributable to the small local model rather than
the pipeline, and stated here rather than left for a reader to notice:

- **Problem and differentiation come back "Not mentioned"**, which is wrong — the
  abstract's first sentence describes both. That is the checklist extraction in
  `extract.py`, a separate path from claim classification.
- **The performance field's citation is mismatched**: it quotes the opening
  sentence to support a statement about success rates. The citation verifies
  against the source, so the guardrail passes it; the guardrail proves
  provenance, not relevance.
- **The most interesting claim did not survive.** The abstract's "a probability
  of about 40%" is a figure with no stated method — the textbook `Plausible`
  case. It is the one claim discarded here for an unverifiable citation, which
  the report declares rather than hides.

The local model is also markedly non-deterministic: across five runs the discard
rate ranged from 0 to 1 claims, and one run returned none at all. Tightening the
citation instruction cut the discard rate from roughly 50% to 6%, but did not
eliminate the variance. A capable model would close all three gaps above.
