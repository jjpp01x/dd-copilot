"""Classification of the quantitative claims a source makes.

This is the module that encodes the judgement an analyst is actually paid for:
telling a strong technical claim from a weak one, and being able to say why.

The verdict is deliberately about the *evidence the source offers*, not about
whether the technology works. "Unsupported" means the source gives no basis for
the number, not that the number is false — a distinction that matters when the
report ends up in front of an investment committee.
"""

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex

from dd_copilot.citation_check import verify_citation
from dd_copilot.index import retrieve_relevant_chunks
from dd_copilot.jsonio import parse_json_response
from dd_copilot.models import Citation, Claim, ClaimVerdict
from dd_copilot.providers import LLMProvider

@dataclass(frozen=True)
class ClaimExtraction:
    """Claims that survived, and how many did not.

    The count matters as much as the list. A model can find every claim in a
    source and still leave the citation field blank; dropping those silently
    made the report assert that the source made no quantitative claims, which
    was false. Reporting the number keeps the no-hallucination guarantee while
    admitting what was lost to it.
    """

    claims: list[Claim]
    discarded: int


VALID_VERDICTS: tuple[ClaimVerdict, ...] = ("verifiable", "plausible", "unsupported")

CLAIMS_QUESTION = (
    "What quantitative claims about performance, cost, accuracy or scalability "
    "does the source make?"
)

CLAIMS_SYSTEM_PROMPT = (
    "You are a technical due-diligence analyst for deep-tech investment. "
    "You extract quantitative claims and judge how well the source itself backs "
    "each one up.\n\n"
    "VERDICTS:\n"
    "- 'verifiable': the source states the measurement method or conditions "
    "alongside the figure (a benchmark, a study, a baseline, a shot count, a "
    "fleet size, a named platform).\n"
    "- 'plausible': a figure is given with no method or conditions attached.\n"
    "- 'unsupported': the claim contradicts the established state of the art, "
    "or rests on nothing at all.\n\n"
    "CITATION — the most important field. It must be a span of text COPIED "
    "CHARACTER FOR CHARACTER from the source excerpts you were given. Do not "
    "paraphrase it, do not summarise it, do not write an empty string. Every "
    "claim whose citation cannot be found verbatim in the source is discarded "
    "and never reaches the report, so a missing citation loses the claim.\n\n"
    "Always respond in valid JSON, with no additional text."
)

#: A worked example beats a schema for a small model: it shows the citation is
#: a verbatim span rather than a description of one, which is the field that
#: was most often left blank.
CLAIMS_RESPONSE_SCHEMA = (
    '{"claims": [{"text": str, "figure": str or null, '
    '"verdict": "verifiable" | "plausible" | "unsupported", '
    '"method": str or null, "justification": str, "citation": str}]}\n\n'
    "Worked example. If the source contained the sentence "
    '"Measured over 10,000 episodes on the PushT benchmark, latency fell to 8 ms.", '
    "you would return:\n"
    '{"claims": [{"text": "Latency fell to 8 ms.", "figure": "8 ms", '
    '"verdict": "verifiable", "method": "10,000 episodes on the PushT benchmark", '
    '"justification": "The source states the benchmark and the episode count.", '
    '"citation": "Measured over 10,000 episodes on the PushT benchmark, latency fell to 8 ms."}]}\n\n'
    "Note the citation is the sentence itself, copied exactly — not a description of it."
)


def classify_claim(payload: dict, source_text: str) -> Claim | None:
    """Turns one raw model payload into a Claim, applying the deterministic
    guardrails. Returns None when the claim should not survive at all.

    Two rules are enforced in code rather than left to the model:

    1. A claim whose citation cannot be found in the source is dropped. If it
       isn't in the text, it isn't a claim the startup made.
    2. A claim cannot be 'verifiable' without a stated measurement method. A
       number with no conditions attached is at best plausible — this is the
       single most common way a pitch deck overstates its evidence.
    """
    citation_text = (payload.get("citation") or "").strip()
    if not verify_citation(citation_text, source_text):
        return None

    verdict = payload.get("verdict")
    justification = (payload.get("justification") or "").strip()

    if verdict not in VALID_VERDICTS:
        verdict = "plausible"
        justification = _append_note(
            justification, "Verdict returned by the model was not recognised; defaulted to plausible."
        )

    method = (payload.get("method") or "").strip()
    if verdict == "verifiable" and not method:
        verdict = "plausible"
        justification = _append_note(
            justification, "Downgraded: no measurement method or conditions are stated in the source."
        )

    return Claim(
        text=payload.get("text", ""),
        figure=_normalise_figure(payload.get("figure")),
        verdict=verdict,
        justification=justification or "No justification provided.",
        citations=[Citation(text=citation_text, source_chunk_id="retrieved")],
    )


#: Models emit these as a *string* about as often as they emit JSON null.
#: Left alone, "null" prints literally in the report's Figure column.
_ABSENT_FIGURES = frozenset({"null", "none", "n/a", "na", "-", "—", ""})


def _normalise_figure(raw: object) -> str | None:
    text = str(raw or "").strip()
    return None if text.lower() in _ABSENT_FIGURES else text


def _append_note(justification: str, note: str) -> str:
    return f"{justification} {note}".strip() if justification else note


def extract_claims(
    provider: LLMProvider, index: VectorStoreIndex, source_text: str
) -> ClaimExtraction:
    """Extracts and classifies every quantitative claim in the source.

    Runs on the cheap tier: enumerating claims is a classification task, and the
    judgement itself is constrained by the guardrails in `classify_claim`.
    """
    nodes = retrieve_relevant_chunks(index, CLAIMS_QUESTION, top_k=5)
    context = "\n\n".join(node.get_content() for node in nodes)
    prompt = (
        f"Question: {CLAIMS_QUESTION}\n\nSource text (relevant excerpts):\n{context}\n\n"
        f"Respond in JSON: {CLAIMS_RESPONSE_SCHEMA}"
    )
    payload = parse_json_response(
        provider.complete(CLAIMS_SYSTEM_PROMPT, prompt, tier="classify")
    )

    raw_claims = payload.get("claims", [])
    classified = [classify_claim(raw, source_text) for raw in raw_claims]
    kept = [claim for claim in classified if claim is not None]
    return ClaimExtraction(claims=kept, discarded=len(classified) - len(kept))
