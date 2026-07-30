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
