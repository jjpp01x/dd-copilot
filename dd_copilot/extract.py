from llama_index.core import VectorStoreIndex

from dd_copilot.citation_check import verify_citation
from dd_copilot.claims import extract_claims
from dd_copilot.index import retrieve_relevant_chunks
from dd_copilot.jsonio import parse_json_response
from dd_copilot.providers import LLMProvider
from dd_copilot.models import (
    Citation,
    ChecklistField,
    RiskChecklistItem,
    RiskName,
    ExtractionResult,
    ReportInput,
)

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
    "scaling_bottleneck": "Is any bottleneck to industrial scaling mentioned (manufacturing, data, energy, supply chain, launch capacity)?",
    "talent_dependency": "Is dependency on specific critical people or scarce specialist talent mentioned?",
}


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


def extract_field(provider: LLMProvider, field_name: str, question: str, index: VectorStoreIndex, source_text: str) -> ChecklistField:
    nodes = retrieve_relevant_chunks(index, question, top_k=5)
    context = "\n\n".join(node.get_content() for node in nodes)
    prompt = (
        f"Question: {question}\n\nSource text (relevant excerpts):\n{context}\n\n"
        'Respond in JSON: {"value": str, "citation": str, "mentioned": bool}'
    )
    payload = parse_json_response(provider.complete(SYSTEM_PROMPT, prompt, tier="classify"))
    return _build_field_from_response(payload, source_text)


def extract_risks(provider: LLMProvider, index: VectorStoreIndex, source_text: str) -> list[RiskChecklistItem]:
    risks = []
    for risk_name, question in RISK_QUESTIONS.items():
        nodes = retrieve_relevant_chunks(index, question, top_k=3)
        context = "\n\n".join(node.get_content() for node in nodes)
        prompt = (
            f"Question: {question}\n\nSource text (relevant excerpts):\n{context}\n\n"
            'Respond in JSON: {"mentioned": bool, "detail": str or null, "citation": str}'
        )
        payload = parse_json_response(provider.complete(SYSTEM_PROMPT, prompt, tier="classify"))
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


def synthesize_confidence(provider: LLMProvider, extraction: ExtractionResult) -> tuple[int, str]:
    summary = extraction.model_dump_json()
    prompt = (
        f"Already-structured extracted data (not raw text):\n{summary}\n\n"
        'Give a confidence level for the analysis (1-5) and its justification. '
        'Respond in JSON: {"confidence_score": int, "confidence_justification": str}'
    )
    payload = parse_json_response(provider.complete(SYSTEM_PROMPT, prompt, tier="synthesis"))
    score = payload.get("confidence_score", 1)
    score = max(1, min(5, int(score)))
    justification = payload.get("confidence_justification", "No justification provided.")
    return score, justification


def run_extraction(provider: LLMProvider, index: VectorStoreIndex, source_text: str, source_name: str) -> ReportInput:
    """Runs the Haiku (per-field) -> Sonnet (final synthesis) cascade."""
    problem = extract_field(provider, "problem", FIELD_QUESTIONS["problem"], index, source_text)
    differentiation = extract_field(provider, "differentiation", FIELD_QUESTIONS["differentiation"], index, source_text)
    performance = extract_field(provider, "performance", FIELD_QUESTIONS["performance"], index, source_text)
    risks = extract_risks(provider, index, source_text)
    claims = extract_claims(provider, index, source_text)

    extraction = ExtractionResult(
        problem=problem,
        differentiation=differentiation,
        performance=performance,
        risks=risks,
        claims=claims,
    )

    confidence_score, confidence_justification = synthesize_confidence(provider, extraction)

    return ReportInput(
        source_name=source_name,
        extraction=extraction,
        confidence_score=confidence_score,
        confidence_justification=confidence_justification,
    )
