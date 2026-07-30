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
        return f"- **{label}:** Not mentioned in the source."
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
