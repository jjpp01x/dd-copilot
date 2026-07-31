from dd_copilot.models import ReportInput, ChecklistField, Claim, RiskChecklistItem

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
    "scaling_bottleneck": "Bottleneck to industrial scaling",
    "talent_dependency": "Dependency on critical talent",
}

VERDICT_LABELS = {
    "verifiable": "Verifiable",
    "plausible": "Plausible",
    "unsupported": "Unsupported",
}


def _render_field(label: str, field: ChecklistField) -> str:
    if not field.mentioned:
        return f"- **{label}:** Not mentioned in the source."
    citations = "; ".join(f'"{c.text}"' for c in field.citations)
    return f"- **{label}:** {field.value} (citation: {citations})"


def _render_risk(risk: RiskChecklistItem) -> str:
    label = RISK_LABELS[risk.risk_name]
    if not risk.mentioned:
        return f"- **{label}:** not mentioned — pending question for the founder."
    return f"- **{label}:** {risk.detail}"


def _escape_cell(text: str) -> str:
    """Markdown tables break on unescaped pipes, and quoted source text has them."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _render_claims_table(claims: list[Claim]) -> str:
    if not claims:
        return "- The source makes no quantitative claims that could be assessed."
    header = (
        "| Claim | Figure | Verdict | Why |\n"
        "| --- | --- | --- | --- |"
    )
    rows = [
        "| {} | {} | {} | {} |".format(
            _escape_cell(c.text),
            _escape_cell(c.figure or "—"),
            VERDICT_LABELS[c.verdict],
            _escape_cell(c.justification),
        )
        for c in claims
    ]
    return "\n".join([header, *rows])


def _claim_questions(claims: list[Claim]) -> list[str]:
    """The claims worth asking about are the ones the source does not back up."""
    return [
        f'- On "{_escape_cell(c.text)}": under what conditions was this measured, and by whom?'
        for c in claims
        if c.verdict != "verifiable"
    ]


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

    question_lines = _claim_questions(extraction.claims) + [
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
            "## 4. Claims Assessed\n\n" + _render_claims_table(extraction.claims),
            "## 5. Questions for the Next Founder Call\n\n" + "\n".join(question_lines),
            "## 6. Confidence Level\n\n"
            f"**{report_input.confidence_score}/5** — {report_input.confidence_justification}",
        ]
    )
