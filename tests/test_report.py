from dd_copilot.models import Citation, ChecklistField, Claim, RiskChecklistItem, ExtractionResult, ReportInput
from dd_copilot.report import render_report

def test_render_report_includes_all_six_fixed_sections():
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
    assert "## 4. Claims Assessed" in markdown
    assert "## 5. Questions for the Next Founder Call" in markdown
    assert "## 6. Confidence Level" in markdown
    assert "accelerates drug discovery" in markdown
    assert "Technology readiness level (TRL)" in markdown


def test_render_report_treats_mentioned_risk_as_covered_and_omits_it_from_gaps():
    problem = ChecklistField(value="Solves X.", citations=[], mentioned=True)
    empty = ChecklistField(value="", citations=[], mentioned=False)
    mentioned_risk = RiskChecklistItem(
        risk_name="hardware_dependency",
        mentioned=True,
        detail="Requires proprietary TPU clusters.",
        citations=[Citation(text="requires proprietary TPU clusters", source_chunk_id="c2")],
    )
    extraction = ExtractionResult(problem=problem, differentiation=empty, performance=empty, risks=[mentioned_risk])
    report_input = ReportInput(source_name="demo", extraction=extraction, confidence_score=4, confidence_justification="Good coverage.")

    markdown = render_report(report_input)

    # A mentioned risk is treated as covered: it must not appear as a gap or a pending question.
    assert "All checklist risks are covered by the source." in markdown
    assert "No pending questions from the fixed checklist" in markdown
    assert "hardware_dependency" not in markdown


def _extraction_with_claims(claims):
    empty = ChecklistField(value="", citations=[], mentioned=False)
    covered_risk = RiskChecklistItem(risk_name="trl_maturity", mentioned=True, detail="TRL 6.")
    return ExtractionResult(problem=empty, differentiation=empty, performance=empty, risks=[covered_risk], claims=claims)


def test_claims_table_renders_verdicts_and_escapes_pipes():
    claim = Claim(
        text="Inference in 8 ms | measured on-device",
        figure="8 ms",
        verdict="verifiable",
        justification="Benchmark and protocol are stated.",
        citations=[Citation(text="8 ms", source_chunk_id="c1")],
    )
    report_input = ReportInput(
        source_name="demo",
        extraction=_extraction_with_claims([claim]),
        confidence_score=4,
        confidence_justification="Good coverage.",
    )

    markdown = render_report(report_input)

    assert "| Claim | Figure | Verdict | Why |" in markdown
    assert "| Verifiable |" in markdown
    # An unescaped pipe would silently split the row into extra columns.
    assert "8 ms \\| measured on-device" in markdown


def test_only_claims_that_are_not_verifiable_become_founder_questions():
    verifiable = Claim(text="Runs in 8 ms.", verdict="verifiable", justification="Protocol stated.")
    weak = Claim(text="Scales to 1,000 robots.", verdict="plausible", justification="No method stated.")
    report_input = ReportInput(
        source_name="demo",
        extraction=_extraction_with_claims([verifiable, weak]),
        confidence_score=3,
        confidence_justification="Mixed evidence.",
    )

    markdown = render_report(report_input)
    questions = markdown.split("## 5. Questions for the Next Founder Call")[1]

    assert "Scales to 1,000 robots." in questions
    assert "Runs in 8 ms." not in questions


def test_report_states_plainly_when_there_are_no_quantitative_claims():
    report_input = ReportInput(
        source_name="demo",
        extraction=_extraction_with_claims([]),
        confidence_score=2,
        confidence_justification="Marketing material only.",
    )

    assert "no quantitative claims that could be assessed" in render_report(report_input)


def test_discarded_claims_are_declared_not_hidden():
    """The report must never say 'no quantitative claims' when claims were
    found and thrown away — that is an assertion the evidence doesn't support."""
    extraction = _extraction_with_claims([])
    extraction.claims_discarded = 3
    report_input = ReportInput(
        source_name="demo", extraction=extraction,
        confidence_score=2, confidence_justification="Extraction was lossy.",
    )

    markdown = render_report(report_input)

    assert "3 further claim(s) were found but could not be assessed" in markdown
    assert "not a finding that the source makes no quantitative claims" in markdown


def test_a_clean_run_carries_no_discarded_note():
    claim = Claim(text="Runs in 8 ms.", verdict="verifiable", justification="Protocol stated.")
    report_input = ReportInput(
        source_name="demo", extraction=_extraction_with_claims([claim]),
        confidence_score=4, confidence_justification="Good coverage.",
    )

    assert "could not be assessed" not in render_report(report_input)


def test_surviving_claims_still_report_the_ones_that_did_not():
    claim = Claim(text="Runs in 8 ms.", verdict="verifiable", justification="Protocol stated.")
    extraction = _extraction_with_claims([claim])
    extraction.claims_discarded = 2
    report_input = ReportInput(
        source_name="demo", extraction=extraction,
        confidence_score=3, confidence_justification="Partly lossy.",
    )

    markdown = render_report(report_input)

    assert "| Verifiable |" in markdown
    assert "2 further claim(s)" in markdown
