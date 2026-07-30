from dd_copilot.models import Citation, ChecklistField, RiskChecklistItem, ExtractionResult, ReportInput
from dd_copilot.report import render_report

def test_render_report_includes_all_five_fixed_sections():
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
    assert "## 4. Questions for the Next Founder Call" in markdown
    assert "## 5. Confidence Level" in markdown
    assert "accelerates drug discovery" in markdown
    assert "trl_maturity" in markdown
