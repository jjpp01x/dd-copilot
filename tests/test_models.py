from dd_copilot.models import Citation, ChecklistField, RiskChecklistItem, ExtractionResult, ReportInput
import pytest
from pydantic import ValidationError


def test_checklist_field_defaults_to_not_mentioned():
    field = ChecklistField(value="", citations=[], mentioned=False)
    assert field.mentioned is False
    assert field.citations == []


def test_extraction_result_holds_all_checklist_fields():
    problem = ChecklistField(value="Solves X", citations=[Citation(text="literal citation", source_chunk_id="chunk-1")], mentioned=True)
    differentiation = ChecklistField(value="", citations=[], mentioned=False)
    performance = ChecklistField(value="", citations=[], mentioned=False)
    risk = RiskChecklistItem(risk_name="trl_maturity", mentioned=False)
    result = ExtractionResult(problem=problem, differentiation=differentiation, performance=performance, risks=[risk])
    assert result.problem.value == "Solves X"
    assert result.risks[0].risk_name == "trl_maturity"


def test_report_input_confidence_score_range():
    problem = ChecklistField(value="x", citations=[], mentioned=True)
    result = ExtractionResult(problem=problem, differentiation=problem, performance=problem, risks=[])
    with pytest.raises(ValidationError):
        ReportInput(source_name="demo", extraction=result, confidence_score=6, confidence_justification="out of range")
