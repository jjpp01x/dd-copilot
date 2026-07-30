from dd_copilot.models import Citation, ChecklistField, RiskChecklistItem, ExtractionResult, ReportInput
import pytest
from pydantic import ValidationError


def test_checklist_field_defaults_to_not_mentioned():
    field = ChecklistField(value="", citations=[], mentioned=False)
    assert field.mentioned is False
    assert field.citations == []


def test_extraction_result_holds_all_checklist_fields():
    problema = ChecklistField(value="Resuelve X", citations=[Citation(text="cita literal", source_chunk_id="chunk-1")], mentioned=True)
    diferenciacion = ChecklistField(value="", citations=[], mentioned=False)
    rendimiento = ChecklistField(value="", citations=[], mentioned=False)
    riesgo = RiskChecklistItem(risk_name="madurez_trl", mentioned=False)
    result = ExtractionResult(problema=problema, diferenciacion=diferenciacion, rendimiento=rendimiento, riesgos=[riesgo])
    assert result.problema.value == "Resuelve X"
    assert result.riesgos[0].risk_name == "madurez_trl"


def test_report_input_confidence_score_range():
    problema = ChecklistField(value="x", citations=[], mentioned=True)
    result = ExtractionResult(problema=problema, diferenciacion=problema, rendimiento=problema, riesgos=[])
    with pytest.raises(ValidationError):
        ReportInput(source_name="demo", extraction=result, confidence_score=6, confidence_justification="fuera de rango")
