from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    text: str
    source_chunk_id: str


class ChecklistField(BaseModel):
    value: str
    citations: list[Citation] = Field(default_factory=list)
    mentioned: bool


RiskName = Literal[
    "madurez_trl",
    "dependencia_hardware",
    "reproducibilidad",
    "riesgo_regulatorio",
]


class RiskChecklistItem(BaseModel):
    risk_name: RiskName
    mentioned: bool
    detail: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    problema: ChecklistField
    diferenciacion: ChecklistField
    rendimiento: ChecklistField
    riesgos: list[RiskChecklistItem]


class ReportInput(BaseModel):
    source_name: str
    extraction: ExtractionResult
    confidence_score: int = Field(ge=1, le=5)
    confidence_justification: str
