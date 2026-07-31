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
    "trl_maturity",
    "hardware_dependency",
    "reproducibility",
    "regulatory_risk",
    "scaling_bottleneck",
    "talent_dependency",
]

ClaimVerdict = Literal["verifiable", "plausible", "unsupported"]


class Claim(BaseModel):
    """A quantitative claim made by the source, with a verdict on how well the
    source itself backs it up. The verdict is about the evidence offered, never
    about whether the underlying technology works."""

    text: str
    figure: str | None = None
    verdict: ClaimVerdict
    justification: str
    citations: list[Citation] = Field(default_factory=list)


class RiskChecklistItem(BaseModel):
    risk_name: RiskName
    mentioned: bool
    detail: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    problem: ChecklistField
    differentiation: ChecklistField
    performance: ChecklistField
    risks: list[RiskChecklistItem]
    claims: list[Claim] = Field(default_factory=list)


class ReportInput(BaseModel):
    source_name: str
    extraction: ExtractionResult
    confidence_score: int = Field(ge=1, le=5)
    confidence_justification: str
