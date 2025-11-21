# app/predictive_lead_score/models.py
from pydantic import BaseModel, Field
from typing import TypedDict

class LeadScoreResponse(BaseModel):
    score: int = Field(..., ge=0, le=100)
    explanation: str
    # transcription: str
    filename: str


class LeadAnalysisResult(TypedDict):
    transcription: str
    score: int
    explanation: str
