"""Pydantic schemas used by the output parsers."""
from typing import List, Literal

from pydantic import BaseModel, Field


class CandidateEvaluation(BaseModel):
    """Structured evaluation of ONE resume against a job description."""

    candidate_name: str = Field(description="Candidate's name as written in the resume")
    match_score: int = Field(description="Overall fit with the JD, integer from 0 to 100")
    matching_skills: List[str] = Field(description="JD skills clearly evidenced in the resume")
    missing_skills: List[str] = Field(description="JD skills NOT evidenced in the resume")
    candidate_summary: str = Field(description="2-3 sentence professional summary")
    strengths: List[str] = Field(description="Key strengths relevant to the JD")
    weaknesses: List[str] = Field(description="Gaps or concerns relevant to the JD")
    recommendation: Literal["Strong Hire", "Hire", "Maybe", "No Hire"]
    justification: str = Field(description="Short justification grounded in resume evidence")


class RankedCandidate(BaseModel):
    candidate: str
    rank: int
    reason: str = Field(description="One or two sentences on why they hold this rank")


class ComparisonResult(BaseModel):
    """Structured comparison across several candidates."""

    ranking: List[RankedCandidate] = Field(description="Candidates ordered best to worst")
    best_candidate: str
    final_recommendation: str = Field(description="Who to hire/interview and next steps")
    justification: str = Field(description="Evidence-based reasoning for the choice")
