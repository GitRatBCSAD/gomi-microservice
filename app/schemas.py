from pydantic import BaseModel
from typing import List

# Request Model Here
class AnalyzeRequest(BaseModel):
    filename: str
    code_content: str
    commits: List[str]

#  Response Models
class SHAPBreakdown(BaseModel):
    base_rate: float
    sentiment_contrib: float
    complexity_contrib: float
    low_info_contrib: float

class AnalyzeResponse(BaseModel):
    filename: str
    risk_score: float
    shap_breakdown: SHAPBreakdown
