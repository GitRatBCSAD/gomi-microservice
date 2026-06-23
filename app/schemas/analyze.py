from pydantic import BaseModel
from typing import List, Optional

# Request schemas

class AnalyzeRepoRequest(BaseModel):
    repo_url: str
    access_token: Optional[str] = None # optional.. only for private repos
    branch: Optional[str] = None # optional.. default to main/master

# Response schemas

class CommitSentiment(BaseModel):
    message: str
    label: str

class SHAPBreakdown(BaseModel):
    base_rate: float
    sentiment_contrib: float
    complexity_contrib: float
    low_info_contrib: float
    entropy_contrib: Optional[float] = 0.0
    ndev_contrib: Optional[float] = 0.0
    age_contrib: Optional[float] = 0.0
    commits_contrib: Optional[float] = 0.0

class FileRiskResult(BaseModel):
    filename: str
    risk_score: float
    sentiment_score: float
    complexity_score: float
    low_confidence: bool
    shap_breakdown: Optional[SHAPBreakdown]
    commit_sentiments: List[CommitSentiment] = []

class AnalyzeRepoResponse(BaseModel):
    repo_url: str
    status: str
    file_results: List[FileRiskResult]





