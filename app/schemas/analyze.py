from pydantic import BaseModel

# Request schemas

class AnalyzeRepoRequest(BaseModel):
    repo_url: str
    access_token: str | None = None # optional.. only for private repos
    branch: str | None = None # optional.. default to main/master

# Response schemas

class CommitSentiment(BaseModel):
    message: str
    label: str

class SHAPBreakdown(BaseModel):
    base_rate: float
    sentiment_contrib: float
    complexity_contrib: float
    low_info_contrib: float
    entropy_contrib: float | None = 0.0
    ndev_contrib: float | None = 0.0
    age_contrib: float | None = 0.0
    commits_contrib: float | None = 0.0

class FileRiskResult(BaseModel):
    filename: str
    risk_score: float
    sentiment_score: float
    complexity_score: float
    low_confidence: bool
    shap_breakdown: SHAPBreakdown | None = None
    commit_sentiments: list[CommitSentiment] = []

class AnalyzeRepoResponse(BaseModel):
    repo_url: str
    status: str
    file_results: list[FileRiskResult]





