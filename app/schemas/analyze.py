from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

# Request schemas


class AnalyzeRepoRequest(BaseModel):
    repo_url: str
    access_token: str | None = None
    branch: str | None = None


# Response schemas — alias_generator outputs camelCase keys in JSON


class CommitSentiment(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    hash: str
    message: str
    committed_at: int
    code: str | None = None
    low_info: bool
    risk_probability: float | None = None


class SHAPBreakdown(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    base_rate: float
    sentiment_contrib: float
    complexity_contrib: float
    low_info_contrib: float
    entropy_contrib: float | None = 0.0
    ndev_contrib: float | None = 0.0
    age_contrib: float | None = 0.0
    commits_contrib: float | None = 0.0


class FileRiskResult(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    filename: str
    risk_score: float
    sentiment_score: float
    complexity_score: float
    change_entropy: float
    ndev_score: float
    age_score: float
    low_info_ratio: float
    commit_count: int
    avg_ccn: float
    avg_nloc: float
    low_confidence: bool
    shap_breakdown: SHAPBreakdown | None = None
    commit_sentiments: list[CommitSentiment] = []


class AnalyzeRepoResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    repo_url: str
    threshold: float
    file_results: list[FileRiskResult]
