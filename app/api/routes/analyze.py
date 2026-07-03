from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ...schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from ...services.analyzer import AnalyzerService, ml_engine

router = APIRouter()


@router.get("/threshold")
async def threshold_handler():
    return {"threshold": ml_engine.threshold}


@router.post("/analyze-repo")
async def analyze_repo_handler(request: AnalyzeRepoRequest):
    try:
        result = AnalyzerService.analyze_repository(
            repo_url=request.repo_url,
            branch=request.branch,
            access_token=request.access_token,
        )

        response = AnalyzeRepoResponse(
            repo_url=request.repo_url,
            head_sha=result.head_sha,
            default_branch=result.default_branch,
            threshold=result.threshold,
            file_results=result.file_results,
        )

        return JSONResponse(content=response.model_dump(by_alias=True, mode="json"))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
