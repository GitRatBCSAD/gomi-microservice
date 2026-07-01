from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ...schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from ...services.analyzer import AnalyzerService

router = APIRouter()


@router.post("/analyze-repo")
async def analyze_repo_handler(request: AnalyzeRepoRequest):
    try:
        results, threshold = AnalyzerService.analyze_repository(
            repo_url=request.repo_url,
            branch=request.branch,
            access_token=request.access_token,
        )

        response = AnalyzeRepoResponse(
            repo_url=request.repo_url,
            status="success",
            threshold=threshold,
            file_results=results,
        )

        return JSONResponse(content=response.model_dump(by_alias=True, mode="json"))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
