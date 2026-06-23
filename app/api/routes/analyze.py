from fastapi import APIRouter, HTTPException
from ...schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from ...services.analyzer import AnalyzerService

router = APIRouter()

@router.post("/analyze-repo", response_model=AnalyzeRepoResponse)
async def analyze_repo_handler(request: AnalyzeRepoRequest):
    try:
        # Pass the data to the service layer
        results = AnalyzerService.analyze_repository(
            repo_url=request.repo_url,
            branch=request.branch,
            access_token=request.access_token
        )
        
        return AnalyzeRepoResponse(
            repo_url=request.repo_url,
            status="success",
            file_results=results
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
