import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from ...schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from ...services.analyzer import AnalyzerService, ml_engine

router = APIRouter()

# ponytail: in-memory job store, lost on restart; swap for Redis/DB if persistence needed
_jobs: dict[str, dict[str, Any]] = {}


@router.get("/threshold")
async def threshold_handler():
    return {"threshold": ml_engine.threshold}


@router.post("/analyze-repo")
async def analyze_repo_handler(request: AnalyzeRepoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "step": "queued", "progress": 0.0, "current_file": None, "result": None, "error": None}
    background_tasks.add_task(_run_analysis, job_id, request)
    return JSONResponse(content={"job_id": job_id}, status_code=202)


@router.get("/jobs/{job_id}")
async def get_job_handler(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=job)


async def _run_analysis(job_id: str, request: AnalyzeRepoRequest) -> None:
    def on_progress(step: str, progress: float, current_file: str | None = None) -> None:
        _jobs[job_id]["step"] = step
        _jobs[job_id]["progress"] = progress
        _jobs[job_id]["current_file"] = current_file

    try:
        _jobs[job_id]["status"] = "running"
        result = await asyncio.to_thread(
            lambda: AnalyzerService.analyze_repository(
                repo_url=request.repo_url,
                branch=request.branch,
                access_token=request.access_token,
                on_progress=on_progress,
            )
        )
        response = AnalyzeRepoResponse(
            repo_url=request.repo_url,
            head_sha=result.head_sha,
            default_branch=result.default_branch,
            threshold=result.threshold,
            file_results=result.file_results,
        )
        _jobs[job_id].update({
            "status": "done",
            "step": "done",
            "progress": 1.0,
            "result": response.model_dump(by_alias=True, mode="json"),
        })
    except Exception as e:
        _jobs[job_id].update({"status": "failed", "error": str(e)})
