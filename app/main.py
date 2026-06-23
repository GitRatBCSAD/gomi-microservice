import tempfile
import subprocess
import os

from fastapi import FastAPI, HTTPException
from .schemas import AnalyzeRepoRequest, AnalyzeRepoResponse, SHAPBreakdown


app = FastAPI(
    title="Gomi ML Microservice",
)

@app.post("/analyze-repo", response_model=AnalyzeRepoResponse)
async def analyze_file(request: AnalyzeRequest):
    pass

@app.get("/health")
async def health_check():
    return {"status": "ok"}
