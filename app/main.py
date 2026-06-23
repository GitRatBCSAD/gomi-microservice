import tempfile
import subprocess
import os

from fastapi import FastAPI, HTTPException
from .schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse, SHAPBreakdown

from .api.routes import analyze


app = FastAPI(
    title="Gomi ML Microservice",
)


app.include_router(analyze.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
