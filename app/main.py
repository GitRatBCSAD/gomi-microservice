import os
from dotenv import load_dotenv
load_dotenv() 

from contextlib import asynccontextmanager
from fastapi import FastAPI
from .api.routes import analyze
from .services.analyzer import ml_engine

import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up ML Engine...")

    hf_sentiment_repo = os.getenv("HF_SENTIMENT_REPO", "GitRatBCSAD/gomi-sentiment")
    hf_risk_repo = os.getenv("HF_RISK_REPO", "GitRatBCSAD/gomi-risk")
    hf_token = os.getenv("HF_TOKEN")

    ml_engine.load_models(hf_sentiment_repo, hf_risk_repo, hf_token)

    yield

    logger.info("Shutting down ML Engine...")

app = FastAPI(
    title="Gomi ML Microservice",
    lifespan=lifespan
)

app.include_router(analyze.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
