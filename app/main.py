from turtle import Shape

from fastapi import FastAPI
from .schemas import AnalyzeRequest, AnalyzeResponse, SHAPBreakdown


app = FastAPI(
    title="Gomi ML Microservice",
)

# Dummy endpoint for now
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeRequest):
    print(f"Received file {request.filename} with {len(request.commits)} commits")

    # dummy data for now 

    return AnalyzeResponse(
        filename=request.filename,
        risk_score=0.75,
        shap_breakdown=SHAPBreakdown(
            base_rate = 0.4,
            sentiment_contrib = 0.25,
            complexity_contrib = 0.20,
            low_info_contrib = 0.00,
            )
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}
