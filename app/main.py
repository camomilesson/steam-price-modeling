from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import ModelLoader
from app.schemas import GameFeatures, HealthResponse, ModelInfoResponse, PricePrediction

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_loader.load()
    except Exception as exc:
        print(f"WARNING: Model failed to load at startup: {exc}")
        print("/health will report model_loaded=false; /predict and /model-info will return 503.")
    yield


app = FastAPI(
    title="Steam Price Modeling API",
    description="Serves the final XGBoost model for Steam game market-price predictions.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_loader.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(**model_loader.model_info)


@app.post("/predict", response_model=PricePrediction, tags=["inference"])
def predict(features: GameFeatures) -> PricePrediction:
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_loader.predict(features)
