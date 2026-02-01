from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Union
import pickle
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="MedOptix Admission Predictor")

MODEL = None
FEATURE_SCHEMA = FEATURE_SCHEMA = None

# --------------------------------------------------
# Load environment (.env at project root)
# --------------------------------------------------
from dotenv import load_dotenv
import os

# Load .env from current working directory or parents
load_dotenv()

HF_TOKEN = os.getenv("HF_token")

print("🔍 ENV CHECK — HF_token:", "FOUND" if HF_TOKEN else "MISSING")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_token not found. Ensure it is set in .env or environment variables."
    )

# --------------------------------------------------
# Startup: load model from Hugging Face (PRODUCTION)
# --------------------------------------------------
@app.on_event("startup")
def load_artifacts():
    global MODEL, FEATURE_SCHEMA

    print("🚀 Startup event triggered")
    print("🔑 HF token present:", bool(HF_TOKEN))

    try:
        model_path = hf_hub_download(
            repo_id="edabam2026/medoptix_admission_model",
            filename="sarimax_model.pkl",
            token=HF_TOKEN,
        )

        schema_path = hf_hub_download(
            repo_id="edabam2026/medoptix_admission_model",
            filename="sarimax_schema.json",
            token=HF_TOKEN,
        )

        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)

        with open(schema_path, "r") as f:
            FEATURE_SCHEMA = json.load(f)

        print("✅ Model and schema loaded successfully from Hugging Face")

    except Exception as e:
        print("❌ Model loading failed:", e)
        raise RuntimeError("Startup failed – model not available")
    

    print("🌐 Open API docs at: http://localhost:8000/docs")

# --------------------------------------------------
# Request schema
# --------------------------------------------------
#class PredictRequest(BaseModel):
 #   steps: int = Field(default=1, ge=1, description="Number of time steps to forecast")
  #  features: Dict[str, Union[int, float]] = Field(
   #     ..., description="Feature name → value mapping"
    #)


class PredictRequest(BaseModel):
    steps: int = Field(..., ge=1, description="Number of time steps to forecast (must be >=1)")
    features: Dict[str, Union[int, float]]
    scenario: Dict[str, float] = {}



# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
class PredictRequest(BaseModel):
    steps: int = Field(..., ge=1, description="Number of time steps to forecast (>=1)")
    features: Dict[str, Union[int, float]] = Field(..., description="Feature name → value mapping")
    scenario: Dict[str, float] = Field(default_factory=dict, description="Optional trend adjustments")

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(request: PredictRequest):
    if MODEL is None or FEATURE_SCHEMA is None:
        raise HTTPException(status_code=503, detail="Model or schema not loaded")

    try:
        steps = request.steps
        f = request.features
        s = request.scenario or {}

        # Simple trend adjustments if provided
        occupancy_trend = s.get("occupancy_trend", 0.0)
        overflow_trend = s.get("overflow_trend", 0.0)
        staffing_trend = s.get("staffing_trend", 0.0)

        # Build exogenous dataframe for forecasting
        exog_df = pd.DataFrame([
            {
                feature: f.get(feature, 0) + (i * occupancy_trend if "occupancy" in feature else 0)
                for feature in FEATURE_SCHEMA
            }
            for i in range(steps)
        ])

        # Optional: other trends
        if any("overflow" in feat for feat in FEATURE_SCHEMA):
            for i in range(steps):
                for col in exog_df.columns:
                    if "overflow" in col:
                        exog_df.loc[i, col] = f.get(col, 0) + i * overflow_trend
                    if "staffing_index" in col:
                        exog_df.loc[i, col] = max(0, min(1.5, f.get("staffing_index", 0) + i * staffing_trend))

        # Forecast
        forecast = MODEL.forecast(steps=steps, exog=exog_df)
        
        #predictions = [max(0, round(float(p))) for p in forecast.tolist()]

        predictions = [max(0, float(p)) for p in forecast.tolist()]

        # Check missing features
        missing_features = [feat for feat in FEATURE_SCHEMA if feat not in f]

        return {
            "predictions": predictions,
            "steps": steps,
            "features_used": list(exog_df.columns),
            "features_provided": list(f.keys()),
            "missing_features": missing_features,
            "note": "Missing features auto-filled with 0; trends applied if provided in scenario"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


