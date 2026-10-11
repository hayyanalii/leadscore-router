"""
main.py
FastAPI service that loads model.pkl and exposes a /predict endpoint.
Also logs every prediction to predictions_log.csv.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from logger import log_prediction   # <-- our new logger

# --- Load the trained model ONCE at startup ---
print("Loading model...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded. API ready.")

app = FastAPI(
    title="LeadScore Router",
    description="Predicts call conversion and routes leads to the right agent queue.",
    version="1.1.0",
)


class CallerData(BaseModel):
    call_duration_sec: int = Field(..., ge=0, le=7200)
    time_of_day: int = Field(..., ge=0, le=2)
    ivr_option: int = Field(..., ge=0, le=2)
    agent_experience_yrs: int = Field(..., ge=0, le=50)
    previous_calls: int = Field(..., ge=0, le=100)
    wait_time_sec: int = Field(..., ge=0, le=3600)


class PredictionResponse(BaseModel):
    conversion_probability: float
    score: int
    route: str
    message: str
    timestamp: str


@app.get("/")
def root():
    return {"service": "LeadScore Router", "status": "live", "version": "1.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(caller: CallerData):
    try:
        # Use a DataFrame with named columns — fixes the warning from before
        features = pd.DataFrame([{
            "call_duration_sec": caller.call_duration_sec,
            "time_of_day": caller.time_of_day,
            "ivr_option": caller.ivr_option,
            "agent_experience_yrs": caller.agent_experience_yrs,
            "previous_calls": caller.previous_calls,
            "wait_time_sec": caller.wait_time_sec,
        }])

        prob = float(model.predict_proba(features)[0][1])
        score = int(round(prob * 100))

        if score >= 70:
            route = "HIGH_VALUE_QUEUE"
            message = "Route to senior sales agent immediately."
        elif score >= 40:
            route = "STANDARD_QUEUE"
            message = "Route to standard sales queue."
        else:
            route = "SELF_SERVICE"
            message = "Offer self-service options to reduce agent load."

        # --- LOG the prediction ---
        log_prediction(caller.model_dump(), score, route)

        return PredictionResponse(
            conversion_probability=round(prob, 4),
            score=score,
            route=route,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")