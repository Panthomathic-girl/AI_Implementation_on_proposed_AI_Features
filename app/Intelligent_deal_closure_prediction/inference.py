# app/Intelligent_deal_closure_prediction/inference.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
from datetime import datetime

# === Load Model Once ===
MODEL_FILE = "medivant_deal_closure_model.pkl"
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

print("Loading Medivant Deal Closure AI Model...")
pipeline = joblib.load(MODEL_FILE)
print("Model loaded & ready for batch predictions!")

# === Router ===
router = APIRouter(
    prefix="/deal-closure-predict",
    tags=["Deal Closure Prediction (XGBoost) - Batch"]
)

# === Single Lead Input Model ===
class LeadInput(BaseModel):
    lead_id: str
    vertical: str
    territory: str
    lead_source: str
    product_stage: str
    target_price: float
    proposed_price: float
    price_discount_pct: float
    expected_order_volume: float
    expected_frequency: str
    hod_approval: int
    emails_sent: int
    emails_opened: int
    calls_made: int
    meetings_held: int
    avg_response_time_hours: float
    last_contact_age_days: int
    complaint_logged: int
    buying_trend_percent: float
    previous_orders: int
    inactive_flag: int
    overdue_payments: int
    license_expiry_days_left: int
    training_completed: int
    deal_age_days: int

# === Batch Request Model ===
class BatchPredictionRequest(BaseModel):
    leads: List[LeadInput]

# === Batch Response Item ===
class BatchPredictionResponse(BaseModel):
    lead_id: str
    prediction: str
    score: float

# === Prediction Logic ===
def predict_single(prob: float) -> str:
    return "Likely to Close" if prob >= 0.5 else "Unlikely to Close"

# === Batch Prediction Endpoint ===
@router.post(
    "/predict-batch",
    response_model=List[BatchPredictionResponse],
    summary="Predict up to 10+ leads at once",
    description="Send a list of leads and get clean, minimal predictions"
)
async def predict_batch(request: BatchPredictionRequest):
    try:
        if len(request.leads) == 0:
            raise HTTPException(status_code=400, detail="No leads provided")
        if len(request.leads) > 50:  # optional limit
            raise HTTPException(status_code=400, detail="Maximum 50 leads per request")

        # Convert to DataFrame
        df = pd.DataFrame([lead.dict() for lead in request.leads])

        # Predict probabilities
        probabilities = pipeline.predict_proba(df)[:, 1].tolist()

        # Build clean response
        results = [
            BatchPredictionResponse(
                lead_id=lead.lead_id,
                prediction=predict_single(prob),
                score=round(prob, 4)
            )
            for lead, prob in zip(request.leads, probabilities)
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Optional: Keep single prediction too
@router.post("/predict", summary="Single Lead Prediction (Legacy)")
async def predict_single_endpoint(lead: LeadInput):
    result = await predict_batch(BatchPredictionRequest(leads=[lead]))
    return result[0]