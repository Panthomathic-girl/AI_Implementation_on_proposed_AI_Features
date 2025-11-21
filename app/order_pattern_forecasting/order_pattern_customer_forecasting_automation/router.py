# app/order_pattern_forecasting/order_pattern_customer_forecasting_automation/router.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from .service import fine_tune_with_new_csv
import shutil
from pathlib import Path

router = APIRouter(prefix="/model", tags=["Fine-Tuning"])

@router.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """
    Upload CSV → Retrain customer forecast model
    Returns exactly the response you wanted
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Only CSV files allowed")

    temp_path = Path("temp_retrain_upload.csv")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        metrics = fine_tune_with_new_csv(str(temp_path))

        # FINAL RESPONSE — EXACTLY AS YOU SPECIFIED
        return {
            "new_model_path": "app/order_pattern_forecasting/customer_forecasting/models/customer_forecast_model_v4.pkl",
            "total_training_records": metrics["total_training_records"],
            "order_prediction_accuracy": metrics["order_prediction_accuracy"],
            "quantity_mae": metrics["quantity_mae"],
            "message": "Model successfully fine-tuned and deployed!"
        }

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(500, f"Training failed: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()