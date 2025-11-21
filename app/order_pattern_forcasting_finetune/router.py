# app/order_pattern_forecasting_finetune/router.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from .service import fine_tune_model_from_csv
from .schema import FineTuneResponse
from .utils import save_uploaded_csv

router = APIRouter(prefix="/order-forecast-finetune", tags=["Model Fine-Tuning"])

@router.post("/upload-and-finetune", response_model=FineTuneResponse)
async def upload_csv_and_finetune(file: UploadFile = File(...)):
    """
    Upload latest Order_Pattern_Forecasting_dataset.csv
    → Automatically fine-tunes the current LSTM model
    → Overwrites the old .pth file with improved version
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    content = await file.read()
    save_uploaded_csv(content)

    result = fine_tune_model_from_csv()
    return result