# router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from .service import train_model_from_csv
import shutil, os

router = APIRouter(prefix="/deal-closure", tags=["AI Deal Closure"])

@router.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files allowed")
    
    temp = f"temp_{file.filename}"
    with open(temp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        result = train_model_from_csv(temp)
        return result
    finally:
        if os.path.exists(temp):
            os.remove(temp)