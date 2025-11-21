# app/predictive_lead_score/views.py
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from .models import LeadScoreResponse
from app.predictive_lead_score.agent import lead_score_agent
router = APIRouter(prefix="/lead", tags=["Lead Scoring"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# FIXED: Added "audio/wave"
ALLOWED_MIME = {
    "audio/mp3", "audio/mpeg",
    "audio/wav", "audio/x-wav", "audio/wave",   # <-- ADDED
    "audio/flac", "audio/x-flac",
    "audio/aac",
    "audio/ogg",
    "audio/webm",
}
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".webm"}

MAX_SIZE = 20 * 1024 * 1024  # 20 MB

@router.post("/predict", response_model=LeadScoreResponse)
async def predict_lead_score(audio: UploadFile = File(...)):
    filename = audio.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if audio.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Invalid MIME: {audio.content_type}")

    content = await audio.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large (>20MB)")

    file_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        temp_path.write_bytes(content)
        result = await lead_score_agent(str(temp_path), audio.content_type)
        return LeadScoreResponse(**result, filename=filename)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass