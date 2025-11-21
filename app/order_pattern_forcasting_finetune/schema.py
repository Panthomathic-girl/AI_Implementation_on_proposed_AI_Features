from pydantic import BaseModel
from typing import Dict

class FineTuneResponse(BaseModel):
    message: str
    model_path: str
    training_time_seconds: float
    evaluation_metrics: Dict[str, float]
    rows_processed: int
    model_updated_at: str