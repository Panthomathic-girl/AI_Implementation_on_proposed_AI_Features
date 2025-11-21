# app/order_pattern_forecasting/order_pattern_customer_forecasting_automation/schema.py
from pydantic import BaseModel
from typing import Optional

class RetrainResponse(BaseModel):
    new_model_path: str
    total_training_records: int
    order_prediction_accuracy: float
    quantity_mae: float
    message: str