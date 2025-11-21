# app/order_pattern_forecasting/schema.py
from pydantic import BaseModel
from typing import Dict
from datetime import date

class ProductForecast(BaseModel):
    product_id: str
    monthly_forecast: Dict[str, int]
    total_orders_predicted: int

class VolumeForecastResponse(BaseModel):
    forecast_from: str
    forecast_to: str
    total_forecast_months: int
    generated_on: date
    model_status: str = "loaded_from_disk"  # or "trained_on_this_call"
    products: Dict[str, ProductForecast]
    grand_total_predicted_orders: int
    message: str
    
    