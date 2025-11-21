# app/order_pattern_forecasting/customer_forecasting/schema.py
from pydantic import BaseModel
from typing import Dict, List

class CustomerOrder(BaseModel):
    customer_id: str
    products: Dict[str, int]  # product_id → quantity

class CustomerForecastResponse(BaseModel):
    forecast_month: str
    total_predicted_orders: int
    total_customers_expected_to_order: int
    customer_orders: List[CustomerOrder]
    generated_on: str
    message: str