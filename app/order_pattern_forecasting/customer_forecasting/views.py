# app/order_pattern_forecasting/customer_forecasting/views.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from .inference import predict_raw
from .schema import CustomerForecastResponse

router = APIRouter(prefix="/customer-forecast", tags=["Customer-Level Forecasting"])

class PredictRequest(BaseModel):
    year: int
    month: int

@router.post("/predict", response_model=CustomerForecastResponse)
async def predict_customer_orders(request: PredictRequest):
    year, month = request.year, request.month
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be 1–12")

    raw = predict_raw(year, month)

    customer_dict = {}
    for p in raw["predictions"]:
        cust = p["customer_id"]
        prod = p["product_id"]
        qty = p["predicted_quantity"]
        customer_dict.setdefault(cust, {})[prod] = qty

    customer_orders = [
        {"customer_id": c, "products": p} for c, p in customer_dict.items()
    ]

    return {
        "forecast_month": f"{year}-{month:02d}",
        "total_predicted_orders": int(raw["total_predicted_orders"]),
        "total_customers_expected_to_order": len(customer_orders),
        "customer_orders": customer_orders[:100],
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"Customer forecast for {month:02d}/{year}"
    }