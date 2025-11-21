# app/order_pattern_forecasting/router.py
from fastapi import APIRouter, Query, HTTPException
from .service import forecast_volume_up_to_date
from .schema import VolumeForecastResponse

router = APIRouter(prefix="/order-forecast", tags=["Order Volume Forecasting"])

@router.get("/volume-by-product", response_model=VolumeForecastResponse)
async def get_forecast(
    year: int = Query(..., ge=2025, le=2030, description="Target year (e.g., 2026)"),
    month: int = Query(..., ge=1, le=12, description="Target month (1=Jan, 12=Dec)")
):
    """
    Forecast total order volume from next month after data ends → up to given year/month
    Example: year=2026&month=4 → Jan 2025 to Apr 2026
    """
    if year == 2025 and month <= 12:
        if month <= 12:
            pass  # allow Jan–Dec 2025
        else:
            raise HTTPException(400, "Month must be 1–12")

    result = forecast_volume_up_to_date(target_year=year, target_month=month)
    return result

