# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# FIXED: lowercase "i" in folder name
from app.predictive_lead_score import router as lead_router
from app.sentiment_analysis import router as sentiment_router
from app.intelligent_deal_closure_automation.router import router as deal_router  # ← FIXED: lowercase "i"
from app.intelligent_deal_closure_automation.scheduler import start_scheduler
from app.Intelligent_deal_closure_prediction.inference import router as prediction_router
from app.order_pattern_forecasting import router as order_pattern_forecast_router
from app.order_pattern_forcasting_finetune import router as order_pattern_finetune_router
from app.order_pattern_forecasting.customer_forecasting.views import router as customer_router
from app.order_pattern_forecasting.order_pattern_customer_forecasting_automation.router import router as order_patter_customer_retrain_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    print("Deal Closure Scheduler → Weekly auto-retrain enabled")
    yield

app = FastAPI(title="AI Sales Intelligence", docs_url="/docs", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lead_router)
app.include_router(sentiment_router)
app.include_router(deal_router)
app.include_router(prediction_router)
app.include_router(order_pattern_forecast_router)
app.include_router(order_pattern_finetune_router)
app.include_router(customer_router)
app.include_router(order_patter_customer_retrain_router)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)