# app/intelligent_deal_closure_automation/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
from .service import train_model_from_csv
import os

def start_scheduler():
    scheduler = BackgroundScheduler()
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "master_training_data.csv")

    def weekly_retrain():
        if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 100:
            logging.info("Running weekly auto-retrain...")
            try:
                train_model_from_csv(DATA_PATH)
                logging.info("Weekly retraining completed.")
            except Exception as e:
                logging.error(f"Weekly retrain failed: {e}")

    # Every Sunday at 2:00 AM
    scheduler.add_job(weekly_retrain, 'cron', day_of_week='sun', hour=2, minute=0)
    scheduler.start()
    logging.info("Deal closure scheduler started (weekly retrain)")