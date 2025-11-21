# Reuses same robust loading from main module
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/Order_Pattern_Forecasting_dataset.csv")

def save_uploaded_csv(file_bytes: bytes):
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        f.write(file_bytes)

def load_and_clean() -> pd.DataFrame:
    from app.order_pattern_forecasting.utils import clean_and_preprocess
    return clean_and_preprocess()