# app/order_pattern_forecasting/utils.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path("data/Order_Pattern_Forecasting_dataset.csv")

def load_raw_data() -> pd.DataFrame:
    """Load CSV even if it has no header or messed up format"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    try:
        # Try with header first
        df = pd.read_csv(DATA_PATH)
    except:
        # If fails → no header
        df = pd.read_csv(DATA_PATH, header=None)

    # Force correct column names
    expected_cols = ["customer_id", "product_id", "order_date", "quantity", "last_refill_date"]
    if df.shape[1] == 5:
        df.columns = expected_cols
    elif "customer_id" not in df.columns:
        df.columns = expected_cols

    return df

def clean_and_preprocess() -> pd.DataFrame:
    """Full cleaning pipeline"""
    df = load_raw_data()

    # Convert dates
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    # df['last_refill_date'] = pd.to_datetime(df['last_refill_date'], errors='coerce')
    
    df = df.drop(columns=['last_refill_date'])

    # Drop completely invalid rows
    df = df.dropna(subset=['order_date', 'customer_id', 'product_id', 'quantity'])
    df = df[df['quantity'] > 0]

    # Convert categorical → numerical IDs (required for ML)
    df['customer_id_num'] = pd.Categorical(df['customer_id']).codes
    df['product_id_num'] = pd.Categorical(df['product_id']).codes

    # Create month period for aggregation
    df['order_month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

    # Sort chronologically
    df = df.sort_values('order_date').reset_index(drop=True)

    print(f"Data cleaned: {len(df):,} orders from {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    return df