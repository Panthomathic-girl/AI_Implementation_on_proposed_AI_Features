# app/order_pattern_forecasting/order_pattern_customer_forecasting_automation/service.py
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from ..utils import clean_and_preprocess

MODEL_DIR = Path("app/order_pattern_forecasting/customer_forecasting/models")
MODEL_DIR.mkdir(exist_ok=True)
CURRENT_MODEL = MODEL_DIR / "customer_forecast_model_v4.pkl"


def fine_tune_with_new_csv(csv_path: str) -> dict:
    """
    Fine-tune model using uploaded CSV + historical data.
    Returns ONLY the metrics you want.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting fine-tuning from: {csv_path}")

    # Load new CSV
    new_df = pd.read_csv(csv_path)
    if new_df.shape[1] == 5:
        new_df.columns = ["customer_id", "product_id", "order_date", "quantity", "last_refill_date"]
    new_df['order_date'] = pd.to_datetime(new_df['order_date'], errors='coerce')
    new_df = new_df.dropna(subset=['order_date', 'quantity', 'customer_id', 'product_id'])
    new_df = new_df[new_df['quantity'] > 0].copy()

    # Merge with historical
    base_df = clean_and_preprocess()
    df = pd.concat([base_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=['customer_id', 'product_id', 'order_date', 'quantity'])

    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month

    # Aggregate
    monthly = df.groupby(['customer_id_num', 'product_id_num', 'year', 'month'], as_index=False)['quantity'].sum()
    cust_ids = sorted(df['customer_id_num'].unique())
    prod_ids = sorted(df['product_id_num'].unique())

    # Train on 2023–2024 only
    grid = pd.DataFrame([
        (c, p, y, m)
        for c in cust_ids
        for p in prod_ids
        for y in [2023, 2024]
        for m in range(1, 13)
    ], columns=['customer_id_num', 'product_id_num', 'year', 'month'])

    full = grid.merge(monthly, on=['customer_id_num', 'product_id_num', 'year', 'month'], how='left')
    full['quantity'] = full['quantity'].fillna(0).astype(int)
    full['ordered'] = (full['quantity'] > 0).astype(int)

    X = full[['customer_id_num', 'product_id_num', 'year', 'month']]
    y_order = full['ordered']
    y_qty = full[full['ordered'] == 1]['quantity']
    X_qty = X[full['ordered'] == 1]

    # Train
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
    reg = LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
    clf.fit(X, y_order)
    reg.fit(X_qty, y_qty)

    # Evaluation (last 3 months of 2024)
    val = full[(full['year'] == 2024) & (full['month'] >= 10)]
    acc = accuracy_score(val['ordered'], clf.predict(val[['customer_id_num', 'product_id_num', 'year', 'month']])) if len(val) > 0 else 0.0
    mae = mean_absolute_error(val[val['ordered'] == 1]['quantity'], reg.predict(X_qty.iloc[-len(val[val['ordered'] == 1]):])) if (val['ordered'] == 1).sum() > 0 else 0.0

    # Save mapping from current clean data
    mapping = clean_and_preprocess()
    cust_map = dict(zip(mapping['customer_id_num'], mapping['customer_id']))
    prod_map = dict(zip(mapping['product_id_num'], mapping['product_id']))

    # Save model
    joblib.dump({
        'classifier': clf,
        'regressor': reg,
        'cust_mapping': cust_map,
        'prod_mapping': prod_map,
        'trained_on': datetime.now().isoformat(),
        'records_used': len(df)
    }, CURRENT_MODEL)

    print(f"[{datetime.now():%H:%M:%S}] Model fine-tuned and saved.")

    # Return raw metrics — formatting happens in router
    return {
        "total_training_records": len(df),
        "order_prediction_accuracy": round(acc, 4),
        "quantity_mae": round(mae, 2)
    }