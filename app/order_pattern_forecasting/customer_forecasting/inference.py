# app/order_pattern_forecasting/customer_forecasting/inference.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from fastapi import HTTPException
from ..utils import clean_and_preprocess

MODEL_PATH = Path("app/order_pattern_forecasting/customer_forecasting/models/customer_forecast_model_v4.pkl")

def predict_raw(year: int, month: int):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run train_model.py first.")

    pkg = joblib.load(MODEL_PATH)
    clf = pkg['classifier']
    reg = pkg['regressor']
    cust_map = pkg['cust_mapping']
    prod_map = pkg['prod_mapping']

    df = clean_and_preprocess()
    cust_ids = sorted(df['customer_id_num'].unique())
    prod_ids = sorted(df['product_id_num'].unique())

    # PREDICT FOR FUTURE (2025+) — model extrapolates
    grid = [(c, p, year, month) for c in cust_ids for p in prod_ids]
    X_pred = pd.DataFrame(grid, columns=['customer_id_num', 'product_id_num', 'year', 'month'])

    # Predict probability and quantity
    prob = clf.predict_proba(X_pred)[:, 1]
    threshold = 0.48  # Slightly lower for future months
    will_order = prob > threshold

    qty_pred = np.zeros(len(X_pred))
    if will_order.sum() > 0:
        qty_pred[will_order] = reg.predict(X_pred[will_order])

    predictions = []
    total = 0
    for i, row in X_pred.iterrows():
        qty = max(1, int(round(qty_pred[i])))  # Minimum 1 if predicted
        if qty > 0:
            total += qty
            predictions.append({
                "customer_id": cust_map.get(row['customer_id_num'], "Unknown"),
                "product_id": prod_map.get(row['product_id_num'], "Unknown"),
                "predicted_quantity": qty
            })

    return {"predictions": predictions, "total_predicted_orders": total}