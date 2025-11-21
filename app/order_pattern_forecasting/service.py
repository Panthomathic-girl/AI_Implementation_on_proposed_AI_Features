# app/order_pattern_forecasting/service.py
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from .utils import get_monthly_volume_by_product, get_product_list

MODEL_PATH = Path("app/order_pattern_forecasting/lstm_volume_model.pth")
scaler = MinMaxScaler()

class LSTMVolumeForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def get_or_train_model():
    """
    Returns a ready-to-use model.
    - If .pth exists → load it (fast)
    - If not → train once and save
    Called fresh on every API request.
    """
    model = LSTMVolumeForecaster()

    # Step 1: Try to load existing model
    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] LSTM model loaded from {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Model corrupted ({e}). Retraining...")

    # Step 2: No valid model → train from scratch (only happens once)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] No valid model found. Training LSTM from scratch...")

    monthly = get_monthly_volume_by_product()
    sequences = []

    for product in get_product_list():
        data = monthly[monthly['product_id'] == product]['quantity'].values
        if len(data) < 10:
            continue
        scaled = scaler.fit_transform(data.reshape(-1, 1))
        for i in range(6, len(scaled)):
            sequences.append(scaled[i-6:i+1])

    if not sequences:
        print("Warning: Not enough data to train LSTM. Using dummy model.")
        # Still save a dummy model so future calls are fast
        torch.save(model.state_dict(), MODEL_PATH)
        model.eval()
        return model

    X = torch.tensor(np.array([s[:-1] for s in sequences]), dtype=torch.float32)
    y = torch.tensor(np.array([s[-1] for s in sequences]), dtype=torch.float32)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("   Training in progress...")
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print(f"   → Epoch {epoch:3d} | Loss: {loss.item():.6f}")

    # Save so next call is instant
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model trained and saved → {MODEL_PATH}")

    return model

def forecast_volume_up_to_date(target_year: int, target_month: int) -> dict:
    # This is the only entry point — fresh model per call, but fast if .pth exists
    model = get_or_train_model()

    monthly = get_monthly_volume_by_product()
    last_historical_date = monthly['order_month'].max()
    start_forecast_date = last_historical_date + pd.DateOffset(months=1)

    current = pd.Timestamp(year=start_forecast_date.year, month=start_forecast_date.month, day=1)
    target = pd.Timestamp(year=target_year, month=target_month, day=1)
    forecast_months = []
    while current <= target:
        forecast_months.append(current)
        current += pd.DateOffset(months=1)

    total_months = len(forecast_months)

    result = {
        "forecast_from": start_forecast_date.strftime("%Y-%m"),
        "forecast_to": target.strftime("%Y-%m"),
        "total_forecast_months": total_months,
        "generated_on": datetime.now().date(),
        "model_status": "loaded_from_disk" if MODEL_PATH.exists() else "trained_on_this_call",
        "products": {},
        "grand_total_predicted_orders": 0
    }

    for product in get_product_list():
        prod_data = monthly[monthly['product_id'] == product].sort_values('order_month')
        values = prod_data['quantity'].values.reshape(-1, 1)

        if len(values) < 6:
            avg = int(values.mean()) if len(values) > 0 else 150
            forecast = {m.strftime("%Y-%m"): avg for m in forecast_months}
            total = avg * total_months
        else:
            scaled = scaler.fit_transform(values)
            seq = torch.tensor(scaled[-6:], dtype=torch.float32).unsqueeze(0)
            preds = []
            current_seq = seq
            model.eval()
            with torch.no_grad():
                for _ in range(total_months):
                    pred = model(current_seq)
                    pred_val = float(scaler.inverse_transform(pred.numpy())[0][0])
                    preds.append(int(round(pred_val)))
                    current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(0)], dim=1)

            forecast = {m.strftime("%Y-%m"): p for m, p in zip(forecast_months, preds)}
            total = sum(preds)

        result["products"][product] = {
            "product_id": product,
            "monthly_forecast": forecast,
            "total_orders_predicted": total
        }
        result["grand_total_predicted_orders"] += total

    result["message"] = f"Forecast from {result['forecast_from']} to {result['forecast_to']} (LSTM • {'cached' if MODEL_PATH.exists() else 'fresh'})"
    return result