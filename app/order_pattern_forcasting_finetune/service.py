# app/order_pattern_forecasting_finetune/service.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
from pathlib import Path
from .utils import load_and_clean

# Use same model architecture
MODEL_PATH = Path("app/order_pattern_forecasting/lstm_volume_model.pth")

class LSTMVolumeForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def fine_tune_model_from_csv() -> dict:
    start_time = time.time()

    # Load fresh data
    df_monthly = load_and_clean()
    monthly = (
        df_monthly.groupby(['product_id', 'order_month'])['quantity']
        .sum().reset_index()
        .sort_values(['product_id', 'order_month'])
    )

    scaler = MinMaxScaler()
    sequences = []

    for product in monthly['product_id'].unique():
        data = monthly[monthly['product_id'] == product]['quantity'].values
        if len(data) < 10:
            continue
        scaled = scaler.fit_transform(data.reshape(-1, 1))
        for i in range(6, len(scaled)):
            sequences.append(scaled[i-6:i+1])

    if not sequences:
        raise ValueError("Not enough data to train (need at least 10 months per product)")

    X = torch.tensor(np.array([s[:-1] for s in sequences]), dtype=torch.float32)
    y = torch.tensor(np.array([s[-1] for s in sequences]), dtype=torch.float32)

    # Load existing model (or create new)
    model = LSTMVolumeForecaster()
    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            print("Loaded existing model for fine-tuning...")
        except:
            print("Failed to load old model → starting fresh")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower LR for fine-tuning
    loss_fn = nn.MSELoss()

    # Fine-tune for fewer epochs (fast + safe)
    for epoch in range(40):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X)
        mse = float(loss_fn(preds, y))
        mae = float(nn.L1Loss()(preds, y))

    # Save new improved model
    torch.save(model.state_dict(), MODEL_PATH)
    training_time = time.time() - start_time

    return {
        "message": "Model successfully fine-tuned and updated!",
        "model_path": str(MODEL_PATH),
        "training_time_seconds": round(training_time, 2),
        "evaluation_metrics": {
            "MSE": round(mse, 6),
            "MAE": round(mae, 2),
            "RMSE": round(mse ** 0.5, 2)
        },
        "rows_processed": len(df_monthly),
        "model_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }