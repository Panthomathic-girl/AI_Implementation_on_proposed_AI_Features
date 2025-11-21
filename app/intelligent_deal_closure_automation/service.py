# app/intelligent_deal_closure_automation/service.py
import pandas as pd
import joblib
import os
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# ────────────────────── PATHS ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # ← This is your project root (where main.py is)

# THIS IS THE EXACT FILE YOUR PREDICTION API USES — IN ROOT FOLDER
TARGET_MODEL_PATH = os.path.join(ROOT_DIR, "medivant_deal_closure_model.pkl")

# Optional: Keep history inside automation folder
HISTORY_DIR = os.path.join(BASE_DIR, "models", "history")
DATA_PATH = os.path.join(BASE_DIR, "data", "master_training_data.csv")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────────── AUTO PREPROCESS ──────────────────────
def auto_preprocess(df: pd.DataFrame):
    df = df.copy()
    target = next((col for col in ["class", "deal_closed", "won", "target"] if col in df.columns), None)
    if not target:
        raise ValueError("Target column not found (class, deal_closed, won, target)")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Drop ID columns
    X = X.drop(columns=[c for c in X.columns if "id" in c.lower()], errors="ignore")

    # Encode categoricals
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].fillna("missing")
        X[col] = pd.factorize(X[col])[0]

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y

# ────────────────────── TRAIN & REPLACE ROOT MODEL ──────────────────────
def train_model_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    logger.info(f"Training on {len(df)} rows from {csv_path}")

    X, y = auto_preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
        "f1_score": round(f1_score(y_test, preds, average='weighted'), 4),
        "total_samples": len(df),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 1. Save to history (optional backup)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(HISTORY_DIR, f"model_backup_{timestamp}.pkl")
    joblib.dump(model, backup_path)

    # 2. OVERWRITE THE MAIN MODEL IN ROOT FOLDER
    joblib.dump(model, TARGET_MODEL_PATH)
    logger.info(f"SUCCESS: Model overwritten → {TARGET_MODEL_PATH}")

    # 3. Append training data
    header = not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0
    df.to_csv(DATA_PATH, mode='a', header=header, index=False)

    return {
        "status": "Model retrained & deployed",
        "model_path": TARGET_MODEL_PATH,
        "metrics": metrics,
        "backup_saved": backup_path
    }