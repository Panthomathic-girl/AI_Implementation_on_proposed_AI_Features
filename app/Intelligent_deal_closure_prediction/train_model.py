# train_model.py
# FINAL – 100% WORKING – NO MORE CLASS IN FEATURES

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

from config import RANDOM_SEED, TEST_SIZE, MODEL_FILENAME, REPORT_FILENAME, XGB_PARAMS
from utils import setup_logging, clean_column_names, save_json_report, print_success_banner

logger = setup_logging()

def main():
    logger.info("Starting Deal Closure AI Training")

    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        logger.error("No CSV found!")
        return
    csv_path = csv_files[0]
    logger.info(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Data loaded: {df.shape}")

    df = clean_column_names(df)

    # Drop IDs
    id_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', 'lead_id'])]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        logger.info(f"Dropped: {id_cols}")

    # FIND AND REMOVE TARGET COLUMN EARLY
    target_col = None
    for col in df.columns:
        if col.lower() in ['class', 'target', 'label', 'outcome', 'won', 'status']:
            target_col = col
            break

    if not target_col:
        logger.error("Target column 'class' not found!")
        return

    logger.info(f"Target column: {target_col}")
    logger.info(f"Distribution:\n{df[target_col].value_counts()}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])  # class is GONE from features

    logger.info(f"Final features: {X.shape[1]} columns")

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include='number').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='drop')

    model = XGBClassifier(**XGB_PARAMS)
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify)

    logger.info("Training...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete!")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # Save model
    joblib.dump(pipeline, MODEL_FILENAME)
    logger.info(f"Model saved → {MODEL_FILENAME}")

    # Save report
    report = {
        "model": MODEL_FILENAME,
        "target": target_col,
        "accuracy": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "date": datetime.now().isoformat()
    }
    save_json_report(report, REPORT_FILENAME)

    print_success_banner(MODEL_FILENAME, acc)

if __name__ == "__main__":
    main()