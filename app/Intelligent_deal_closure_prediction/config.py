# config.py
# Configuration file – Easy to change settings

RANDOM_SEED = 42
TEST_SIZE = 0.20

# Output files
MODEL_FILENAME = "medivant_deal_closure_model.pkl"
METADATA_FILENAME = "model_metadata.json"
REPORT_FILENAME = "training_report.json"

# XGBoost settings (best for tabular data)
XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "n_jobs": -1
}