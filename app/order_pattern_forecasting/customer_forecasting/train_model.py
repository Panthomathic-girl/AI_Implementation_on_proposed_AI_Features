# app/order_pattern_forecasting/customer_forecasting/train_model.py
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from lightgbm import LGBMClassifier, LGBMRegressor

# Direct import — no relative path issues
from utils import clean_and_preprocess

MODEL_DIR = Path("app/order_pattern_forecasting/customer_forecasting/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "customer_forecast_model_v4.pkl"


def train_from_csv(csv_path: str) -> str:
    """
    Train customer forecasting model using ONLY the provided CSV.
    Saves model to .pkl and prints the path.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting training from CSV: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load and clean the CSV exactly like main system
    df = pd.read_csv(csv_path)

    # Handle headerless or standard CSV
    expected_cols = ["customer_id", "product_id", "order_date", "quantity", "last_refill_date"]
    if df.shape[1] == 5:
        df.columns = expected_cols

    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df = df.dropna(subset=['order_date', 'quantity', 'customer_id', 'product_id'])
    df = df[df['quantity'] > 0].copy()

    # Apply the same preprocessing (adds customer_id_num, product_id_num)
    df = clean_and_preprocess()  # This re-runs on the whole dataset, but we override below
    # → We need to re-apply encoding from the NEW data only
    df = df.copy()  # break reference
    df['customer_id_num'] = pd.Categorical(df['customer_id']).codes
    df['product_id_num'] = pd.Categorical(df['product_id']).codes

    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month

    # Monthly aggregation
    monthly = df.groupby(['customer_id_num', 'product_id_num', 'year', 'month'], as_index=False)['quantity'].sum()

    cust_ids = sorted(df['customer_id_num'].unique())
    prod_ids = sorted(df['product_id_num'].unique())

    # Train only on real months present in this CSV
    grid = pd.DataFrame([
        (c, p, y, m)
        for c in cust_ids
        # Only customers in this CSV
        for p in prod_ids            # Only products in this CSV
        for y in df['year'].unique()
        for m in df['month'].unique()
    ], columns=['customer_id_num', 'product_id_num', 'year', 'month'])

    full = grid.merge(monthly, on=['customer_id_num', 'product_id_num', 'year', 'month'], how='left')
    full['quantity'] = full['quantity'].fillna(0).astype(int)
    full['ordered'] = (full['quantity'] > 0).astype(int)

    X = full[['customer_id_num', 'product_id_num', 'year', 'month']]
    y_order = full['ordered']
    y_qty = full[full['ordered'] == 1]['quantity']
    X_qty = X[full['ordered'] == 1]

    print("   Training classifier (will customer order?)...")
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
    clf.fit(X, y_order)

    print("   Training regressor (how much will they order?)...")
    reg = LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42, verbose=-1)
    reg.fit(X_qty, y_qty)

    # Create mapping from THIS CSV only
    cust_map = dict(zip(df['customer_id_num'], df['customer_id']))
    prod_map = dict(zip(df['product_id_num'], df['product_id']))

    # Save model
    joblib.dump({
        'classifier': clf,
        'regressor': reg,
        'cust_mapping': cust_map,
        'prod_mapping': prod_map,
        'trained_on': datetime.now().isoformat(),
        'source_csv': csv_path,
        'model_type': 'customer_forecast_v4_from_single_csv'
    }, MODEL_PATH)

    print("\n" + "="*80)
    print("MODEL TRAINING FROM CSV COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Model saved → {MODEL_PATH}")
    print(f"From CSV    → {csv_path}")
    print(f"Records     → {len(df):,}")
    print(f"Customers   → {len(cust_ids)}")
    print(f"Products    → {len(prod_ids)}")
    print("="*80)

    return str(MODEL_PATH)


# Run when executed directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <path_to_csv>")
        print("Example: python train_model.py data/my_orders_2025.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    train_from_csv(csv_file)