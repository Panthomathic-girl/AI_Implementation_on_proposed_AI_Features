# analyze_historical_monthly_volume.py
import pandas as pd
from pathlib import Path
from datetime import datetime

# 1. Load the CSV (handles broken format)
DATA_PATH = Path("data/Order_Pattern_Forecasting_dataset.csv")

if not DATA_PATH.exists():
    print("CSV not found! Put it in data/Order_Pattern_Forecasting_dataset.csv")
    exit()

try:
    df = pd.read_csv(DATA_PATH)
except:
    df = pd.read_csv(DATA_PATH, header=None)

# 2. Force correct column names
cols = ["customer_id", "product_id", "order_date", "quantity", "last_refill_date"]
if df.shape[1] == 5:
    df.columns = cols
elif "customer_id" not in df.columns:
    df.columns = cols[:df.shape[1]]

# 3. Convert dates properly
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['last_refill_date'] = pd.to_datetime(df['last_refill_date'], errors='coerce')

# 4. Filter date range: 2023-01-01 to 2024-12-29
start_date = "2023-01-01"
end_date = "2024-12-29"

df = df.dropna(subset=['order_date'])
df = df[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)]

print(f"Total orders in 2023–2024: {len(df):,} rows")

# 5. Preprocessing: Convert categorical → numerical
df['customer_id_num'] = pd.Categorical(df['customer_id']).codes
df['product_id_num'] = pd.Categorical(df['product_id']).codes

# 6. Create clean monthly bucket
df['order_month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

# 7. Group by product and month → total quantity sold
monthly_volume = (
    df.groupby(['product_id', 'order_month'])['quantity']
    .sum()
    .reset_index()
    .sort_values(['product_id', 'order_month'])
)

# 8. Pivot for easy reading
pivot = monthly_volume.pivot(index='order_month', columns='product_id', values='quantity').fillna(0)
pivot = pivot.astype(int)

# 9. Print beautiful table
print("\n" + "="*60)
print("       HISTORICAL MONTHLY ORDER VOLUME (2023–2024)")
print("="*60)
print(pivot.to_string())

# 10. Total per product
print("\n" + "-"*60)
print("TOTAL QUANTITY SOLD (2023–2024)")
print("-"*60)
totals = pivot.sum()
print(totals.to_string())

# 11. Save to CSV (optional)
pivot.to_csv("historical_monthly_volume_2023_2024.csv")
print("\nSaved to: historical_monthly_volume_2023_2024.csv")