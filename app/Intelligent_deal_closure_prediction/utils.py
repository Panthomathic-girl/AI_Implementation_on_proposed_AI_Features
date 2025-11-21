# utils.py
import logging
import json
import os
import re
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def clean_column_names(df):
    """Clean column names for safe processing"""
    df.columns = [
        re.sub(r'\W+', '_', col.strip()).lower() 
        for col in df.columns
    ]
    return df

def save_json_report(data: dict, filename: str):
    """Save dictionary as pretty JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Report saved → {filename}")

def print_success_banner(model_file, accuracy):
    print("\n" + "="*70)
    print(" TRAINING SUCCESSFUL - MODEL READY!")
    print("="*70)
    print(f"Model File     : {model_file}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Location       : {os.path.abspath(model_file)}")
    print("="*70)