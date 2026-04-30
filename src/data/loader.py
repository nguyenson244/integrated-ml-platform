"""
Data Loader Module
Handles downloading and loading data from APIs (like Yahoo Finance) or databases/local files.
"""
import pandas as pd
import yfinance as yf
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical stock prices."""
    pass

def load_local_dataset(filename: str) -> pd.DataFrame:
    """
    Reads a local static dataset from the data/raw/ directory.
    """
    file_path = os.path.join("data", "raw", filename)
    
    try:
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Data successfully loaded from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File {filename} not found in data/raw/.")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return pd.DataFrame()
