#!/usr/bin/env python3
"""Debug the underlying returns data."""

from analysis.weights.beta_builder import _underlying_log_returns
from data.db_utils import get_conn
import pandas as pd

def debug_underlying_data():
    """Check what underlying data is available."""
    print("Debugging underlying returns data...")
    
    # Check what's in the database directly
    conn = get_conn()
    
    # Check underlying_prices table
    try:
        underlying_df = pd.read_sql_query(
            "SELECT asof_date, ticker, close FROM underlying_prices ORDER BY asof_date", 
            conn
        )
        print(f"underlying_prices table: {underlying_df.shape}")
        if not underlying_df.empty:
            print(f"Date range: {underlying_df['asof_date'].min()} to {underlying_df['asof_date'].max()}")
            print(f"Tickers: {sorted(underlying_df['ticker'].unique())}")
            print(f"Sample:\n{underlying_df.tail()}")
        else:
            print("underlying_prices table is empty")
    except Exception as e:
        print(f"underlying_prices table error: {e}")
    
    # Check options_quotes fallback
    try:
        options_df = pd.read_sql_query(
            "SELECT DISTINCT asof_date, ticker FROM options_quotes ORDER BY asof_date", 
            conn
        )
        print(f"\noptions_quotes table: {options_df.shape}")
        if not options_df.empty:
            print(f"Date range: {options_df['asof_date'].min()} to {options_df['asof_date'].max()}")
            print(f"Tickers: {sorted(options_df['ticker'].unique())}")
    except Exception as e:
        print(f"options_quotes table error: {e}")
    
    # Test the actual function
    print(f"\nTesting _underlying_log_returns...")
    ret = _underlying_log_returns(get_conn)
    if ret is not None and not ret.empty:
        print(f"Returns shape: {ret.shape}")
        print(f"Date range: {ret.index.min()} to {ret.index.max()}")
        print(f"Tickers: {list(ret.columns)}")
        print(f"Sample returns:\n{ret.tail()}")
    else:
        print("Returns data is empty or None")

if __name__ == "__main__":
    debug_underlying_data()
