#!/usr/bin/env python3
"""
Debug script to investigate why only 2 ATM rows are being found.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from analysis.analysis_pipeline import get_smile_slice, available_dates
from analysis.weights.correlation_utils import compute_atm_corr_pillar_free
# One-time manual fix (e.g. run in a script or Python shell)
import sqlite3
conn = sqlite3.connect("data/calculations.db")
conn.execute("DROP TABLE IF EXISTS calc_cache")
conn.commit()
conn.close()

def debug_atm_extraction():
    """Debug ATM data extraction to see why only 2 rows are found."""
    
    # Get most recent date
    dates = available_dates(ticker="SPY", most_recent_only=True)
    if not dates:
        print("No dates available")
        return
    
    asof = dates[0]
    print(f"Using date: {asof}")
    
    # Test with common tickers
    tickers = ["SPY", "QQQ", "IWM"]
    
    print(f"\nTesting tickers: {tickers}")
    print("=" * 50)
    
    # Test each ticker individually first
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        try:
            df = get_smile_slice(ticker, asof, T_target_years=None)
            if df is None or df.empty:
                print(f"  No data returned for {ticker}")
                continue
                
            print(f"  Raw data shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check for required columns
            need_cols = {"T", "moneyness", "sigma"}
            missing_cols = need_cols - set(df.columns)
            if missing_cols:
                print(f"  Missing columns: {missing_cols}")
                continue
                
            # Check data quality
            df_clean = df.dropna(subset=["T", "moneyness", "sigma"])
            print(f"  Clean data shape: {df_clean.shape}")
            
            # Check unique expiries (T values)
            unique_T = df_clean["T"].unique()
            print(f"  Unique T values: {len(unique_T)} -> {sorted(unique_T)}")
            
            # Check moneyness distribution
            mny_range = (df_clean["moneyness"].min(), df_clean["moneyness"].max())
            print(f"  Moneyness range: {mny_range}")
            
            # Check ATM candidates
            atm_candidates = df_clean[abs(df_clean["moneyness"] - 1.0) <= 0.05]
            print(f"  ATM candidates (±5%): {len(atm_candidates)}")
            
            # Show ATM extraction per expiry
            print("  ATM extraction per expiry:")
            for T_val in sorted(unique_T)[:6]:  # max_expiries=6
                grp = df_clean[df_clean["T"] == T_val]
                in_band = grp[abs(grp["moneyness"] - 1.0) <= 0.05]
                if not in_band.empty:
                    atm_vol = in_band["sigma"].median()
                    atm_count = len(in_band)
                    print(f"    T={T_val:.4f}: {atm_count} ATM options, median vol={atm_vol:.4f}")
                else:
                    # Find closest to ATM
                    closest_idx = abs(grp["moneyness"] - 1.0).idxmin()
                    closest_mny = grp.loc[closest_idx, "moneyness"]
                    closest_vol = grp.loc[closest_idx, "sigma"]
                    print(f"    T={T_val:.4f}: No ATM, closest mny={closest_mny:.4f}, vol={closest_vol:.4f}")
                    
        except Exception as e:
            print(f"  Error with {ticker}: {e}")
    
    print(f"\n{'='*50}")
    print("Now testing compute_atm_corr_pillar_free...")
    
    # Test the full function
    try:
        atm_df, corr_df = compute_atm_corr_pillar_free(
            get_smile_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=6,
            atm_band=0.05,
        )
        
        print(f"ATM DataFrame shape: {atm_df.shape}")
        print(f"ATM DataFrame index: {list(atm_df.index)}")
        print(f"ATM DataFrame columns: {list(atm_df.columns)}")
        print("\nATM DataFrame:")
        print(atm_df)
        
        print(f"\nCorrelation DataFrame shape: {corr_df.shape}")
        print("Correlation DataFrame:")
        print(corr_df)
        
        # Count non-NaN values per ticker
        print("\nNon-NaN ATM values per ticker:")
        for ticker in atm_df.index:
            non_nan_count = atm_df.loc[ticker].notna().sum()
            print(f"  {ticker}: {non_nan_count} non-NaN values")
            
    except Exception as e:
        print(f"Error in compute_atm_corr_pillar_free: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_atm_extraction()
