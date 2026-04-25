"""
Data pipeline:
- takes raw downloader output
- computes T, moneyness, log_moneyness
- computes BS Greeks (r,q configurable)
- flags ATM (closest to |Δ|-0.5 per date/ticker/expiry/CP)
- applies light sanity filters
- renames to DB schema fields ready for insert
- auto-fetches underlying price history for weight computation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Set, Optional
import logging

from .db_utils import get_conn
from .interest_rates import (
    STANDARD_RISK_FREE_RATE,
    STANDARD_DIVIDEND_YIELD,
    get_ticker_interest_rate,
)
from .greeks import compute_all_greeks_df
from .underlying_prices import update_underlying_prices
from .quote_quality import (
    STORE_MAX_MONEYNESS,
    STORE_MIN_MONEYNESS,
    filter_quotes,
    normalize_market_fields,
)

# Set up logging
logger = logging.getLogger(__name__)


def check_and_update_underlying_prices(
    tickers: Set[str], 
    lookback_days: int = 365,
    force_update: bool = False
) -> int:
    """
    Check underlying price data coverage and fetch missing/stale data.
    
    Args:
        tickers: Set of ticker symbols to check
        lookback_days: How many days of history we want
        force_update: If True, always fetch latest data regardless of coverage
        
    Returns:
        Number of rows updated/inserted
    """
    if not tickers:
        return 0
        
    logger.info(f"Checking underlying price coverage for {len(tickers)} tickers...")
    
    conn = get_conn()
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Check what data we already have
        existing_df = pd.read_sql_query("""
            SELECT ticker, MIN(asof_date) as earliest, MAX(asof_date) as latest, COUNT(*) as row_count
            FROM underlying_prices 
            WHERE ticker IN ({})
            GROUP BY ticker
        """.format(','.join([f"'{t}'" for t in tickers])), conn)
        
        tickers_to_update = set()
        
        if existing_df.empty or force_update:
            # No data at all, fetch everything
            tickers_to_update = tickers
            logger.info(f"No existing data found, will fetch all {len(tickers)} tickers")
        else:
            existing_tickers = set(existing_df['ticker'])
            missing_tickers = tickers - existing_tickers
            
            if missing_tickers:
                logger.info(f"Missing tickers: {sorted(missing_tickers)}")
                tickers_to_update.update(missing_tickers)
            
            # Check for stale or incomplete data
            for _, row in existing_df.iterrows():
                ticker = row['ticker']
                latest = row['latest']
                earliest = row['earliest']
                row_count = row['row_count']
                
                # Check if data is stale (more than 2 days old)
                days_stale = (datetime.now() - pd.to_datetime(latest)).days
                
                # Check if we have insufficient history (less than 80% of expected days)
                expected_days = min(lookback_days, 250)  # max ~1 trading year
                coverage_ratio = row_count / expected_days
                
                if days_stale > 2:
                    logger.info(f"Ticker {ticker} data is {days_stale} days stale (latest: {latest})")
                    tickers_to_update.add(ticker)
                elif coverage_ratio < 0.8:
                    logger.info(f"Ticker {ticker} has insufficient coverage: {row_count}/{expected_days} days ({coverage_ratio:.1%})")
                    tickers_to_update.add(ticker)
                elif earliest > cutoff_date:
                    logger.info(f"Ticker {ticker} history too short: starts {earliest}, need {cutoff_date}")
                    tickers_to_update.add(ticker)
        
        if not tickers_to_update:
            logger.info("All underlying price data is current and complete")
            return 0
        
        logger.info(f"Updating underlying prices for {len(tickers_to_update)} tickers: {sorted(tickers_to_update)}")
        
        # Fetch missing/stale data
        total_updated = update_underlying_prices(tickers_to_update, period="1y")
        
        logger.info(f"Successfully updated {total_updated} underlying price records")
        return total_updated
        
    except Exception as e:
        logger.error(f"Error checking/updating underlying prices: {e}")
        return 0


def ensure_underlying_price_data(tickers, force_update: bool = False) -> bool:
    """
    Convenience function to ensure underlying price data is available.
    
    Args:
        tickers: List or set of ticker symbols
        force_update: Force update even if data seems current
        
    Returns:
        True if data is available/updated successfully, False otherwise
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    try:
        updated_count = check_and_update_underlying_prices(
            set(str(t).upper() for t in tickers), 
            force_update=force_update
        )
        logger.info(f"Underlying data check complete. Updated {updated_count} records.")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure underlying price data: {e}")
        return False


def enrich_quotes(
    raw_df: pd.DataFrame,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    auto_update_underlying: bool = True,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return raw_df

    df = raw_df.copy()
    
    # Auto-update underlying price data for all tickers in this batch
    if auto_update_underlying and 'ticker' in df.columns:
        tickers_in_batch = set(df['ticker'].unique())
        logger.info(f"Auto-updating underlying prices for {len(tickers_in_batch)} tickers from options data")
        try:
            updated_count = check_and_update_underlying_prices(tickers_in_batch)
            if updated_count > 0:
                logger.info(f"Updated {updated_count} underlying price records")
        except Exception as e:
            logger.warning(f"Failed to auto-update underlying prices: {e}")

    # Parse dates and compute time to maturity in years
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    df["asof_date"] = pd.to_datetime(df["asof_date"], utc=True)
    df["T"] = (df["expiry"] - df["asof_date"]).dt.days / 365.25
    df = df[df["T"] > 0]
    
    # Convert timestamps back to ISO strings for database storage
    df["asof_date"] = df["asof_date"].dt.strftime('%Y-%m-%d')
    df["expiry"] = df["expiry"].dt.strftime('%Y-%m-%d')

    # Vendor IV -> sigma, spot
    df["sigma"] = df["iv_raw"]
    df["S"] = df["spot_raw"]
    df["K"] = df["strike"]

    # Moneyness and log-moneyness
    df["moneyness"] = df["K"] / df["S"]
    df["log_moneyness"] = np.log(df["moneyness"])

    # Normalize market fields before Greeks/persistence. Crossed markets are
    # removed below via the shared quality filter.
    cleaned_market = df.apply(
        lambda row: normalize_market_fields(
            row.get("bid_raw"),
            row.get("ask_raw"),
            mid=None,
            last=row.get("last_raw"),
        ),
        axis=1,
        result_type="expand",
    )
    cleaned_market.columns = ["bid_raw", "ask_raw", "mid", "_market_quality_reason"]
    for col in cleaned_market.columns:
        df[col] = cleaned_market[col]

    # Compute Greeks in bulk (adds: price, delta, gamma, vega, theta, rho, d1, d2)
    # Uses ticker-specific rates if available, falls back to provided r
    df = compute_all_greeks_df(df, r=r, q=q, use_ticker_rates=True)
    
    # Store the rates that were actually used in the calculation
    # The compute_all_greeks_df function handles ticker-specific rates internally
    df["q"] = q

    # ATM flag per (date, ticker, expiry, call_put)
    group_cols = ["asof_date", "ticker", "expiry", "call_put"]
    # distance from ATM using delta when available, otherwise moneyness
    df["_atm_dist"] = np.where(
        df["delta"].notna(),
        (df["delta"].abs() - 0.5).abs(),
        (df["moneyness"] - 1.0).abs(),
    )
    df["is_atm"] = 0
    atm_idx = df.groupby(group_cols)["_atm_dist"].idxmin()
    df.loc[atm_idx.to_numpy(), "is_atm"] = 1
    df = df.drop(columns="_atm_dist")

    # Shared store-quality filters. Analytics can apply tighter bounds later.
    df = filter_quotes(
        df,
        min_moneyness=STORE_MIN_MONEYNESS,
        max_moneyness=STORE_MAX_MONEYNESS,
        require_uncrossed=True,
    )

    # Keep expiries with enough quotes per CP
    counts = df.groupby(["ticker", "expiry", "call_put"]).size()
    valid = counts[counts >= 3].index
    df = df.set_index(["ticker", "expiry", "call_put"]).loc[valid].reset_index()

    # Map to DB column names and keep only what DB expects
    out = df.rename(
        columns={
            "strike": "K",  # ensure K exists (already set above)
        }
    )

    # final selection in DB field order (db_utils.insert expects these keys)
    cols = [
        "asof_date", "ticker", "expiry", "K", "call_put",
        "sigma", "S", "T", "moneyness", "log_moneyness", "delta", "is_atm",
        "volume_raw", "open_interest_raw", "bid_raw", "ask_raw", "mid", "last_raw",
        "r", "q", "price", "gamma", "vega", "theta", "rho", "d1", "d2",
        "vendor"
    ]
    # Some raw cols may be missing; add if needed
    for c in ["volume_raw", "open_interest_raw", "bid_raw", "ask_raw", "last_raw"]:
        if c not in out.columns:
            out[c] = None


    return out[cols]
