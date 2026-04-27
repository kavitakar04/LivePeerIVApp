"""Utilities for fetching and storing underlying price history."""

from __future__ import annotations

from typing import Iterable
import pandas as pd
import yfinance as yf

from .db_utils import get_conn, ensure_initialized


def _fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download historical daily close prices for ``ticker`` from yfinance."""
    tk = yf.Ticker(ticker)
    try:
        hist = tk.history(period=period)
    except Exception:
        return pd.DataFrame()
    if hist.empty or "Close" not in hist.columns:
        return pd.DataFrame()
    df = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "asof_date", "Close": "close"})
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date.astype(str)
    df["ticker"] = ticker.upper()
    return df[["asof_date", "ticker", "close"]]


def update_underlying_prices(tickers: Iterable[str], period: str = "1y") -> int:
    """Fetch and upsert historical prices for ``tickers`` into the DB.

    Returns the total number of rows inserted.
    """
    conn = get_conn()
    ensure_initialized(conn)

    total = 0
    for t in tickers:
        df = _fetch_history(t, period=period)
        if df.empty:
            continue
        rows = [tuple(x) for x in df.itertuples(index=False, name=None)]
        conn.executemany(
            """
            INSERT OR REPLACE INTO underlying_prices(asof_date, ticker, close)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        total += len(rows)
    return total


def get_available_tickers():
    """Get all tickers that have options data."""
    conn = get_conn()
    df = pd.read_sql_query("SELECT DISTINCT ticker FROM options_quotes", conn)
    return sorted(df["ticker"].tolist())


def update_all_underlying_prices():
    """Update underlying prices for all available tickers."""
    print("Getting available tickers...")
    tickers = get_available_tickers()
    print(f"Found {len(tickers)} tickers: {tickers}")

    print(f"\nFetching 1 year of historical data for {len(tickers)} tickers...")
    print("This may take a few minutes due to API rate limits...")

    total_rows = update_underlying_prices(tickers, period="1y")
    print(f"\nSuccessfully updated {total_rows} price records")

    # Check the result
    conn = get_conn()
    result_df = pd.read_sql_query(
        "SELECT COUNT(*) as total_rows, COUNT(DISTINCT ticker) as unique_tickers, "
        "MIN(asof_date) as earliest, MAX(asof_date) as latest FROM underlying_prices",
        conn,
    )
    print("\nDatabase now contains:")
    print(f"  Total rows: {result_df['total_rows'].iloc[0]}")
    print(f"  Unique tickers: {result_df['unique_tickers'].iloc[0]}")
    print(f"  Date range: {result_df['earliest'].iloc[0]} to {result_df['latest'].iloc[0]}")


if __name__ == "__main__":
    update_all_underlying_prices()


__all__ = ["update_underlying_prices"]
