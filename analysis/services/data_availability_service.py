"""Data availability and lightweight data-fetch service for GUI workflows."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import logging
import numpy as np
import pandas as pd

from data.db_utils import get_conn
from data.historical_saver import save_for_tickers
from data.interest_rates import STANDARD_DIVIDEND_YIELD, STANDARD_RISK_FREE_RATE
from data.quote_quality import (
    ANALYTICS_MAX_MONEYNESS,
    ANALYTICS_MIN_MONEYNESS,
)
from analysis.config.settings import DEFAULT_MAX_EXPIRIES, DEFAULT_UNDERLYING_LOOKBACK_DAYS


logger = logging.getLogger(__name__)
_RO_CONN = None


def _get_ro_conn():
    global _RO_CONN
    if _RO_CONN is None:
        _RO_CONN = get_conn()
    return _RO_CONN


def ingest_and_process(
    tickers: Iterable[str],
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    underlying_lookback_days: int = DEFAULT_UNDERLYING_LOOKBACK_DAYS,
) -> int:
    """Download raw chains, enrich via pipeline, and persist to DB."""
    tickers = [t.upper() for t in tickers]
    logger.info(
        "Ingesting: %s (max_expiries=%s underlying_lookback_days=%s)",
        ",".join(tickers),
        max_expiries,
        underlying_lookback_days,
    )
    return save_for_tickers(
        tickers,
        max_expiries=max_expiries,
        r=r,
        q=q,
        underlying_lookback_days=underlying_lookback_days,
    )


@lru_cache(maxsize=1)
def available_tickers() -> list[str]:
    """Unique tickers present in DB for GUI dropdowns."""
    conn = _get_ro_conn()
    return pd.read_sql_query("SELECT DISTINCT ticker FROM options_quotes ORDER BY 1", conn)["ticker"].tolist()


@lru_cache(maxsize=None)
def available_dates(ticker: Optional[str] = None, most_recent_only: bool = False) -> list[str]:
    """Get available asof_date strings globally or for a ticker."""
    conn = _get_ro_conn()

    if most_recent_only:
        from data.db_utils import get_most_recent_date

        recent = get_most_recent_date(conn, ticker)
        return [recent] if recent else []

    base = "SELECT DISTINCT asof_date FROM options_quotes"
    if ticker:
        df = pd.read_sql_query(f"{base} WHERE ticker = ? ORDER BY 1", conn, params=[ticker])
    else:
        df = pd.read_sql_query(f"{base} ORDER BY 1", conn)
    return df["asof_date"].tolist()


def get_daily_iv_for_spillover(
    tickers: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return daily ATM IV per (date, ticker) for the spillover engine."""
    conn = get_conn()
    sql = (
        "SELECT asof_date AS date, ticker, AVG(iv) AS atm_iv "
        "FROM options_quotes "
        "WHERE is_atm = 1 AND iv IS NOT NULL AND iv > 0 "
        "AND spot > 0 AND strike > 0 AND ttm_years > 0 "
        "AND moneyness BETWEEN ? AND ? "
        "AND (bid IS NULL OR ask IS NULL OR bid <= ask)"
    )
    params: list = [ANALYTICS_MIN_MONEYNESS, ANALYTICS_MAX_MONEYNESS]
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        sql += f" AND ticker IN ({placeholders})"
        params.extend([t.upper() for t in tickers])
    if start_date:
        sql += " AND asof_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND asof_date <= ?"
        params.append(end_date)
    sql += " GROUP BY asof_date, ticker ORDER BY ticker, asof_date"

    df = pd.read_sql_query(sql, conn, params=params if params else None)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_daily_hv_for_spillover(
    tickers: Optional[list[str]] = None,
    hv_window: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return daily annualized historical volatility per (date, ticker)."""
    conn = get_conn()
    sql = "SELECT asof_date AS date, ticker, close FROM underlying_prices WHERE close > 0"
    params: list = []
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        sql += f" AND ticker IN ({placeholders})"
        params.extend([t.upper() for t in tickers])
    sql += " ORDER BY ticker, asof_date"

    raw = pd.read_sql_query(sql, conn, params=params if params else None)
    conn.close()
    raw["date"] = pd.to_datetime(raw["date"])

    raw = raw.sort_values(["ticker", "date"])
    raw["log_ret"] = raw.groupby("ticker")["close"].transform(lambda s: np.log(s).diff())
    raw["atm_iv"] = raw.groupby("ticker")["log_ret"].transform(
        lambda s: s.rolling(int(hv_window), min_periods=max(5, int(hv_window) // 2)).std() * np.sqrt(252.0)
    )
    df = raw.dropna(subset=["atm_iv"])[["date", "ticker", "atm_iv"]].copy()
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


def get_most_recent_date_global() -> Optional[str]:
    """Get the most recent date across all tickers."""
    conn = _get_ro_conn()
    from data.db_utils import get_most_recent_date

    return get_most_recent_date(conn)

__all__ = [
    "available_dates",
    "available_tickers",
    "get_daily_hv_for_spillover",
    "get_daily_iv_for_spillover",
    "get_most_recent_date_global",
    "ingest_and_process",
]
