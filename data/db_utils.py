from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

import pandas as pd

from .db_schema import init_db
from .quote_quality import normalize_market_fields, to_float

import os

# Allow override via environment variable; fallback to the old logic
DB_PATH = os.getenv("DB_PATH", __file__.replace("db_utils.py", "iv_data.db"))


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes used by common query patterns if they don't exist."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_options_quotes_ticker ON options_quotes(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_options_quotes_asof_date ON options_quotes(asof_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_options_quotes_is_atm ON options_quotes(is_atm)")
    conn.commit()


def ensure_initialized(conn: sqlite3.Connection) -> None:
    init_db(conn)
    ensure_indexes(conn)


def check_db_health(conn: sqlite3.Connection) -> None:
    """Run a quick integrity check and raise if the database is corrupt."""
    status = conn.execute("PRAGMA quick_check").fetchone()
    if not status or status[0] != "ok":
        raise sqlite3.DatabaseError(f"Database health check failed: {status[0] if status else 'unknown'}")


def insert_quotes(conn: sqlite3.Connection, quotes: Iterable[dict]) -> int:
    rows = []
    for q in quotes:
        # Ensure dates are strings, not Timestamp objects
        asof_date = q["asof_date"]
        if hasattr(asof_date, "strftime"):  # pandas Timestamp or datetime
            asof_date = asof_date.strftime("%Y-%m-%d") if hasattr(asof_date, "strftime") else str(asof_date)

        expiry = q["expiry"]
        if hasattr(expiry, "strftime"):  # pandas Timestamp or datetime
            expiry = expiry.strftime("%Y-%m-%d") if hasattr(expiry, "strftime") else str(expiry)

        volume = q.get("volume", q.get("volume_raw"))
        bid = q.get("bid", q.get("bid_raw"))
        ask = q.get("ask", q.get("ask_raw"))
        mid = q.get("mid")
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2
        if mid is None:
            mid = q.get("last_raw")
        open_interest = q.get("open_interest", q.get("open_interest_raw"))
        bid, ask, mid, market_reason = normalize_market_fields(
            bid,
            ask,
            mid=mid,
            last=q.get("last_raw"),
        )
        if market_reason in {"negative_bid", "negative_ask", "crossed_market"}:
            continue
        volume = to_float(volume)
        open_interest = to_float(open_interest)

        rows.append(
            (
                asof_date,
                q["ticker"],
                expiry,
                float(q["K"]),
                q["call_put"],
                q.get("sigma"),
                q.get("S"),
                q.get("T"),
                q.get("moneyness"),
                q.get("log_moneyness"),
                q.get("delta"),
                1 if q.get("is_atm") else 0,
                volume,
                open_interest,
                bid,
                ask,
                mid,
                q.get("r"),
                q.get("q"),
                q.get("price"),
                q.get("gamma"),
                q.get("vega"),
                q.get("theta"),
                q.get("rho"),
                q.get("d1"),
                q.get("d2"),
                q.get("vendor", "yfinance"),
            )
        )

    if rows:
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO options_quotes (
                    asof_date, ticker, expiry, strike, call_put,
                    iv, spot, ttm_years, moneyness, log_moneyness, delta, is_atm,
                    volume, open_interest, bid, ask, mid,
                    r, q, price, gamma, vega, theta, rho, d1, d2,
                    vendor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    check_db_health(conn)
    return len(rows)


def get_most_recent_date(conn: sqlite3.Connection, ticker: Optional[str] = None) -> Optional[str]:
    """Get the most recent asof_date in the database, optionally for a specific ticker."""
    if ticker:
        sql = "SELECT MAX(asof_date) FROM options_quotes WHERE ticker = ?"
        params = [ticker]
    else:
        sql = "SELECT MAX(asof_date) FROM options_quotes"
        params = []

    result = conn.execute(sql, params).fetchone()
    return result[0] if result and result[0] else None


def fetch_quotes(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    asof_date: Optional[str] = None,
    use_most_recent: bool = True,
):
    """
    Fetch options quotes from database.

    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    ticker : Optional[str]
        Specific ticker to filter by
    asof_date : Optional[str]
        Specific date to filter by. If None and use_most_recent=True, uses most recent date
    use_most_recent : bool
        If True and asof_date is None, automatically use the most recent date
    """
    sql = "SELECT * FROM options_quotes WHERE 1=1"
    params: list = []

    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker)

    if asof_date:
        sql += " AND asof_date = ?"
        params.append(asof_date)
    elif use_most_recent:
        # Use most recent date if no specific date provided
        recent_date = get_most_recent_date(conn, ticker)
        if recent_date:
            sql += " AND asof_date = ?"
            params.append(recent_date)

    sql += " ORDER BY ticker, asof_date, expiry, strike, call_put"
    return conn.execute(sql, params).fetchall()


def fetch_vol_shifts(
    conn: sqlite3.Connection,
    tickers: Optional[Iterable[str]] = None,
    threshold: float = 0.0,
):
    """Return implied volatility changes between the two most recent dates.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    tickers : Optional[Iterable[str]]
        Specific tickers to examine. If ``None`` all distinct tickers in the
        database are considered.
    threshold : float
        Minimum absolute change in implied volatility required for a row to be
        included in the result.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns
        ``[ticker, asof_date_new, asof_date_old, expiry, strike, call_put,
        iv_new, iv_old, iv_shift]``. The DataFrame will be empty if fewer than
        two distinct dates exist for a ticker or if no shifts exceed the
        threshold.
    """

    if tickers is None:
        tickers = [  # type: ignore[assignment]
            row[0] for row in conn.execute("SELECT DISTINCT ticker FROM options_quotes")
        ]

    results: list[pd.DataFrame] = []
    for t in tickers:
        rows = conn.execute(
            "SELECT DISTINCT asof_date FROM options_quotes WHERE ticker = ? ORDER BY asof_date DESC LIMIT 2",
            (t,),
        ).fetchall()
        if len(rows) < 2:
            continue

        date_new, date_old = rows[0][0], rows[1][0]
        df_new = pd.read_sql_query(
            "SELECT expiry, strike, call_put, iv FROM options_quotes WHERE ticker = ? AND asof_date = ?",
            conn,
            params=(t, date_new),
        )
        df_old = pd.read_sql_query(
            "SELECT expiry, strike, call_put, iv FROM options_quotes WHERE ticker = ? AND asof_date = ?",
            conn,
            params=(t, date_old),
        )
        merged = df_new.merge(df_old, on=["expiry", "strike", "call_put"], suffixes=("_new", "_old"))
        if merged.empty:
            continue
        merged["iv_shift"] = merged["iv_new"] - merged["iv_old"]
        shifted = merged.loc[merged["iv_shift"].abs() > threshold].copy()
        if shifted.empty:
            continue
        shifted.insert(0, "ticker", t)
        shifted.insert(1, "asof_date_new", date_new)
        shifted.insert(2, "asof_date_old", date_old)
        results.append(shifted)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "ticker",
            "asof_date_new",
            "asof_date_old",
            "expiry",
            "strike",
            "call_put",
            "iv_new",
            "iv_old",
            "iv_shift",
        ]
    )
