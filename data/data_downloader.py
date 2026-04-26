"""
Thin Yahoo Finance downloader.
- Fetches raw option chains only (no calculations, no filters).
- Returns a DataFrame with minimal raw fields + asof_date, spot.
"""
from __future__ import annotations
from collections.abc import Iterable
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

def _get_spot(tk: yf.Ticker) -> float | None:
    spot = None
    try:
        spot = tk.info.get("regularMarketPrice")
    except Exception:
        pass
    if spot is None:
        try:
            hist = tk.history(period="1d")
            if not hist.empty:
                spot = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return float(spot) if spot is not None else None


def get_available_expiries(ticker: str) -> list[str]:
    """Return the provider expiry list for a ticker without fetching chains."""
    tk = yf.Ticker(ticker)
    return [str(expiry) for expiry in (tk.options or [])]


def download_raw_option_data(
    ticker: str,
    max_expiries: int = 8,
    expiries: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    tk = yf.Ticker(ticker)
    provider_expiries = [str(expiry) for expiry in (tk.options or [])]
    if expiries is None:
        expiries_to_fetch = provider_expiries[:max_expiries]
    else:
        requested = [str(expiry) for expiry in expiries]
        available = set(provider_expiries)
        expiries_to_fetch = [expiry for expiry in requested if expiry in available]
        missing = [expiry for expiry in requested if expiry not in available]
        if missing:
            print(f"{ticker}: requested expiries not listed by provider: {missing}")

    expiries = expiries_to_fetch
    if not expiries:
        return None

    spot = _get_spot(tk)
    if spot is None:
        return None

    asof_iso = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict] = []

    for expiry in expiries[:max_expiries]:
        try:
            opt = tk.option_chain(expiry)
        except Exception:
            continue

        for df, cp in ((opt.calls, "C"), (opt.puts, "P")):
            if df is None or df.empty:
                continue
            # keep only raw vendor columns we need
            sub = df.loc[:, ["strike", "impliedVolatility", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "asof_date": asof_iso,
                        "ticker": ticker,
                        "expiry": pd.to_datetime(expiry).date().isoformat(),
                        "call_put": cp,
                        "strike": float(r["strike"]),
                        "iv_raw": None if pd.isna(r["impliedVolatility"]) else float(r["impliedVolatility"]),
                        "bid_raw": None if pd.isna(r["bid"]) else float(r["bid"]),
                        "ask_raw": None if pd.isna(r["ask"]) else float(r["ask"]),
                        "last_raw": None if pd.isna(r["lastPrice"]) else float(r["lastPrice"]),
                        "volume_raw": None if pd.isna(r["volume"]) else float(r["volume"]),
                        "open_interest_raw": None if pd.isna(r["openInterest"]) else float(r["openInterest"]),
                        "spot_raw": float(spot),
                        "vendor": "yfinance",
                    }
                )

    if not rows:
        return None
    return pd.DataFrame(rows)
