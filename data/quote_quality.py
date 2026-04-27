"""Shared quote quality rules for raw options data and analytics reads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STORE_MIN_MONEYNESS = 0.10
STORE_MAX_MONEYNESS = 5.00
ANALYTICS_MIN_MONEYNESS = 0.20
ANALYTICS_MAX_MONEYNESS = 3.00
MIN_IV = 0.01
MAX_IV = 2.00


def to_float(value: Any) -> float | None:
    """Convert numeric-ish values to float, treating non-finite as missing."""
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def normalize_market_fields(
    bid: Any,
    ask: Any,
    mid: Any = None,
    last: Any = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Return clean bid/ask/mid and a quality reason when the quote is unusable.

    If both bid and ask are present, they must be non-negative and uncrossed.
    The midpoint is always recomputed from valid bid/ask. When bid/ask are not
    available, ``mid`` falls back to a supplied mid or last trade.
    """

    bid_f = to_float(bid)
    ask_f = to_float(ask)
    mid_f = to_float(mid)
    last_f = to_float(last)

    if bid_f is not None and bid_f < 0:
        return bid_f, ask_f, mid_f, "negative_bid"
    if ask_f is not None and ask_f < 0:
        return bid_f, ask_f, mid_f, "negative_ask"
    if bid_f is not None and ask_f is not None:
        if bid_f > ask_f:
            return bid_f, ask_f, (bid_f + ask_f) / 2.0, "crossed_market"
        return bid_f, ask_f, (bid_f + ask_f) / 2.0, None
    return bid_f, ask_f, mid_f if mid_f is not None else last_f, None


def _series(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def quality_mask(
    df: pd.DataFrame,
    *,
    min_moneyness: float = ANALYTICS_MIN_MONEYNESS,
    max_moneyness: float = ANALYTICS_MAX_MONEYNESS,
    min_iv: float = MIN_IV,
    max_iv: float = MAX_IV,
    require_uncrossed: bool = True,
) -> pd.Series:
    """Boolean mask for option rows safe enough for modeling/plotting."""

    mask = pd.Series(True, index=df.index)
    iv = _series(df, "sigma", "iv", "iv_raw")
    spot = _series(df, "S", "spot", "spot_raw")
    strike = _series(df, "K", "strike")
    ttm = _series(df, "T", "ttm_years")
    mny = _series(df, "moneyness")

    if iv is not None:
        mask &= iv.notna() & (iv > min_iv) & (iv < max_iv)
    if spot is not None:
        mask &= spot.notna() & (spot > 0)
    if strike is not None:
        mask &= strike.notna() & (strike > 0)
    if ttm is not None:
        mask &= ttm.notna() & (ttm > 0)
    if mny is not None:
        mask &= mny.notna() & (mny > min_moneyness) & (mny < max_moneyness)

    if require_uncrossed:
        bid = _series(df, "bid", "bid_raw")
        ask = _series(df, "ask", "ask_raw")
        if bid is not None:
            mask &= bid.isna() | (bid >= 0)
        if ask is not None:
            mask &= ask.isna() | (ask >= 0)
        if bid is not None and ask is not None:
            both = bid.notna() & ask.notna()
            mask &= ~both | (bid <= ask)

    return mask.fillna(False)


def filter_quotes(
    df: pd.DataFrame,
    *,
    min_moneyness: float = ANALYTICS_MIN_MONEYNESS,
    max_moneyness: float = ANALYTICS_MAX_MONEYNESS,
    require_uncrossed: bool = True,
) -> pd.DataFrame:
    """Return rows passing the shared quality mask."""

    if df is None or df.empty:
        return df
    return df.loc[
        quality_mask(
            df,
            min_moneyness=min_moneyness,
            max_moneyness=max_moneyness,
            require_uncrossed=require_uncrossed,
        )
    ].copy()
