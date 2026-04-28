# tests/test_no_errors.py
from __future__ import annotations
import math
import os
import pandas as pd
import pytest

# Core APIs we want to defend
from analysis.weights.unified_weights import compute_unified_weights
from analysis.analysis_pipeline import (
    available_tickers,
    available_dates,
    get_smile_slice,
    build_surfaces,
    latest_relative_snapshot_corrweighted,
    sample_smile_curve,
)
from data.db_utils import get_conn

# -----------------------
# Helpers
# -----------------------

def _db_has_options_data() -> bool:
    try:
        conn = get_conn()
        n = pd.read_sql_query("SELECT COUNT(*) n FROM options_quotes", conn)["n"].iloc[0]
        return n > 0
    except Exception:
        return False

def _pick_three_tickers() -> list[str]:
    # Prefer real list from DB; fall back to common ETFs if available
    try:
        tickers = available_tickers()
    except Exception:
        tickers = []
    pref = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"]
    have = [t for t in pref if t in tickers]
    if len(have) >= 3:
        return have[:3]
    # fallback: if DB has at least 3 in any order
    if len(tickers) >= 3:
        return tickers[:3]
    # last resort: use whatever is there (tests will xfail if <2)
    return tickers

def _choose_asof_coverage(tickers: list[str]) -> str | None:
    """Pick a date with max simultaneous coverage among tickers."""
    if not tickers:
        return None
    conn = get_conn()
    q = """
      SELECT asof_date, COUNT(DISTINCT ticker) AS n
      FROM options_quotes
      WHERE ticker IN ({})
      GROUP BY asof_date
      ORDER BY n DESC, asof_date DESC
      LIMIT 1
    """.format(",".join("?" * len(tickers)))
    df = pd.read_sql_query(q, conn, params=tickers)
    if df.empty:
        return None
    return df["asof_date"].iloc[0]

def _assert_prob_series(w: pd.Series, peers: list[str]):
    assert isinstance(w, pd.Series)
    # must be defined for all peers, even if zero
    for p in peers:
        assert p in w.index
        assert math.isfinite(float(w[p]))
    total = float(w.sum())
    assert math.isfinite(total)
    # allow tiny tolerance due to rounding
    assert abs(total - 1.0) < 1e-6 or abs(total) < 1e-6  # equal-weights or all-zero edge
    # if all-zero (shouldn't happen with our fallbacks), at least not NaNs
    if abs(total) < 1e-12:
        assert (w.fillna(0.0) == 0.0).all()

# -----------------------
# Skip conditions
# -----------------------

pytestmark = pytest.mark.skipif(
    not _db_has_options_data(),
    reason="No options data in DB; skipping no-error smoke tests."
)

# -----------------------
# Weight modes (smoke)
# -----------------------

@pytest.mark.parametrize(
    "mode",
    [
        "equal",
        "oi",                   # open interest
        "corr_iv_atm",
        "corr_surface",
        "corr_surface_grid",
        "corr_ul",
        "cosine_iv_atm",
        "cosine_surface",
        "cosine_ul",
        "pca_iv_atm",
        "pca_surface",
    ],
)
def test_weight_modes_no_exceptions(mode):
    tickers = _pick_three_tickers()
    if len(tickers) < 3:
        pytest.xfail("Need >=3 tickers for the smoke test")
    target, peers = tickers[0], tickers[1:]

    # Choose a coverage-heavy as-of for option-based modes
    asof = None
    if any(key in mode for key in ["atm", "surface"]):
        asof = _choose_asof_coverage([target] + peers) or (
            available_dates(target, most_recent_only=True)[0]
            if available_dates(target, most_recent_only=True) else None
        )
        if asof is None:
            pytest.xfail("No asof date found for option-based mode")

    w = compute_unified_weights(
        target=target,
        peers=peers,
        mode=mode,
        asof=asof,
    )
    _assert_prob_series(w, peers)

# -----------------------
# Surface build (smoke)
# -----------------------

def test_build_surfaces_no_exceptions():
    tickers = _pick_three_tickers()
    if len(tickers) < 2:
        pytest.xfail("Need >=2 tickers")
    # both most_recent_only paths
    s1 = build_surfaces(tickers=tickers, most_recent_only=True)
    assert isinstance(s1, dict)
    s2 = build_surfaces(tickers=tickers, most_recent_only=False)
    assert isinstance(s2, dict)

# -----------------------
# Smile fit / sample (smoke)
# -----------------------

def test_smile_fit_and_sample_no_exceptions():
    tickers = _pick_three_tickers()
    if not tickers:
        pytest.xfail("No tickers available")
    t = tickers[0]
    dates = available_dates(t, most_recent_only=True)
    if not dates:
        pytest.xfail("No dates for smile test")
    asof = dates[0]
    # Ensure we can fetch a slice
    df = get_smile_slice(t, asof_date=asof, max_expiries=6)
    # It's okay if df is empty (skip), but the call should not raise.
    if df.empty:
        pytest.skip(f"No smile data for {t} on {asof}")
    # Sample a curve (should not raise; may return empty if sparse)
    curve = sample_smile_curve(t, asof_date=asof, T_target_years=30/365.25, model="svi")
    assert isinstance(curve, pd.DataFrame)

# -----------------------
# RV snapshot (ATM-only path)
# -----------------------

def test_latest_relative_snapshot_no_exceptions():
    tickers = _pick_three_tickers()
    if len(tickers) < 3:
        pytest.xfail("Need >=3 tickers")
    target, peers = tickers[0], tickers[1:]
    rv, w = latest_relative_snapshot_corrweighted(
        target=target,
        peers=peers,
        mode="iv_atm",
        pillar_days=(7, 30, 60, 90),
        lookback=60,
        tolerance_days=10.0,
    )
    # No exception is the main check; results may be empty on sparse data
    assert isinstance(w, pd.Series)
    assert isinstance(rv, pd.DataFrame)
    # If non-empty, ensure expected columns exist
    if not rv.empty:
        for c in ["asof_date", "pillar_days"]:
            assert c in rv.columns
