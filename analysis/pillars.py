# analysis/pillars.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Dict, List
import numpy as np
import pandas as pd


from data.db_utils import get_conn
from analysis.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_EXTENDED_PILLAR_DAYS,
    DEFAULT_NEAR_TERM_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
)

DEFAULT_PILLARS_DAYS = list(DEFAULT_NEAR_TERM_PILLAR_DAYS)
# Extended pillars for when longer-term data is available
EXTENDED_PILLARS_DAYS = list(DEFAULT_EXTENDED_PILLAR_DAYS)


def detect_available_pillars(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    candidate_pillars: Iterable[int] = EXTENDED_PILLARS_DAYS,
    min_tickers_per_pillar: int = 2,
    tol_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    slices: Optional[Dict[str, "pd.DataFrame"]] = None,
) -> List[int]:
    """Detect which pillars have sufficient data across the given tickers.

    Pass ``slices`` (a dict of pre-loaded ticker DataFrames from
    ``get_smile_slices_batch``) to skip per-ticker DB queries.
    """
    candidate_pillars = list(candidate_pillars)
    pillar_coverage = {p: 0 for p in candidate_pillars}

    for ticker in tickers:
        ticker_up = ticker.upper()
        if slices is not None:
            day_df = slices.get(ticker_up, pd.DataFrame())
        else:
            day_df = get_smile_slice(ticker_up, asof, T_target_years=None)
        if day_df is None or day_df.empty:
            continue

        for pillar in candidate_pillars:
            atm_val = _atm_by_pillar_from_day_slice(day_df, pillar, tol_days=tol_days)
            if atm_val is not None and np.isfinite(atm_val):
                pillar_coverage[pillar] += 1

    good_pillars = [p for p, count in pillar_coverage.items() if count >= min_tickers_per_pillar]
    if not good_pillars:
        good_pillars = [p for p, count in pillar_coverage.items() if count > 0]
    return sorted(good_pillars)

def _atm_by_pillar_from_day_slice(day_df: pd.DataFrame, pillar_days: int,
                                  atm_band: float = DEFAULT_ATM_BAND, tol_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS) -> Optional[float]:
    """Return ATM IV for the expiry nearest to ``pillar_days`` (within ``tol_days``).

    Priority order:
      1. is_atm=1 rows (flagged at ingestion via delta proximity — most accurate).
      2. Median IV within |moneyness-1| <= atm_band.
      3. Nearest-to-ATM single row (last resort).
    """
    if day_df is None or day_df.empty:
        return None
    target_T = pillar_days / 365.25
    Tvals = pd.to_numeric(day_df["T"], errors="coerce").to_numpy()
    if Tvals.size == 0 or not np.isfinite(Tvals).any():
        return None
    j = int(np.nanargmin(np.abs(Tvals - target_T)))
    if abs(Tvals[j] - target_T) > (tol_days / 365.25):
        return None
    near = day_df.loc[np.isclose(day_df["T"], Tvals[j])].copy()
    if near.empty:
        return None

    # 1. DB-flagged ATM option (delta-based, set at ingestion)
    if "is_atm" in near.columns:
        atm_iv = pd.to_numeric(near.loc[near["is_atm"] == 1, "sigma"], errors="coerce").dropna()
        if not atm_iv.empty:
            return float(atm_iv.median())

    # 2. Moneyness band
    if "moneyness" in near.columns:
        cand = pd.to_numeric(
            near.loc[(near["moneyness"] - 1.0).abs() <= atm_band, "sigma"], errors="coerce"
        ).dropna()
        if not cand.empty:
            return float(cand.median())
        # 3. Nearest-to-ATM single row
        k = int(np.argmin(np.abs(near["moneyness"] - 1.0)))
        return float(near["sigma"].iloc[k])

    return float(pd.to_numeric(near["sigma"], errors="coerce").median())

def build_atm_matrix(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = DEFAULT_ATM_BAND,
    tol_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    min_pillars: int = 3,
    corr_method: str = "pearson",   # "pearson" | "spearman" | "kendall"
    demean_rows: bool = False,
    slices: Optional[Dict[str, "pd.DataFrame"]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (atm_df, corr_df).

    Pass ``slices`` (dict from ``get_smile_slices_batch``) to avoid N separate
    DB queries — the batch was already loaded by the caller.
    """
    tickers = [t.upper() for t in tickers]
    pillars_days = [int(p) for p in pillars_days]

    # Build ATM-by-pillar table
    rows = []
    for t in tickers:
        day_df = (
            slices.get(t, pd.DataFrame())
            if slices is not None
            else get_smile_slice(t, asof, T_target_years=None)
        )
        row = {}
        for d in pillars_days:
            row[d] = _atm_by_pillar_from_day_slice(day_df, d, atm_band=atm_band, tol_days=tol_days)
        rows.append(pd.Series(row, name=t))
    atm_df = pd.DataFrame(rows)

    # Filter rows with insufficient pillars
    valid_mask = atm_df.count(axis=1) >= int(min_pillars)
    atm_valid = atm_df.loc[valid_mask].copy()

    if atm_valid.empty or atm_valid.shape[0] < 2:
        # Not enough data to form correlations
        empty = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
        return atm_df, empty

    # Optional de-mean across pillars (per row) to remove level bias
    if demean_rows:
        atm_valid = atm_valid.sub(atm_valid.mean(axis=1), axis=0)

    # Compute correlations across pillars (pairwise, min periods = min_pillars)
    # Use min_periods=1 for more lenient correlation calculation
    corr_df = atm_valid.transpose().corr(method=corr_method, min_periods=max(1, int(min_pillars) - 1))

    # Reindex to original ticker universe (preserve shape)
    corr_df = corr_df.reindex(index=tickers, columns=tickers)

    return atm_df, corr_df

# ----------------------------
# ATM extraction compatibility exports
# ----------------------------
from analysis.atm_extraction import (
    _fit_smile_get_atm,
    atm_curve_for_ticker_on_date,
    compute_atm_by_expiry,
    fit_smile_get_atm,
)

# ----------------------------
# DB helpers (optional)
# ----------------------------
def load_atm(conn=None) -> pd.DataFrame:
    """Load rows flagged as ATM from DB (if you persist such a flag)."""
    if conn is None:
        conn = get_conn()
        should_close = True
    else:
        should_close = False
    
    try:
        # First try to get pre-flagged ATM data
        df = pd.read_sql_query("SELECT * FROM options_quotes WHERE is_atm = 1", conn)
        
        # If we don't have enough pre-flagged data, dynamically identify ATM options
        if df.empty or len(df) < 100:  # Threshold for sufficient data
            print(f"Warning: Only {len(df)} pre-flagged ATM rows found. Dynamically identifying ATM options...")
            
            # Get all options data with key columns
            all_data = pd.read_sql_query(
                "SELECT asof_date, ticker, expiry, strike, call_put, iv, spot, ttm_years, moneyness, delta, volume, bid, ask, mid, price, gamma, vega, theta, rho, d1, d2, r, q FROM options_quotes WHERE iv IS NOT NULL AND moneyness IS NOT NULL AND ttm_years IS NOT NULL", 
                conn
            )
            
            if not all_data.empty:
                # Identify ATM options (moneyness close to 1.0)
                atm_mask = (all_data['moneyness'] >= 0.95) & (all_data['moneyness'] <= 1.05)
                df = all_data[atm_mask].copy()
                print(f"Dynamically identified {len(df)} ATM options (moneyness 0.95-1.05)")
        
        if not df.empty:
            # Rename columns to match expected interface
            if 'ttm_years' in df.columns and 'T' not in df.columns:
                df = df.rename(columns={'ttm_years': 'T'})
        
        return df
    finally:
        if should_close:
            try:
                conn.close()
            except Exception:
                pass

def nearest_pillars(
    df: pd.DataFrame,
    pillars_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """
    From a DataFrame that includes ['asof_date','ticker','T', ...],
    pick, per (asof,ticker) and per pillar day D, the nearest expiry to D (in days),
    keeping it only if within tolerance_days.
    """
    need = {"asof_date", "ticker", "T"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=list(df.columns) + ["pillar_days", "within_tol"])

    d = df.copy()
    d["ttm_days"] = pd.to_numeric(d["T"], errors="coerce") * 365.25
    d = d.dropna(subset=["ttm_days"])
    out_rows = []

    pillars_days = [int(p) for p in pillars_days]
    for (asof, ticker), g in d.groupby(["asof_date", "ticker"]):
        tt = g["ttm_days"].to_numpy()
        if tt.size == 0:
            continue
        for P in pillars_days:
            j = int(np.argmin(np.abs(tt - P)))
            row = g.iloc[j].to_dict()
            row["pillar_days"] = P
            row["pillar_diff_days"] = float(tt[j] - P)  # Add the difference column
            row["within_tol"] = bool(abs(float(tt[j]) - P) <= float(tolerance_days))
            out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    # keep only within tolerance by default
    return out.loc[out["within_tol"]].reset_index(drop=True)

# ----------------------------
# End of file - duplicates removed
# ----------------------------
