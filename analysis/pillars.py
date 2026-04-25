# analysis/pillars.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Dict, List
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
# Optional model fits with graceful fallbacks
# ----------------------------
_HAS_SVI = False
_HAS_SABR = False
try:
    # from your upgraded SVI module (kept compatible via aliases)
    from volModel.sviFit import fit_svi_smile as _fit_svi_smile, svi_implied_vol as _svi_iv
    _HAS_SVI = True
except Exception:
    pass

try:
    # prefer the slice fitter and vectorized predictor
    from volModel.sabrFit import fit_sabr_slice as _fit_sabr_slice, sabr_smile_iv as _sabr_iv
    _HAS_SABR = True
except Exception:
    def _sabr_iv(S: float, K: np.ndarray | float, T: float, params) -> np.ndarray | float:
        # Extremely crude fallback: constant vol ~ alpha/F^(1-beta)
        if isinstance(K, (list, tuple, np.ndarray)):
            K = np.asarray(K, float)
            return np.full_like(K, 0.2, dtype=float)
        return 0.2
    def _fit_sabr_slice(*_a, **_k):
        raise RuntimeError("SABR fitter unavailable")

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
# Numeric helpers for ATM extraction
# ----------------------------
def _ensure_numeric(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in ("T", "sigma", "moneyness", "K", "S", "vega"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["T", "sigma", "moneyness", "K", "S"])

def _vega_weights_if_any(g: pd.DataFrame) -> Optional[np.ndarray]:
    if "vega" in g.columns:
        w = pd.to_numeric(g["vega"], errors="coerce").to_numpy()
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        if np.isfinite(w).sum() >= 3:
            w = np.nan_to_num(w, nan=np.nanmedian(w))
            s = w.sum()
            if s > 0:
                return w / s
    return None

def _local_poly_fit_atm(
    k: np.ndarray, iv: np.ndarray, weights: Optional[np.ndarray] = None, band: float = 0.25
) -> Dict[str, float]:
    # quadratic in k around 0 → f(k)=a + b k + c k^2 ; ATM=a, skew=b, curv=2c
    mask = np.abs(k) <= band
    if mask.sum() < 3:
        mask = np.argsort(np.abs(k))[:max(3, min(7, k.size))]
    x = k[mask]
    y = iv[mask]
    X = np.column_stack([np.ones_like(x), x, x**2])
    if weights is not None:
        w = weights[mask]
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b, c = beta
    rmse = float(np.sqrt(np.mean((X @ beta - y) ** 2)))
    return {"atm_vol": float(a), "skew": float(b), "curv": float(2 * c), "rmse": rmse, "model": "poly2"}

def _fit_smile_get_atm(
    g: pd.DataFrame,
    model: str = "svi",
    vega_weighted: bool = True,
    atm_band_for_poly: float = 0.25,
) -> Dict[str, float]:
    """
    Fit SVI/SABR for one expiry and return ATM vol, slope, curvature, rmse.
    Fallback to local quadratic if models unavailable or fail.
    """
    S = float(np.nanmedian(g["S"]))
    T = float(np.nanmedian(g["T"]))
    if not np.isfinite(S) or not np.isfinite(T) or T <= 0:
        return {"atm_vol": np.nan, "skew": np.nan, "curv": np.nan, "rmse": np.nan, "model": "invalid"}

    mny = g["moneyness"].to_numpy(float)
    iv = g["sigma"].to_numpy(float)
    k = np.log(np.clip(mny, 1e-12, None))
    w = _vega_weights_if_any(g) if vega_weighted else None

    def _finite_diff(f, x0: float, h: float = 1e-3):
        f_p = f(x0 + h)
        f_m = f(x0 - h)
        f0 = f(x0)
        first = (f_p - f_m) / (2 * h)
        second = (f_p - 2 * f0 + f_m) / (h * h)
        return float(first), float(second)

    use_svi = (model == "svi") or (model == "auto")
    use_sabr = (model == "sabr") or (model == "auto")

    # --- SVI path ---
    if use_svi and _HAS_SVI:
        try:
            params = _fit_svi_smile(k, iv, T)  # (a,b,rho,m,sigma)
            def f(kx: float) -> float:
                return float(_svi_iv(kx, T, *params))
            atm = f(0.0)
            sk, cu = _finite_diff(f, 0.0, 1e-3)
            yhat = np.array([f(kk) for kk in k])
            rmse = float(np.sqrt(np.mean((yhat - iv) ** 2)))
            return {"atm_vol": atm, "skew": sk, "curv": cu, "rmse": rmse, "model": "svi"}
        except Exception:
            pass

    # --- SABR path ---
    if use_sabr and _HAS_SABR:
        try:
            # Convert to strikes and fit
            K = mny * S
            params = _fit_sabr_slice(S=S, K=np.asarray(K, float), T=T, iv_obs=np.asarray(iv, float))
            def f(kx: float) -> float:
                Kx = S * math.exp(kx)
                ivp = _sabr_iv(S, np.array([Kx], float), T, params)
                return float(np.asarray(ivp, float)[0])
            atm = f(0.0)
            sk, cu = _finite_diff(f, 0.0, 1e-3)
            yhat = np.array([f(kk) for kk in k])
            rmse = float(np.sqrt(np.mean((yhat - iv) ** 2)))
            return {"atm_vol": atm, "skew": sk, "curv": cu, "rmse": rmse, "model": "sabr"}
        except Exception:
            pass

    # --- Fallback local quadratic around ATM ---
    return _local_poly_fit_atm(k, iv, weights=w, band=atm_band_for_poly)

# ----------------------------
# Public API: ATM per expiry
# ----------------------------
def compute_atm_by_expiry(
    df: pd.DataFrame,
    atm_band: float = DEFAULT_ATM_BAND,
    method: str = "fit",        # "fit" (SVI/SABR/poly) or "median"
    model: str = "auto",        # when method="fit": "svi" | "sabr" | "auto"
    vega_weighted: bool = True,
    min_points: int = 4,
    n_boot: int = 100,          # bootstrap reps per expiry (100 = default for CI)
    ci_level: float = 0.68,     # used if n_boot>0
) -> pd.DataFrame:
    """
    Compute ATM vol per expiry for a single ticker-date DataFrame.

    Expects columns: ['T','moneyness','sigma','K','S'] (optional: 'vega','expiry').

    Returns columns:
      ['T','atm_vol','count','mny_min','mny_max','rmse','model','atm_lo','atm_hi','skew','curv',('expiry')]
    """
    need = {"T", "moneyness", "sigma"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=["T", "atm_vol", "count", "mny_min", "mny_max"])
    d = _ensure_numeric(df)
    if d.empty:
        return pd.DataFrame(columns=["T", "atm_vol", "count", "mny_min", "mny_max"])

    rows = []
    for T_val, g in d.groupby("T"):
        g = g.dropna(subset=["moneyness", "sigma"])

        # Thin slice → simple ATM median/nearest
        if len(g) < max(3, min_points) or method == "median":
            inb = g.loc[(g["moneyness"] - 1.0).abs() <= atm_band]
            if not inb.empty:
                atm_vol = float(inb["sigma"].median()); cnt = int(len(inb))
            else:
                i = int((g["moneyness"] - 1.0).abs().idxmin())
                atm_vol = float(g.loc[i, "sigma"]); cnt = 1
            row = {
                "T": float(T_val), "atm_vol": atm_vol, "count": cnt,
                "mny_min": float(g["moneyness"].min()), "mny_max": float(g["moneyness"].max()),
                "rmse": np.nan, "model": "median", "atm_lo": np.nan, "atm_hi": np.nan,
                "skew": np.nan, "curv": np.nan,
            }
            if "expiry" in g.columns:
                row["expiry"] = g["expiry"].mode().iloc[0]
            rows.append(row)
            continue

        # Fit-based ATM
        res = _fit_smile_get_atm(g, model=model, vega_weighted=vega_weighted)

        # optional bootstrap CI
        atm_lo = atm_hi = np.nan
        if n_boot and n_boot > 0:
            rng = np.random.default_rng(42)
            boots = []
            for _ in range(int(n_boot)):
                gb = g.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1_000_000)))
                try:
                    rb = _fit_smile_get_atm(gb, model=model, vega_weighted=vega_weighted)
                    boots.append(rb.get("atm_vol", np.nan))
                except Exception:
                    boots.append(np.nan)
            boots = np.array([b for b in boots if np.isfinite(b)], float)
            if boots.size >= 10:
                alpha = (1.0 - float(ci_level)) / 2.0
                atm_lo = float(np.quantile(boots, alpha))
                atm_hi = float(np.quantile(boots, 1 - alpha))

        row = {
            "T": float(T_val),
            "atm_vol": float(res["atm_vol"]),
            "count": int(len(g)),
            "mny_min": float(g["moneyness"].min()),
            "mny_max": float(g["moneyness"].max()),
            "rmse": float(res.get("rmse", np.nan)),
            "model": str(res.get("model", "unknown")),
            "atm_lo": atm_lo,
            "atm_hi": atm_hi,
            "skew": float(res.get("skew", np.nan)),
            "curv": float(res.get("curv", np.nan)),
        }
        if "expiry" in g.columns:
            row["expiry"] = g["expiry"].mode().iloc[0]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

# Public tiny helper: build ATM curve for GUI/plots
def atm_curve_for_ticker_on_date(
    get_smile_slice,
    ticker: str,
    asof: str,
    **kw,
) -> pd.DataFrame:
    """
    Convenience: fetch full day slice and compute its ATM-by-expiry curve.
    """
    df = get_smile_slice(ticker, asof, T_target_years=None)
    if df is None or df.empty:
        return pd.DataFrame(columns=["T", "atm_vol"])
    return compute_atm_by_expiry(df, **kw)[["T", "atm_vol"]].dropna().sort_values("T").reset_index(drop=True)

# ----------------------------
# End of file - duplicates removed
# ----------------------------
