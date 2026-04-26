"""ATM extraction service.

New analysis code should import ATM extraction helpers from here.  The legacy
``analysis.pillars`` module re-exports these names during migration.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from analysis.settings import DEFAULT_ATM_BAND
from volModel.quality import validate_model_fit, warn_model_fallback

_HAS_SVI = False
_HAS_SABR = False
_HAS_TPS = False
try:
    from volModel.sviFit import fit_svi_slice as _fit_svi_slice, svi_smile_iv as _svi_iv

    _HAS_SVI = True
except Exception as exc:
    logging.getLogger(__name__).warning("SVI fitter unavailable: %s", exc)

try:
    from volModel.sabrFit import fit_sabr_slice as _fit_sabr_slice, sabr_smile_iv as _sabr_iv

    _HAS_SABR = True
except Exception:

    def _sabr_iv(S: float, K: np.ndarray | float, T: float, params) -> np.ndarray | float:
        if isinstance(K, (list, tuple, np.ndarray)):
            K = np.asarray(K, float)
            return np.full_like(K, 0.2, dtype=float)
        return 0.2

    def _fit_sabr_slice(*_a, **_k):
        raise RuntimeError("SABR fitter unavailable")

try:
    from volModel.polyFit import fit_tps_slice as _fit_tps_slice, tps_smile_iv as _tps_iv

    _HAS_TPS = True
except Exception as exc:
    logging.getLogger(__name__).warning("TPS fitter unavailable: %s", exc)


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
    mask = np.abs(k) <= band
    if mask.sum() < 3:
        mask = np.argsort(np.abs(k))[: max(3, min(7, k.size))]
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


def fit_smile_get_atm(
    g: pd.DataFrame,
    model: str = "svi",
    vega_weighted: bool = True,
    atm_band_for_poly: float = 0.25,
) -> Dict[str, float]:
    """Fit one expiry smile and return ATM vol, slope, curvature, RMSE, and model."""
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

    requested_model = str(model).lower()
    tried: list[str] = []

    def _result_from_curve(model_name: str, f, yhat: np.ndarray, n: int) -> Dict[str, float]:
        atm = f(0.0)
        sk, cu = _finite_diff(f, 0.0, 1e-3)
        rmse = float(np.sqrt(np.mean((yhat - iv) ** 2)))
        return {
            "atm_vol": float(atm),
            "skew": float(sk),
            "curv": float(cu),
            "rmse": rmse,
            "model": model_name,
            "n": int(n),
        }

    def _accept_or_warn(model_name: str, params: Dict[str, float], yhat: np.ndarray) -> Optional[Dict[str, float]]:
        quality = validate_model_fit(
            model_name,
            params,
            lambda _p: yhat,
            iv_obs=iv,
            max_abs_curv=500.0,
        )
        if quality.ok:
            return params
        warn_model_fallback(
            requested_model=requested_model,
            failed_model=model_name,
            fallback_model="next" if requested_model == "auto" else "poly2",
            message=quality.reason,
            quality=quality,
        )
        return None

    if requested_model in ("svi", "auto") and _HAS_SVI:
        tried.append("svi")
        try:
            K = mny * S
            params = _fit_svi_slice(S, np.asarray(K, float), T, np.asarray(iv, float))

            def f(kx: float) -> float:
                Kx = S * math.exp(kx)
                ivp = _svi_iv(S, np.array([Kx], float), T, params)
                return float(np.asarray(ivp, float)[0])

            yhat = np.array([f(kk) for kk in k])
            res = _result_from_curve("svi", f, yhat, len(k))
            accepted = _accept_or_warn("svi", res, yhat)
            if accepted is not None:
                return accepted
        except Exception as exc:
            warn_model_fallback(
                requested_model=requested_model,
                failed_model="svi",
                fallback_model="next" if requested_model == "auto" else "poly2",
                message=str(exc),
            )

    if requested_model in ("sabr", "auto") and _HAS_SABR:
        tried.append("sabr")
        try:
            K = mny * S
            params = _fit_sabr_slice(S=S, K=np.asarray(K, float), T=T, iv_obs=np.asarray(iv, float))

            def f(kx: float) -> float:
                Kx = S * math.exp(kx)
                ivp = _sabr_iv(S, np.array([Kx], float), T, params)
                return float(np.asarray(ivp, float)[0])

            yhat = np.array([f(kk) for kk in k])
            res = _result_from_curve("sabr", f, yhat, len(k))
            accepted = _accept_or_warn("sabr", res, yhat)
            if accepted is not None:
                return accepted
        except Exception as exc:
            warn_model_fallback(
                requested_model=requested_model,
                failed_model="sabr",
                fallback_model="next" if requested_model == "auto" else "poly2",
                message=str(exc),
            )

    if requested_model in ("tps", "auto") and _HAS_TPS:
        tried.append("tps")
        try:
            K = mny * S
            params = _fit_tps_slice(S, np.asarray(K, float), T, np.asarray(iv, float))

            def f(kx: float) -> float:
                Kx = S * math.exp(kx)
                ivp = _tps_iv(S, np.array([Kx], float), T, params)
                return float(np.asarray(ivp, float)[0])

            yhat = np.array([f(kk) for kk in k])
            res = _result_from_curve("tps", f, yhat, len(k))
            accepted = _accept_or_warn("tps", res, yhat)
            if accepted is not None:
                return accepted
        except Exception as exc:
            warn_model_fallback(
                requested_model=requested_model,
                failed_model="tps",
                fallback_model="poly2",
                message=str(exc),
            )

    res = _local_poly_fit_atm(k, iv, weights=w, band=atm_band_for_poly)
    res["n"] = int(len(k))
    if tried:
        warn_model_fallback(
            requested_model=requested_model,
            failed_model=",".join(tried),
            fallback_model=res.get("model", "poly2"),
            message="all requested model fits failed quality checks",
        )
    return res


# Compatibility alias for older callers. New code should use fit_smile_get_atm.
_fit_smile_get_atm = fit_smile_get_atm


def compute_atm_by_expiry(
    df: pd.DataFrame,
    atm_band: float = DEFAULT_ATM_BAND,
    method: str = "fit",
    model: str = "auto",
    vega_weighted: bool = True,
    min_points: int = 4,
    n_boot: int = 100,
    ci_level: float = 0.68,
) -> pd.DataFrame:
    """Compute ATM vol per expiry for a single ticker-date DataFrame."""
    need = {"T", "moneyness", "sigma"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=["T", "atm_vol", "count", "mny_min", "mny_max"])
    d = _ensure_numeric(df)
    if d.empty:
        return pd.DataFrame(columns=["T", "atm_vol", "count", "mny_min", "mny_max"])

    rows = []
    for T_val, g in d.groupby("T"):
        g = g.dropna(subset=["moneyness", "sigma"])

        if len(g) < max(3, min_points) or method == "median":
            inb = g.loc[(g["moneyness"] - 1.0).abs() <= atm_band]
            if not inb.empty:
                atm_vol = float(inb["sigma"].median())
                cnt = int(len(inb))
            else:
                i = int((g["moneyness"] - 1.0).abs().idxmin())
                atm_vol = float(g.loc[i, "sigma"])
                cnt = 1
            atm_idx = int((g["moneyness"] - 1.0).abs().idxmin())
            row = {
                "T": float(T_val),
                "atm_vol": atm_vol,
                "count": cnt,
                "mny_min": float(g["moneyness"].min()),
                "mny_max": float(g["moneyness"].max()),
                "rmse": np.nan,
                "model": "median",
                "atm_lo": np.nan,
                "atm_hi": np.nan,
                "skew": np.nan,
                "curv": np.nan,
                "spot": float(np.nanmedian(g["S"])) if "S" in g.columns else np.nan,
                "atm_strike": float(g.loc[atm_idx, "K"]) if "K" in g.columns else np.nan,
                "iv_source": "sigma",
                "extraction_status": "fallback_median",
            }
            if "expiry" in g.columns:
                row["expiry"] = g["expiry"].mode().iloc[0]
            rows.append(row)
            continue

        res = fit_smile_get_atm(g, model=model, vega_weighted=vega_weighted)

        atm_lo = atm_hi = np.nan
        if n_boot and n_boot > 0:
            rng = np.random.default_rng(42)
            boots = []
            for _ in range(int(n_boot)):
                gb = g.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1_000_000)))
                try:
                    rb = fit_smile_get_atm(gb, model=model, vega_weighted=vega_weighted)
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
            "spot": float(np.nanmedian(g["S"])) if "S" in g.columns else np.nan,
            "atm_strike": float(g.loc[int((g["moneyness"] - 1.0).abs().idxmin()), "K"])
            if "K" in g.columns
            else np.nan,
            "iv_source": "sigma",
            "extraction_status": str(res.get("model", "unknown")),
        }
        if "expiry" in g.columns:
            row["expiry"] = g["expiry"].mode().iloc[0]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def atm_curve_for_ticker_on_date(
    get_smile_slice,
    ticker: str,
    asof: str,
    **kw,
) -> pd.DataFrame:
    """Fetch a full day slice and compute its ATM-by-expiry curve."""
    df = get_smile_slice(ticker, asof, T_target_years=None)
    if df is None or df.empty:
        return pd.DataFrame(columns=["T", "atm_vol"])
    return compute_atm_by_expiry(df, **kw)[["T", "atm_vol"]].dropna().sort_values("T").reset_index(drop=True)


__all__ = [
    "fit_smile_get_atm",
    "_fit_smile_get_atm",
    "compute_atm_by_expiry",
    "atm_curve_for_ticker_on_date",
]
