"""Smile data preparation service.

Owns the implementations of smile-slice fetching, fitting, and composite
payload building.  ``analysis.analysis_pipeline`` re-exports these names for
backward compatibility while the pipeline is being split.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import logging
import numpy as np
import pandas as pd

from data.db_utils import get_conn
from data.quote_quality import (
    ANALYTICS_MAX_MONEYNESS,
    ANALYTICS_MIN_MONEYNESS,
    filter_quotes,
)
from volModel.volModel import VolModel

from analysis.peer_composite_builder import build_surface_grids, combine_surfaces
from analysis.model_fit_service import fit_model_params, quality_checked_result
from analysis.model_params_logger import append_params, load_model_params
from analysis.atm_extraction import fit_smile_get_atm
from analysis.settings import DEFAULT_CI, DEFAULT_MAX_EXPIRIES

logger = logging.getLogger(__name__)


def get_smile_slice(
    ticker: str,
    asof_date: Optional[str] = None,
    T_target_years: float | None = None,
    call_put: Optional[str] = None,
    nearest_by: str = "T",
    max_expiries: Optional[int] = None,
) -> pd.DataFrame:
    """Return a slice of quotes for plotting a smile (one date, one ticker).

    If asof_date is None, uses the most recent date for the ticker.
    If T_target_years is given, returns the nearest expiry; otherwise returns all expiries.
    """
    conn = get_conn()
    ticker = ticker.upper()

    if asof_date is None:
        from data.db_utils import get_most_recent_date
        asof_date = get_most_recent_date(conn, ticker)
        if asof_date is None:
            return pd.DataFrame()

    q = """
        SELECT asof_date, ticker, expiry, call_put, strike AS K, spot AS S, ttm_years AS T,
               moneyness, iv AS sigma, delta, is_atm
        FROM options_quotes
        WHERE asof_date = ? AND ticker = ?
    """
    df = pd.read_sql_query(q, conn, params=[asof_date, ticker])
    if df.empty:
        return df
    df = filter_quotes(
        df,
        min_moneyness=ANALYTICS_MIN_MONEYNESS,
        max_moneyness=ANALYTICS_MAX_MONEYNESS,
        require_uncrossed=True,
    )
    if df.empty:
        return df

    if call_put in ("C", "P"):
        df = df[df["call_put"] == call_put]

    if max_expiries is not None and max_expiries > 0 and not df.empty:
        unique_expiries = df.groupby("expiry")["T"].first().sort_values()
        limited_expiries = unique_expiries.head(max_expiries).index.tolist()
        df = df[df["expiry"].isin(limited_expiries)]

    if T_target_years is not None and not df.empty:
        abs_diff = (df["T"] - float(T_target_years)).abs()
        nearest_mask = abs_diff.groupby(df["expiry"]).transform("min") == abs_diff
        df = df[nearest_mask]
        if df["expiry"].nunique() > 1:
            first_expiry = df.groupby("expiry").size().sort_values(ascending=False).index[0]
            df = df[df["expiry"] == first_expiry]

    return df.sort_values(["call_put", "T", "moneyness", "K"]).reset_index(drop=True)


def get_smile_slices_batch(
    tickers: list[str],
    asof_date: str,
    max_expiries: Optional[int] = None,
    call_put: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Load ALL tickers' option quotes for one date in a single SQL query.

    Replaces N calls to get_smile_slice with one round-trip.
    """
    tickers_up = [t.upper() for t in tickers]
    if not tickers_up or not asof_date:
        return {t: pd.DataFrame() for t in tickers_up}

    conn = get_conn()
    ph = ",".join("?" * len(tickers_up))
    q = f"""
        SELECT asof_date, ticker, expiry, call_put, strike AS K, spot AS S, ttm_years AS T,
               moneyness, iv AS sigma, delta, is_atm
        FROM options_quotes
        WHERE asof_date = ? AND ticker IN ({ph})
    """
    df = pd.read_sql_query(q, conn, params=[asof_date] + tickers_up)
    conn.close()

    if not df.empty:
        df = filter_quotes(
            df,
            min_moneyness=ANALYTICS_MIN_MONEYNESS,
            max_moneyness=ANALYTICS_MAX_MONEYNESS,
            require_uncrossed=True,
        )

    result: dict[str, pd.DataFrame] = {}
    for t in tickers_up:
        if df.empty:
            result[t] = pd.DataFrame()
            continue
        tdf = df[df["ticker"] == t].copy()
        if tdf.empty:
            result[t] = pd.DataFrame()
            continue
        if call_put in ("C", "P"):
            tdf = tdf[tdf["call_put"] == call_put]
        if max_expiries is not None and max_expiries > 0 and not tdf.empty:
            unique_expiries = tdf.groupby("expiry")["T"].first().sort_values()
            limited_expiries = unique_expiries.head(max_expiries).index.tolist()
            tdf = tdf[tdf["expiry"].isin(limited_expiries)]
        result[t] = tdf.sort_values(["call_put", "T", "moneyness", "K"]).reset_index(drop=True)

    return result


def prepare_smile_data(
    target: str,
    asof: str,
    T_days: float,
    model: str = "svi",
    ci: float = DEFAULT_CI * 100.0,
    overlay_synth: bool = False,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    overlay_peers: bool = False,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
) -> Dict[str, Any]:
    """Precompute smile data and fitted parameters for plotting."""
    peers = list(peers or [])

    asof_ts = pd.to_datetime(asof).normalize()
    try:
        params_cache = load_model_params()
        params_cache = params_cache[
            (params_cache["ticker"] == target)
            & (params_cache["asof_date"] == asof_ts)
            & (params_cache["model"].isin(["svi", "sabr", "tps", "sens"]))
        ]
    except Exception:
        params_cache = pd.DataFrame()

    df = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df is None or df.empty:
        return {}

    T_arr = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
    K_arr = pd.to_numeric(df["K"], errors="coerce").to_numpy(float)
    sigma_arr = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(float)
    S_arr = pd.to_numeric(df["S"], errors="coerce").to_numpy(float)
    expiry_arr = pd.to_datetime(df.get("expiry"), errors="coerce").to_numpy()
    cp_arr = df["call_put"].to_numpy() if "call_put" in df.columns else None

    Ts = np.sort(np.unique(T_arr[np.isfinite(T_arr)]))
    if Ts.size == 0:
        return {}
    idx0 = int(np.argmin(np.abs(Ts * 365.25 - float(T_days))))
    T0 = float(Ts[idx0])

    fit_by_expiry: Dict[float, Dict[str, Any]] = {}
    for T_val in Ts:
        mask = np.isclose(T_arr, T_val)
        if not np.any(mask):
            tol = 1e-6
            mask = (T_arr >= T_val - tol) & (T_arr <= T_val + tol)
        if not np.any(mask):
            continue

        S = float(np.nanmedian(S_arr[mask])) if np.any(mask) else float("nan")
        K = K_arr[mask]
        IV = sigma_arr[mask]

        expiry_dt = None
        if expiry_arr.size and np.any(mask):
            try:
                expiry_dt = pd.to_datetime(expiry_arr[mask][0])
            except Exception:
                expiry_dt = None

        tenor_d = None
        if expiry_dt is not None:
            try:
                tenor_d = int((expiry_dt - asof_ts).days)
            except Exception:
                tenor_d = None
        if tenor_d is None:
            tenor_d = int(round(float(T_val) * 365.25))

        def _cached(model: str) -> Optional[Dict[str, float]]:
            if params_cache.empty:
                return None
            sub = params_cache[
                (params_cache["tenor_d"] == tenor_d)
                & (params_cache["model"] == model)
            ]
            if sub.empty:
                return None
            return sub.set_index("param")["value"].to_dict()

        quality_map: Dict[str, Dict[str, Any]] = {}
        fallback_map = {"svi": "none", "sabr": "none", "tps": "none"}

        svi_params = _cached("svi")
        if not svi_params:
            svi_params, quality_map["svi"] = quality_checked_result(
                "svi", fit_model_params("svi", S, K, T_val, IV), S, K, T_val, IV,
            )
            try:
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                if svi_params:
                    append_params(asof, target, exp_str, "svi", svi_params, meta={"rmse": svi_params.get("rmse")})
            except Exception:
                pass

        sabr_params = _cached("sabr")
        if not sabr_params:
            sabr_params, quality_map["sabr"] = quality_checked_result(
                "sabr", fit_model_params("sabr", S, K, T_val, IV), S, K, T_val, IV,
            )
            try:
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                if sabr_params:
                    append_params(asof, target, exp_str, "sabr", sabr_params, meta={"rmse": sabr_params.get("rmse")})
            except Exception:
                pass

        tps_params = _cached("tps")
        if not tps_params:
            try:
                tps_params, quality_map["tps"] = quality_checked_result(
                    "tps", fit_model_params("tps", S, K, T_val, IV), S, K, T_val, IV,
                )
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                if tps_params:
                    append_params(asof, target, exp_str, "tps", tps_params, meta={"rmse": tps_params.get("rmse")})
            except Exception:
                tps_params = {}
                quality_map["tps"] = {
                    "ok": False, "reason": "fit failed", "rmse": np.nan,
                    "min_iv": np.nan, "max_iv": np.nan, "n": int(np.isfinite(IV).sum()),
                }

        sens_params = _cached("sens")
        if not sens_params:
            dfe = df[mask].copy()
            try:
                dfe["moneyness"] = dfe["K"].astype(float) / float(S)
            except Exception:
                dfe["moneyness"] = np.nan
            sens = fit_smile_get_atm(dfe, model="auto")
            sens_params = {k: sens[k] for k in ("atm_vol", "skew", "curv") if k in sens}
            try:
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                append_params(asof, target, exp_str, "sens", sens_params)
            except Exception:
                pass

        fit_by_expiry[T_val] = {
            "svi": svi_params,
            "sabr": sabr_params,
            "tps": tps_params,
            "sens": sens_params,
            "expiry": str(expiry_dt) if expiry_dt is not None else None,
            "quality": quality_map,
            "fallback": fallback_map,
        }

    fit_entry = fit_by_expiry.get(T0, {})
    fit_info = {
        "ticker": target,
        "asof": asof,
        "expiry": fit_entry.get("expiry"),
        "svi": fit_entry.get("svi", {}),
        "sabr": fit_entry.get("sabr", {}),
        "tps": fit_entry.get("tps", {}),
        "sens": fit_entry.get("sens", {}),
    }

    tgt_surface = None
    syn_surface = None
    if peers:
        try:
            tickers = list({target, *peers})
            surfaces = build_surface_grids(
                tickers=tickers,
                use_atm_only=False,
                max_expiries=max_expiries,
            )
            asof_ts = pd.Timestamp(asof).normalize()
            if target in surfaces and asof_ts in surfaces[target]:
                tgt_surface = surfaces[target][asof_ts]
            peer_surfaces = {p: surfaces[p] for p in peers if p in surfaces}
            if peer_surfaces:
                w = {p: float(weights.get(p, 1.0)) for p in peer_surfaces} if weights else {p: 1.0 for p in peer_surfaces}
                synth_by_date = combine_surfaces(peer_surfaces, w)
                syn_surface = synth_by_date.get(asof_ts)
        except Exception:
            tgt_surface = None
            syn_surface = None

    peer_slices: Dict[str, Dict[str, np.ndarray]] = {}
    if (overlay_peers or overlay_synth) and peers:
        for p in peers:
            df_p = get_smile_slice(p, asof, T_target_years=None, max_expiries=max_expiries)
            if df_p is None or df_p.empty:
                continue
            T_p = pd.to_numeric(df_p["T"], errors="coerce").to_numpy(float)
            K_p = pd.to_numeric(df_p["K"], errors="coerce").to_numpy(float)
            sigma_p = pd.to_numeric(df_p["sigma"], errors="coerce").to_numpy(float)
            S_p = pd.to_numeric(df_p["S"], errors="coerce").to_numpy(float)
            peer_slices[p.upper()] = {"T_arr": T_p, "K_arr": K_p, "sigma_arr": sigma_p, "S_arr": S_p}

    return {
        "T_arr": T_arr,
        "K_arr": K_arr,
        "sigma_arr": sigma_arr,
        "S_arr": S_arr,
        "cp_arr": cp_arr,
        "Ts": Ts,
        "idx0": idx0,
        "tgt_surface": tgt_surface,
        "syn_surface": syn_surface,
        "peer_slices": peer_slices,
        "expiry_arr": expiry_arr,
        "fit_info": fit_info,
        "fit_by_expiry": fit_by_expiry,
    }


def fit_smile_for(
    ticker: str,
    asof_date: Optional[str] = None,
    model: str = "svi",
    min_quotes_per_expiry: int = 3,
    beta: float = 0.5,
    max_expiries: Optional[int] = None,
) -> VolModel:
    """Fit a volatility smile model for one day/ticker using all available expiries."""
    conn = get_conn()
    ticker = ticker.upper()

    if asof_date is None:
        from data.db_utils import get_most_recent_date
        asof_date = get_most_recent_date(conn, ticker)
        if asof_date is None:
            return VolModel(model=model)

    q = """
        SELECT spot AS S, strike AS K, ttm_years AS T, iv AS sigma
        FROM options_quotes
        WHERE asof_date = ? AND ticker = ?
    """
    df = pd.read_sql_query(q, conn, params=[asof_date, ticker])
    if df.empty:
        return VolModel(model=model)

    S = float(df["S"].median())
    df = df.dropna(subset=["K", "T", "sigma"]).copy()
    if df.empty:
        return VolModel(model=model)

    if max_expiries is not None and max_expiries > 0:
        unique_T = df.groupby("T")["T"].first().sort_values()
        limited_T = unique_T.head(max_expiries).values
        df = df[df["T"].isin(limited_T)]
        if df.empty:
            return VolModel(model=model)

    counts = df.groupby("T").size()
    good_T = counts[counts >= max(1, int(min_quotes_per_expiry))].index
    df = df[df["T"].isin(good_T)]
    if df.empty:
        return VolModel(model=model)

    Ks = df["K"].to_numpy()
    Ts = df["T"].to_numpy()
    IVs = df["sigma"].to_numpy()

    return VolModel(model=model).fit(S, Ks, Ts, IVs, beta=beta)


def sample_smile_curve(
    ticker: str,
    asof_date: Optional[str] = None,
    T_target_years: float = 30 / 365.25,
    model: str = "svi",
    moneyness_grid: tuple[float, float, int] = (0.6, 1.4, 81),
    beta: float = 0.5,
    max_expiries: Optional[int] = None,
) -> pd.DataFrame:
    """Fit a smile then return a tidy curve at the nearest expiry to T_target_years."""
    ticker = ticker.upper()
    actual_date = asof_date
    if actual_date is None:
        conn = get_conn()
        from data.db_utils import get_most_recent_date
        actual_date = get_most_recent_date(conn, ticker)

    vm = fit_smile_for(ticker, asof_date, model=model, beta=beta, max_expiries=max_expiries)
    if not vm.available_expiries() or vm.S is None:
        return pd.DataFrame(columns=["asof_date", "ticker", "model", "T_used", "moneyness", "K", "IV"])

    Ts = np.array(vm.available_expiries(), dtype=float)
    T_used = float(Ts[np.argmin(np.abs(Ts - float(T_target_years)))])

    lo, hi, n = moneyness_grid
    m_grid = np.linspace(float(lo), float(hi), int(n))
    K_grid = m_grid * vm.S
    iv = vm.smile(K_grid, T_used)

    return pd.DataFrame(
        {
            "asof_date": actual_date,
            "ticker": ticker,
            "model": model.upper(),
            "T_used": T_used,
            "moneyness": m_grid,
            "K": K_grid,
            "IV": iv,
        }
    )


__all__ = [
    "get_smile_slice",
    "get_smile_slices_batch",
    "prepare_smile_data",
    "sample_smile_curve",
    "fit_smile_for",
]
