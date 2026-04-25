# analysis/analysis_pipeline.py
"""
GUI-ready analysis orchestrator.

This module wires together:
- ingest + enrich (data.historical_saver)
- surface grid building
- synthetic ETF surface construction (surface & ATM pillars)
- vol betas (UL, IV-ATM, Surface)
- lightweight snapshot/smile helpers for GUI

All public functions are fast to call from a GUI. Heavy work is cached
in-memory and can optionally be dumped to disk (parquet) for fast reloads.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List, Mapping, Union

import json
import logging

import numpy as np
import pandas as pd
import os
from data.historical_saver import save_for_tickers
from data.db_utils import get_conn
from data.interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD
from data.data_pipeline import enrich_quotes
from data.quote_quality import (
    ANALYTICS_MAX_MONEYNESS,
    ANALYTICS_MIN_MONEYNESS,
    filter_quotes,
)
from volModel.volModel import VolModel

from .syntheticETFBuilder import (
    build_surface_grids,
    DEFAULT_TENORS,
    DEFAULT_MNY_BINS,
    combine_surfaces,
    build_synthetic_iv as build_synthetic_iv_pillars,
)

from .beta_builder import (
    peer_weights_from_correlations,
    build_vol_betas,
    save_correlations,
)
from .pillars import load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS, _fit_smile_get_atm, compute_atm_by_expiry, DEFAULT_PILLARS_DAYS, atm_curve_for_ticker_on_date
from .correlation_utils import (
    compute_atm_corr_pillar_free,
    corr_weights,
)
from volModel.sviFit import fit_svi_slice
from volModel.sabrFit import fit_sabr_slice
from .model_params_logger import append_params, load_model_params
from .confidence_bands import synthetic_etf_pillar_bands
from .settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CI,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
    DEFAULT_RV_LOOKBACK_DAYS,
)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Global connection cache
# -----------------------------------------------------------------------------
_RO_CONN = None

# -----------------------------------------------------------------------------
# Config (GUI friendly)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for surface building and caching (GUI friendly)."""
    tenors: Tuple[int, ...] = DEFAULT_TENORS
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS
    pillar_days: Tuple[int, ...] = tuple(DEFAULT_PILLARS_DAYS)
    use_atm_only: bool = False
    max_expiries: Optional[int] = None  # Limit number of expiries in smiles/surfaces
    cache_dir: str = "data/cache"       # Optional disk cache for GUI speed

    def ensure_cache_dir(self) -> None:
        """Ensure the on-disk cache directory exists."""
        p = Path(self.cache_dir)
        if self.cache_dir and not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------
def ingest_and_process(
    tickers: Iterable[str],
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> int:
    """Download raw chains, enrich via pipeline, and persist to DB."""
    tickers = [t.upper() for t in tickers]
    logger.info("Ingesting: %s (max_expiries=%s)", ",".join(tickers), max_expiries)
    return save_for_tickers(tickers, max_expiries=max_expiries, r=r, q=q)

# -----------------------------------------------------------------------------
# Surfaces (for GUI)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def get_surface_grids_cached(
    cfg: PipelineConfig,
    tickers_key: str,  # lru_cache needs hashables → pass a joined string of tickers
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    tickers = tickers_key.split(",") if tickers_key else None
    logger.debug("Building surface grids (cached), tickers=%s", tickers_key)
    return build_surface_grids(
        tickers=tickers,
        tenors=cfg.tenors,
        mny_bins=cfg.mny_bins,
        use_atm_only=cfg.use_atm_only,
        max_expiries=cfg.max_expiries,
    )


def build_surfaces(
    tickers: Iterable[str] | None = None,
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = False,
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Return dict[ticker][date] -> IV grid DataFrame.

    Parameters
    ----------
    tickers : Iterable[str] | None
        List of tickers to build surfaces for.
    cfg : PipelineConfig
        Configuration for surface building.
    most_recent_only : bool
        If True, only return surfaces for each ticker's most recent date
        (prefers global most recent if present for that ticker).
    """
    key = ",".join(sorted([t.upper() for t in tickers])) if tickers else ""
    all_surfaces = get_surface_grids_cached(cfg, key)

    if most_recent_only and all_surfaces:
        # Prefer the most recent global date; fall back to each ticker's last date
        most_recent = get_most_recent_date_global()
        filtered: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
        if most_recent:
            most_recent_ts = pd.to_datetime(most_recent)
            for ticker, date_dict in all_surfaces.items():
                if most_recent_ts in date_dict:
                    filtered[ticker] = {most_recent_ts: date_dict[most_recent_ts]}
                else:
                    if date_dict:
                        latest = max(date_dict.keys())
                        filtered[ticker] = {latest: date_dict[latest]}
            return filtered
    return all_surfaces


def list_surface_dates(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]]
) -> List[pd.Timestamp]:
    """All unique dates available across tickers (sorted)."""
    dates: set[pd.Timestamp] = set()
    for dct in surfaces.values():
        dates.update(dct.keys())
    return sorted(dates)


def surface_to_frame_for_date(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Extract the grid for a single date across tickers."""
    return {t: dct[date] for t, dct in surfaces.items() if date in dct}

# -----------------------------------------------------------------------------
# Synthetic ETF (surface & ATM pillars)
# -----------------------------------------------------------------------------
def build_synthetic_surface(
    weights: Mapping[str, float],
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = True,  # default to True for performance
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Create a synthetic ETF surface from ticker grids + weights."""
    w = {k.upper(): float(v) for k, v in weights.items()}
    surfaces = build_surfaces(tickers=list(w.keys()), cfg=cfg, most_recent_only=most_recent_only)
    return combine_surfaces(surfaces, w)


def build_synthetic_iv_series(
    weights: Mapping[str, float],
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Create a weighted ATM pillar IV time series."""
    w = {k.upper(): float(v) for k, v in weights.items()}
    return build_synthetic_iv_pillars(w, pillar_days=pillar_days, tolerance_days=tolerance_days)

# -----------------------------------------------------------------------------
# Weights (modern unified first, legacy fallback kept)
# -----------------------------------------------------------------------------
def compute_peer_weights(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_iv_atm",
    asof: str | None = None,
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tenor_days: Iterable[int] = DEFAULT_TENORS,
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS,
) -> pd.Series:
    """Compute portfolio weights via unified weight computation."""
    target = target.upper()
    peers = [p.upper() for p in peers]

    from analysis.unified_weights import compute_unified_weights

    return compute_unified_weights(
        target=target,
        peers=peers,
        mode=weight_mode,
        asof=asof,
        pillars_days=pillar_days,
        tenors=tenor_days,
        mny_bins=mny_bins,
    )


# -----------------------------------------------------------------------------
# Betas
# -----------------------------------------------------------------------------
def build_synthetic_surface_corrweighted(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_iv_atm",
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = True,
    asof: str | None = None,
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], pd.Series]:
    """Build synthetic surface where peer weights derive from correlation/PCA metrics."""
    w = compute_peer_weights(
        target=target,
        peers=peers,
        weight_mode=weight_mode,
        asof=asof,
        pillar_days=cfg.pillar_days,
        tenor_days=cfg.tenors,
        mny_bins=cfg.mny_bins,
    )
    surfaces = build_surfaces(tickers=list(w.index), cfg=cfg, most_recent_only=most_recent_only)
    synth = combine_surfaces(surfaces, w.to_dict())
    return synth, w


def build_synthetic_iv_series_corrweighted(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_iv_atm",
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLARS_DAYS,
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    asof: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build correlation/PCA-weighted synthetic ATM pillar IV series."""
    w = compute_peer_weights(
        target=target,
        peers=peers,
        weight_mode=weight_mode,
        asof=asof,
        pillar_days=pillar_days,
    )
    df = build_synthetic_iv_pillars(w.to_dict(), pillar_days=pillar_days, tolerance_days=tolerance_days)
    return df, w


def compute_betas(
    mode: str,  # 'ul' | 'iv_atm' | 'surface'
    benchmark: str,
    cfg: PipelineConfig = PipelineConfig(),
):
    """Compute vol betas per requested mode (GUI can show table/heatmap)."""
    if mode in ("surface", "surface_grid"):
        return build_vol_betas(
            mode=mode, benchmark=benchmark,
            tenor_days=cfg.tenors, mny_bins=cfg.mny_bins
        )
    if mode == "iv_atm":
        return build_vol_betas(mode=mode, benchmark=benchmark, pillar_days=cfg.pillar_days)
    return build_vol_betas(mode=mode, benchmark=benchmark)


def save_betas(
    mode: str,
    benchmark: str,
    base_path: str = "data",
    cfg: PipelineConfig = PipelineConfig(),
) -> list[str]:
    """Persist betas to Parquet files."""
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    if mode == "surface":
        res = build_vol_betas(
            mode=mode, benchmark=benchmark,
            tenor_days=cfg.tenors, mny_bins=cfg.mny_bins
        )
        filename = f"betas_{mode}_vs_{benchmark}.parquet"
        p = os.path.join(base_path, filename)
        df = res.sort_index().to_frame(name="beta").reset_index().rename(columns={"index": "ticker"})
        df.to_parquet(p, index=False)
        return [p]
    return save_correlations(mode=mode, benchmark=benchmark, base_path=base_path)
# =========================
# Relative value (target vs synthetic peers by corr)
# -----------------------------------------------------------------------------
def _fetch_target_atm(
    target: str,
    pillar_days: Iterable[int],
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS
) -> pd.DataFrame:
    atm = load_atm()
    atm = atm[atm["ticker"] == target].copy()
    if atm.empty:
        return pd.DataFrame(columns=["asof_date", "pillar_days", "iv"])
    piv = nearest_pillars(atm, pillars_days=list(pillar_days), tolerance_days=tolerance_days)
    out = piv.groupby(["asof_date", "pillar_days"])["iv"].mean().rename("iv").reset_index()
    return out[["asof_date", "pillar_days", "iv"]]


def _rv_metrics_join(target_iv: pd.DataFrame, synth_iv: pd.DataFrame, lookback: int = DEFAULT_RV_LOOKBACK_DAYS) -> pd.DataFrame:
    tgt = target_iv.rename(columns={"iv": "iv_target"})
    syn = synth_iv.rename(columns={"iv": "iv_synth"})
    df = pd.merge(tgt, syn, on=["asof_date", "pillar_days"], how="inner").sort_values(["pillar_days", "asof_date"])
    if df.empty:
        return df

    def per_pillar(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["spread"] = g["iv_target"] - g["iv_synth"]
        roll = max(5, int(lookback // 5))
        m = g["spread"].rolling(lookback, min_periods=roll).mean()
        s = g["spread"].rolling(lookback, min_periods=roll).std(ddof=1)
        g["z"] = (g["spread"] - m) / s
        # percentile rank of latest value within window
        def _pct_rank(x: pd.Series) -> float:
            return x.rank(pct=True).iloc[-1]
        g["pct_rank"] = g["spread"].rolling(lookback, min_periods=roll).apply(_pct_rank, raw=False)
        return g

    return df.groupby("pillar_days", group_keys=False).apply(per_pillar, include_groups=False)


def relative_value_atm_report_corrweighted(
    target: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",                # 'ul' | 'iv_atm' | 'surface'  (weights source)
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    lookback: int = DEFAULT_RV_LOOKBACK_DAYS,
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    weight_power: float = 1.0,
    clip_negative: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute relative value (spread/z/pct) using correlation-based peer weights.
    Returns (rv_dataframe, weights_used).
    """
    w = peer_weights_from_correlations(
        benchmark=target,
        peers=peers,
        mode=mode,
        pillar_days=pillar_days,
        tenor_days=DEFAULT_TENORS,
        mny_bins=DEFAULT_MNY_BINS,
        clip_negative=clip_negative,
        power=weight_power,
    )
    if w.empty:
        empty_cols = ["asof_date", "pillar_days", "iv_target", "iv_synth", "spread", "z", "pct_rank"]
        return pd.DataFrame(columns=empty_cols), w

    synth = build_synthetic_iv_series(weights=w.to_dict(), pillar_days=pillar_days, tolerance_days=tolerance_days)
    tgt = _fetch_target_atm(target, pillar_days=pillar_days, tolerance_days=tolerance_days)
    rv = _rv_metrics_join(tgt, synth, lookback=lookback)
    return rv, w


def latest_relative_snapshot_corrweighted(
    target: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",
    pillar_days: Iterable[int] = DEFAULT_PILLAR_DAYS,
    lookback: int = DEFAULT_RV_LOOKBACK_DAYS,
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience: last date per pillar with RV metrics and the weights used.
    """
    rv, w = relative_value_atm_report_corrweighted(
        target=target, peers=peers, mode=mode, pillar_days=pillar_days,
        lookback=lookback, tolerance_days=tolerance_days, **kwargs
    )
    # If nothing came back, just return early
    if rv.empty:
        return rv, w

    # Ensure required columns exist even if upstream fallback omitted them
    # - pillar_days: create a placeholder if missing (single bucket)
    # - asof_date: ensure it exists and is sortable
    if "pillar_days" not in rv.columns:
        # treat whole snapshot as a single synthetic pillar bucket (e.g., 0)
        rv = rv.copy()
        rv["pillar_days"] = 0
    if "asof_date" not in rv.columns:
        # cannot manufacture real dates; create a constant so sort/group don't explode
        rv = rv.copy()
        rv["asof_date"] = pd.Timestamp("1970-01-01")

    # Keep only columns we know how to present; tolerate missing metric columns
    base_cols = ["asof_date", "pillar_days", "iv_target", "iv_synth", "spread", "z", "pct_rank"]
    present_cols = [c for c in base_cols if c in rv.columns]

    # Group safely by pillar and take the latest per pillar
    last = rv.sort_values("asof_date").groupby("pillar_days", dropna=False).tail(1)
    # Guarantee pillar_days in the projected set
    proj_cols = sorted(set(["pillar_days", "asof_date"] + present_cols))
    return last[proj_cols].sort_values("pillar_days"), w

# -----------------------------------------------------------------------------
# Basic DB helpers + caches
# -----------------------------------------------------------------------------
def _get_ro_conn():
    global _RO_CONN
    if _RO_CONN is None:
        _RO_CONN = get_conn()
    return _RO_CONN


@lru_cache(maxsize=8)
def get_atm_pillars_cached() -> pd.DataFrame:
    """Tidy ATM rows from DB (asof_date, ticker, expiry, ttm_years, iv, spot, moneyness, delta, pillar_days, ...)"""
    return load_atm()


@lru_cache(maxsize=1)
def available_tickers() -> List[str]:
    """Unique tickers present in DB (for GUI dropdowns)."""
    conn = _get_ro_conn()
    return pd.read_sql_query(
        "SELECT DISTINCT ticker FROM options_quotes ORDER BY 1", conn
    )["ticker"].tolist()


@lru_cache(maxsize=None)
def available_dates(ticker: Optional[str] = None, most_recent_only: bool = False) -> List[str]:
    """Get available asof_date strings (globally or for a ticker)."""
    conn = _get_ro_conn()

    if most_recent_only:
        from data.db_utils import get_most_recent_date
        recent = get_most_recent_date(conn, ticker)
        return [recent] if recent else []

    base = "SELECT DISTINCT asof_date FROM options_quotes"
    if ticker:
        df = pd.read_sql_query(
            f"{base} WHERE ticker = ? ORDER BY 1", conn, params=[ticker]
        )
    else:
        df = pd.read_sql_query(f"{base} ORDER BY 1", conn)
    return df["asof_date"].tolist()


def get_daily_iv_for_spillover(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return daily ATM IV per (date, ticker) for the spillover engine.

    Queries the options_quotes table directly — no parquet file needed.
    Returns a DataFrame with columns: date (datetime), ticker (str), atm_iv (float).
    """
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
    tickers: Optional[List[str]] = None,
    hv_window: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Return daily annualised historical volatility per (date, ticker).

    Computes HV as the rolling standard deviation of log returns scaled by
    sqrt(252).  Returns a DataFrame with columns: date (datetime), ticker (str),
    atm_iv (float) — named ``atm_iv`` so the spillover engine works unchanged.
    """
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
    raw["log_ret"] = raw.groupby("ticker")["close"].transform(
        lambda s: np.log(s).diff()
    )
    raw["atm_iv"] = (
        raw.groupby("ticker")["log_ret"]
        .transform(lambda s: s.rolling(hv_window, min_periods=max(2, hv_window // 2)).std())
        * np.sqrt(252)
    )

    df = raw.dropna(subset=["atm_iv"])[["date", "ticker", "atm_iv"]].copy()
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


def invalidate_cache() -> None:
    """Clear cached ticker/date queries and reset shared connection."""
    clear_all_caches()


def invalidate_config_caches() -> None:
    """Clear only configuration-dependent caches (lighter operation)."""
    clear_config_dependent_caches()


def get_most_recent_date_global() -> Optional[str]:
    """Get the most recent date across all tickers."""
    conn = _get_ro_conn()
    from data.db_utils import get_most_recent_date
    return get_most_recent_date(conn)

# -----------------------------------------------------------------------------
# Smile helpers (GUI plotting)
# -----------------------------------------------------------------------------
def get_smile_slice(
    ticker: str,
    asof_date: Optional[str] = None,
    T_target_years: float | None = None,
    call_put: Optional[str] = None,  # 'C' or 'P' or None for both
    nearest_by: str = "T",           # 'T' or 'moneyness' (reserved; not used here)
    max_expiries: Optional[int] = None,  # Limit number of expiries
) -> pd.DataFrame:
    """
    Return a slice of quotes for plotting a smile (one date, one ticker).
    If asof_date is None, uses the most recent date for the ticker.
    If T_target_years is given, returns the nearest expiry; otherwise returns all expiries that day.
    Optionally filter by call_put ('C'/'P').
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

    # Optionally limit number of expiries (smallest T first)
    if max_expiries is not None and max_expiries > 0 and not df.empty:
        unique_expiries = df.groupby("expiry")["T"].first().sort_values()
        limited_expiries = unique_expiries.head(max_expiries).index.tolist()
        df = df[df["expiry"].isin(limited_expiries)]

    if T_target_years is not None and not df.empty:
        # pick nearest expiry to T_target_years
        abs_diff = (df["T"] - float(T_target_years)).abs()
        nearest_mask = abs_diff.groupby(df["expiry"]).transform("min") == abs_diff
        df = df[nearest_mask]
        # if multiple expiries tie, keep the largest bucket (most rows)
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

    Replaces N calls to get_smile_slice with one round-trip.  The returned dict
    maps upper-cased ticker → filtered DataFrame (same schema as get_smile_slice).
    Missing or empty tickers map to an empty DataFrame.
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
    """Precompute smile data and fitted parameters for plotting.

    Returns a dictionary with raw quote arrays, prebuilt target and synthetic
    surfaces when peers are supplied, peer slices, and a ``fit_info`` mapping
    suitable for parameter summaries.
    """
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

        svi_params = _cached("svi")
        if not svi_params:
            svi_params = fit_svi_slice(S, K, T_val, IV)
            try:
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                append_params(asof, target, exp_str, "svi", svi_params, meta={"rmse": svi_params.get("rmse")})
            except Exception:
                pass

        sabr_params = _cached("sabr")
        if not sabr_params:
            sabr_params = fit_sabr_slice(S, K, T_val, IV)
            try:
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                append_params(asof, target, exp_str, "sabr", sabr_params, meta={"rmse": sabr_params.get("rmse")})
            except Exception:
                pass

        tps_params = _cached("tps")
        if not tps_params:
            try:
                from volModel.polyFit import fit_tps_slice
                tps_params = fit_tps_slice(S, K, T_val, IV)
                exp_str = str(expiry_dt) if expiry_dt is not None else None
                append_params(asof, target, exp_str, "tps", tps_params, meta={"rmse": tps_params.get("rmse")})
            except Exception:
                tps_params = {}

        sens_params = _cached("sens")
        if not sens_params:
            dfe = df[mask].copy()
            try:
                dfe["moneyness"] = dfe["K"].astype(float) / float(S)
            except Exception:
                dfe["moneyness"] = np.nan
            sens = _fit_smile_get_atm(dfe, model="auto")
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
            # build_surface_grids stores keys as pd.Timestamp; normalise to match.
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
    if overlay_peers and peers:
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

def prepare_term_data(
    target: str,
    asof: str,
    ci: float = DEFAULT_CI * 100.0,
    overlay_synth: bool = False,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    atm_band: float = DEFAULT_ATM_BAND,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
) -> Dict[str, Any]:
    """Precompute ATM term structure and synthetic overlay data.

    Returns a dictionary with ``atm_curve`` and ``synth_curve`` DataFrames ready
    for plotting. When peers are supplied, the synthetic curve is always
    constructed, regardless of whether it will be rendered.
    """

    df_all = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df_all is None or df_all.empty:
        return {}

    min_boot = 64 if (ci and ci > 0) else 0
    atm_curve = compute_atm_by_expiry(
        df_all,
        atm_band=atm_band,
        method="fit",
        model="auto",
        vega_weighted=True,
        n_boot=min_boot,
        ci_level=ci,
    )

    synth_curve = None
    synth_bands = None
    peer_curves: Dict[str, pd.DataFrame] = {}
    weight_series = pd.Series(dtype=float)

    if peers:
        w = pd.Series(weights if weights else {p: 1.0 for p in peers}, dtype=float)
        if w.sum() <= 0:
            w = pd.Series({p: 1.0 for p in peers}, dtype=float)
        w = (w / w.sum()).astype(float)
        weight_series = w.copy()
        peers = [p for p in w.index if p in peers]

        curves: Dict[str, pd.DataFrame] = {}
        for p in peers:
            c = atm_curve_for_ticker_on_date(
                get_smile_slice,
                p,
                asof,
                atm_band=atm_band,
                method="median",
                model="auto",
                vega_weighted=False,
            )
            if not c.empty:
                curves[p] = c
        peer_curves = curves

        if curves:
            # Determine common expiries across target and peers
            tol_years = 10.0 / 365.25
            arrays = [atm_curve["T"].to_numpy(float)] + [c["T"].to_numpy(float) for c in curves.values()]
            common_T = arrays[0]
            for arr in arrays[1:]:
                common_T = np.array([t for t in common_T if np.any(np.abs(arr - t) <= tol_years)], float)
                if common_T.size == 0:
                    break

            if common_T.size > 0:
                common_T = np.sort(common_T)
                # Filter target curve to common expiries
                atm_curve = atm_curve[
                    atm_curve["T"].apply(lambda x: np.any(np.abs(common_T - x) <= tol_years))
                ]

                # Build per-peer ATM arrays aligned to common_T
                atm_data: Dict[str, np.ndarray] = {}
                for p, c in curves.items():
                    arr_T = c["T"].to_numpy(float)
                    arr_v = c["atm_vol"].to_numpy(float)
                    vals = []
                    for t in common_T:
                        j = int(np.argmin(np.abs(arr_T - t)))
                        if np.abs(arr_T[j] - t) <= tol_years:
                            vals.append(arr_v[j])
                    if len(vals) == len(common_T):
                        atm_data[p] = np.array(vals, float)

                if atm_data:
                    pillar_days = common_T * 365.25
                    level = float(ci) if ci and float(ci) <= 1.0 else float(ci) / 100.0 if ci and ci > 0 else 0.68
                    n_boot = max(min_boot, 1)
                    synth_bands = synthetic_etf_pillar_bands(
                        atm_data,
                        w.to_dict(),
                        pillar_days,
                        level=level,
                        n_boot=n_boot,
                    )
                    synth_curve = pd.DataFrame(
                        {
                            "T": common_T,
                            "atm_vol": synth_bands.mean,
                            "atm_lo": synth_bands.lo,
                            "atm_hi": synth_bands.hi,
                        }
                    )

    return {
        "atm_curve": atm_curve,
        "synth_curve": synth_curve,
        "synth_bands": synth_bands,
        "peer_curves": peer_curves,
        "weights": weight_series,
    }


def fit_smile_for(
    ticker: str,
    asof_date: Optional[str] = None,
    model: str = "svi",             # "svi" or "sabr"
    min_quotes_per_expiry: int = 3, # skip super sparse expiries
    beta: float = 0.5,              # SABR beta (fixed)
    max_expiries: Optional[int] = None,  # Limit number of expiries
) -> VolModel:
    """
    Fit a volatility smile model (SVI, SABR, or polynomial) for one day/ticker using all expiries available that day.
    If asof_date is None, uses the most recent date for the ticker.

    Returns a VolModel you can query/plot from the GUI:
      vm.available_expiries() -> list of T (years)
      vm.predict_iv(K, T)     -> IV at (K, T) using nearest fitted expiry
      vm.smile(Ks, T)         -> vectorized IVs across Ks at nearest fitted expiry
      vm.plot(T)              -> quick plot (if matplotlib installed)
    """
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

    # Median spot for the day
    S = float(df["S"].median())

    # Drop junk, enforce per-expiry density
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

    vm = VolModel(model=model).fit(S, Ks, Ts, IVs, beta=beta)
    return vm


def sample_smile_curve(
    ticker: str,
    asof_date: Optional[str] = None,
    T_target_years: float = 30 / 365.25,  # ~30 days
    model: str = "svi",
    moneyness_grid: tuple[float, float, int] = (0.6, 1.4, 81),  # (lo, hi, n)
    beta: float = 0.5,
    max_expiries: Optional[int] = None,  # Limit number of expiries
) -> pd.DataFrame:
    """
    Convenience: fit a smile then return a tidy curve at the nearest expiry to T_target_years.

    Returns DataFrame with columns:
      ['asof_date','ticker','model','T_used','moneyness','K','IV']
    """
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

# -----------------------------------------------------------------------------
# Enhanced cache management
# -----------------------------------------------------------------------------
def get_cache_info() -> dict:
    """Get information about current cache state."""
    s = get_surface_grids_cached.cache_info()
    a = get_atm_pillars_cached.cache_info()
    t = available_tickers.cache_info()
    d = available_dates.cache_info()
    return {
        "surface_grids_cache": {"size": s.currsize, "hits": s.hits, "misses": s.misses, "maxsize": s.maxsize},
        "atm_pillars_cache": {"size": a.currsize, "hits": a.hits, "misses": a.misses, "maxsize": a.maxsize},
        "available_tickers_cache": {"size": t.currsize, "hits": t.hits, "misses": t.misses, "maxsize": t.maxsize},
        "available_dates_cache": {"size": d.currsize, "hits": d.hits, "misses": d.misses, "maxsize": d.maxsize},
    }


def clear_all_caches() -> None:
    """Clear all in-memory caches and reset shared connection."""
    get_surface_grids_cached.cache_clear()
    get_atm_pillars_cached.cache_clear()
    available_tickers.cache_clear()
    available_dates.cache_clear()
    global _RO_CONN
    if _RO_CONN is not None:
        try:
            _RO_CONN.close()
        except Exception:
            pass
        _RO_CONN = None


def clear_config_dependent_caches() -> None:
    """Clear caches that depend on configuration settings."""
    get_surface_grids_cached.cache_clear()
    get_atm_pillars_cached.cache_clear()


def get_disk_cache_info(cfg: PipelineConfig) -> dict:
    """Get information about disk cache files."""
    base = Path(cfg.cache_dir)
    if not base.exists():
        return {"exists": False, "files": []}

    files = []
    for p in base.iterdir():
        if p.suffix.lower() in (".parquet", ".json"):
            files.append(
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "modified": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return {"exists": True, "files": files}


def cleanup_disk_cache(cfg: PipelineConfig, max_age_days: int = 30) -> list[str]:
    """Clean up old disk cache files."""
    base = Path(cfg.cache_dir)
    if not base.exists():
        return []

    import time
    now = time.time()
    cutoff = max_age_days * 86400
    removed: list[str] = []

    for p in base.iterdir():
        if p.suffix.lower() not in (".parquet", ".json"):
            continue
        if now - p.stat().st_mtime > cutoff:
            try:
                p.unlink()
                removed.append(p.name)
            except OSError:
                pass
    return removed

# -----------------------------------------------------------------------------
# Lightweight disk cache (enhanced)
# -----------------------------------------------------------------------------
def dump_surface_to_cache(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    cfg: PipelineConfig,
    tag: str = "default",
) -> str:
    """Store surfaces as parquet for fast GUI reloads."""
    cfg.ensure_cache_dir()
    base = Path(cfg.cache_dir)
    rows: list[pd.DataFrame] = []

    for t, dct in surfaces.items():
        for date, grid in dct.items():
            g = grid.copy()
            g.insert(0, "mny_bin", g.index.astype(str))
            tidy = g.melt(id_vars="mny_bin", var_name="tenor_days", value_name="iv")
            tidy["ticker"] = t
            tidy["asof_date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
            rows.append(tidy)

    path = base / f"surfaces_{tag}.parquet"
    if not rows:
        pd.DataFrame(columns=["ticker", "asof_date", "mny_bin", "tenor_days", "iv"]).to_parquet(path, index=False)
    else:
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)

    # Save config next to it
    (base / f"surfaces_{tag}.json").write_text(json.dumps(asdict(cfg), indent=2, default=list))
    return str(path)


def load_surface_from_cache(path: str) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Reload cached surfaces (for GUI cold start)."""
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_parquet(p)
    out: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    if df.empty:
        return out
    for (t, date), g in df.groupby(["ticker", "asof_date"]):
        grid = g.pivot(index="mny_bin", columns="tenor_days", values="iv").sort_index(axis=1)
        out.setdefault(t, {})[pd.to_datetime(date)] = grid
    return out


def is_cache_valid(cfg: PipelineConfig, tag: str = "default") -> bool:
    """Check if disk cache is valid for the given configuration."""
    base = Path(cfg.cache_dir)
    cache_path = base / f"surfaces_{tag}.parquet"
    config_path = base / f"surfaces_{tag}.json"

    if not (cache_path.is_file() and config_path.is_file()):
        return False

    try:
        cached_config = json.loads(config_path.read_text())
        current_config = asdict(cfg)

        # Normalize tuples -> lists for comparison
        def _normalize(v):
            if isinstance(v, tuple):
                return [_normalize(x) for x in v]
            if isinstance(v, list):
                return [_normalize(x) for x in v]
            return v

        fields = ["tenors", "mny_bins", "pillar_days", "use_atm_only", "max_expiries"]
        for f in fields:
            if _normalize(cached_config.get(f)) != _normalize(current_config.get(f)):
                return False
        return True
    except Exception:
        return False


def load_surface_from_cache_if_valid(cfg: PipelineConfig, tag: str = "default") -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Load cached surfaces only if the cache is valid for the current configuration."""
    if not is_cache_valid(cfg, tag):
        return {}
    return load_surface_from_cache(str(Path(cfg.cache_dir) / f"surfaces_{tag}.parquet"))

# -----------------------------------------------------------------------------
# __main__ demo path (safe, optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = PipelineConfig()

    print("=== IVCorrelation Analysis Pipeline Demo ===\n")

    # Cache info
    print("Cache Status:")
    cache_info = get_cache_info()
    for name, info in cache_info.items():
        print(f"  {name}: {info['size']}/{info['maxsize']} entries, {info['hits']} hits")
    print()

    # 1) Ingest (optional)
    try:
        inserted = ingest_and_process(["SPY", "QQQ"], max_expiries=6)
        print(f"Data ingestion: {inserted} rows inserted")
    except Exception as e:
        print(f"Data ingestion skipped: {e}")

    # 2) Surfaces + cache
    try:
        surfaces = build_surfaces(["SPY", "QQQ"], cfg=cfg)
        cache_path = dump_surface_to_cache(surfaces, cfg, tag="spyqqq")
        print(f"Surface cache created: {Path(cache_path).name}")
    except Exception as e:
        print(f"Surface building skipped: {e}")

    # 3) Betas
    try:
        paths = save_betas(mode="iv_atm", benchmark="SPY", base_path="data", cfg=cfg)
        print(f"Betas saved to: {[Path(p).name for p in paths]} (Parquet format)")
    except Exception as e:
        print(f"Beta computation skipped: {e}")

    # 4) Quick smile slice
    try:
        dates = available_dates("SPY")
        if dates:
            d = dates[-1]
            df_smile = get_smile_slice("SPY", asof_date=d, T_target_years=30 / 365.25, call_put="C")
            print(f"Smile data: {len(df_smile)} quotes for SPY on {d}")
        else:
            print("No dates available for SPY")
    except Exception as e:
        print(f"Smile data skipped: {e}")

    # 5) Disk cache info
    print("\nDisk Cache Information:")
    disk_info = get_disk_cache_info(cfg)
    if disk_info["exists"] and disk_info["files"]:
        for fi in disk_info["files"]:
            print(f"  {fi['name']}: {fi['size']:,} bytes")
    else:
        print("  No disk cache files found")

    # Optional: smile modeling
    try:
        dates = available_dates("SPY")
        if dates:
            d = dates[-1]
            vm = fit_smile_for("SPY", d, model="svi")
            print(f"\nFitted expiries for SPY on {d}: {vm.available_expiries()}")
            curve = sample_smile_curve("SPY", d, T_target_years=30 / 365.25, model="svi")
            print(f"Smile curve data points: {len(curve)}")
    except Exception as e:
        print(f"Smile modeling skipped: {e}")
