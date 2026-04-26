# analysis/analysis_pipeline.py
"""
GUI-ready analysis orchestrator.

This module wires together:
- ingest + enrich (data.historical_saver)
- surface grid building
- peer composite surface construction (surface & ATM pillars)
- vol betas (UL, IV-ATM, Surface)
- lightweight snapshot/smile helpers for GUI

All public functions are fast to call from a GUI. Heavy work is cached
in-memory and can optionally be dumped to disk (parquet) for fast reloads.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, Mapping, Union

import json
import logging

import numpy as np
import pandas as pd
import os
from data.historical_saver import save_for_tickers
from data.db_utils import get_conn
from data.interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD
from data.quote_quality import (
    ANALYTICS_MAX_MONEYNESS,
    ANALYTICS_MIN_MONEYNESS,
)

from .peer_composite_builder import (
    build_surface_grids,
    DEFAULT_TENORS,
    DEFAULT_MNY_BINS,
    combine_surfaces,
    build_synthetic_iv as build_synthetic_iv_pillars,
)

from .weight_service import compute_peer_weights
from .beta_builder import build_vol_betas, save_correlations
from .pillar_selection import nearest_pillars
from .pillar_selection import load_atm, DEFAULT_PILLARS_DAYS
from .settings import (
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
    DEFAULT_RV_LOOKBACK_DAYS,
    DEFAULT_UNDERLYING_LOOKBACK_DAYS,
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
# Peer-composite surface & ATM pillars
# -----------------------------------------------------------------------------
def build_synthetic_surface(
    weights: Mapping[str, float],
    cfg: PipelineConfig = PipelineConfig(),
    most_recent_only: bool = True,  # default to True for performance
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Create a peer composite surface from ticker grids + weights."""
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

    def per_pillar(g: pd.DataFrame, pillar_days: int | float | None = None) -> pd.DataFrame:
        g = g.copy()
        if "pillar_days" not in g.columns:
            g["pillar_days"] = pillar_days
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

    parts = [per_pillar(g, pillar) for pillar, g in df.groupby("pillar_days", group_keys=False)]
    return pd.concat(parts, ignore_index=True) if parts else df


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
    weight_mode = mode if mode.startswith(("corr_", "pca_", "cosine_")) or mode in ("oi", "equal") else f"corr_{mode}"
    w = compute_peer_weights(
        target=target,
        peers=peers,
        weight_mode=weight_mode,
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
# Smile helpers, term data, and RV heatmap
# Implementations live in service modules; re-exported here for backward compat.
# -----------------------------------------------------------------------------
from .smile_data_service import (  # noqa: E402
    get_smile_slice,
    get_smile_slices_batch,
    prepare_smile_data,
    fit_smile_for,
    sample_smile_curve,
)
from .term_data_service import prepare_term_data  # noqa: E402
from .rv_heatmap_service import prepare_rv_heatmap_data  # noqa: E402
from .model_fit_service import fit_model_params, quality_checked_result  # noqa: E402  (re-export for backward compat)


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
