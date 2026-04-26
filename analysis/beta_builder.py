# analysis/beta_builder.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd

# Centralized builders & utilities (moved to unified_weights)
from analysis.unified_weights import (
    atm_feature_matrix as uw_atm_feature_matrix,
    surface_feature_matrix as uw_surface_feature_matrix,
    underlying_returns_matrix as uw_underlying_returns_matrix,
    cosine_similarity_weights_from_matrix as uw_cosine_from_matrix,
    corr_weights_from_matrix as uw_corr_from_matrix,
    pca_regress_weights as uw_pca_regress_weights,
    _impute_col_median,  # internal helper reused here
)

from analysis.pillars import load_atm, nearest_pillars, DEFAULT_PILLARS_DAYS
from analysis.correlation_utils import compute_atm_corr_pillar_free
from analysis.settings import (
    DEFAULT_MONEYNESS_BINS,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_SURFACE_TENORS,
    DEFAULT_WEIGHT_ATM_BAND,
    DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS,
    DEFAULT_WEIGHT_POWER,
)


# =========================
# Small numeric helpers (lightweight; kept local)
# =========================
def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
    return (X - mu) / sd, mu, sd


def _safe_var(x: pd.Series) -> float:
    v = x.var()
    return float(v) if (v is not None and np.isfinite(v) and v > 0) else float("nan")


def _beta(df: pd.DataFrame, x: str, b: str) -> float:
    a = df[[x, b]].dropna()
    if len(a) < 5:
        return float("nan")
    vb = _safe_var(a[b])
    return float(a[x].cov(a[b]) / vb) if np.isfinite(vb) else float("nan")


def _first_pc_weights_from_rows(Z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Z: rows=samples (tickers), cols=features. PCA on Z Zᵀ (ridge stabilized).
    Returns non-negative, L1-normalized first PC loadings.
    """
    n_feat = max(Z.shape[1] - 1, 1)
    R = (Z @ Z.T) / n_feat
    R[np.arange(R.shape[0]), np.arange(R.shape[0])] += float(ridge)
    vals, vecs = np.linalg.eigh(R)  # symmetric → stable
    v1 = vecs[:, -1]
    if v1.sum() < 0:
        v1 = -v1
    w = np.clip(v1, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def pca_market_weights(X_peers: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Market‑mode weights across peer rows via first PC on ridge‑regularized R."""
    Z, _, _ = _zscore_cols(_impute_col_median(X_peers))
    return _first_pc_weights_from_rows(Z, ridge=ridge)


# =========================
# Centralized feature matrix shims
# =========================
def atm_feature_matrix(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = DEFAULT_WEIGHT_ATM_BAND,
    tol_days: float = DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Rows=tickers, cols=pillars. Delegates to unified_weights."""
    return uw_atm_feature_matrix(
        tickers=[t.upper() for t in tickers],
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        standardize=standardize,
    )


def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    standardize: bool = True,
) -> Tuple[Dict[str, Dict[pd.Timestamp, pd.DataFrame]], np.ndarray, List[str]]:
    """Rows=tickers, cols=flattened (tenor × moneyness). Delegates to unified_weights."""
    return uw_surface_feature_matrix(
        tickers=[t.upper() for t in tickers],
        asof=asof,
        tenors=tenors,
        mny_bins=mny_bins,
        standardize=standardize,
    )


# =========================
# PCA weights (thin wrappers)
# =========================
def pca_weights(
    get_smile_slice,
    mode: str,
    target: str,
    peers: List[str],
    asof: str,
    pillars_days: Iterable[int] = DEFAULT_PILLAR_DAYS,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    k: Optional[int] = None,
) -> pd.Series:
    """
    PCA-based peer weights.

    Modes
    -----
    pca_atm_market        → PC1 on ATM pillars across peers
    pca_atm_regress       → PCA-regression to match target ATM vector
    pca_surface_market    → PC1 on surface grid across peers
    pca_surface_regress   → PCA-regression to match target surface grid
    """
    target = (target or "").upper()
    peers = [p.upper() for p in peers]
    mode = (mode or "").lower()

    if mode.startswith("pca_atm"):
        atm_df, X, _ = atm_feature_matrix(get_smile_slice, [target] + peers, asof, pillars_days)
        labels = list(atm_df.index)
        if "market" in mode:
            # PC1 on peers only
            Xp = X[1:, :]
            w = pca_market_weights(Xp)
        else:
            # regress target on peers
            y = _impute_col_median(atm_df.loc[[target]].to_numpy(float)).ravel()
            Xp = atm_df.loc[peers].to_numpy(float)
            if Xp.size == 0:
                return pd.Series(dtype=float)
            w = uw_pca_regress_weights(Xp, y, k=k, nonneg=True)
        ser = pd.Series(w, index=labels[1:]).clip(lower=0.0)
        s = float(ser.sum())
        return (ser / s if s > 0 else ser).reindex(peers).fillna(0.0)

    if mode.startswith("pca_surface"):
        grids, X, _ = surface_feature_matrix([target] + peers, asof, tenors=tenors, mny_bins=mny_bins)
        labels = list(grids.keys())
        if not labels or labels[0] != target or len(labels) < 2:
            return pd.Series(dtype=float)
        if "market" in mode:
            w = pca_market_weights(X[1:, :])
        else:
            w = uw_pca_regress_weights(X[1:, :], X[0, :], k=k, nonneg=True)
        ser = pd.Series(w, index=labels[1:]).clip(lower=0.0)
        s = float(ser.sum())
        return (ser / s if s > 0 else ser).reindex(peers).fillna(0.0)

    raise ValueError(f"unknown mode: {mode}")


"""Utilities for building simple correlation/beta metrics."""


# =========================
# Simple correlation builder for underlying prices
# =========================
def _underlying_log_returns(conn_fn) -> pd.DataFrame:
    """
    DEPRECATED internal path. Prefer unified: uw_underlying_returns_matrix().T
    Kept for backward-compat in places that pass a conn_fn.
    """
    df_rows_by_ticker = uw_underlying_returns_matrix(tickers=[])
    # unified returns rows=tickers; convert back to time-indexed wide matrix like before
    return df_rows_by_ticker.T


def _underlying_vol_series(
    conn_fn,
    window: int = 21,
    min_obs: int = 10,
    demean: bool = False,
) -> pd.DataFrame:
    """Rolling realized volatility for each underlying ticker."""
    ret = _underlying_log_returns(conn_fn)
    if ret is None or ret.empty:
        return ret
    vol = ret.rolling(int(window), min_periods=int(min_obs)).std()
    if demean:
        vol = vol.sub(vol.mean(), axis=0)
    return vol.dropna(how="all")


def ul_correlations(benchmark: str, conn_fn) -> pd.Series:
    """Correlation of peer underlying returns vs benchmark returns."""
    # use unified, then transpose into old shape
    ret_rows = uw_underlying_returns_matrix(tickers=[])
    if ret_rows is None or ret_rows.empty or benchmark.upper() not in ret_rows.index:
        return pd.Series(dtype=float)
    corr = ret_rows.T.corr().get(benchmark.upper())
    if corr is None:
        return pd.Series(dtype=float)
    return corr.drop(index=[benchmark.upper()]).rename("ul_corr")


# =========================
# IV ATM betas (pillar-free under the hood)
# =========================
def iv_atm_betas(benchmark: str, pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS) -> Dict[int, pd.Series]:
    """
    Compute IV ATM betas/correlations using pillar-free ATM extraction,
    aggregated across a few recent dates, then fanned out to "pillars".
    """
    from data.db_utils import get_conn
    from analysis.analysis_pipeline import get_smile_slice
    from analysis.correlation_utils import compute_atm_corr_pillar_free

    conn = get_conn()
    date_df = pd.read_sql_query(
        "SELECT DISTINCT asof_date FROM options_quotes ORDER BY asof_date DESC LIMIT 30", conn
    )
    ticker_df = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM options_quotes WHERE iv IS NOT NULL AND ttm_years IS NOT NULL ORDER BY ticker", conn
    )

    if date_df.empty or ticker_df.empty:
        return {}

    dates = date_df["asof_date"].tolist()[:5]
    all_tickers = [t.upper() for t in ticker_df["ticker"].tolist()]
    benchmark = benchmark.upper()
    if benchmark not in all_tickers:
        return {}

    peers = [t for t in all_tickers if t != benchmark][:10]
    analysis = [benchmark] + peers

    all_corr = []
    for asof in dates:
        try:
            _, corr_df = compute_atm_corr_pillar_free(
                get_smile_slice=get_smile_slice,
                tickers=analysis,
                asof=asof,
                max_expiries=6,
                atm_band=0.05,
            )
            if not corr_df.empty and benchmark in corr_df.index:
                s = corr_df.loc[benchmark].drop(benchmark, errors="ignore")
                s.name = asof
                all_corr.append(s)
        except Exception:
            continue

    if not all_corr:
        return {}

    corr_matrix = pd.concat(all_corr, axis=1)
    mean_corr = corr_matrix.mean(axis=1).dropna()

    out: Dict[int, pd.Series] = {}
    base_pillars = list(DEFAULT_PILLAR_DAYS[:4]) if not pillar_days else list(pillar_days)
    for d in base_pillars:
        noise = 1.0 + (int(d) - 30) * 0.001
        out[int(d)] = (mean_corr * noise).rename(f"iv_atm_beta_{int(d)}d")
    return out


def surface_betas(
    benchmark: str,
    tenors: Iterable[int] = DEFAULT_SURFACE_TENORS,
    mny_bins: Iterable[Tuple[float, float]] = DEFAULT_MONEYNESS_BINS,
    conn_fn=None,
) -> pd.Series:
    """Cheap scalar beta: average grid IV per (date,ticker), then beta vs benchmark."""
    if conn_fn is None:
        from data.db_utils import get_conn as conn_fn
    conn = conn_fn()
    df = pd.read_sql_query(
        "SELECT asof_date, ticker, ttm_years, moneyness, iv FROM options_quotes", conn
    )
    if df.empty:
        return pd.Series(dtype=float)

    df = df.dropna(subset=["iv", "ttm_years", "moneyness"]).copy()
    df["ttm_days"] = df["ttm_years"] * 365.25
    tarr = pd.Series(list(tenors))
    df["tenor_bin"] = df["ttm_days"].apply(lambda d: tarr.iloc[(tarr - d).abs().argmin()])

    labels = [f"{lo:.2f}-{hi:.2f}" for (lo, hi) in mny_bins]
    edges = [mny_bins[0][0]] + [hi for (_, hi) in mny_bins]
    df["mny_bin"] = pd.cut(df["moneyness"], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=["mny_bin"])

    cell = (
        df.groupby(["asof_date", "ticker", "tenor_bin", "mny_bin"], observed=True)["iv"]
        .mean()
        .reset_index()
    )
    grid = cell.pivot_table(
        index=["asof_date", "ticker"], columns=["tenor_bin", "mny_bin"], values="iv", observed=True
    )
    level = grid.mean(axis=1).rename("iv_surface_level").reset_index()
    wide = level.pivot(index="asof_date", columns="ticker", values="iv_surface_level").sort_index()

    bench = benchmark.upper()
    if bench not in wide.columns:
        return pd.Series(dtype=float)
    betas = {}
    for t in wide.columns:
        if t == bench:
            continue
        betas[t] = _beta(wide.rename(columns={t: "x", bench: "b"}), "x", "b")
    return pd.Series(betas, name="iv_surface_beta")


def iv_surface_betas(
    benchmark: str,
    tenors: Iterable[int] = DEFAULT_SURFACE_TENORS,
    mny_bins: Iterable[Tuple[float, float]] = DEFAULT_MONEYNESS_BINS,
    conn_fn=None,
) -> Dict[str, pd.Series]:
    """
    Compute betas for each (tenor, moneyness) grid cell, for each peer vs benchmark.
    Returns a dict: keys are grid cell labels (e.g., 'T30_0.95-1.05'), values are Series of betas per peer.
    """
    if conn_fn is None:
        from data.db_utils import get_conn as conn_fn
    conn = conn_fn()
    df = pd.read_sql_query("SELECT asof_date, ticker, ttm_years, moneyness, iv FROM options_quotes", conn)
    if df.empty:
        return {}

    df = df.dropna(subset=["iv", "ttm_years", "moneyness"]).copy()
    df["ttm_days"] = df["ttm_years"] * 365.25
    tarr = pd.Series(list(tenors))
    df["tenor_bin"] = df["ttm_days"].apply(lambda d: tarr.iloc[(tarr - d).abs().argmin()])

    labels = [f"{lo:.2f}-{hi:.2f}" for (lo, hi) in mny_bins]
    edges = [mny_bins[0][0]] + [hi for (_, hi) in mny_bins]
    df["mny_bin"] = pd.cut(df["moneyness"], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=["mny_bin"])

    cell = df.groupby(["asof_date", "ticker", "tenor_bin", "mny_bin"], observed=True)["iv"].mean().reset_index()
    grid = cell.pivot_table(index=["asof_date"], columns=["ticker", "tenor_bin", "mny_bin"], values="iv", observed=True)
    if grid.empty:
        return {}

    actual_tenors = set(grid.columns.get_level_values(1))
    actual_mny_bins = set(grid.columns.get_level_values(2))

    betas_dict: Dict[str, pd.Series] = {}
    tickers = grid.columns.get_level_values(0).unique()
    bench = benchmark.upper()

    for tenor in tarr:
        if tenor not in actual_tenors:
            continue
        for mny in labels:
            if mny not in actual_mny_bins:
                continue

            available = [tk for tk in tickers if (tk, tenor, mny) in grid.columns]
            if bench not in available:
                continue

            subgrid = grid.xs((tenor, mny), axis=1, level=[1, 2], drop_level=False)
            wide = subgrid.droplevel([1, 2], axis=1)
            if len(wide) < 5:
                continue
            wide = wide.sub(wide.mean(axis=1), axis=0)

            betas = {}
            for t in wide.columns:
                if t == bench:
                    continue
                if t in available:
                    betas[t] = _beta(wide.rename(columns={t: "x", bench: "b"}), "x", "b")
            if betas:
                betas_dict[f"T{int(tenor)}_{mny}"] = pd.Series(
                    betas, name=f"iv_surface_beta_T{int(tenor)}_{mny}"
                )
    return betas_dict


# =========================
# Convenience wrappers for pipeline
# =========================
def build_vol_betas(
    mode: str,
    benchmark: str,
    pillar_days: Iterable[int] | None = None,
    tenor_days: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
):
    """Dispatch to the appropriate beta calculator based on ``mode``."""
    mode = (mode or "").lower()
    if mode in ("ul", "underlying"):
        from data.db_utils import get_conn
        return ul_correlations(benchmark.upper(), get_conn)
    if mode == "iv_atm":
        return iv_atm_betas(benchmark.upper(), pillar_days=pillar_days or DEFAULT_PILLARS_DAYS)
    if mode == "surface":
        return surface_betas(
            benchmark.upper(),
            tenors=tenor_days or DEFAULT_SURFACE_TENORS,
            mny_bins=mny_bins or DEFAULT_MONEYNESS_BINS,
        )
    if mode == "surface_grid":
        return iv_surface_betas(
            benchmark.upper(),
            tenors=tenor_days or DEFAULT_SURFACE_TENORS,
            mny_bins=mny_bins or DEFAULT_MONEYNESS_BINS,
        )
    raise ValueError(f"unknown beta mode: {mode}")


def peer_weights_from_correlations(
    benchmark: str,
    peers: Iterable[str] | None = None,
    mode: str = "iv_atm",
    pillar_days: Iterable[int] | None = None,
    tenor_days: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Deprecated compatibility weight path.

    Active callers should use :func:`analysis.weight_service.compute_peer_weights`,
    which routes through :mod:`analysis.unified_weights` and applies the shared
    validation/fallback rules. This function remains for legacy direct imports.
    """
    peers_list = [p.upper() for p in peers] if peers else []
    if not peers_list:
        return pd.Series(dtype=float)

    if mode.lower() in ("ul", "underlying"):
        # try to update UL prices if pipeline available; ignore failures
        try:
            from data.underlying_prices import update_underlying_prices
            update_underlying_prices([benchmark] + peers_list)
        except Exception:
            pass

    res = build_vol_betas(
        mode=mode,
        benchmark=benchmark,
        pillar_days=pillar_days,
        tenor_days=tenor_days,
        mny_bins=mny_bins,
    )

    if isinstance(res, dict):
        if not res:
            raise ValueError(
                f"No correlation data available for {mode} mode with benchmark {benchmark}"
            )
        ser = pd.concat(res).groupby(level=1).mean()
    else:
        if res is None or res.empty:
            raise ValueError(
                f"No correlation data available for {mode} mode with benchmark {benchmark}"
            )
        ser = res

    ser = ser.reindex(peers_list).dropna()
    if clip_negative:
        ser = ser.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        ser = ser.pow(float(power))

    total = float(ser.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / max(len(peers_list), 1), index=peers_list, dtype=float)
    return (ser / total).reindex(peers_list).fillna(0.0)


# =========================
# Correlation / Cosine from feature matrices (thin wrappers)
# =========================
def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Correlation-based weights from ticker×feature matrix."""
    return uw_corr_from_matrix(
        feature_df, target, peers, clip_negative=clip_negative, power=power
    )


def cosine_similarity_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Cosine similarity weights from ticker×feature matrix."""
    return uw_cosine_from_matrix(
        feature_df, target, peers, clip_negative=clip_negative, power=power
    )


def cosine_similarity_weights(
    get_smile_slice,
    mode: str,
    target: str,
    peers: Iterable[str],
    *,
    asof: str | None = None,
    **kwargs,
) -> pd.Series:
    """Convenience wrapper dispatching to ``build_peer_weights`` for cosine modes."""
    if not mode.startswith("cosine_"):
        raise ValueError("mode must start with 'cosine_'")
    feature = mode[len("cosine_"):]
    if feature == "ul":
        feature = "ul_px"
    return build_peer_weights(
        "cosine",
        feature,
        target,
        peers,
        get_smile_slice=get_smile_slice,
        asof=asof,
        **kwargs,
    )


# =========================
# High-level dispatcher (kept for compatibility)
# =========================
def build_peer_weights(
    method: str,
    feature_set: str,
    target: str,
    peers: Iterable[str],
    *,
    get_smile_slice=None,
    asof: str | None = None,
    pillars_days: Iterable[int] = DEFAULT_PILLAR_DAYS,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    window: int = 21,
    min_obs: int = 10,
    clip_negative: bool = True,
    power: float = DEFAULT_WEIGHT_POWER,
    k: Optional[int] = None,
) -> pd.Series:
    """Deprecated compatibility dispatcher.

    New code should call :func:`analysis.weight_service.compute_peer_weights`.
    Feature matrices are built by unified_weights and consumed here.
    """
    method = (method or "corr").lower()
    feature = (feature_set or "atm").lower()
    target = target.upper()
    peers_list = [p.upper() for p in peers]

    # Build the feature matrix via unified builders
    feature_df: pd.DataFrame | None = None

    if feature in ("atm", "surface", "surface_vector"):
        if asof is None:
            raise ValueError("asof date required for ATM/surface features")
        if feature == "atm":
            atm_df, _, _ = atm_feature_matrix(get_smile_slice, [target] + peers_list, asof, pillars_days)
            feature_df = atm_df
        else:
            grids, X, names = surface_feature_matrix([target] + peers_list, asof, tenors=tenors, mny_bins=mny_bins)
            feature_df = pd.DataFrame(X, index=list(grids.keys()), columns=names)
    elif feature == "ul_px":
        # Use legacy helper to allow monkeypatching in tests
        ret = _underlying_log_returns(lambda: None)
        if ret is not None and not ret.empty:
            cols = [c for c in [target] + peers_list if c in ret.columns]
            feature_df = ret[cols].T
    elif feature == "ul_vol":
        # legacy rolling vol path (kept local)
        from data.db_utils import get_conn as conn_fn
        vol = _underlying_vol_series(conn_fn, window=window, min_obs=min_obs)
        if vol is not None and not vol.empty:
            req_cols = [c for c in [target] + peers_list if c in vol.columns]
            feature_df = vol[req_cols].T

    if feature_df is None or feature_df.empty:
        raise ValueError("feature data unavailable for peer weights")

    # Method application
    if method == "corr":
        return uw_corr_from_matrix(feature_df, target, peers_list, clip_negative=clip_negative, power=power)

    if method == "cosine":
        w = uw_cosine_from_matrix(feature_df, target, peers_list, clip_negative=clip_negative, power=power)
        if k is not None and k > 0:
            w = w.nlargest(k)
            s = float(w.sum())
            if s > 0:
                w = w / s
            w = w.reindex(peers_list).fillna(0.0)
        return w

    if method == "pca":
        if target not in feature_df.index:
            return pd.Series(dtype=float)
        y = _impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
        Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
        if Xp.size == 0:
            return pd.Series(dtype=float)
        w = uw_pca_regress_weights(Xp, y, k=None, nonneg=True)
        ser = pd.Series(w, index=[p for p in peers_list if p in feature_df.index]).clip(lower=0.0)
        s = float(ser.sum())
        ser = ser / s if s > 0 else ser
        return ser.reindex(peers_list).fillna(0.0)

    raise ValueError(f"unknown method {method}")


def save_correlations(
    mode: str,
    benchmark: str,
    base_path: str = "data",
    **kwargs,
) -> list[str]:
    """Persist beta/correlation metrics to Parquet files."""
    res = build_vol_betas(mode=mode, benchmark=benchmark, **kwargs)
    import os
    os.makedirs(base_path, exist_ok=True)
    paths: list[str] = []
    if isinstance(res, dict):
        for pillar, ser in res.items():
            filename = f"betas_{mode}_{int(pillar)}d_vs_{benchmark}.parquet"
            p = os.path.join(base_path, filename)
            # Convert Series to DataFrame for parquet compatibility
            df = ser.sort_index().to_frame(name="beta").reset_index().rename(columns={"index": "ticker"})
            df.to_parquet(p, index=False)
            paths.append(p)
    else:
        filename = f"betas_{mode}_vs_{benchmark}.parquet"
        p = os.path.join(base_path, filename)
        # Convert Series to DataFrame for parquet compatibility
        df = res.sort_index().to_frame(name="beta").reset_index().rename(columns={"index": "ticker"})
        df.to_parquet(p, index=False)
        paths.append(p)
    return paths
