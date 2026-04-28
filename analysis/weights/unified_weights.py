"""
Unified Weight Computation System

Single, consistent interface for computing portfolio weights
across methods (correlation, PCA, cosine, equal, open-interest)
and features (ATM, surface, underlying returns).

- Options-based features (ATM, surface*) are single as-of snapshots.
- Underlying returns (UL) use time-series panels (no as-of required).
- Open-interest is a method (no feature matrix).

This module also centralizes *feature building* helpers so other modules
can reuse them without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple, Union, List, Any
import logging

import numpy as np
import pandas as pd

# Delayed imports to avoid circular dependencies
# from analysis.analysis_pipeline import get_smile_slice, available_dates
from analysis.surfaces.pillar_selection import build_atm_matrix, DEFAULT_PILLARS_DAYS
from analysis.config.settings import (
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_MONEYNESS_BINS,
    DEFAULT_ATM_BAND,
    DEFAULT_SURFACE_TENORS,
    DEFAULT_WEIGHT_ATM_BAND,
    DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS,
)

# from analysis.surfaces.peer_composite_builder import build_surface_grids
# from analysis.weights.correlation_utils import compute_atm_corr_pillar_free

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class WeightMethod(Enum):
    CORRELATION = "corr"
    PCA = "pca"
    COSINE = "cosine"
    EQUAL = "equal"
    OPEN_INTEREST = "oi"  # implemented via _open_interest_weights


class FeatureSet(Enum):
    ATM = "iv_atm"  # options ATM features
    ATM_RANKS = "iv_atm_ranks"  # pillar-free ATM by expiry rank
    SURFACE = "surface"  # market-native expiry-rank x moneyness-bin surface
    SURFACE_VECTOR = "surface_grid"  # standardized tenor-grid x moneyness-bin surface
    UNDERLYING_PX = "ul"  # underlying price returns (time-series)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class WeightConfig:
    method: WeightMethod
    feature_set: FeatureSet
    pillars_days: Tuple[int, ...] = tuple(DEFAULT_PILLARS_DAYS)
    tenors: Tuple[int, ...] = DEFAULT_SURFACE_TENORS
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MONEYNESS_BINS
    clip_negative: bool = True
    power: float = 1.0
    asof: Optional[str] = None
    atm_band: float = DEFAULT_WEIGHT_ATM_BAND
    atm_tol_days: float = DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS
    max_expiries: int = DEFAULT_MAX_EXPIRIES
    max_abs_weight: float = 0.98
    max_l1_norm: float = 3.0
    corr_shrinkage: float = 0.05
    pca_ridge: float = 1e-3
    surface_missing_policy: str = "median_impute"
    surface_min_coverage: float = 0.70
    surface_source: str = "fit"
    surface_model: str = "svi"

    @classmethod
    def from_mode(cls, mode: str, **kwargs) -> "WeightConfig":
        """Parse a canonical mode string into a :class:`WeightConfig`.

        Parameters
        ----------
        mode:
            Strings like ``"corr_iv_atm"`` or ``"pca_surface"``.  If only a
            method is supplied (e.g. ``"oi"``) the feature set defaults to
            ``"iv_atm"``.
        """
        mode = (mode or "").lower().strip()
        if "_" in mode:
            method_str, feature_str = mode.split("_", 1)
        else:
            method_str, feature_str = mode, "iv_atm"

        method = WeightMethod(method_str)
        feature_set = FeatureSet(feature_str)
        return cls(method=method, feature_set=feature_set, **kwargs)


@dataclass
class WeightDiagnostics:
    mode: str
    feature_set: str
    peers: list[str]
    input_shape: Optional[tuple[int, ...]] = None
    weights: dict[str, float] = field(default_factory=dict)
    rejected_weights: dict[str, float] = field(default_factory=dict)
    weight_min: float = np.nan
    weight_max: float = np.nan
    weight_mean: float = np.nan
    weight_sum: float = np.nan
    weight_l1: float = np.nan
    weight_l2: float = np.nan
    normalization: str = "sum"
    normalization_scale: float = np.nan
    nonfinite_count: int = 0
    negative_count: int = 0
    condition_number: Optional[float] = None
    clipped_count: int = 0
    fallback: Optional[str] = None
    issues: list[str] = field(default_factory=list)

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feature_set": self.feature_set,
            "peers": self.peers,
            "input_shape": self.input_shape,
            "weights": self.weights,
            "rejected_weights": self.rejected_weights,
            "min": self.weight_min,
            "max": self.weight_max,
            "mean": self.weight_mean,
            "sum": self.weight_sum,
            "l1": self.weight_l1,
            "l2": self.weight_l2,
            "normalization": self.normalization,
            "normalization_scale": self.normalization_scale,
            "nonfinite": self.nonfinite_count,
            "negative": self.negative_count,
            "condition": self.condition_number,
            "clipped": self.clipped_count,
            "fallback": self.fallback,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class FeatureDiagnostics:
    feature_set: str
    coordinate_system: str
    value_type: str
    shape: tuple[int, int]
    tickers_requested: list[str]
    tickers_included: list[str]
    tickers_excluded: list[str]
    missing_policy: str
    normalization: str
    asof: Optional[str] = None
    n_expiries: Optional[int] = None
    n_grid_points: Optional[int] = None

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "coordinate_system": self.coordinate_system,
            "value_type": self.value_type,
            "shape": self.shape,
            "tickers_requested": self.tickers_requested,
            "tickers_included": self.tickers_included,
            "tickers_excluded": self.tickers_excluded,
            "missing_policy": self.missing_policy,
            "normalization": self.normalization,
            "asof": self.asof,
            "n_expiries": self.n_expiries,
            "n_grid_points": self.n_grid_points,
        }


# -----------------------------------------------------------------------------
# Centralized feature builders (reusable; no circular imports)
# -----------------------------------------------------------------------------
def _impute_col_median(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    med = np.zeros((1, X.shape[1]), dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            med[0, j] = float(np.median(finite))
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.broadcast_to(med, X.shape)[mask]
    return X


def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    finite = np.isfinite(X)
    counts = finite.sum(axis=0, keepdims=True)
    sums = np.where(finite, X, 0.0).sum(axis=0, keepdims=True)
    mu = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    centered = np.where(finite, X - mu, 0.0)
    var = np.divide(
        (centered * centered).sum(axis=0, keepdims=True),
        counts - 1,
        out=np.zeros_like(mu, dtype=float),
        where=counts > 1,
    )
    sd = np.sqrt(var)
    sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
    return (X - mu) / sd, mu, sd


def _apply_surface_missing_policy(
    feature_df: pd.DataFrame,
    *,
    policy: str = "median_impute",
    min_coverage: float = 0.70,
) -> tuple[pd.DataFrame, str]:
    """Apply surface feature coverage policy before similarity/weighting."""
    if feature_df is None or feature_df.empty:
        return feature_df, "empty surface feature frame"
    df = feature_df.apply(pd.to_numeric, errors="coerce")
    policy = str(policy or "median_impute").lower()
    min_cov = float(np.clip(min_coverage, 0.0, 1.0))
    if policy == "require_shared":
        out = df.dropna(axis=1, how="any")
        return out, f"require_shared dropped {df.shape[1] - out.shape[1]} non-shared cells"
    if policy == "drop_sparse":
        coverage = df.notna().mean(axis=0)
        out = df.loc[:, coverage >= min_cov]
        return out, f"drop_sparse kept {out.shape[1]}/{df.shape[1]} cells at min coverage {min_cov:.0%}"
    return df, "median_impute retained sparse cells for downstream column-median imputation"


def _condition_number(arr: np.ndarray) -> Optional[float]:
    arr = np.asarray(arr, float)
    if arr.size == 0 or min(arr.shape) == 0:
        return None
    try:
        vals = np.linalg.svd(arr, compute_uv=False)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        smallest = float(np.min(vals))
        largest = float(np.max(vals))
        if smallest <= 1e-12:
            return float("inf")
        return largest / smallest
    except Exception:
        return None


def _diagnose_weights(
    weights: pd.Series,
    *,
    config: WeightConfig,
    peers: list[str],
    input_shape: Optional[tuple[int, ...]] = None,
    normalization: str = "sum",
    normalization_scale: float = np.nan,
    condition_number: Optional[float] = None,
    clipped_count: int = 0,
    fallback: Optional[str] = None,
    issues: Optional[list[str]] = None,
    rejected_weights: Optional[pd.Series] = None,
) -> WeightDiagnostics:
    w = pd.Series(weights, dtype=float).reindex(peers).fillna(0.0)
    rejected = (
        pd.Series(rejected_weights, dtype=float).reindex(peers)
        if rejected_weights is not None
        else pd.Series(dtype=float)
    )
    arr = w.to_numpy(float)
    finite = arr[np.isfinite(arr)]
    return WeightDiagnostics(
        mode=config.method.value,
        feature_set=config.feature_set.value,
        peers=peers,
        input_shape=input_shape,
        weights={str(k): float(v) for k, v in w.items()},
        rejected_weights={str(k): float(v) for k, v in rejected.replace([np.inf, -np.inf], np.nan).dropna().items()},
        weight_min=float(np.min(finite)) if finite.size else np.nan,
        weight_max=float(np.max(finite)) if finite.size else np.nan,
        weight_mean=float(np.mean(finite)) if finite.size else np.nan,
        weight_sum=float(np.sum(finite)) if finite.size else np.nan,
        weight_l1=float(np.sum(np.abs(finite))) if finite.size else np.nan,
        weight_l2=float(np.linalg.norm(finite)) if finite.size else np.nan,
        normalization=normalization,
        normalization_scale=float(normalization_scale) if np.isfinite(normalization_scale) else np.nan,
        nonfinite_count=int((~np.isfinite(arr)).sum()),
        negative_count=int((arr < -1e-12).sum()),
        condition_number=condition_number,
        clipped_count=int(clipped_count),
        fallback=fallback,
        issues=list(issues or []),
    )


def _validate_and_normalize_weights(
    weights: pd.Series,
    *,
    peers: list[str],
    config: WeightConfig,
    allow_signed: bool = False,
    input_shape: Optional[tuple[int, ...]] = None,
    condition_number: Optional[float] = None,
    clipped_count: int = 0,
) -> tuple[pd.Series, WeightDiagnostics]:
    issues: list[str] = []
    raw = pd.Series(weights, dtype=float).reindex(peers)
    arr = raw.to_numpy(float)
    if raw.empty:
        issues.append("empty weights")
    if not np.isfinite(arr).all():
        issues.append("nonfinite weights")

    clean = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if not allow_signed and (clean < -1e-12).any():
        issues.append("negative weights")
        clean = clean.clip(lower=0.0)
        clipped_count += int((raw < -1e-12).sum())

    if allow_signed:
        scale = float(clean.abs().sum())
        normalization = "l1"
    else:
        scale = float(clean.sum())
        normalization = "sum"
    if not np.isfinite(scale) or scale <= 1e-12:
        issues.append("zero normalization scale")
    else:
        clean = clean / scale

    arr2 = clean.to_numpy(float)
    if not np.isfinite(arr2).all():
        issues.append("nonfinite normalized weights")
    if not allow_signed and (arr2 < -1e-12).any():
        issues.append("negative normalized weights")
    if np.isfinite(arr2).all() and arr2.size:
        max_abs = float(np.max(np.abs(arr2)))
        l1 = float(np.sum(np.abs(arr2)))
        if arr2.size > 1 and max_abs > float(config.max_abs_weight):
            issues.append(f"max weight {max_abs:.4f} exceeds {config.max_abs_weight:.4f}")
        if l1 > float(config.max_l1_norm):
            issues.append(f"l1 norm {l1:.4f} exceeds {config.max_l1_norm:.4f}")
        if not allow_signed and abs(float(arr2.sum()) - 1.0) > 1e-6:
            issues.append("weights do not sum to 1")

    diag = _diagnose_weights(
        clean,
        config=config,
        peers=peers,
        input_shape=input_shape,
        normalization=normalization,
        normalization_scale=scale,
        condition_number=condition_number,
        clipped_count=clipped_count,
        issues=issues,
    )
    if issues:
        raise ValueError("; ".join(issues))
    return clean.reindex(peers).fillna(0.0), diag


def _log_weight_diagnostics(diag: WeightDiagnostics, level: int = logging.INFO) -> None:
    logger.log(level, "weight diagnostics: %s", diag.as_log_dict())


def _attach_output_diagnostics(weights: pd.Series, feature_df: pd.DataFrame, diag: WeightDiagnostics) -> pd.Series:
    weights.attrs["feature_diagnostics"] = dict(getattr(feature_df, "attrs", {}).get("feature_diagnostics", {}))
    weights.attrs["weight_diagnostics"] = diag.as_log_dict()
    return weights


def atm_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    *,
    atm_band: float = DEFAULT_WEIGHT_ATM_BAND,
    tol_days: float = DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Rows=tickers, cols=pillars (days). Values=ATM IVs for a single as-of date."""
    from analysis.services.smile_data_service import get_smile_slice, get_smile_slices_batch

    tickers_up = [t.upper() for t in tickers]
    # One DB round-trip for all tickers; slices passed into build_atm_matrix
    slices = get_smile_slices_batch(tickers_up, asof)

    atm_df, _ = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=tickers_up,
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        slices=slices,
    )
    X = _impute_col_median(atm_df.to_numpy(dtype=float))
    if standardize:
        X, _, _ = _zscore_cols(X)
    return atm_df, X, list(atm_df.columns)


def _atm_rank_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    *,
    atm_band: float = DEFAULT_ATM_BAND,
) -> tuple[pd.DataFrame, list[int]]:
    """Build expiry-rank ATM feature matrix (rows=tickers, cols=ranks)."""
    from analysis.services.smile_data_service import get_smile_slice  # delayed import
    from analysis.weights.correlation_utils import compute_atm_corr_pillar_free

    tickers = [t.upper() for t in tickers]
    atm_df, _ = compute_atm_corr_pillar_free(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        max_expiries=max_expiries,
        atm_band=atm_band,
    )
    # ensure all requested tickers present
    atm_df = atm_df.reindex(tickers)
    return atm_df, list(atm_df.columns)


def native_surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    *,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Market-native surface features: expiry rank x moneyness bin IV levels."""
    from analysis.services.smile_data_service import get_smile_slices_batch
    from analysis.surfaces.peer_composite_builder import DEFAULT_MNY_BINS

    tickers_up = [str(t).upper() for t in tickers]
    bins = tuple(tuple(b) for b in (mny_bins or DEFAULT_MNY_BINS))
    labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in bins]
    edges = [bins[0][0]] + [hi for _, hi in bins]
    feature_names = [f"rank{rank}_{label}" for rank in range(int(max_expiries)) for label in labels]
    slices = get_smile_slices_batch(tickers_up, asof, max_expiries=max_expiries)

    rows: list[pd.Series] = []
    for ticker in tickers_up:
        df = slices.get(ticker, pd.DataFrame())
        values = {name: np.nan for name in feature_names}
        if df is not None and not df.empty and {"T", "moneyness", "sigma"}.issubset(df.columns):
            d = df.copy()
            d["T"] = pd.to_numeric(d["T"], errors="coerce")
            d["moneyness"] = pd.to_numeric(d["moneyness"], errors="coerce")
            d["sigma"] = pd.to_numeric(d["sigma"], errors="coerce")
            d = d.dropna(subset=["T", "moneyness", "sigma"]).sort_values("T")
            Ts = np.sort(d["T"].unique())[: int(max_expiries)]
            for rank, T_val in enumerate(Ts):
                g = d.loc[np.isclose(d["T"], T_val)].copy()
                if g.empty:
                    continue
                g["mny_bin"] = pd.cut(g["moneyness"], bins=edges, labels=labels, include_lowest=True)
                cell = g.dropna(subset=["mny_bin"]).groupby("mny_bin", observed=True)["sigma"].mean()
                for label, value in cell.items():
                    values[f"rank{rank}_{label}"] = float(value)
        rows.append(pd.Series(values, name=ticker))
    return pd.DataFrame(rows), feature_names


def surface_feature_matrix(
    tickers: Iterable[str],
    asof: str,
    *,
    tenors: Iterable[int] | None = None,
    mny_bins: Iterable[Tuple[float, float]] | None = None,
    surface_source: str = "fit",
    model: str = "svi",
    max_expiries: int | None = DEFAULT_MAX_EXPIRIES,
    standardize: bool = True,
    missing_policy: str = "median_impute",
    min_coverage: float = 0.70,
) -> Tuple[Dict[str, Dict[pd.Timestamp, pd.DataFrame]], np.ndarray, List[str]]:
    """Rows=tickers, cols=flattened (tenor × moneyness) grid for a single as-of date."""
    # Route through the LRU-cached builder so repeated calls on same tickers/date
    # skip the full DB query.
    from analysis.analysis_pipeline import PipelineConfig, get_surface_grids_cached
    from analysis.surfaces.peer_composite_builder import DEFAULT_TENORS, DEFAULT_MNY_BINS

    asof_ts = pd.Timestamp(asof).normalize()
    req = [t.upper() for t in tickers]

    _tenors = tuple(int(t) for t in tenors) if tenors else DEFAULT_TENORS
    _mny_bins = tuple(tuple(b) for b in mny_bins) if mny_bins else DEFAULT_MNY_BINS
    mny_labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in _mny_bins]
    cfg = PipelineConfig(
        tenors=_tenors,
        mny_bins=_mny_bins,
        surface_source=str(surface_source or "fit"),
        surface_model=str(model or "svi"),
        max_expiries=max_expiries,
    )
    key = ",".join(sorted(req))

    logger.debug("Building surface grids (cached) for %s on %s", req, asof_ts)
    grids = get_surface_grids_cached(cfg, key)
    if not grids:
        logger.debug("No surface grids returned for %s on %s", req, asof_ts)

    feats: list[np.ndarray] = []
    ok: list[str] = []
    feat_names: list[str] | None = None

    for t in req:
        if t not in grids:
            logger.debug("Ticker %s missing from surface grids", t)
            continue
        if asof_ts not in grids[t]:
            logger.debug(
                "Ticker %s has no surface for %s (available=%s)",
                t,
                asof_ts,
                sorted(grids[t].keys()),
            )
            continue
        df = grids[t][asof_ts]  # index=mny labels, columns=tenor (days)
        df = df.reindex(index=mny_labels, columns=list(_tenors))
        logger.debug("Ticker %s surface grid aligned shape %s", t, df.shape)
        arr = df.to_numpy(float).T.reshape(-1)
        if feat_names is None:
            feat_names = [f"T{c}_{r}" for c in _tenors for r in mny_labels]
        feats.append(arr)
        ok.append(t)

    if not feats:
        logger.debug("No surface features constructed for %s on %s", req, asof)
        return {}, np.empty((0, 0)), []

    raw = pd.DataFrame(np.vstack(feats), index=ok, columns=feat_names or [])
    raw, _ = _apply_surface_missing_policy(
        raw,
        policy=missing_policy,
        min_coverage=min_coverage,
    )
    if raw.empty or raw.shape[1] == 0:
        logger.debug("Surface feature matrix empty after missing policy %s", missing_policy)
        return {t: grids[t] for t in ok}, np.empty((0, 0)), []
    X = _impute_col_median(raw.to_numpy(float))
    if standardize:
        X, _, _ = _zscore_cols(X)
    logger.debug("Surface feature matrix shape %s for tickers %s", X.shape, ok)
    return {t: grids[t] for t in ok}, X, list(raw.columns)


def underlying_returns_matrix(tickers: Iterable[str]) -> pd.DataFrame:
    """Ticker×time matrix of log returns (rows=tickers). Pairwise-ready (keeps partial rows)."""
    from data.db_utils import get_conn

    # Auto-update underlying price data if needed
    tickers_set = set(str(t).upper() for t in tickers)
    logger.info(f"Checking underlying price data for {len(tickers_set)} tickers...")

    try:
        from data.data_pipeline import ensure_underlying_price_data

        if not ensure_underlying_price_data(tickers_set):
            logger.warning("Failed to ensure underlying price data is available")
    except Exception as e:
        logger.warning(f"Could not auto-update underlying prices: {e}")

    conn = get_conn()
    # prefer dedicated table; fallback to options_quotes spot
    placeholders = ",".join("?" * len(tickers_set))
    ticker_params = sorted(tickers_set)
    try:
        df = pd.read_sql_query(
            f"SELECT asof_date, ticker, close FROM underlying_prices WHERE ticker IN ({placeholders})",
            conn,
            params=ticker_params,
        )
        logger.debug(f"Loaded {len(df)} underlying price records from dedicated table")
    except Exception:
        df = pd.DataFrame()
        logger.debug("No underlying_prices table, trying fallback")

    if df.empty:
        df = pd.read_sql_query(
            f"SELECT asof_date, ticker, spot AS close FROM options_quotes WHERE ticker IN ({placeholders})",
            conn,
            params=ticker_params,
        )
        logger.debug(f"Loaded {len(df)} price records from options_quotes fallback")
    if df.empty:
        return pd.DataFrame()

    px = df.groupby(["asof_date", "ticker"])["close"].median().unstack("ticker").sort_index()
    ret = np.log(px / px.shift(1))
    # drop only all-NaN rows; keep partial rows for pairwise stats
    ret = ret.dropna(how="all")
    # return in rows=tickers layout
    return ret.T


# -----------------------------------------------------------------------------
# Public similarity/weight utilities (reused by beta_builder)
# -----------------------------------------------------------------------------
def cosine_similarity_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    target = target.upper()
    peers = [p.upper() for p in peers]
    if target not in feature_df.index:
        raise ValueError(f"target {target} not in feature matrix")

    df = feature_df.apply(pd.to_numeric, errors="coerce")
    X = _impute_col_median(df.to_numpy(float))
    tickers = list(df.index)
    t_idx = tickers.index(target)
    # Compare shape rather than level; de-mean each ticker vector and guard
    # degenerate near-zero norms explicitly.
    X = X - np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isfinite(X), X, 0.0)
    t_vec = X[t_idx]
    t_norm = float(np.linalg.norm(t_vec))
    if t_norm <= 1e-12:
        raise ValueError("target feature norm is near zero for cosine weights")

    sims: dict[str, float] = {}
    for i, peer in enumerate(tickers):
        if peer == target or peer not in peers:
            continue
        p_vec = X[i]
        p_norm = float(np.linalg.norm(p_vec))
        denom = t_norm * p_norm
        sims[peer] = float(np.dot(t_vec, p_vec) / denom) if denom > 1e-12 else 0.0

    ser = pd.Series(sims, dtype=float)
    if clip_negative:
        ser = ser.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        ser = ser.pow(float(power))
    total = float(ser.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("cosine similarity weights sum to zero")
    return (ser / total).reindex(peers).fillna(0.0)


def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: list[str],
    *,
    clip_negative: bool = True,
    power: float = 1.0,
    shrinkage: float = 0.05,
) -> pd.Series:
    target = target.upper()
    peers = [p.upper() for p in peers]
    corr_df = feature_df.apply(pd.to_numeric, errors="coerce").T.corr()
    corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr_df = corr_df.clip(lower=-1.0, upper=1.0)
    for t in corr_df.index.intersection(corr_df.columns):
        corr_df.loc[t, t] = 1.0
    shrinkage = min(max(float(shrinkage), 0.0), 1.0)
    if shrinkage > 0.0 and corr_df.shape[0] == corr_df.shape[1]:
        corr_df = (1.0 - shrinkage) * corr_df + shrinkage * pd.DataFrame(
            np.eye(corr_df.shape[0]), index=corr_df.index, columns=corr_df.columns
        )
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0]
    s = s.apply(pd.to_numeric, errors="coerce")
    if clip_negative:
        s = s.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        s = s.pow(float(power))
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("correlation weights sum to zero")
    return (s / total).reindex(peers).fillna(0.0)


def pca_regress_weights(
    X_peers: np.ndarray,
    y_target: np.ndarray,
    k: Optional[int] = None,
    *,
    nonneg: bool = True,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Ridge-stabilized least squares for ``X_peers.T @ w ~= y_target``."""
    X = _impute_col_median(np.asarray(X_peers, float))
    y = np.asarray(y_target, float).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("empty PCA peer feature matrix")
    if y.shape[0] != X.shape[1]:
        raise ValueError("PCA target/peer feature dimensions do not align")

    # Stable row-wise shape normalization. This avoids one high-volatility
    # feature scale dominating the regression and makes sign conventions stable.
    X = X - np.nanmean(X, axis=1, keepdims=True)
    y = y - float(np.nanmean(y))
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    y_norm = float(np.linalg.norm(y))
    if y_norm <= 1e-12 or np.any(X_norm <= 1e-12):
        raise ValueError("near-zero vector in PCA weighting")
    X = X / X_norm
    y = y / y_norm

    A = X.T  # features x peers
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if k is None or k <= 0 or k > len(s):
        k = len(s)
    U, s, Vt = U[:, :k], s[:k], Vt[:k, :]
    filt = s / (s * s + max(float(ridge), 0.0))
    w = Vt.T @ (filt * (U.T @ y))
    if nonneg:
        w = np.clip(w, 0.0, None)
        ssum = float(w.sum())
        if not np.isfinite(ssum) or ssum <= 1e-12:
            raise ValueError("PCA produced zero non-negative weights")
        return w / ssum
    scale = float(np.sum(np.abs(w)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("PCA produced zero signed weights")
    return w / scale


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
class UnifiedWeightComputer:
    """Unified weight computation engine with centralized feature building."""

    def _choose_asof(self, target: str, peers: list[str], config: WeightConfig) -> Optional[str]:
        """Robust as-of for options features; UL ignores."""
        if config.asof:
            return config.asof
        if config.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            from data.db_utils import get_conn

            tickers = [target] + peers
            conn = get_conn()
            placeholders = ",".join("?" * len(tickers))
            q = (
                "SELECT asof_date, COUNT(DISTINCT ticker) AS n "
                "FROM options_quotes WHERE ticker IN (" + placeholders + ") "
                "GROUP BY asof_date "
                "HAVING SUM(CASE WHEN ticker = ? THEN 1 ELSE 0 END) > 0 "
                "ORDER BY n DESC, asof_date DESC LIMIT 1"
            )
            params = [t.upper() for t in tickers] + [target.upper()]
            df = pd.read_sql_query(q, conn, params=params)
            if not df.empty:
                best_date = pd.to_datetime(df["asof_date"].iloc[0])
                best_n = int(df["n"].iloc[0])
                logger.debug(
                    "_choose_asof picked %s with %d/%d ticker coverage",
                    best_date,
                    best_n,
                    len(tickers),
                )
                return best_date.strftime("%Y-%m-%d")
            from analysis.services.data_availability_service import get_most_recent_date_global, available_dates

            d = get_most_recent_date_global()
            if d:
                return d
            dates = available_dates(ticker=target, most_recent_only=True)
            if dates:
                return dates[0]
        return None  # UL or if nothing found

    def _log_option_counts(self, tickers: list[str], asof: Optional[str], atm_df: Optional[pd.DataFrame]) -> None:
        if asof is None:
            return
        from data.db_utils import get_conn

        conn = get_conn()
        q = (
            "SELECT ticker, COUNT(*) AS n FROM options_quotes WHERE asof_date = ? "
            f"AND ticker IN ({','.join('?' * len(tickers))}) GROUP BY ticker"
        )
        df = pd.read_sql_query(q, conn, params=[asof] + [t.upper() for t in tickers])
        counts = {row["ticker"].upper(): int(row["n"]) for _, row in df.iterrows()}
        for t in tickers:
            total = counts.get(t.upper(), 0)
            atm_rows = 0
            if atm_df is not None and t.upper() in atm_df.index:
                atm_rows = int(atm_df.loc[t.upper()].count())
            logger.debug("asof %s %s: quotes=%d, atm_rows=%d", asof, t.upper(), total, atm_rows)

    # ---- public API ----
    def compute_weights(
        self,
        target: str,
        peers: Iterable[str],
        config: WeightConfig,
    ) -> pd.Series:
        target = (target or "").upper()
        peers_list = [p.upper() for p in peers]
        if not peers_list:
            return pd.Series(dtype=float)
        feature_df = None

        def fallback(
            reason: str,
            condition_number: Optional[float] = None,
            input_shape: Optional[tuple[int, ...]] = None,
            rejected_weights: Optional[pd.Series] = None,
        ) -> pd.Series:
            w_eq = self._fallback_equal(peers_list)
            w_eq.attrs["weight_warning"] = (
                f"{config.method.value} weights failed validation; using equal weights ({reason})"
            )
            diag = _diagnose_weights(
                w_eq,
                config=config,
                peers=peers_list,
                input_shape=input_shape or (len(peers_list),),
                normalization="sum",
                normalization_scale=float(w_eq.sum()),
                condition_number=condition_number,
                fallback="equal",
                issues=[reason],
                rejected_weights=rejected_weights,
            )
            _log_weight_diagnostics(diag, logging.WARNING)
            if feature_df is not None:
                w_eq.attrs["feature_diagnostics"] = dict(
                    getattr(feature_df, "attrs", {}).get("feature_diagnostics", {})
                )
            w_eq.attrs["weight_diagnostics"] = diag.as_log_dict()
            return w_eq

        # Equal weights
        if config.method == WeightMethod.EQUAL:
            w = 1.0 / len(peers_list)
            ser = pd.Series(w, index=peers_list, dtype=float)
            try:
                ser, diag = _validate_and_normalize_weights(
                    ser,
                    peers=peers_list,
                    config=config,
                    allow_signed=False,
                    input_shape=(len(peers_list),),
                )
                _log_weight_diagnostics(diag)
                return ser
            except Exception as e:
                return fallback(f"equal validation failed: {e}")

        # Open interest method
        if config.method == WeightMethod.OPEN_INTEREST:
            try:
                w_oi, condition = self._open_interest_weights(peers_list, config.asof)
                w_oi, diag = _validate_and_normalize_weights(
                    w_oi,
                    peers=peers_list,
                    config=config,
                    allow_signed=False,
                    input_shape=(len(peers_list), 1),
                    condition_number=condition,
                )
                _log_weight_diagnostics(diag)
                return w_oi
            except Exception as e:
                return fallback(f"open-interest validation failed: {e}", input_shape=(len(peers_list), 1))

        # Pick as-of only for options features
        asof = None
        if config.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
            asof = self._choose_asof(target, peers_list, config)
            if asof is None:
                raise ValueError("no surface/ATM date available to build features")

        # Build features
        feature_df = self._build_feature_matrix(target, peers_list, asof, config)
        if feature_df is None or feature_df.empty:
            return fallback(f"empty {config.feature_set.value} feature matrix")

        return self._compute_weights_from_features(feature_df, target, peers_list, config)

    def _compute_weights_from_features(
        self,
        feature_df: pd.DataFrame,
        target: str,
        peers_list: list[str],
        config: WeightConfig,
    ) -> pd.Series:
        """Compute weights from an already-built, contract-tagged feature matrix."""

        def fallback(
            reason: str,
            condition_number: Optional[float] = None,
            input_shape: Optional[tuple[int, ...]] = None,
            rejected_weights: Optional[pd.Series] = None,
        ) -> pd.Series:
            w_eq = self._fallback_equal(peers_list)
            w_eq.attrs["weight_warning"] = (
                f"{config.method.value} weights failed validation; using equal weights ({reason})"
            )
            diag = _diagnose_weights(
                w_eq,
                config=config,
                peers=peers_list,
                input_shape=input_shape or (len(peers_list),),
                normalization="sum",
                normalization_scale=float(w_eq.sum()),
                condition_number=condition_number,
                fallback="equal",
                issues=[reason],
                rejected_weights=rejected_weights,
            )
            _log_weight_diagnostics(diag, logging.WARNING)
            return w_eq

        try:
            condition_number = None
            attempted_weights: Optional[pd.Series] = None
            if config.method == WeightMethod.CORRELATION:
                corr_df = feature_df.apply(pd.to_numeric, errors="coerce").T.corr()
                corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                condition_number = _condition_number(corr_df.to_numpy(float))
                attempted_weights = corr_weights_from_matrix(
                    feature_df,
                    target,
                    peers_list,
                    clip_negative=config.clip_negative,
                    power=config.power,
                    shrinkage=config.corr_shrinkage,
                )
                w, diag = _validate_and_normalize_weights(
                    attempted_weights,
                    peers=peers_list,
                    config=config,
                    allow_signed=False,
                    input_shape=feature_df.shape,
                    condition_number=condition_number,
                )
                _log_weight_diagnostics(diag)
                return _attach_output_diagnostics(w, feature_df, diag)
            if config.method == WeightMethod.COSINE:
                condition_number = _condition_number(feature_df.to_numpy(float))
                attempted_weights = cosine_similarity_weights_from_matrix(
                    feature_df,
                    target,
                    peers_list,
                    clip_negative=config.clip_negative,
                    power=config.power,
                )
                w, diag = _validate_and_normalize_weights(
                    attempted_weights,
                    peers=peers_list,
                    config=config,
                    allow_signed=False,
                    input_shape=feature_df.shape,
                    condition_number=condition_number,
                )
                _log_weight_diagnostics(diag)
                return _attach_output_diagnostics(w, feature_df, diag)
            if config.method == WeightMethod.PCA:
                # PCA regression of target on peers
                y = _impute_col_median(feature_df.loc[[target]].to_numpy(float)).ravel()
                Xp = feature_df.loc[[p for p in peers_list if p in feature_df.index]].to_numpy(float)
                if Xp.size == 0:
                    raise ValueError("No peer data available for PCA weighting")
                condition_number = _condition_number(Xp)
                pca_fn = pca_regress_weights
                try:
                    w = pca_fn(
                        Xp,
                        y,
                        k=min(Xp.shape[0], Xp.shape[1]),
                        nonneg=config.clip_negative,
                        ridge=config.pca_ridge,
                    )
                except TypeError:
                    w = pca_fn(
                        Xp,
                        y,
                        k=min(Xp.shape[0], Xp.shape[1]),
                        nonneg=config.clip_negative,
                    )
                ser = pd.Series(w, index=[p for p in peers_list if p in feature_df.index])
                if config.clip_negative:
                    ser = ser.clip(lower=0.0)
                attempted_weights = ser.copy()
                ser, diag = _validate_and_normalize_weights(
                    ser,
                    peers=peers_list,
                    config=config,
                    allow_signed=not config.clip_negative,
                    input_shape=feature_df.shape,
                    condition_number=condition_number,
                )
                _log_weight_diagnostics(diag)
                return _attach_output_diagnostics(ser, feature_df, diag)
            raise ValueError(f"Unsupported method: {config.method}")
        except Exception as e:
            logger.warning("Weight computation failed (%s); using equal weights", e)
            return fallback(
                f"{config.method.value} failed: {e}",
                condition_number=condition_number,
                input_shape=feature_df.shape if feature_df is not None else None,
                rejected_weights=attempted_weights,
            )

    def _build_surface_features(self, tickers: list[str], asof: Optional[str], config: WeightConfig) -> pd.DataFrame:
        surface_df, names = native_surface_feature_matrix(
            tickers,
            asof,
            max_expiries=config.max_expiries,
            mny_bins=config.mny_bins,
        )
        surface_df, policy_note = _apply_surface_missing_policy(
            surface_df,
            policy=config.surface_missing_policy,
            min_coverage=config.surface_min_coverage,
        )
        self._log_option_counts(tickers, asof, None)
        self._attach_feature_diagnostics(
            surface_df,
            config=config,
            requested=tickers,
            asof=asof,
            coordinate_system="native_expiry_rank_x_moneyness_bin",
            value_type="iv_levels",
            missing_policy=policy_note,
            normalization="none",
            n_expiries=config.max_expiries,
            n_grid_points=surface_df.shape[1],
        )
        return surface_df

    # ---- feature assembly ----
    def _build_feature_matrix(
        self,
        target: str,
        peers_list: list[str],
        asof: Optional[str],
        config: WeightConfig,
    ) -> Optional[pd.DataFrame]:
        tickers = [target] + peers_list
        if config.feature_set == FeatureSet.ATM:
            atm_df, _, _ = atm_feature_matrix(
                tickers,
                asof,
                config.pillars_days,
                atm_band=config.atm_band,
                tol_days=config.atm_tol_days,
            )
            self._log_option_counts(tickers, asof, atm_df)
            self._attach_feature_diagnostics(
                atm_df,
                config=config,
                requested=tickers,
                asof=asof,
                coordinate_system="native_expiry_pillars",
                value_type="iv_levels",
                missing_policy="column median imputation for weighting; raw NaNs retained in frame",
                normalization="none",
                n_expiries=len(atm_df.columns),
            )
            return atm_df
        if config.feature_set == FeatureSet.ATM_RANKS:
            atm_df, _ = _atm_rank_feature_matrix(
                tickers,
                asof,
                max_expiries=config.max_expiries,
                atm_band=config.atm_band,
            )
            self._log_option_counts(tickers, asof, atm_df)
            self._attach_feature_diagnostics(
                atm_df,
                config=config,
                requested=tickers,
                asof=asof,
                coordinate_system="native_expiry_ranks",
                value_type="iv_levels",
                missing_policy="pairwise available ranks; raw NaNs retained in frame",
                normalization="none",
                n_expiries=len(atm_df.columns),
            )
            return atm_df
        if config.feature_set == FeatureSet.SURFACE:
            return self._build_surface_features(tickers, asof, config)
        if config.feature_set == FeatureSet.SURFACE_VECTOR:
            try:
                grids, X, names = surface_feature_matrix(
                    tickers,
                    asof,
                    tenors=config.tenors,
                    mny_bins=config.mny_bins,
                    surface_source=config.surface_source,
                    model=config.surface_model,
                    max_expiries=config.max_expiries,
                    missing_policy=config.surface_missing_policy,
                    min_coverage=config.surface_min_coverage,
                )
            except TypeError:
                grids, X, names = surface_feature_matrix(
                    tickers,
                    asof,
                    tenors=config.tenors,
                    mny_bins=config.mny_bins,
                )
            logger.debug(
                "surface_feature_matrix returned shape %s for tickers %s",
                X.shape,
                list(grids.keys()),
            )
            self._log_option_counts(tickers, asof, None)
            if X.size == 0 or not names:
                out = pd.DataFrame(index=list(grids.keys()))
                self._attach_feature_diagnostics(
                    out,
                    config=config,
                    requested=tickers,
                    asof=asof,
                    coordinate_system="standardized_tenor_grid_x_moneyness_bin",
                    value_type="standardized_iv_levels",
                    missing_policy=f"{config.surface_missing_policy}; min coverage {config.surface_min_coverage:.0%}",
                    normalization="column_zscore",
                    n_expiries=len(config.tenors),
                    n_grid_points=0,
                )
                return out
            out = pd.DataFrame(X, index=list(grids.keys()), columns=names)
            self._attach_feature_diagnostics(
                out,
                config=config,
                requested=tickers,
                asof=asof,
                coordinate_system="standardized_tenor_grid_x_moneyness_bin",
                value_type="standardized_iv_levels",
                missing_policy=f"{config.surface_missing_policy}; min coverage {config.surface_min_coverage:.0%}",
                normalization="column_zscore",
                n_expiries=len(config.tenors),
                n_grid_points=len(names),
            )
            return out
        if config.feature_set == FeatureSet.UNDERLYING_PX:
            df = underlying_returns_matrix(tickers)
            # Need at least two time rows (pairwise) to compute correlation
            if df.shape[1] < 2:
                return None
            self._attach_feature_diagnostics(
                df,
                config=config,
                requested=tickers,
                asof=None,
                coordinate_system="calendar_time_series",
                value_type="underlying_log_returns",
                missing_policy="pairwise returns; all-NaN dates dropped",
                normalization="returns",
                n_grid_points=df.shape[1],
            )
            return df
        return None

    @staticmethod
    def _attach_feature_diagnostics(
        feature_df: pd.DataFrame,
        *,
        config: WeightConfig,
        requested: list[str],
        asof: Optional[str],
        coordinate_system: str,
        value_type: str,
        missing_policy: str,
        normalization: str,
        n_expiries: Optional[int] = None,
        n_grid_points: Optional[int] = None,
    ) -> None:
        included = [str(x).upper() for x in feature_df.index]
        diag = FeatureDiagnostics(
            feature_set=config.feature_set.value,
            coordinate_system=coordinate_system,
            value_type=value_type,
            shape=tuple(feature_df.shape),
            tickers_requested=[str(t).upper() for t in requested],
            tickers_included=included,
            tickers_excluded=[str(t).upper() for t in requested if str(t).upper() not in included],
            missing_policy=missing_policy,
            normalization=normalization,
            asof=asof,
            n_expiries=n_expiries,
            n_grid_points=n_grid_points,
        )
        feature_df.attrs["feature_diagnostics"] = diag.as_log_dict()
        logger.info("feature diagnostics: %s", diag.as_log_dict())

    # ---- OI method ----
    def _open_interest_weights(self, peers_list: list[str], asof: Optional[str]) -> tuple[pd.Series, Optional[float]]:
        from data.db_utils import get_conn

        if not peers_list:
            return pd.Series(dtype=float), None
        conn = get_conn()
        # choose date if missing
        if asof is None:
            row = conn.execute(
                "SELECT MAX(asof_date) FROM options_quotes WHERE ticker IN ({})".format(
                    ",".join("?" * len(peers_list))
                ),
                peers_list,
            ).fetchone()
            asof = row[0] if row and row[0] else None
            if asof is None:
                raise ValueError("no open-interest date available")

        q = (
            "SELECT ticker, SUM(open_interest) AS oi, COUNT(open_interest) AS oi_rows, COUNT(*) AS rows "
            "FROM options_quotes WHERE asof_date = ? AND ticker IN ({}) GROUP BY ticker"
        ).format(",".join("?" * len(peers_list)))
        df = pd.read_sql_query(q, conn, params=[asof] + peers_list)
        if df.empty:
            raise ValueError("no open-interest rows available")
        coverage = float(df["oi_rows"].sum()) / max(float(df["rows"].sum()), 1.0)
        if coverage < 0.80 or set(df["ticker"].str.upper()) != set(peers_list):
            raise ValueError(f"open-interest coverage {coverage:.1%} is too sparse")
        s = pd.Series(df["oi"].values, index=df["ticker"].str.upper())
        s = s.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s = s.clip(lower=0.0)
        total = float(s.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("open-interest sum is zero")
        condition = _condition_number(s.to_numpy(float).reshape(-1, 1))
        return (s / total).reindex(peers_list).fillna(0.0), condition

    @staticmethod
    def _fallback_equal(peers_list: list[str]) -> pd.Series:
        if not peers_list:
            return pd.Series(dtype=float)
        w = 1.0 / len(peers_list)
        return pd.Series(w, index=peers_list, dtype=float)


# Global instance
_weight_computer = UnifiedWeightComputer()


# -----------------------------------------------------------------------------
# Top-level API
# -----------------------------------------------------------------------------
def compute_unified_weights(
    target: str,
    peers: Iterable[str],
    mode: Union[str, WeightConfig],
    **kwargs,
) -> pd.Series:
    """
    Args
    ----
    target : str
    peers  : Iterable[str]
    mode   : weight mode string (e.g., "corr_iv_atm", "pca_surface_grid", "ul", "equal", "oi")
             or WeightConfig
    **kwargs : forwarded to :meth:`WeightConfig.from_mode`
    """
    if isinstance(mode, str):
        cfg = WeightConfig.from_mode(mode, **kwargs)
    else:
        cfg = mode
    return _weight_computer.compute_weights(target, peers, cfg)


def build_weight_feature_matrix(
    target: str,
    peers: Iterable[str],
    mode: Union[str, WeightConfig],
    **kwargs,
) -> pd.DataFrame:
    """Build the exact feature matrix used by unified weights."""
    if isinstance(mode, str):
        cfg = WeightConfig.from_mode(mode, **kwargs)
    else:
        cfg = mode
    target = (target or "").upper()
    peers_list = [str(p).upper() for p in peers]
    asof = cfg.asof
    if cfg.feature_set in (FeatureSet.ATM, FeatureSet.ATM_RANKS, FeatureSet.SURFACE, FeatureSet.SURFACE_VECTOR):
        asof = _weight_computer._choose_asof(target, peers_list, cfg)
    feature_df = _weight_computer._build_feature_matrix(target, peers_list, asof, cfg)
    if feature_df is None:
        return pd.DataFrame()
    return feature_df


def similarity_matrix_from_features(feature_df: pd.DataFrame, method: str = "corr") -> pd.DataFrame:
    """Compute a display similarity matrix from an already-built feature matrix."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()
    method = str(method or "corr").lower()
    df = feature_df.apply(pd.to_numeric, errors="coerce")
    X = _impute_col_median(df.to_numpy(float))
    tickers = list(df.index)
    if method == "cosine":
        Xc = X - np.nanmean(X, axis=1, keepdims=True)
        Xc = np.where(np.isfinite(Xc), Xc, 0.0)
        norms = np.linalg.norm(Xc, axis=1)
        denom = norms[:, None] * norms[None, :]
        sim = np.divide(Xc @ Xc.T, denom, out=np.zeros((len(tickers), len(tickers))), where=denom > 1e-12)
        np.fill_diagonal(sim, 1.0)
        return pd.DataFrame(sim, index=tickers, columns=tickers)
    if method == "pca":
        mat = np.full((len(tickers), len(tickers)), np.nan, dtype=float)
        np.fill_diagonal(mat, 1.0)
        for i, ticker in enumerate(tickers):
            peer_idx = [j for j in range(len(tickers)) if j != i]
            if not peer_idx:
                continue
            try:
                w = pca_regress_weights(
                    X[peer_idx, :],
                    X[i, :],
                    k=min(len(peer_idx), X.shape[1]),
                    nonneg=True,
                )
            except Exception as exc:
                logger.warning("PCA display matrix row failed for %s: %s", ticker, exc)
                continue
            if len(w) != len(peer_idx) or not np.isfinite(w).all():
                logger.warning("PCA display matrix row rejected for %s: invalid weights", ticker)
                continue
            mat[i, peer_idx] = w
        out = pd.DataFrame(mat, index=tickers, columns=tickers)
        out.attrs["display_kind"] = "directed_pca_regression_weights"
        return out
    corr = pd.DataFrame(X, index=tickers, columns=df.columns).T.corr()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-1.0, upper=1.0)
    for t in corr.index.intersection(corr.columns):
        corr.loc[t, t] = 1.0
    return corr
