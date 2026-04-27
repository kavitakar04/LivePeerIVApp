"""Analysis-side preparation for correlation detail views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from analysis.cache_io import compute_or_load
from analysis.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_WEIGHT_METHOD,
    DEFAULT_WEIGHT_POWER,
)
from analysis.feature_health import build_feature_construction_result
from analysis.unified_weights import compute_unified_weights, similarity_matrix_from_features


@dataclass(frozen=True)
class CorrelationViewData:
    """Prepared data needed to render the correlation detail plot."""

    atm_df: pd.DataFrame
    corr_df: pd.DataFrame
    weights: Optional[pd.Series]
    coverage: pd.Series
    overlap: Optional[pd.DataFrame]
    finite_count: int
    total_cells: int
    finite_ratio: float
    method_label: str
    basis_label: str
    context: dict


def compute_atm_curve_simple(df: pd.DataFrame, atm_band: float = DEFAULT_ATM_BAND) -> pd.DataFrame:
    """Compute ATM implied volatility per expiry from option quotes.

    Prefers is_atm=1 rows (flagged at ingestion via delta proximity) before
    falling back to the moneyness-band median.
    """
    need_cols = {"T", "moneyness", "sigma"}
    if df is None or df.empty or not need_cols.issubset(df.columns):
        return pd.DataFrame(columns=["T", "atm_vol"])

    d = df.copy()
    d["T"] = pd.to_numeric(d["T"], errors="coerce")
    d["moneyness"] = pd.to_numeric(d["moneyness"], errors="coerce")
    d["sigma"] = pd.to_numeric(d["sigma"], errors="coerce")
    d = d.dropna(subset=["T", "moneyness", "sigma"])

    rows: list[dict[str, float]] = []
    for T_val, grp in d.groupby("T"):
        g = grp.dropna(subset=["moneyness", "sigma"])

        # 1. DB-flagged ATM option (delta-based, set at ingestion)
        if "is_atm" in g.columns:
            atm_iv = pd.to_numeric(g.loc[g["is_atm"] == 1, "sigma"], errors="coerce").dropna()
            if not atm_iv.empty:
                rows.append({"T": float(T_val), "atm_vol": float(atm_iv.median())})
                continue

        # 2. Moneyness band fallback
        in_band = g.loc[(g["moneyness"] - 1.0).abs() <= atm_band]
        if not in_band.empty:
            atm_vol = float(in_band["sigma"].median())
        else:
            idx = int((g["moneyness"] - 1.0).abs().idxmin())
            atm_vol = float(g.loc[idx, "sigma"])
        rows.append({"T": float(T_val), "atm_vol": atm_vol})

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def corr_by_expiry_rank(
    get_slice: Callable,
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    atm_band: float = DEFAULT_ATM_BAND,
    min_tickers: int = 2,
    corr_method: str = "pearson",
    min_periods: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an ATM-rank matrix and pairwise ticker correlation matrix.

    Batch-loads all ticker slices in one DB query to avoid N separate round-trips.
    """
    tickers = [str(t).upper() for t in tickers]

    # Single DB query for all tickers on this date
    slices: dict = {}
    try:
        from analysis.smile_data_service import get_smile_slices_batch

        slices = get_smile_slices_batch(tickers, asof, max_expiries=max_expiries)
    except Exception:
        pass  # fallback to per-ticker callable below

    rows: list[pd.Series] = []

    for ticker in tickers:
        df = slices.get(ticker) if slices else None
        if df is None or df.empty:
            # fallback: per-ticker callable (preserves backward compat)
            try:
                df = get_slice(ticker, asof_date=asof, T_target_years=None, call_put=None, nearest_by="T")
            except TypeError:
                try:
                    df = get_slice(ticker, asof, T_target_years=None)
                except Exception:
                    df = None
            except Exception:
                df = None

        if df is None or df.empty:
            values = {i: np.nan for i in range(max_expiries)}
            rows.append(pd.Series(values, name=ticker))
            continue

        atm_df = compute_atm_curve_simple(df, atm_band=atm_band)
        values = {}
        for i in range(max_expiries):
            if i < len(atm_df):
                v = atm_df.at[i, "atm_vol"]
                values[i] = float(v) if pd.notna(v) else np.nan
            else:
                values[i] = np.nan
        rows.append(pd.Series(values, name=ticker))

    atm_rank_df = pd.DataFrame(rows)
    if atm_rank_df.empty or len(atm_rank_df.index) < int(min_tickers):
        corr_df = pd.DataFrame(index=atm_rank_df.index, columns=atm_rank_df.index, dtype=float)
    else:
        corr_df = atm_rank_df.transpose().corr(
            method=corr_method,
            min_periods=int(min_periods),
        )
        corr_df = corr_df.reindex(index=atm_rank_df.index, columns=atm_rank_df.index)
    return atm_rank_df, corr_df


def maybe_compute_weights(
    target: Optional[str],
    peers: Optional[Iterable[str]],
    *,
    asof: str,
    weight_mode: str,
    weight_power: float,
    clip_negative: bool,
    **weight_config,
) -> Optional[pd.Series]:
    """Compute unified peer weights, returning None when unavailable."""
    if not target or not peers:
        return None
    peers_list = [str(p).upper() for p in peers]
    try:
        return compute_unified_weights(
            target=str(target).upper(),
            peers=peers_list,
            mode=weight_mode,
            asof=asof,
            clip_negative=clip_negative,
            power=weight_power,
            **weight_config,
        )
    except Exception:
        return None


def coverage_by_ticker(corr_df: pd.DataFrame, atm_df: Optional[pd.DataFrame]) -> pd.Series:
    """Count finite ATM-rank features available for each ticker."""
    if atm_df is None or atm_df.empty:
        return pd.Series(dtype=float)
    coverage = atm_df.apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    return coverage.reindex(corr_df.index).fillna(0).astype(int)


def overlap_counts(corr_df: pd.DataFrame, atm_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Pairwise count of shared finite ATM-rank observations."""
    if atm_df is None or atm_df.empty:
        return None
    aligned = atm_df.reindex(corr_df.index)
    valid = aligned.apply(pd.to_numeric, errors="coerce").notna().astype(int)
    counts = valid.to_numpy() @ valid.to_numpy().T
    return pd.DataFrame(counts, index=aligned.index, columns=aligned.index)


def finite_cell_summary(corr_df: pd.DataFrame) -> tuple[int, int, float]:
    """Return finite matrix cell count, total cells, and finite ratio."""
    data = corr_df.to_numpy(dtype=float)
    finite_count = int(np.sum(np.isfinite(data)))
    total_cells = int(data.size)
    ratio = finite_count / total_cells if total_cells else 0.0
    return finite_count, total_cells, ratio


def split_weight_mode(weight_mode: Optional[str]) -> tuple[str, str]:
    """Return user-facing method and basis labels from a canonical mode."""
    mode = str(weight_mode or "").strip().lower()
    if not mode:
        return "not selected", "expiry-rank ATM IV"
    if mode == "oi":
        return "open interest", "contracts"
    if "_" in mode:
        method, basis = mode.split("_", 1)
    else:
        method, basis = mode, "iv_atm"

    method_labels = {
        "corr": "correlation",
        "pca": "PCA",
        "cosine": "cosine",
        "equal": "equal",
        "oi": "open interest",
    }
    basis_labels = {
        "iv_atm": "ATM IV",
        "iv_atm_ranks": "ATM IV expiry ranks",
        "ul": "underlying returns",
        "surface": "IV surface",
        "surface_grid": "IV surface grid",
    }
    return method_labels.get(method, method), basis_labels.get(basis, basis)


def prepare_correlation_view(
    get_smile_slice: Callable,
    tickers: Iterable[str],
    asof: str,
    *,
    target: Optional[str] = None,
    peers: Optional[Iterable[str]] = None,
    atm_band: float = DEFAULT_ATM_BAND,
    clip_negative: bool = DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    weight_power: float = DEFAULT_WEIGHT_POWER,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    weight_mode: str = DEFAULT_WEIGHT_METHOD,
    show_values: bool = True,
    use_cache: bool = True,
    **weight_config,
) -> CorrelationViewData:
    """Prepare all non-visual data for the correlation detail plot."""
    tickers = [str(t).upper() for t in tickers]
    payload = {
        "tickers": sorted(tickers),
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "atm_band": float(atm_band),
        "max_expiries": int(max_expiries),
        "weight_mode": str(weight_mode),
        "clip_negative": bool(clip_negative),
        "weight_power": float(weight_power),
        "weight_config": {k: str(v) for k, v in sorted(weight_config.items())},
    }

    def _builder() -> tuple[pd.DataFrame, pd.DataFrame]:
        method_label, _basis_label = split_weight_mode(weight_mode)
        method = str(weight_mode or "corr").split("_", 1)[0]
        feature_result = build_feature_construction_result(
            target=target or tickers[0],
            peers=[t for t in tickers if t != str(target or tickers[0]).upper()],
            weight_mode=weight_mode,
            asof=asof,
            atm_band=atm_band,
            max_expiries=max_expiries,
            use_cache=use_cache,
            **weight_config,
        )
        feature_df = feature_result.feature_matrix
        if feature_df.empty:
            return pd.DataFrame(index=tickers), pd.DataFrame(index=tickers, columns=tickers, dtype=float)
        display_method = method if method in {"corr", "cosine", "pca"} else "corr"
        sim_df = similarity_matrix_from_features(feature_df, method=display_method)
        sim_df.attrs["display_method"] = display_method
        sim_df.attrs["requested_method"] = method_label
        return feature_df, sim_df

    if use_cache:
        atm_df, corr_df = compute_or_load("corr", payload, _builder)
    else:
        atm_df, corr_df = _builder()

    weights = maybe_compute_weights(
        target=target,
        peers=peers,
        asof=asof,
        weight_mode=weight_mode,
        weight_power=weight_power,
        clip_negative=clip_negative,
        **weight_config,
    )
    coverage = coverage_by_ticker(corr_df, atm_df)
    overlap = overlap_counts(corr_df, atm_df)
    finite_count, total_cells, finite_ratio = finite_cell_summary(corr_df)
    method_label, basis_label = split_weight_mode(weight_mode)

    return CorrelationViewData(
        atm_df=atm_df,
        corr_df=corr_df,
        weights=weights,
        coverage=coverage,
        overlap=overlap,
        finite_count=finite_count,
        total_cells=total_cells,
        finite_ratio=finite_ratio,
        method_label=method_label,
        basis_label=basis_label,
        context={
            "target": str(target).upper() if target else None,
            "asof": asof,
            "weight_mode": weight_mode,
            "max_expiries": int(max_expiries),
            "atm_band": float(atm_band),
            "clip_negative": bool(clip_negative),
            "weight_power": float(weight_power),
            "tickers": tickers,
            "feature_diagnostics": getattr(atm_df, "attrs", {}).get("feature_diagnostics", {}),
            "feature_health": build_feature_construction_result(
                target=target or tickers[0],
                peers=[t for t in tickers if t != str(target or tickers[0]).upper()],
                weight_mode=weight_mode,
                asof=asof,
                atm_band=atm_band,
                max_expiries=max_expiries,
                use_cache=True,
                **weight_config,
            ).feature_health,
            "similarity_display_method": getattr(corr_df, "attrs", {}).get("display_method", "corr"),
        },
    )
