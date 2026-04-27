from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analysis.cache_io import compute_or_load


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def summarize_feature_health(feature_df: pd.DataFrame | None, *, target: str | None = None) -> dict[str, Any]:
    """Summarize feature construction quality for similarity/weight diagnostics."""
    if feature_df is None or feature_df.empty:
        return {
            "available": False,
            "warnings": ["Feature matrix is unavailable."],
            "summary": {},
            "distribution": [],
            "pairs": [],
            "alignment": {"shared_grid": False},
            "transformation_log": [],
        }

    df = feature_df.apply(pd.to_numeric, errors="coerce")
    diag = dict(getattr(feature_df, "attrs", {}).get("feature_diagnostics", {}) or {})
    total_points = int(df.shape[1])
    finite = df.notna()
    coverage = finite.sum(axis=1).astype(float) / max(total_points, 1)
    warnings: list[str] = []

    distribution: list[dict[str, Any]] = []
    for ticker, row in df.iterrows():
        vals = pd.to_numeric(row, errors="coerce").dropna().to_numpy(float)
        distribution.append(
            {
                "ticker": str(ticker),
                "coverage": float(coverage.loc[ticker]),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            }
        )

    stds = np.array([_safe_float(r["std"]) for r in distribution], dtype=float)
    finite_stds = stds[np.isfinite(stds) & (stds > 1e-12)]
    if finite_stds.size >= 2 and float(np.nanmax(finite_stds) / np.nanmin(finite_stds)) > 5.0:
        warnings.append("Scale mismatch: per-ticker feature standard deviations differ by more than 5x.")
    if (coverage < 0.70).any():
        sparse = ", ".join(str(x) for x in coverage[coverage < 0.70].index)
        warnings.append(f"Missing regions: feature coverage below 70% for {sparse}.")

    target_u = str(target or df.index[0]).upper()
    if target_u not in df.index:
        target_u = str(df.index[0])
    target_row = pd.to_numeric(df.loc[target_u], errors="coerce")
    pairs: list[dict[str, Any]] = []
    for ticker, row in df.iterrows():
        ticker_s = str(ticker)
        if ticker_s == target_u:
            continue
        peer = pd.to_numeric(row, errors="coerce")
        both = target_row.notna() & peer.notna()
        n = int(both.sum())
        if n >= 2:
            a = target_row[both].to_numpy(float)
            b = peer[both].to_numpy(float)
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
            mean_diff = float(np.mean(a - b))
            sign_consistency = float((np.sign(a - np.mean(a)) == np.sign(b - np.mean(b))).mean())
        else:
            corr = np.nan
            mean_diff = np.nan
            sign_consistency = np.nan
        flag = ""
        if np.isfinite(corr) and corr < 0.20:
            flag = "low correlation"
        if np.isfinite(sign_consistency) and sign_consistency < 0.50:
            flag = "low sign consistency" if not flag else f"{flag}; low sign consistency"
        pairs.append(
            {
                "ticker": ticker_s,
                "common_points": n,
                "correlation": corr,
                "mean_difference": mean_diff,
                "sign_consistency": sign_consistency,
                "flag": flag,
            }
        )
    if any(p.get("flag") for p in pairs):
        warnings.append("Pair diagnostics flagged unusual target/peer feature relationships.")

    coordinate_system = str(diag.get("coordinate_system", "unknown"))
    normalization = str(diag.get("normalization", "unknown"))
    missing_policy = str(diag.get("missing_policy", "unknown"))
    shared_grid = "standardized" in coordinate_system or "grid" in coordinate_system
    if "surface_grid" in str(diag.get("feature_set", "")) and not shared_grid:
        warnings.append(
            "Inconsistent normalization/alignment: surface_grid is not marked as a shared standardized grid."
        )
    if "interpolation" in missing_policy.lower() or "imputation" in missing_policy.lower():
        sparse_cols = int((finite.sum(axis=0) < max(2, int(0.5 * len(df)))).sum())
    else:
        sparse_cols = int((finite.sum(axis=0) == 0).sum())
    if sparse_cols:
        warnings.append(f"Sparse areas: {sparse_cols} feature point(s) have weak or no cross-ticker coverage.")

    transformation_log = [
        "raw option quotes / underlying observations",
        "filtered to valid finite inputs",
    ]
    if "surface" in str(diag.get("feature_set", "")):
        transformation_log.extend(
            ["aligned by moneyness/tenor definition", "resampled to configured feature coordinates"]
        )
    if normalization not in {"", "none", "unknown"}:
        transformation_log.append(f"normalized using {normalization}")
    else:
        transformation_log.append("levels retained without normalization")

    return {
        "available": True,
        "warnings": warnings,
        "summary": {
            "feature_set": diag.get("feature_set", ""),
            "coordinate_system": coordinate_system,
            "normalization": normalization,
            "missing_policy": missing_policy,
            "shape": tuple(df.shape),
            "total_points": total_points,
            "tickers": [str(x) for x in df.index],
        },
        "distribution": distribution,
        "pairs": pairs,
        "alignment": {
            "shared_grid": bool(shared_grid),
            "coordinate_system": coordinate_system,
            "sparse_points": sparse_cols,
        },
        "transformation_log": transformation_log,
    }


@dataclass(frozen=True)
class FeatureConstructionResult:
    feature_matrix: pd.DataFrame
    alignment_metadata: dict[str, Any]
    coverage: dict[str, float]
    warnings: list[str]
    feature_health: dict[str, Any]


def build_feature_construction_result(
    *,
    target: str,
    peers: list[str] | tuple[str, ...],
    asof: str | None,
    weight_mode: str,
    atm_band: float | None = None,
    max_expiries: int | None = None,
    use_cache: bool = True,
    **weight_config,
) -> FeatureConstructionResult:
    """Canonical feature-construction service for weights, matrices, and health."""
    target_u = str(target or "").upper()
    peers_u = [str(p).upper() for p in peers if str(p).strip()]
    payload = {
        "target": target_u,
        "peers": sorted(peers_u),
        "asof": str(asof or ""),
        "weight_mode": str(weight_mode or ""),
        "atm_band": "" if atm_band is None else float(atm_band),
        "max_expiries": "" if max_expiries is None else int(max_expiries),
        "weight_config": {k: str(v) for k, v in sorted(weight_config.items())},
    }

    def _builder() -> FeatureConstructionResult:
        from analysis.unified_weights import build_weight_feature_matrix

        kwargs = dict(weight_config)
        if asof is not None:
            kwargs["asof"] = asof
        if atm_band is not None:
            kwargs["atm_band"] = atm_band
        if max_expiries is not None:
            kwargs["max_expiries"] = max_expiries
        feature_df = build_weight_feature_matrix(
            target=target_u,
            peers=peers_u,
            mode=weight_mode,
            **kwargs,
        )
        health = summarize_feature_health(feature_df, target=target_u)
        coverage = {
            str(row.get("ticker")): float(row.get("coverage"))
            for row in health.get("distribution", [])
            if row.get("ticker") is not None and np.isfinite(_safe_float(row.get("coverage")))
        }
        alignment = dict(health.get("alignment") or {})
        alignment["feature_diagnostics"] = dict(getattr(feature_df, "attrs", {}).get("feature_diagnostics", {}) or {})
        return FeatureConstructionResult(
            feature_matrix=feature_df,
            alignment_metadata=alignment,
            coverage=coverage,
            warnings=[str(w) for w in health.get("warnings") or []],
            feature_health=health,
        )

    if use_cache:
        return compute_or_load("feature_construction", payload, _builder)
    return _builder()
