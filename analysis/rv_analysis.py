"""
Relative Value analysis: surface residuals, skew/curvature spreads,
term-structure shape dislocation, ranked signal generation, and
weight stability tracking.

All public functions return plain pandas DataFrames or plain dicts so
callers are not forced into any custom class.  Heavy imports are delayed
inside functions to avoid circular-import issues with analysis_pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import numpy as np
import pandas as pd


def _safe_float(val) -> float:
    """Convert *val* to float; return ``np.nan`` if not finite or not convertible."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _empty_dashboard() -> dict[str, Any]:
    cols = [
        "rank", "opportunity", "direction", "metric", "feature", "maturity", "spread",
        "z_score", "percentile", "confidence", "event_context",
        "spillover_support", "data_quality", "why", "what_differs",
        "why_matters", "statistical_read", "comparability", "warnings",
        "tradeability_score", "signal",
    ]
    return {
        "executive_summary": [],
        "context_cards": {
            "strongest_dislocation": "Unavailable",
            "most_tradeable": "Unavailable",
            "most_systemic": "Unavailable",
            "weakest_signal": "Unavailable",
            "data_quality_warnings": 0,
        },
        "opportunities": pd.DataFrame(columns=cols),
        "warnings": [],
        "integration_status": {},
    }


def _label_feature(signal_type: str) -> str:
    text = str(signal_type or "").lower()
    if "atm" in text:
        return "ATM"
    if "skew" in text:
        return "Skew"
    if "curv" in text:
        return "Curvature"
    if "event" in text or "bump" in text:
        return "Event Bump"
    if "slope" in text or "level" in text or "term" in text or text.startswith("ts"):
        return "Term Structure"
    return "Surface"


def _metric_family(signal_type: str) -> str:
    text = str(signal_type or "").lower()
    if "atm" in text or "level" in text:
        return "level"
    if "slope" in text or "term" in text or text.startswith("ts"):
        return "slope"
    if "skew" in text:
        return "asymmetry"
    if "curv" in text:
        return "convexity"
    if "event" in text or "bump" in text:
        return "timing"
    return "level"


def _metric_label(metric_family: str) -> str:
    labels = {
        "level": "Vol level",
        "slope": "Term structure",
        "asymmetry": "Smile asymmetry",
        "convexity": "Smile convexity",
        "timing": "Event timing",
    }
    return labels.get(metric_family, "Vol surface")


def _direction_from_spread(spread: float, eps: float = 1e-8) -> str:
    if not np.isfinite(spread) or abs(spread) <= eps:
        return "Neutral"
    return "Rich" if spread > 0 else "Cheap"


def _format_maturity(days: Any) -> str:
    d = _safe_float(days)
    if not np.isfinite(d) or d <= 0:
        return "All"
    return f"{int(round(d))}d"


def _fmt_signed_pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.2%}"


def _fmt_metric_value(metric_family: str, value: float, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "n/a"
    family = str(metric_family)
    if family in {"level", "timing"}:
        return f"{value:+.2%}" if signed else f"{value:.2%}"
    sign = "+" if signed else ""
    return f"{value:{sign}.4f}"


def _calculation_details(
    *,
    target: str,
    metric: str,
    metric_family: str,
    target_value: float,
    synthetic_value: float,
    spread: float,
    direction: str,
    reference_label: str = "Weighted peer synthetic",
) -> dict[str, Any]:
    value_unit = "IV" if metric_family in {"level", "timing"} else "fit coefficient"
    reference_text = str(reference_label or "Weighted peer synthetic")
    reference_formula = "weighted peer" if reference_text == "Weighted peer synthetic" else reference_text.lower()
    level_reference_formula = (
        "weighted peer synthetic"
        if reference_text == "Weighted peer synthetic"
        else reference_text.lower()
    )
    formula = f"spread = target value - {level_reference_formula} value"
    if metric_family == "convexity":
        formula = f"convexity spread = target smile curvature - {reference_formula} smile curvature"
    elif metric_family == "asymmetry":
        formula = f"asymmetry spread = target smile skew - {reference_formula} smile skew"
    elif metric_family == "slope":
        formula = f"term spread = target term-structure coefficient - {reference_formula} term-structure coefficient"
    return {
        "metric": metric,
        "unit": value_unit,
        "target_label": f"{target} {metric.lower()}",
        "target_value": target_value,
        "synthetic_label": reference_text,
        "synthetic_value": synthetic_value,
        "spread": spread,
        "direction": direction,
        "formula": formula,
        "display": (
            f"{formula}: "
            f"{_fmt_metric_value(metric_family, spread, signed=True)} = "
            f"{_fmt_metric_value(metric_family, target_value)} - "
            f"{_fmt_metric_value(metric_family, synthetic_value)}"
        ),
    }


def _alignment_score(comparability: str, surface_meta: Mapping[str, Any] | None = None) -> float:
    meta = dict(surface_meta or {})
    corr = _safe_float(meta.get("avg_surface_corr"))
    if np.isfinite(corr):
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    return {
        "Comparable": 0.85,
        "Mixed": 0.55,
        "Poor": 0.20,
        "Unknown": 0.35,
    }.get(str(comparability), 0.35)


def _spillover_score(spill_meta: Mapping[str, Any] | None, spillover_label: str) -> float:
    meta = dict(spill_meta or {})
    hit = _safe_float(meta.get("hit_rate"))
    same = _safe_float(meta.get("same_direction_probability"))
    parts = [v for v in (hit, same) if np.isfinite(v)]
    if parts:
        return float(np.clip(np.mean(parts), 0.0, 1.0))
    if "Strong" in spillover_label:
        return 0.80
    if "Suggestive" in spillover_label:
        return 0.65
    if "Weak" in spillover_label:
        return 0.35
    return 0.25


def _data_quality_score(data_quality: str) -> float:
    return {
        "Good": 1.0,
        "Acceptable": 0.75,
        "Unknown": 0.45,
        "Degraded": 0.25,
        "Poor": 0.10,
    }.get(str(data_quality), 0.45)


def _confidence_label(
    z: float,
    percentile: float,
    spillover_label: str,
    data_quality: str,
    *,
    alignment: float = 0.50,
    spillover_strength: float | None = None,
) -> tuple[str, float]:
    z_abs = abs(z) if np.isfinite(z) else 0.0
    pct_edge = abs(percentile - 50.0) / 50.0 if np.isfinite(percentile) else 0.0
    statistical = min(1.0, 0.75 * min(z_abs / 3.0, 1.0) + 0.25 * pct_edge)
    if not np.isfinite(z):
        statistical = min(0.45, abs(pct_edge))
    spill = _spillover_score({}, spillover_label) if spillover_strength is None else spillover_strength
    quality = _data_quality_score(data_quality)
    score = 0.42 * statistical + 0.24 * float(np.clip(alignment, 0.0, 1.0))
    score += 0.18 * float(np.clip(spill, 0.0, 1.0)) + 0.16 * quality
    if quality <= 0.25:
        score *= 0.65
    if alignment < 0.35:
        score *= 0.75
    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.75:
        return "High", score
    if score >= 0.45:
        return "Medium", score
    return "Low", score


def _load_model_quality(target: str, peers: list[str], asof: str | None) -> tuple[str, list[str], dict[str, Any]]:
    warnings: list[str] = []
    meta: dict[str, Any] = {"model": "Unknown", "degraded_rows": 0, "rmse_max": np.nan}
    try:
        from analysis.model_params_logger import load_model_params

        df = load_model_params()
    except Exception as exc:
        return "Unknown", ["Model parameter log is unavailable."], meta

    if df is None or df.empty:
        return "Unknown", ["No logged model-fit parameters available."], meta

    tickers = [target] + peers
    work = df[df["ticker"].astype(str).str.upper().isin(tickers)].copy()
    if asof is not None and "asof_date" in work:
        asof_ts = pd.to_datetime(asof, errors="coerce")
        if pd.notna(asof_ts):
            work = work[pd.to_datetime(work["asof_date"], errors="coerce") <= asof_ts]
    if work.empty:
        return "Unknown", ["No model-fit parameters found for this peer group/date."], meta

    latest_date = pd.to_datetime(work["asof_date"], errors="coerce").max()
    latest = work[pd.to_datetime(work["asof_date"], errors="coerce") == latest_date]
    models = sorted({str(m).upper() for m in latest.get("model", pd.Series(dtype=str)).dropna().unique()})
    meta["model"] = ", ".join(models) if models else "Unknown"

    degraded = 0
    rmse_vals: list[float] = []
    if "fit_meta" in latest:
        for item in latest["fit_meta"]:
            if isinstance(item, dict):
                if bool(item.get("degraded")) or str(item.get("status", "")).lower() in {"degraded", "fallback"}:
                    degraded += 1
                rmse = _safe_float(item.get("rmse", item.get("RMSE", np.nan)))
                if np.isfinite(rmse):
                    rmse_vals.append(rmse)
    if "param" in latest and "value" in latest:
        rmse_rows = latest[latest["param"].astype(str).str.lower().isin({"rmse", "fit_rmse"})]
        rmse_vals.extend([_safe_float(v) for v in rmse_rows["value"]])

    rmse_vals = [v for v in rmse_vals if np.isfinite(v)]
    rmse_max = max(rmse_vals) if rmse_vals else np.nan
    meta["degraded_rows"] = degraded
    meta["rmse_max"] = rmse_max

    if degraded:
        warnings.append(f"{degraded} logged fit rows are marked degraded.")
    if np.isfinite(rmse_max) and rmse_max > 0.15:
        warnings.append(f"High model RMSE observed ({rmse_max:.3f}).")

    if warnings:
        return "Degraded", warnings, meta
    if models:
        return "Good", [], meta
    return "Unknown", ["Model used could not be identified from parameter log."], meta


def _load_spillover_support(target: str, peers: list[str]) -> tuple[str, dict[str, Any], list[str]]:
    path = Path(__file__).resolve().parents[1] / "data" / "spill_summary.parquet"
    if not path.exists():
        return "Unavailable", {}, ["Spillover summary has not been generated."]
    try:
        summary = pd.read_parquet(path)
    except Exception as exc:
        return "Unavailable", {}, [f"Spillover summary could not be read: {exc}"]
    if summary.empty:
        return "Unavailable", {}, ["Spillover summary is empty."]

    target_u = target.upper()
    peer_set = {p.upper() for p in peers}
    work = summary.copy()
    work["ticker"] = work["ticker"].astype(str).str.upper()
    work["peer"] = work["peer"].astype(str).str.upper()
    direct = work[(work["ticker"] == target_u) & (work["peer"].isin(peer_set))]
    reverse = work[(work["peer"] == target_u) & (work["ticker"].isin(peer_set))]
    rel = pd.concat([direct, reverse], ignore_index=True)
    if rel.empty:
        return "Unavailable", {}, ["No spillover rows match this target/peer group."]

    if "h" in rel:
        h1 = rel[pd.to_numeric(rel["h"], errors="coerce") == 1]
        if not h1.empty:
            rel = h1
    hit = float(pd.to_numeric(rel.get("hit_rate"), errors="coerce").mean())
    same = float(pd.to_numeric(rel.get("sign_concord"), errors="coerce").mean())
    med = float(pd.to_numeric(rel.get("median_resp"), errors="coerce").median())
    strength_counts = rel.get("strength", pd.Series(dtype=str)).astype(str).value_counts()
    strength = strength_counts.index[0] if not strength_counts.empty else "Unknown"
    meta = {
        "hit_rate": hit,
        "same_direction_probability": same,
        "median_response": med,
        "strength": strength,
        "rows": int(len(rel)),
    }
    if strength in {"Strong", "Suggestive"}:
        label = f"{strength} ({same:.0%} same-dir, {hit:.0%} hit)"
    elif np.isfinite(same) and np.isfinite(hit):
        label = f"Weak ({same:.0%} same-dir, {hit:.0%} hit)"
    else:
        label = "Weak"
    return label, meta, []


def _load_supporting_contracts(
    target: str,
    asof: str | None,
    maturity_days: Any,
    metric_family: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return real option quotes that ground a synthesized signal."""
    try:
        from data.db_utils import get_conn

        conn = get_conn()
    except Exception:
        return []

    asof_filter = asof
    if not asof_filter:
        try:
            row = conn.execute(
                "SELECT MAX(asof_date) FROM options_quotes WHERE ticker = ?",
                (target,),
            ).fetchone()
            asof_filter = row[0] if row and row[0] else None
        except Exception:
            asof_filter = None
    if not asof_filter:
        return []

    maturity = _safe_float(maturity_days)
    clauses = ["ticker = ?", "asof_date = ?", "iv IS NOT NULL", "spot IS NOT NULL", "strike IS NOT NULL"]
    params: list[Any] = [target.upper(), asof_filter]
    if np.isfinite(maturity) and maturity > 0:
        clauses.append("ABS(ttm_years * 365.25 - ?) <= ?")
        params.extend([float(maturity), max(10.0, float(maturity) * 0.35)])

    family = str(metric_family)
    if family == "level":
        clauses.append("moneyness BETWEEN 0.95 AND 1.05")
    elif family == "asymmetry":
        clauses.append("((call_put = 'P' AND moneyness BETWEEN 0.85 AND 1.00) OR (call_put = 'C' AND moneyness BETWEEN 1.00 AND 1.15))")
    elif family == "convexity":
        clauses.append("(moneyness BETWEEN 0.75 AND 0.90 OR moneyness BETWEEN 1.10 AND 1.30)")
    elif family == "timing":
        clauses.append("moneyness BETWEEN 0.90 AND 1.10")
    elif family == "slope":
        clauses.append("moneyness BETWEEN 0.95 AND 1.05")

    sql = (
        "SELECT expiry, strike, moneyness, call_put, iv, bid, ask, volume, open_interest, ttm_years "
        "FROM options_quotes WHERE "
        + " AND ".join(clauses)
        + " ORDER BY ABS(moneyness - 1.0), volume DESC, open_interest DESC LIMIT ?"
    )
    params.append(int(limit))
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return []
    if df.empty:
        return []

    contracts: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        contracts.append({
            "expiry": str(row.get("expiry", "")),
            "strike": _safe_float(row.get("strike")),
            "moneyness": _safe_float(row.get("moneyness")),
            "call_put": str(row.get("call_put", "")),
            "iv": _safe_float(row.get("iv")),
            "bid": _safe_float(row.get("bid")),
            "ask": _safe_float(row.get("ask")),
            "volume": _safe_float(row.get("volume")),
            "open_interest": _safe_float(row.get("open_interest")),
            "ttm_days": _safe_float(row.get("ttm_years")) * 365.25,
        })
    return contracts


def _contract_summary(target: str, contracts: list[dict[str, Any]], metric_family: str) -> str:
    if not contracts:
        return "No contract-level quotes were available for this signal."
    expiries = sorted({c.get("expiry") for c in contracts if c.get("expiry")})
    cp = sorted({c.get("call_put") for c in contracts if c.get("call_put")})
    mnys = [_safe_float(c.get("moneyness")) for c in contracts]
    mnys = [m for m in mnys if np.isfinite(m)]
    expiry = expiries[0] if len(expiries) == 1 else f"{len(expiries)} expiries"
    side = "puts/calls" if len(cp) > 1 else ("puts" if cp == ["P"] else "calls" if cp == ["C"] else "options")
    if mnys:
        return f"{target} {expiry} {side} around {min(mnys):.2f}-{max(mnys):.2f} K/S anchor this {metric_family} signal."
    return f"{target} {expiry} {side} anchor this {metric_family} signal."


def _build_narrative(
    *,
    target: str,
    metric_family: str,
    direction: str,
    maturity: str,
    context: str,
    dynamics: str,
    data_quality: str,
    contracts: list[dict[str, Any]],
) -> dict[str, str]:
    rich = direction == "Rich"
    family_text = {
        "level": "overall implied volatility",
        "slope": "the term structure",
        "asymmetry": "the smile asymmetry",
        "convexity": "smile convexity",
        "timing": "risk timing",
    }.get(metric_family, "the volatility surface")
    if metric_family == "asymmetry":
        what = "Downside/upside protection is priced higher than peers" if rich else "Downside/upside protection is priced lower than peers"
    elif metric_family == "convexity":
        what = "Extreme-outcome options are priced higher than peers" if rich else "Extreme-outcome options are priced lower than peers"
    elif metric_family == "timing":
        what = "Risk is concentrated in a specific expiry versus peers" if rich else "Event-timing premium is lower than peers"
    elif metric_family == "slope":
        what = "The maturity profile is steeper/richer than peers" if rich else "The maturity profile is flatter/cheaper than peers"
    else:
        what = "Implied volatility is priced higher than peers" if rich else "Implied volatility is priced lower than peers"

    return {
        "headline": f"{target} {maturity} {family_text} is {direction.lower()} vs peers.",
        "what_differs": what,
        "why_matters": (
            "The signal is more tradeable when the dislocation is statistically unusual, "
            "the peer surface comparison is structurally aligned, and supporting contracts are liquid enough to audit."
        ),
        "context": f"Context reads {context.lower()}; dynamics read {dynamics.lower()}; data quality is {data_quality.lower()}.",
        "contracts": _contract_summary(target, contracts, metric_family),
    }


def _surface_comparability(target: str, peers: list[str], asof: str | None, max_expiries: int) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        from analysis.peer_composite_builder import build_surface_grids

        surfaces = build_surface_grids([target] + peers, max_expiries=max_expiries)
    except Exception as exc:
        return "Unknown", {}, [f"Surface comparability unavailable: {exc}"]
    if not surfaces or target not in surfaces:
        return "Unknown", {}, ["No target surface grid available for comparability check."]

    target_dates = sorted(surfaces.get(target, {}).keys())
    if not target_dates:
        return "Unknown", {}, ["No target surface dates available for comparability check."]
    asof_ts = pd.to_datetime(asof, errors="coerce") if asof is not None else pd.NaT
    if pd.notna(asof_ts):
        eligible = [d for d in target_dates if pd.to_datetime(d) <= asof_ts]
        date = eligible[-1] if eligible else target_dates[-1]
    else:
        date = target_dates[-1]
    target_grid = surfaces[target].get(date)
    if target_grid is None or target_grid.empty:
        return "Unknown", {}, ["Target surface grid is empty."]

    corrs: list[float] = []
    common_counts: list[int] = []
    for peer in peers:
        peer_grid = surfaces.get(peer, {}).get(date)
        if peer_grid is None or peer_grid.empty:
            continue
        rows = target_grid.index.intersection(peer_grid.index)
        cols = target_grid.columns.intersection(peer_grid.columns)
        if len(rows) == 0 or len(cols) == 0:
            continue
        a = target_grid.loc[rows, cols].to_numpy(float).ravel()
        b = peer_grid.loc[rows, cols].to_numpy(float).ravel()
        valid = np.isfinite(a) & np.isfinite(b)
        common_counts.append(int(valid.sum()))
        if valid.sum() >= 3:
            corrs.append(float(np.corrcoef(a[valid], b[valid])[0, 1]))
    avg_corr = float(np.nanmean(corrs)) if corrs else np.nan
    avg_cells = float(np.nanmean(common_counts)) if common_counts else 0.0
    meta = {"date": str(pd.to_datetime(date).date()), "avg_surface_corr": avg_corr, "avg_common_cells": avg_cells}
    if not common_counts:
        return "Unknown", meta, ["No common target/peer surface cells found."]
    if np.isfinite(avg_corr) and avg_corr >= 0.75 and avg_cells >= 4:
        return "Comparable", meta, []
    if np.isfinite(avg_corr) and avg_corr >= 0.45:
        warnings.append("Peer surfaces are only moderately similar.")
        return "Mixed", meta, warnings
    warnings.append("Peer surfaces have weak structural similarity.")
    return "Poor", meta, warnings


def _event_context(target: str, peers: list[str], asof: str | None, lookback: int) -> tuple[str, dict[str, Any], list[str]]:
    try:
        from analysis.analysis_pipeline import get_daily_iv_for_spillover

        iv_df = get_daily_iv_for_spillover([target] + peers)
    except Exception as exc:
        return "Unknown", {}, [f"Peer date/event context unavailable: {exc}"]
    if iv_df is None or iv_df.empty:
        return "Unknown", {}, ["No daily ATM IV history available for event context."]
    work = iv_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ticker"] = work["ticker"].astype(str).str.upper()
    if asof is not None:
        asof_ts = pd.to_datetime(asof, errors="coerce")
        if pd.notna(asof_ts):
            work = work[work["date"] <= asof_ts]
    piv = work.pivot(index="date", columns="ticker", values="atm_iv").sort_index().tail(max(3, lookback))
    if target not in piv or len(piv) < 2:
        return "Unknown", {}, ["Insufficient daily ATM IV history for event context."]
    ret = piv.pct_change().iloc[-1]
    target_move = _safe_float(ret.get(target))
    peer_moves = pd.to_numeric(ret.reindex(peers), errors="coerce").dropna()
    if not np.isfinite(target_move) or peer_moves.empty:
        return "Unknown", {}, ["Could not compute target/peer daily IV moves."]
    same_share = float((np.sign(peer_moves) == np.sign(target_move)).mean())
    peer_median = float(peer_moves.median())
    peer_abs = float(peer_moves.abs().median())
    meta = {
        "target_move": target_move,
        "peer_median_move": peer_median,
        "same_direction_share": same_share,
        "peer_abs_median_move": peer_abs,
    }
    if same_share >= 0.70 and peer_abs >= 0.005:
        return "Systemic", meta, []
    if 0.40 <= same_share < 0.70:
        return "Cluster", meta, []
    return "Idiosyncratic", meta, []


def _feature_health_context(
    target: str,
    peers: list[str],
    asof: str | None,
    weight_mode: str,
    max_expiries: int,
) -> tuple[dict[str, Any], list[str]]:
    try:
        from analysis.feature_health import build_feature_construction_result

        result = build_feature_construction_result(
            target=target,
            peers=peers,
            weight_mode=weight_mode,
            asof=asof,
            max_expiries=max_expiries,
        )
        health = result.feature_health
    except Exception as exc:
        return {}, [f"Feature Health unavailable: {exc}"]
    warnings = [str(w) for w in health.get("warnings") or []]
    if warnings:
        warnings = [f"Feature Health: {w}" for w in warnings]
    return health, warnings


# ---------------------------------------------------------------------------
# Phase 1 — Surface residual
# ---------------------------------------------------------------------------

def compute_surface_residual(
    target_surfaces: Dict[pd.Timestamp, pd.DataFrame],
    synthetic_surfaces: Dict[pd.Timestamp, pd.DataFrame],
    lookback: int = 60,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Compute the normalized per-cell residual between target and synthetic surfaces.

    For each date: raw_residual[K, T] = (target[K, T] - synth[K, T]) / synth[K, T].
    When more than one common date exists a rolling z-score over the previous
    ``lookback`` dates is computed per cell; otherwise the raw normalized residual
    is returned.

    Parameters
    ----------
    target_surfaces, synthetic_surfaces : dict[Timestamp -> DataFrame]
        Grids returned by ``build_surface_grids`` / ``combine_surfaces``.
        Rows = moneyness bins, columns = tenor (days).
    lookback : int
        Maximum number of past dates used for the rolling z-score.

    Returns
    -------
    dict[Timestamp -> DataFrame]
        Same moneyness × tenor shape as the inputs (common grid).
        Values are z-scored residuals where history is available, raw
        normalized residuals otherwise.
    """
    common_dates = sorted(
        set(target_surfaces.keys()).intersection(synthetic_surfaces.keys())
    )
    if not common_dates:
        return {}

    # Align all grids to the intersection of rows/cols on the first date
    def _align(tgt: pd.DataFrame, syn: pd.DataFrame):
        rows = tgt.index.intersection(syn.index)
        cols = tgt.columns.intersection(syn.columns)
        if len(rows) < 1 or len(cols) < 1:
            return None, None
        t = tgt.loc[rows, cols].astype(float)
        s = syn.loc[rows, cols].astype(float)
        return t, s

    # Compute raw normalized residual per date
    raw: Dict[pd.Timestamp, pd.DataFrame] = {}
    ref_index = None
    ref_columns = None
    for d in common_dates:
        t, s = _align(target_surfaces[d], synthetic_surfaces[d])
        if t is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(
                np.isfinite(s.to_numpy()) & (s.to_numpy() != 0.0),
                (t.to_numpy() - s.to_numpy()) / s.to_numpy(),
                np.nan,
            )
        raw[d] = pd.DataFrame(r, index=t.index, columns=t.columns)
        if ref_index is None:
            ref_index = t.index
            ref_columns = t.columns

    if not raw:
        return {}

    dates = sorted(raw.keys())
    if len(dates) < 2:
        return dict(raw)

    # Z-score per cell using rolling history
    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for i, d in enumerate(dates):
        curr_df = raw[d].reindex(index=ref_index, columns=ref_columns)
        hist_start = max(0, i - lookback + 1)
        hist_dates = dates[hist_start : i + 1]
        hist_arrays = np.array(
            [raw[h].reindex(index=ref_index, columns=ref_columns).to_numpy(float)
             for h in hist_dates if h in raw],
            dtype=float,
        )
        if hist_arrays.shape[0] < 2:
            result[d] = curr_df
            continue
        mu = np.nanmean(hist_arrays, axis=0)
        sd = np.nanstd(hist_arrays, axis=0, ddof=1)
        sd = np.where(~np.isfinite(sd) | (sd <= 0.0), 1.0, sd)
        z_arr = (curr_df.to_numpy(float) - mu) / sd
        result[d] = pd.DataFrame(z_arr, index=curr_df.index, columns=curr_df.columns)

    return result


# ---------------------------------------------------------------------------
# Phase 2 — Skew and curvature spread
# ---------------------------------------------------------------------------

def compute_skew_spread(
    target: str,
    peers: Iterable[str],
    asof: str,
    weights: Optional[Mapping[str, float]] = None,
    atm_band: float = 0.05,
    max_expiries: int = 6,
) -> pd.DataFrame:
    """Compute target-vs-synthetic skew and curvature spread per expiry.

    Parameters
    ----------
    target : str
        Ticker of the target asset.
    peers : iterable of str
        Peer tickers used to build the synthetic composite.
    asof : str
        As-of date string (ISO format).
    weights : mapping, optional
        Per-peer weights.  Equal weights used if omitted.
    atm_band : float
        Moneyness band around 1.0 used when extracting ATM vol.
    max_expiries : int
        Maximum number of expiries to load per ticker.

    Returns
    -------
    DataFrame with columns:
        T, T_days,
        target_atm, synth_atm, atm_spread,
        target_skew, synth_skew, skew_spread,
        target_curv, synth_curv, curv_spread
    """
    from analysis.analysis_pipeline import get_smile_slice  # delayed import
    from analysis.atm_extraction import compute_atm_by_expiry

    target = target.upper()
    peers = [p.upper() for p in peers]

    # Normalise weights
    if weights:
        w: Dict[str, float] = {
            k.upper(): float(v)
            for k, v in weights.items()
            if k.upper() in peers and float(v) > 0
        }
    else:
        w = {p: 1.0 for p in peers}
    total = sum(w.values())
    if total <= 0:
        return pd.DataFrame()
    w = {k: v / total for k, v in w.items()}

    # Target ATM curve (with skew/curv)
    df_tgt = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df_tgt is None or df_tgt.empty:
        return pd.DataFrame()
    tgt_atm = compute_atm_by_expiry(df_tgt, atm_band=atm_band, method="fit", n_boot=0)
    if tgt_atm.empty or "skew" not in tgt_atm.columns:
        return pd.DataFrame()

    # Per-peer ATM curves
    tol_years = 10.0 / 365.25
    peer_atm: Dict[str, pd.DataFrame] = {}
    for p in peers:
        if w.get(p, 0.0) <= 0:
            continue
        df_p = get_smile_slice(p, asof, T_target_years=None, max_expiries=max_expiries)
        if df_p is None or df_p.empty:
            continue
        c = compute_atm_by_expiry(df_p, atm_band=atm_band, method="fit", n_boot=0)
        if not c.empty and "skew" in c.columns:
            peer_atm[p] = c

    if not peer_atm:
        return pd.DataFrame()

    rows = []
    for _, tgt_row in tgt_atm.iterrows():
        T_val = float(tgt_row["T"])
        target_atm_v = float(tgt_row["atm_vol"]) if pd.notna(tgt_row.get("atm_vol")) else np.nan
        target_skew_v = float(tgt_row["skew"]) if pd.notna(tgt_row.get("skew")) else np.nan
        target_curv_v = float(tgt_row["curv"]) if pd.notna(tgt_row.get("curv")) else np.nan

        sum_w = sum_atm = sum_skew = sum_curv = 0.0
        for p, patm in peer_atm.items():
            p_T = patm["T"].to_numpy(float)
            j = int(np.argmin(np.abs(p_T - T_val)))
            if abs(p_T[j] - T_val) > tol_years:
                continue
            p_row = patm.iloc[j]
            p_w = w.get(p, 0.0)
            if p_w <= 0:
                continue
            if pd.notna(p_row.get("atm_vol")):
                sum_atm += p_w * float(p_row["atm_vol"])
            if pd.notna(p_row.get("skew")):
                sum_skew += p_w * float(p_row["skew"])
            if pd.notna(p_row.get("curv")):
                sum_curv += p_w * float(p_row["curv"])
            sum_w += p_w

        if sum_w <= 0:
            continue

        synth_atm_v = sum_atm / sum_w
        synth_skew_v = sum_skew / sum_w
        synth_curv_v = sum_curv / sum_w

        rows.append({
            "T": T_val,
            "T_days": int(round(T_val * 365.25)),
            "target_atm": target_atm_v,
            "synth_atm": synth_atm_v,
            "atm_spread": (target_atm_v - synth_atm_v) if np.isfinite(target_atm_v) else np.nan,
            "target_skew": target_skew_v,
            "synth_skew": synth_skew_v,
            "skew_spread": (target_skew_v - synth_skew_v) if np.isfinite(target_skew_v) else np.nan,
            "target_curv": target_curv_v,
            "synth_curv": synth_curv_v,
            "curv_spread": (target_curv_v - synth_curv_v) if np.isfinite(target_curv_v) else np.nan,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 3 — Term-structure shape dislocation
# ---------------------------------------------------------------------------

def compute_term_shape_dislocation(
    target: str,
    peers: Iterable[str],
    asof: str,
    weights: Optional[Mapping[str, float]] = None,
    max_expiries: int = 6,
    atm_band: float = 0.05,
) -> Dict[str, float]:
    """Fit a log-linear term structure to target and synthetic; return shape dislocation.

    The model is: ``atm_vol ≈ level + slope * log(T)`` where ``T`` is in years.
    Residuals after subtracting each fitted line are used to identify
    localised event-vol bumps.

    Returns
    -------
    dict with keys:
        target_level, target_slope,
        synth_level, synth_slope,
        level_spread, slope_spread,
        max_event_bump, event_bump_T_days
    Empty dict if there are fewer than 2 aligned expiries.
    """
    spread_df = compute_skew_spread(
        target, peers, asof,
        weights=weights, atm_band=atm_band, max_expiries=max_expiries,
    )
    if spread_df.empty or len(spread_df) < 2:
        return {}

    T = spread_df["T"].to_numpy(float)
    valid = np.isfinite(T) & (T > 0)
    if valid.sum() < 2:
        return {}

    log_T = np.log(T[valid])
    tgt_atm = spread_df["target_atm"].to_numpy(float)[valid]
    syn_atm = spread_df["synth_atm"].to_numpy(float)[valid]

    def _fit_log_linear(log_t, iv):
        finite = np.isfinite(log_t) & np.isfinite(iv)
        if finite.sum() < 2:
            return np.nan, np.nan, np.full(len(log_t), np.nan)
        X = np.column_stack([np.ones(finite.sum()), log_t[finite]])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, iv[finite], rcond=None)
        except Exception:
            return np.nan, np.nan, np.full(len(log_t), np.nan)
        level, slope = float(coef[0]), float(coef[1])
        fitted_all = coef[0] + coef[1] * log_t
        residuals = np.where(np.isfinite(iv), iv - fitted_all, np.nan)
        return level, slope, residuals

    tgt_level, tgt_slope, tgt_resid = _fit_log_linear(log_T, tgt_atm)
    syn_level, syn_slope, syn_resid = _fit_log_linear(log_T, syn_atm)

    bump_diff = tgt_resid - syn_resid
    finite_bump = np.isfinite(bump_diff)
    max_event_bump = float(np.nanmax(np.abs(bump_diff))) if finite_bump.any() else np.nan
    if finite_bump.any() and np.isfinite(max_event_bump):
        bump_idx = int(np.nanargmax(np.abs(np.where(finite_bump, bump_diff, np.nan))))
        event_bump_T_days = float(T[valid][bump_idx] * 365.25)
    else:
        event_bump_T_days = np.nan

    return {
        "target_level": tgt_level,
        "target_slope": tgt_slope,
        "synth_level": syn_level,
        "synth_slope": syn_slope,
        "level_spread": (tgt_level - syn_level)
        if (np.isfinite(tgt_level) and np.isfinite(syn_level))
        else np.nan,
        "slope_spread": (tgt_slope - syn_slope)
        if (np.isfinite(tgt_slope) and np.isfinite(syn_slope))
        else np.nan,
        "max_event_bump": max_event_bump,
        "event_bump_T_days": event_bump_T_days,
    }


def _reference_label_from_signal(row: Mapping[str, Any]) -> str:
    comparison = str(row.get("comparison", "synthetic") or "synthetic")
    peer = str(row.get("peer", "") or "").upper()
    if comparison == "peer" and peer:
        return f"Actual peer {peer}"
    return "Weighted peer synthetic"


# ---------------------------------------------------------------------------
# Phase 4 — Ranked RV signals
# ---------------------------------------------------------------------------

def generate_rv_signals(
    target: str,
    peers: Iterable[str],
    asof: Optional[str] = None,
    weight_mode: str = "corr_iv_atm",
    lookback: int = 60,
    max_expiries: int = 6,
    min_abs_z: float = 1.0,
) -> pd.DataFrame:
    """Generate a ranked DataFrame of implied-volatility dislocation signals.

    Combines three signal sources:

    1. **ATM level** – per-pillar z-scores from the historical
       target-vs-synthetic spread, via the existing
       ``relative_value_atm_report_corrweighted`` infrastructure.
    2. **Skew / Curvature** – per-expiry spread between target and
       synthetic skew / curvature (no historical z-score; single-date).
    3. **Term structure shape** – level, slope, and max event-bump
       from the log-linear term-structure dislocation.

    Parameters
    ----------
    target : str
    peers : iterable of str
    asof : str, optional
        ISO date string.  Uses most-recent DB date when omitted.
    weight_mode : str
        Canonical weight-mode string (e.g. ``"corr_iv_atm"``).
    lookback : int
        Rolling window (days) for ATM-level z-score.
    max_expiries : int
    min_abs_z : float
        Only ATM-level signals with ``|z| >= min_abs_z`` are kept.
        Skew/curvature and shape signals always pass through.

    Returns
    -------
    DataFrame with columns:
        signal_type, asof_date, T_days, value, synth_value,
        spread, z_score, pct_rank, description, comparison, peer, reference_label
    Sorted descending by |z_score| (NaN z-score signals come last).
    """
    from analysis.analysis_pipeline import (  # delayed to avoid circular import
        _fetch_target_atm,
        _rv_metrics_join,
        compute_peer_weights,
        get_most_recent_date_global,
        relative_value_atm_report_corrweighted,
    )

    target = target.upper()
    peers = [p.upper() for p in peers]

    if asof is None:
        asof = get_most_recent_date_global()
    if not asof:
        return pd.DataFrame(columns=[
            "signal_type", "asof_date", "T_days", "value", "synth_value",
            "spread", "z_score", "pct_rank", "description",
            "comparison", "peer", "reference_label",
        ])

    # Compute weights
    try:
        weights = compute_peer_weights(target, peers, weight_mode=weight_mode)
    except Exception:
        eq = 1.0 / max(len(peers), 1)
        weights = pd.Series(eq, index=peers, dtype=float)
    w_dict = weights.to_dict()

    signals: list[dict] = []

    # ---- 1. ATM level (z-scored via historical spread) ----
    try:
        rv_df, _ = relative_value_atm_report_corrweighted(
            target=target,
            peers=peers,
            mode=weight_mode,
            pillar_days=[7, 14, 30, 60, 90],
            lookback=lookback,
        )
        if not rv_df.empty:
            latest = rv_df.sort_values("asof_date").groupby("pillar_days").tail(1)
            for _, row in latest.iterrows():
                z_f = _safe_float(row.get("z", np.nan))
                if not np.isfinite(z_f):
                    continue
                spread_v = row.get("spread", np.nan)
                iv_tgt = row.get("iv_target", np.nan)
                iv_syn = row.get("iv_synth", np.nan)
                pct = row.get("pct_rank", np.nan)
                T_days = int(row.get("pillar_days", 0))
                signals.append({
                    "signal_type": "ATM Level",
                    "asof_date": str(row.get("asof_date", asof)),
                    "T_days": T_days,
                    "value": _safe_float(iv_tgt),
                    "synth_value": _safe_float(iv_syn),
                    "spread": _safe_float(spread_v),
                    "z_score": z_f,
                    "pct_rank": _safe_float(pct),
                    "description": f"{target} ATM vol vs synthetic at {T_days}d",
                    "comparison": "synthetic",
                    "peer": "",
                    "reference_label": "Weighted peer synthetic",
                })
    except Exception:
        pass

    # ---- 1b. ATM level versus each actual peer ----
    for peer in peers:
        try:
            tgt_iv = _fetch_target_atm(target, pillar_days=[7, 14, 30, 60, 90])
            peer_iv = _fetch_target_atm(peer, pillar_days=[7, 14, 30, 60, 90])
            peer_rv = _rv_metrics_join(tgt_iv, peer_iv, lookback=lookback)
            if peer_rv.empty:
                continue
            latest = peer_rv.sort_values("asof_date").groupby("pillar_days").tail(1)
            for _, row in latest.iterrows():
                z_f = _safe_float(row.get("z", np.nan))
                if not np.isfinite(z_f):
                    continue
                spread_v = row.get("spread", np.nan)
                iv_tgt = row.get("iv_target", np.nan)
                iv_peer = row.get("iv_synth", np.nan)
                pct = row.get("pct_rank", np.nan)
                T_days = int(row.get("pillar_days", 0))
                signals.append({
                    "signal_type": "ATM Level",
                    "asof_date": str(row.get("asof_date", asof)),
                    "T_days": T_days,
                    "value": _safe_float(iv_tgt),
                    "synth_value": _safe_float(iv_peer),
                    "spread": _safe_float(spread_v),
                    "z_score": z_f,
                    "pct_rank": _safe_float(pct),
                    "description": f"{target} ATM vol vs actual peer {peer} at {T_days}d",
                    "comparison": "peer",
                    "peer": peer,
                    "reference_label": f"Actual peer {peer}",
                })
        except Exception:
            continue

    # ---- 2. Skew and curvature ----
    try:
        skew_df = compute_skew_spread(
            target, peers, asof, weights=w_dict, max_expiries=max_expiries
        )
        if not skew_df.empty:
            for _, row in skew_df.iterrows():
                T_days = int(row.get("T_days", int(round(row.get("T", 0) * 365.25))))
                for sig_type, (tgt_col, syn_col, spread_col, desc_suffix) in {
                    "Skew": ("target_skew", "synth_skew", "skew_spread", "put/call skew"),
                    "Curvature": ("target_curv", "synth_curv", "curv_spread", "vol curvature"),
                }.items():
                    s_f = _safe_float(row.get(spread_col, np.nan))
                    if not np.isfinite(s_f):
                        continue
                    signals.append({
                        "signal_type": sig_type,
                        "asof_date": asof,
                        "T_days": T_days,
                        "value": _safe_float(row.get(tgt_col, np.nan)),
                        "synth_value": _safe_float(row.get(syn_col, np.nan)),
                        "spread": s_f,
                        "z_score": np.nan,
                        "pct_rank": np.nan,
                        "description": f"{target} {desc_suffix} vs synthetic at {T_days}d",
                        "comparison": "synthetic",
                        "peer": "",
                        "reference_label": "Weighted peer synthetic",
                    })
    except Exception:
        pass

    # ---- 2b. Skew and curvature versus each actual peer ----
    for peer in peers:
        try:
            peer_skew_df = compute_skew_spread(
                target, [peer], asof, weights={peer: 1.0}, max_expiries=max_expiries
            )
            if peer_skew_df.empty:
                continue
            for _, row in peer_skew_df.iterrows():
                T_days = int(row.get("T_days", int(round(row.get("T", 0) * 365.25))))
                for sig_type, (tgt_col, syn_col, spread_col, desc_suffix) in {
                    "Skew": ("target_skew", "synth_skew", "skew_spread", "put/call skew"),
                    "Curvature": ("target_curv", "synth_curv", "curv_spread", "vol curvature"),
                }.items():
                    s_f = _safe_float(row.get(spread_col, np.nan))
                    if not np.isfinite(s_f):
                        continue
                    signals.append({
                        "signal_type": sig_type,
                        "asof_date": asof,
                        "T_days": T_days,
                        "value": _safe_float(row.get(tgt_col, np.nan)),
                        "synth_value": _safe_float(row.get(syn_col, np.nan)),
                        "spread": s_f,
                        "z_score": np.nan,
                        "pct_rank": np.nan,
                        "description": f"{target} {desc_suffix} vs actual peer {peer} at {T_days}d",
                        "comparison": "peer",
                        "peer": peer,
                        "reference_label": f"Actual peer {peer}",
                    })
        except Exception:
            continue

    # ---- 3. Term structure shape ----
    try:
        shape = compute_term_shape_dislocation(
            target, peers, asof, weights=w_dict, max_expiries=max_expiries
        )
        if shape:
            for sig_type, val_key, syn_key, spread_key, desc in [
                ("TS Level", "target_level", "synth_level", "level_spread",
                 f"{target} term-structure level vs synthetic"),
                ("TS Slope", "target_slope", "synth_slope", "slope_spread",
                 f"{target} term-structure slope vs synthetic"),
            ]:
                sv = shape.get(spread_key, np.nan)
                if not np.isfinite(sv):
                    continue
                signals.append({
                    "signal_type": sig_type,
                    "asof_date": asof,
                    "T_days": 0,
                    "value": float(shape.get(val_key, np.nan)),
                    "synth_value": float(shape.get(syn_key, np.nan)),
                    "spread": float(sv),
                    "z_score": np.nan,
                    "pct_rank": np.nan,
                    "description": desc,
                    "comparison": "synthetic",
                    "peer": "",
                    "reference_label": "Weighted peer synthetic",
                })
            bump = shape.get("max_event_bump", np.nan)
            bump_T = shape.get("event_bump_T_days", np.nan)
            if np.isfinite(bump) and bump > 0:
                T_label = int(bump_T) if np.isfinite(bump_T) else "?"
                signals.append({
                    "signal_type": "Event Bump",
                    "asof_date": asof,
                    "T_days": int(bump_T) if np.isfinite(bump_T) else 0,
                    "value": float(bump),
                    "synth_value": 0.0,
                    "spread": float(bump),
                    "z_score": np.nan,
                    "pct_rank": np.nan,
                    "description": f"{target} event-vol bump vs synthetic at {T_label}d",
                    "comparison": "synthetic",
                    "peer": "",
                    "reference_label": "Weighted peer synthetic",
                })
    except Exception:
        pass

    # ---- 3b. Term structure shape versus each actual peer ----
    for peer in peers:
        try:
            shape = compute_term_shape_dislocation(
                target, [peer], asof, weights={peer: 1.0}, max_expiries=max_expiries
            )
            if not shape:
                continue
            for sig_type, val_key, syn_key, spread_key, desc in [
                ("TS Level", "target_level", "synth_level", "level_spread",
                 f"{target} term-structure level vs actual peer {peer}"),
                ("TS Slope", "target_slope", "synth_slope", "slope_spread",
                 f"{target} term-structure slope vs actual peer {peer}"),
            ]:
                sv = shape.get(spread_key, np.nan)
                if not np.isfinite(sv):
                    continue
                signals.append({
                    "signal_type": sig_type,
                    "asof_date": asof,
                    "T_days": 0,
                    "value": float(shape.get(val_key, np.nan)),
                    "synth_value": float(shape.get(syn_key, np.nan)),
                    "spread": float(sv),
                    "z_score": np.nan,
                    "pct_rank": np.nan,
                    "description": desc,
                    "comparison": "peer",
                    "peer": peer,
                    "reference_label": f"Actual peer {peer}",
                })
            bump = shape.get("max_event_bump", np.nan)
            bump_T = shape.get("event_bump_T_days", np.nan)
            if np.isfinite(bump) and bump > 0:
                T_label = int(bump_T) if np.isfinite(bump_T) else "?"
                signals.append({
                    "signal_type": "Event Bump",
                    "asof_date": asof,
                    "T_days": int(bump_T) if np.isfinite(bump_T) else 0,
                    "value": float(bump),
                    "synth_value": 0.0,
                    "spread": float(bump),
                    "z_score": np.nan,
                    "pct_rank": np.nan,
                    "description": f"{target} event-vol bump vs actual peer {peer} at {T_label}d",
                    "comparison": "peer",
                    "peer": peer,
                    "reference_label": f"Actual peer {peer}",
                })
        except Exception:
            continue

    if not signals:
        return pd.DataFrame(columns=[
            "signal_type", "asof_date", "T_days", "value", "synth_value",
            "spread", "z_score", "pct_rank", "description",
            "comparison", "peer", "reference_label",
        ])

    df = pd.DataFrame(signals)

    # Separate z-scored from non-z-scored signals
    has_z = df["z_score"].apply(lambda z: pd.notna(z) and np.isfinite(float(z)))
    z_signals = df[has_z].copy()
    other_signals = df[~has_z].copy()

    # Filter z-scored by min_abs_z
    if not z_signals.empty:
        z_signals = z_signals[z_signals["z_score"].abs() >= min_abs_z]

    # Sort each group: z-scored by |z|, others by |spread|
    if not z_signals.empty:
        z_signals = z_signals.sort_values("z_score", key=lambda s: s.abs(), ascending=False)
    if not other_signals.empty:
        other_signals = other_signals.sort_values(
            "spread", key=lambda s: s.abs(), ascending=False
        )

    return pd.concat([z_signals, other_signals], ignore_index=True)


def generate_rv_opportunity_dashboard(
    target: str,
    peers: Iterable[str],
    asof: Optional[str] = None,
    weight_mode: str = "corr_iv_atm",
    lookback: int = 60,
    max_expiries: int = 6,
    min_abs_z: float = 1.0,
) -> dict[str, Any]:
    """Return the synthesized RV Signals dashboard payload.

    This is the final relative-value interpretation layer.  It consumes the
    raw target-vs-peer signals, then attaches context from model-fit logs,
    spillover summaries, surface-grid comparability, and peer date moves so the
    GUI can show tradeable theses instead of a raw differences table.
    """
    target = target.upper().strip()
    peer_list = [str(p).upper().strip() for p in peers if str(p).strip()]
    dashboard = _empty_dashboard()
    if not target or not peer_list:
        dashboard["warnings"].append("Target and at least one peer are required.")
        return dashboard

    try:
        raw = generate_rv_signals(
            target=target,
            peers=peer_list,
            asof=asof,
            weight_mode=weight_mode,
            lookback=lookback,
            max_expiries=max_expiries,
            min_abs_z=min_abs_z,
        )
    except Exception as exc:
        dashboard["warnings"].append(f"Raw RV signal generation failed: {exc}")
        return dashboard

    data_quality, model_warnings, model_meta = _load_model_quality(target, peer_list, asof)
    spill_label, spill_meta, spill_warnings = _load_spillover_support(target, peer_list)
    comparability, surface_meta, surface_warnings = _surface_comparability(target, peer_list, asof, max_expiries)
    event_ctx, event_meta, event_warnings = _event_context(target, peer_list, asof, lookback)
    feature_health, feature_warnings = _feature_health_context(target, peer_list, asof, weight_mode, max_expiries)
    alignment = _alignment_score(comparability, surface_meta)
    spill_score = _spillover_score(spill_meta, spill_label)

    warnings = model_warnings + spill_warnings + surface_warnings + event_warnings + feature_warnings
    dashboard["warnings"] = warnings
    dashboard["integration_status"] = {
        "model_quality": data_quality,
        "model_meta": model_meta,
        "system_health": {
            "quality": data_quality,
            "warnings": model_warnings + surface_warnings + feature_warnings,
            "confidence_input": "RV confidence is penalized when System Health reports degraded fits, weak structural comparability, or fallbacks.",
        },
        "feature_health": feature_health,
        "spillover": spill_meta,
        "surface_comparability": surface_meta,
        "event_context": event_meta,
        "weight_mode": weight_mode,
    }

    if raw is None or raw.empty:
        dashboard["executive_summary"] = [
            f"No RV opportunities passed the current threshold for {target}.",
            f"Model quality: {data_quality}; surface comparability: {comparability}.",
        ]
        dashboard["context_cards"]["data_quality_warnings"] = len(warnings)
        return dashboard

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        signal_type = str(row.get("signal_type", "Signal"))
        feature = _label_feature(signal_type)
        metric_family = _metric_family(signal_type)
        metric = _metric_label(metric_family)
        spread = _safe_float(row.get("spread"))
        z = _safe_float(row.get("z_score"))
        pct = _safe_float(row.get("pct_rank"))
        direction = _direction_from_spread(spread)
        maturity = _format_maturity(row.get("T_days"))
        target_value = _safe_float(row.get("value"))
        synthetic_value = _safe_float(row.get("synth_value"))
        reference_label = str(row.get("reference_label") or _reference_label_from_signal(row))
        comparison = str(row.get("comparison", "synthetic") or "synthetic")
        peer = str(row.get("peer", "") or "").upper()
        calculation = _calculation_details(
            target=target,
            metric=metric,
            metric_family=metric_family,
            target_value=target_value,
            synthetic_value=synthetic_value,
            spread=spread,
            direction=direction,
            reference_label=reference_label,
        )
        confidence, confidence_score = _confidence_label(
            z,
            pct,
            spill_label,
            data_quality,
            alignment=alignment,
            spillover_strength=spill_score,
        )
        if feature_warnings:
            confidence_score = float(np.clip(confidence_score * 0.80, 0.0, 1.0))
            if confidence_score >= 0.75:
                confidence = "High"
            elif confidence_score >= 0.45:
                confidence = "Medium"
            else:
                confidence = "Low"
        stat = (
            f"Z-score {z:+.2f}; percentile {pct:.0f}."
            if np.isfinite(z)
            else "No historical z-score is available for this feature; rank uses current spread magnitude."
        )
        supporting_contracts = _load_supporting_contracts(
            target,
            asof,
            row.get("T_days"),
            metric_family,
        )
        narrative = _build_narrative(
            target=target,
            metric_family=metric_family,
            direction=direction,
            maturity=maturity,
            context=event_ctx,
            dynamics=spill_label,
            data_quality=data_quality,
            contracts=supporting_contracts,
        )
        opportunity = narrative["headline"]
        if comparison == "peer" and peer:
            opportunity = opportunity.replace("vs peers.", f"vs {peer}.")
        why = f"{metric} spread is {_fmt_signed_pct(spread)} versus {reference_label}."
        if np.isfinite(z) and abs(z) >= 2.0:
            why = f"{why} The move is statistically unusual."
        elif np.isfinite(z):
            why = f"{why} The move is notable but below a 2-sigma threshold."
        if event_ctx == "Systemic":
            why = f"{why} Current peer moves look broad rather than single-name."
        elif event_ctx == "Idiosyncratic":
            why = f"{why} Current peer moves look more single-name."

        row_warnings = list(warnings)
        if comparability in {"Poor", "Unknown"}:
            row_warnings.append(f"Structural comparability is {comparability.lower()}.")
        if data_quality in {"Degraded", "Unknown"}:
            row_warnings.append(f"Model/data quality is {data_quality.lower()}.")

        tradeability = confidence_score
        if direction == "Neutral":
            tradeability -= 0.20
        if event_ctx == "Idiosyncratic":
            tradeability += 0.08
        if event_ctx == "Systemic":
            tradeability -= 0.05
        if comparability == "Comparable":
            tradeability += 0.10
        elif comparability in {"Poor", "Unknown"}:
            tradeability -= 0.20
        tradeability = float(np.clip(tradeability, 0.0, 1.0))

        signal_obj = {
            "location": {
                "metric_family": metric_family,
                "maturity": maturity,
                "T_days": _safe_float(row.get("T_days")),
            },
            "magnitude": {
                "spread": spread,
                "direction": direction,
                "target_value": target_value,
                "synthetic_value": synthetic_value,
                "reference_label": reference_label,
                "comparison": comparison,
                "peer": peer,
            },
            "calculation": calculation,
            "significance": {
                "z_score": z,
                "percentile": pct,
            },
            "structure": {
                "surface_vs_surface_grid_consistency": comparability,
                "similarity_score": alignment,
                "feature_health": feature_health,
                "details": surface_meta,
            },
            "context": {
                "classification": event_ctx,
                "peer_dispersion": _safe_float(event_meta.get("peer_abs_median_move")) if isinstance(event_meta, dict) else np.nan,
                "details": event_meta,
            },
            "dynamics": {
                "spillover_strength": spill_meta.get("strength", spill_label) if isinstance(spill_meta, dict) else spill_label,
                "same_direction_probability": _safe_float(spill_meta.get("same_direction_probability")) if isinstance(spill_meta, dict) else np.nan,
                "hit_rate": _safe_float(spill_meta.get("hit_rate")) if isinstance(spill_meta, dict) else np.nan,
                "lag_profile": "h=1" if isinstance(spill_meta, dict) and spill_meta.get("rows") else "unavailable",
                "label": spill_label,
            },
            "data_quality": {
                "fit_quality": data_quality,
                "rmse": _safe_float(model_meta.get("rmse_max")) if isinstance(model_meta, dict) else np.nan,
                "degraded": data_quality in {"Degraded", "Poor"},
                "coverage": surface_meta.get("avg_common_cells") if isinstance(surface_meta, dict) else np.nan,
                "system_health_warnings": model_warnings + surface_warnings,
                "feature_health_warnings": feature_warnings,
                "details": model_meta,
            },
            "supporting_contracts": supporting_contracts,
            "narrative": narrative,
            "confidence_score": confidence_score,
        }

        rows.append({
            "rank": 0,
            "opportunity": opportunity,
            "direction": direction,
            "metric": metric,
            "feature": feature,
            "maturity": maturity,
            "spread": spread,
            "z_score": z,
            "percentile": pct,
            "confidence": confidence,
            "event_context": event_ctx,
            "spillover_support": spill_label,
            "data_quality": data_quality,
            "why": why,
            "what_differs": (
                f"{target} {metric.lower()} is {_fmt_signed_pct(spread)} "
                f"{'above' if spread > 0 else 'below' if spread < 0 else 'in line with'} "
                f"{reference_label}."
            ),
            "why_matters": narrative["why_matters"],
            "statistical_read": stat,
            "comparability": f"{comparability} surface match; similarity score {alignment:.2f}",
            "warnings": "; ".join(dict.fromkeys(row_warnings)) if row_warnings else "",
            "tradeability_score": tradeability,
            "signal": signal_obj,
        })

    opp = pd.DataFrame(rows)
    opp = opp.sort_values(
        ["tradeability_score", "z_score", "spread"],
        key=lambda s: s.abs() if s.name in {"z_score", "spread"} else s,
        ascending=False,
    ).reset_index(drop=True)
    opp["rank"] = np.arange(1, len(opp) + 1)
    dashboard["opportunities"] = opp

    strongest = opp.iloc[opp["spread"].abs().argmax()] if not opp.empty else None
    tradeable = opp.iloc[0] if not opp.empty else None
    systemic = opp[opp["event_context"] == "Systemic"].head(1)
    weakest = opp.sort_values("tradeability_score", ascending=True).head(1)
    dashboard["context_cards"] = {
        "strongest_dislocation": strongest["opportunity"] if strongest is not None else "Unavailable",
        "most_tradeable": tradeable["opportunity"] if tradeable is not None else "Unavailable",
        "most_systemic": systemic.iloc[0]["opportunity"] if not systemic.empty else "No systemic signal",
        "weakest_signal": weakest.iloc[0]["opportunity"] if not weakest.empty else "Unavailable",
        "data_quality_warnings": len(warnings),
    }

    summary: list[str] = []
    for _, r in opp.head(3).iterrows():
        z_text = f" z={float(r['z_score']):+.1f}" if np.isfinite(_safe_float(r["z_score"])) else ""
        summary.append(f"{r['opportunity']} Confidence is {str(r['confidence']).lower()}{z_text}.")
    if event_ctx == "Systemic":
        summary.append("Latest ATM move appears sector-wide, so idiosyncratic RV conviction is lower.")
    elif event_ctx == "Idiosyncratic":
        summary.append(f"Latest ATM move is concentrated in {target}, supporting an idiosyncratic RV read.")
    if spill_label != "Unavailable":
        summary.append(f"Spillover read: {spill_label}, relevant for convergence or propagation logic.")
    if warnings:
        summary.append(f"{len(warnings)} data/model warning(s) should be reviewed before trading.")
    dashboard["executive_summary"] = summary[:5]
    return dashboard


# ---------------------------------------------------------------------------
# Phase 5 — Weight stability
# ---------------------------------------------------------------------------

def compute_weight_stability(
    target: str,
    peers: Iterable[str],
    lookback: int = 30,
    asof: Optional[str] = None,
) -> pd.DataFrame:
    """Compute rolling IV correlation between target and each peer.

    Uses the daily ATM IV series stored in the database.  Returns a
    DataFrame (rows = peers) with columns ``rolling_corr`` and ``stable``
    (``True`` when ``rolling_corr >= 0.3``).

    Parameters
    ----------
    target : str
    peers : iterable of str
    lookback : int
        Number of most-recent trading days to use.
    asof : str, optional
        Ignored — function always uses the most-recent ``lookback`` dates.

    Returns
    -------
    DataFrame  (index = peer ticker)
    """
    from analysis.analysis_pipeline import get_daily_iv_for_spillover  # delayed

    target = target.upper()
    peers = [p.upper() for p in peers]
    all_tickers = [target] + peers

    empty = pd.DataFrame(
        {"rolling_corr": np.nan, "stable": False},
        index=peers,
    )

    try:
        iv_df = get_daily_iv_for_spillover(tickers=all_tickers)
    except Exception:
        return empty

    if iv_df.empty:
        return empty

    piv = iv_df.pivot(index="date", columns="ticker", values="atm_iv").sort_index()
    piv = piv.iloc[-lookback:] if len(piv) > lookback else piv

    if target not in piv.columns or len(piv) < 5:
        return empty

    tgt_series = piv[target]
    rows: dict = {}
    for p in peers:
        if p not in piv.columns:
            rows[p] = {"rolling_corr": np.nan, "stable": False}
            continue
        common = tgt_series.dropna().index.intersection(piv[p].dropna().index)
        if len(common) < 5:
            rows[p] = {"rolling_corr": np.nan, "stable": False}
            continue
        corr = float(tgt_series.loc[common].corr(piv[p].loc[common]))
        rows[p] = {"rolling_corr": corr, "stable": bool(corr >= 0.3)}

    return pd.DataFrame(rows).T


__all__ = [
    "compute_surface_residual",
    "compute_skew_spread",
    "compute_term_shape_dislocation",
    "generate_rv_signals",
    "generate_rv_opportunity_dashboard",
    "compute_weight_stability",
]
