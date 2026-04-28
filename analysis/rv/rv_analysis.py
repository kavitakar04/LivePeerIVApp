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

HEDGE_RATIO_REL_TOLERANCE = 0.10
MAX_HEDGE_PACKAGE_CONTRACTS = 20


def _safe_float(val) -> float:
    """Convert *val* to float; return ``np.nan`` if not finite or not convertible."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _empty_dashboard() -> dict[str, Any]:
    cols = [
        "rank",
        "opportunity",
        "direction",
        "metric",
        "feature",
        "maturity",
        "spread",
        "z_score",
        "percentile",
        "confidence",
        "event_context",
        "spillover_support",
        "data_quality",
        "why",
        "what_differs",
        "why_matters",
        "statistical_read",
        "comparability",
        "warnings",
        "tradeability_score",
        "signal",
    ]
    trade_cols = [
        "rank",
        "title",
        "judgment",
        "trade_type",
        "direction",
        "target",
        "hedge_or_peer",
        "maturity",
        "confidence",
        "horizon",
        "buy_legs",
        "sell_legs",
        "net_premium",
        "estimated_delta_after_hedge",
        "rationale",
        "supporting_contracts",
        "risks",
        "trade_score",
        "substitutability",
        "trade",
        "source_signal",
    ]
    anomaly_cols = [
        "rank",
        "title",
        "judgment",
        "anomaly_type",
        "affected_names",
        "likely_driver",
        "systemic_or_idiosyncratic",
        "spillover_relevance",
        "why_it_matters",
        "impact_on_trade_confidence",
        "supporting_contracts",
        "severity_score",
        "trade_score",
        "substitutability",
        "group_size",
        "classification_reasons",
        "source_signal",
        "member_signals",
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
        "trade_opportunities": pd.DataFrame(columns=trade_cols),
        "market_anomalies": pd.DataFrame(columns=anomaly_cols),
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
        "weighted peer synthetic" if reference_text == "Weighted peer synthetic" else reference_text.lower()
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
        from analysis.persistence.model_params_logger import load_model_params

        df = load_model_params()
    except Exception:
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
        clauses.append(
            "((call_put = 'P' AND moneyness BETWEEN 0.85 AND 1.00) "
            "OR (call_put = 'C' AND moneyness BETWEEN 1.00 AND 1.15))"
        )
    elif family == "convexity":
        clauses.append("(moneyness BETWEEN 0.75 AND 0.90 OR moneyness BETWEEN 1.10 AND 1.30)")
    elif family == "timing":
        clauses.append("moneyness BETWEEN 0.90 AND 1.10")
    elif family == "slope":
        clauses.append("moneyness BETWEEN 0.95 AND 1.05")

    sql = (
        "SELECT expiry, strike, moneyness, call_put, iv, bid, ask, mid, price, spot, "
        "volume, open_interest, ttm_years, delta, gamma, vega, theta "
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
        contracts.append(
            {
                "expiry": str(row.get("expiry", "")),
                "strike": _safe_float(row.get("strike")),
                "moneyness": _safe_float(row.get("moneyness")),
                "call_put": str(row.get("call_put", "")),
                "iv": _safe_float(row.get("iv")),
                "bid": _safe_float(row.get("bid")),
                "ask": _safe_float(row.get("ask")),
                "mid": _safe_float(row.get("mid")),
                "price": _safe_float(row.get("price")),
                "spot": _safe_float(row.get("spot")),
                "volume": _safe_float(row.get("volume")),
                "open_interest": _safe_float(row.get("open_interest")),
                "ttm_days": _safe_float(row.get("ttm_years")) * 365.25,
                "delta": _safe_float(row.get("delta")),
                "gamma": _safe_float(row.get("gamma")),
                "vega": _safe_float(row.get("vega")),
                "theta": _safe_float(row.get("theta")),
            }
        )
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
        return (
            f"{target} {expiry} {side} around {min(mnys):.2f}-{max(mnys):.2f} K/S anchor this {metric_family} signal."
        )
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
        what = (
            "Downside/upside protection is priced higher than peers"
            if rich
            else "Downside/upside protection is priced lower than peers"
        )
    elif metric_family == "convexity":
        what = (
            "Extreme-outcome options are priced higher than peers"
            if rich
            else "Extreme-outcome options are priced lower than peers"
        )
    elif metric_family == "timing":
        what = (
            "Risk is concentrated in a specific expiry versus peers"
            if rich
            else "Event-timing premium is lower than peers"
        )
    elif metric_family == "slope":
        what = (
            "The maturity profile is steeper/richer than peers"
            if rich
            else "The maturity profile is flatter/cheaper than peers"
        )
    else:
        what = (
            "Implied volatility is priced higher than peers"
            if rich
            else "Implied volatility is priced lower than peers"
        )

    return {
        "headline": f"{target} {maturity} {family_text} is {direction.lower()} vs peers.",
        "what_differs": what,
        "why_matters": (
            "The signal is more tradeable when the dislocation is statistically unusual, "
            "the peer surface comparison is structurally aligned, and supporting contracts are liquid enough to audit."
        ),
        "context": (
            f"Context reads {context.lower()}; dynamics read {dynamics.lower()}; "
            f"data quality is {data_quality.lower()}."
        ),
        "contracts": _contract_summary(target, contracts, metric_family),
    }


def _surface_comparability(
    target: str, peers: list[str], asof: str | None, max_expiries: int
) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        from analysis.surfaces.peer_composite_builder import build_surface_grids

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


def _event_context(
    target: str, peers: list[str], asof: str | None, lookback: int
) -> tuple[str, dict[str, Any], list[str]]:
    try:
        from analysis.services.data_availability_service import get_daily_iv_for_spillover

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
        from analysis.views.feature_health import build_feature_construction_result

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
    common_dates = sorted(set(target_surfaces.keys()).intersection(synthetic_surfaces.keys()))
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
            [raw[h].reindex(index=ref_index, columns=ref_columns).to_numpy(float) for h in hist_dates if h in raw],
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
    from analysis.services.smile_data_service import get_smile_slice  # delayed import
    from analysis.surfaces.atm_extraction import compute_atm_by_expiry

    target = target.upper()
    peers = [p.upper() for p in peers]

    # Normalise weights
    if weights:
        w: Dict[str, float] = {k.upper(): float(v) for k, v in weights.items() if k.upper() in peers and float(v) > 0}
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

        rows.append(
            {
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
            }
        )

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
        target,
        peers,
        asof,
        weights=weights,
        atm_band=atm_band,
        max_expiries=max_expiries,
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
        "level_spread": (tgt_level - syn_level) if (np.isfinite(tgt_level) and np.isfinite(syn_level)) else np.nan,
        "slope_spread": (tgt_slope - syn_slope) if (np.isfinite(tgt_slope) and np.isfinite(syn_slope)) else np.nan,
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
        return pd.DataFrame(
            columns=[
                "signal_type",
                "asof_date",
                "T_days",
                "value",
                "synth_value",
                "spread",
                "z_score",
                "pct_rank",
                "description",
                "comparison",
                "peer",
                "reference_label",
            ]
        )

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
                signals.append(
                    {
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
                    }
                )
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
                signals.append(
                    {
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
                    }
                )
        except Exception:
            continue

    # ---- 2. Skew and curvature ----
    try:
        skew_df = compute_skew_spread(target, peers, asof, weights=w_dict, max_expiries=max_expiries)
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
                    signals.append(
                        {
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
                        }
                    )
    except Exception:
        pass

    # ---- 2b. Skew and curvature versus each actual peer ----
    for peer in peers:
        try:
            peer_skew_df = compute_skew_spread(target, [peer], asof, weights={peer: 1.0}, max_expiries=max_expiries)
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
                    signals.append(
                        {
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
                        }
                    )
        except Exception:
            continue

    # ---- 3. Term structure shape ----
    try:
        shape = compute_term_shape_dislocation(target, peers, asof, weights=w_dict, max_expiries=max_expiries)
        if shape:
            for sig_type, val_key, syn_key, spread_key, desc in [
                (
                    "TS Level",
                    "target_level",
                    "synth_level",
                    "level_spread",
                    f"{target} term-structure level vs synthetic",
                ),
                (
                    "TS Slope",
                    "target_slope",
                    "synth_slope",
                    "slope_spread",
                    f"{target} term-structure slope vs synthetic",
                ),
            ]:
                sv = shape.get(spread_key, np.nan)
                if not np.isfinite(sv):
                    continue
                signals.append(
                    {
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
                    }
                )
            bump = shape.get("max_event_bump", np.nan)
            bump_T = shape.get("event_bump_T_days", np.nan)
            if np.isfinite(bump) and bump > 0:
                T_label = int(bump_T) if np.isfinite(bump_T) else "?"
                signals.append(
                    {
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
                    }
                )
    except Exception:
        pass

    # ---- 3b. Term structure shape versus each actual peer ----
    for peer in peers:
        try:
            shape = compute_term_shape_dislocation(target, [peer], asof, weights={peer: 1.0}, max_expiries=max_expiries)
            if not shape:
                continue
            for sig_type, val_key, syn_key, spread_key, desc in [
                (
                    "TS Level",
                    "target_level",
                    "synth_level",
                    "level_spread",
                    f"{target} term-structure level vs actual peer {peer}",
                ),
                (
                    "TS Slope",
                    "target_slope",
                    "synth_slope",
                    "slope_spread",
                    f"{target} term-structure slope vs actual peer {peer}",
                ),
            ]:
                sv = shape.get(spread_key, np.nan)
                if not np.isfinite(sv):
                    continue
                signals.append(
                    {
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
                    }
                )
            bump = shape.get("max_event_bump", np.nan)
            bump_T = shape.get("event_bump_T_days", np.nan)
            if np.isfinite(bump) and bump > 0:
                T_label = int(bump_T) if np.isfinite(bump_T) else "?"
                signals.append(
                    {
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
                    }
                )
        except Exception:
            continue

    if not signals:
        return pd.DataFrame(
            columns=[
                "signal_type",
                "asof_date",
                "T_days",
                "value",
                "synth_value",
                "spread",
                "z_score",
                "pct_rank",
                "description",
                "comparison",
                "peer",
                "reference_label",
            ]
        )

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
        other_signals = other_signals.sort_values("spread", key=lambda s: s.abs(), ascending=False)

    return pd.concat([z_signals, other_signals], ignore_index=True)


def _spread_threshold(metric_family: str) -> float:
    """Minimum same-day spread that can be meaningful without a z-score."""
    return {
        "level": 0.02,
        "timing": 0.015,
        "slope": 0.010,
        "asymmetry": 0.020,
        "convexity": 0.020,
    }.get(str(metric_family), 0.020)


def _dislocation_evidence(
    *,
    z: float,
    percentile: float,
    spread: float,
    metric_family: str,
) -> dict[str, Any]:
    z_abs = abs(z) if np.isfinite(z) else np.nan
    pct_edge = abs(percentile - 50.0) / 50.0 if np.isfinite(percentile) else np.nan
    spread_abs = abs(spread) if np.isfinite(spread) else np.nan
    threshold = _spread_threshold(metric_family)

    z_meaningful = np.isfinite(z_abs) and z_abs >= 2.0
    pct_meaningful = np.isfinite(percentile) and (percentile >= 90.0 or percentile <= 10.0)
    spread_meaningful = np.isfinite(spread_abs) and spread_abs >= threshold
    if np.isfinite(z_abs):
        meaningful = bool(z_meaningful or (pct_meaningful and spread_meaningful))
    else:
        meaningful = bool(spread_meaningful)

    if np.isfinite(z_abs):
        score = min(1.0, z_abs / 3.0)
    elif np.isfinite(pct_edge):
        score = float(np.clip(pct_edge, 0.0, 1.0))
    elif np.isfinite(spread_abs):
        score = float(np.clip(spread_abs / max(threshold * 2.0, 1e-8), 0.0, 1.0))
    else:
        score = 0.0

    if not meaningful:
        reason = "Dislocation is not statistically meaningful enough for a trade."
    elif np.isfinite(z_abs):
        reason = f"Dislocation is statistically meaningful at |z|={z_abs:.2f}."
    else:
        reason = f"Same-day spread exceeds the {threshold:.2%} audit threshold for this feature."

    return {
        "meaningful": meaningful,
        "score": float(np.clip(score, 0.0, 1.0)),
        "z_abs": z_abs,
        "percentile_edge": pct_edge,
        "spread_abs": spread_abs,
        "spread_threshold": threshold,
        "reason": reason,
    }


def _contract_auditability(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contracts:
        return {
            "auditable": False,
            "score": 0.0,
            "label": "No liquid supporting contracts found.",
            "risks": ["No contract-level quotes were available to audit the signal."],
        }

    liquid = 0
    quoted = 0
    for c in contracts:
        iv = _safe_float(c.get("iv"))
        bid = _safe_float(c.get("bid"))
        ask = _safe_float(c.get("ask"))
        volume = _safe_float(c.get("volume"))
        oi = _safe_float(c.get("open_interest"))
        has_iv = np.isfinite(iv) and 0.0 < iv < 5.0
        has_market = np.isfinite(bid) and np.isfinite(ask) and bid >= 0.0 and ask > 0.0 and ask >= bid
        if has_market:
            quoted += 1
        has_interest = (np.isfinite(volume) and volume > 0.0) or (np.isfinite(oi) and oi >= 25.0)
        if has_iv and has_market and has_interest:
            liquid += 1

    auditable = liquid > 0
    score = min(1.0, liquid / max(3.0, float(len(contracts))))
    risks: list[str] = []
    if not auditable:
        risks.append("Supporting contracts lack usable bid/ask, IV, volume, or open-interest evidence.")
    if quoted < max(1, len(contracts) // 2):
        risks.append("Quote coverage is thin across the contracts backing this signal.")
    return {
        "auditable": auditable,
        "score": float(score),
        "label": f"{liquid}/{len(contracts)} supporting contracts passed the liquidity audit.",
        "risks": risks,
    }


def _data_quality_component(data_quality: str) -> float:
    return {
        "Good": 1.0,
        "Acceptable": 0.80,
        "Unknown": 0.45,
        "Degraded": 0.30,
        "Poor": 0.10,
    }.get(str(data_quality), 0.45)


def _score_from_corr(value: Any) -> float:
    corr = _safe_float(value)
    if not np.isfinite(corr):
        return np.nan
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _mean_finite(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float(default)
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _substitutability_evidence(
    *,
    peer: str,
    peer_list: list[str],
    alignment: float,
    surface_meta: Mapping[str, Any] | None,
    feature_health: Mapping[str, Any] | None,
    spill_score: float,
) -> dict[str, Any]:
    """Estimate whether target/peer surfaces are close substitutes.

    This intentionally combines slow-moving similarity evidence with current
    surface alignment.  A high score is a prior for convergence, especially
    when a signal lacks a historical z-score but the current spread is large.
    """
    peer_u = str(peer or "").upper()
    selected = {peer_u} if peer_u else {str(p).upper() for p in peer_list}
    pair_scores: list[float] = []
    pair_labels: list[str] = []
    pair_items = (feature_health or {}).get("pairs", []) if isinstance(feature_health, Mapping) else []
    for item in pair_items:
        if not isinstance(item, Mapping):
            continue
        ticker = str(item.get("ticker", "")).upper()
        if ticker not in selected:
            continue
        corr_score = _score_from_corr(item.get("correlation"))
        sign_score = _safe_float(item.get("sign_consistency"))
        components = [v for v in (corr_score, sign_score) if np.isfinite(v)]
        if components:
            score = float(np.clip(np.mean(components), 0.0, 1.0))
            pair_scores.append(score)
            pair_labels.append(f"{ticker} feature similarity {score:.2f}")

    long_run = _mean_finite(pair_scores, default=np.nan)
    if not np.isfinite(long_run):
        long_run = _score_from_corr((surface_meta or {}).get("avg_surface_corr"))
    if not np.isfinite(long_run):
        long_run = float(np.clip(spill_score, 0.0, 1.0))

    structural = float(np.clip(alignment if np.isfinite(alignment) else 0.35, 0.0, 1.0))
    spill = float(np.clip(spill_score if np.isfinite(spill_score) else 0.25, 0.0, 1.0))
    score = float(np.clip(0.45 * long_run + 0.35 * structural + 0.20 * spill, 0.0, 1.0))

    if score >= 0.78:
        label = "Near substitutes"
        prior = "Strong convergence prior from long-run similarity and aligned surfaces."
    elif score >= 0.58:
        label = "Related"
        prior = "Moderate convergence prior; trade needs cleaner confirmation."
    else:
        label = "Weak substitutes"
        prior = "Weak convergence prior; treat dislocations mainly as context."

    details = list(pair_labels)
    details.append(f"structural alignment {structural:.2f}")
    details.append(f"spillover support {spill:.2f}")
    return {
        "score": score,
        "label": label,
        "prior": prior,
        "long_run_similarity": long_run,
        "structural_alignment": structural,
        "spillover_component": spill,
        "details": details,
    }


def _trade_judgment(classification: str) -> str:
    if classification == "trade":
        return "Tradeable"
    if classification == "conditional":
        return "Conditional"
    return "Not tradeable"


def _trade_interpretation(
    *,
    classification: str,
    trade_score: float,
    dislocation: dict[str, Any],
    substitutability: dict[str, Any],
    spill_label: str,
    data_quality: str,
    feature_warnings: list[str],
    event_ctx: str,
) -> str:
    sub_label = substitutability.get("label", "Unknown")
    sub_score = _safe_float(substitutability.get("score"))
    z_abs = _safe_float(dislocation.get("z_abs"))
    spread_abs = _safe_float(dislocation.get("spread_abs"))
    if classification == "trade":
        opener = (
            "Tradeable because the dislocation is large enough "
            "and the peer relationship has a strong convergence prior."
        )
    elif classification == "conditional":
        opener = (
            "Conditional because the setup has enough RV evidence to monitor or audit, "
            "but not enough clean support for an outright trade."
        )
    else:
        opener = (
            "Not tradeable because the RV evidence is too weak after similarity, spillover, and quality adjustments."
        )

    stats = (
        f"|z|={z_abs:.2f}"
        if np.isfinite(z_abs)
        else f"spread={spread_abs:.2%}" if np.isfinite(spread_abs) else "magnitude unavailable"
    )
    warning_text = " Feature warnings reduce confidence." if feature_warnings else ""
    return (
        f"{opener} Trade score {trade_score:.2f}; {stats}; "
        f"substitutability is {sub_label.lower()} ({sub_score:.2f}); "
        f"spillover reads {spill_label}; data quality is {str(data_quality).lower()}; "
        f"context is {str(event_ctx).lower()}.{warning_text}"
    )


def _classify_signal(
    *,
    signal_type: str,
    metric_family: str,
    dislocation: dict[str, Any],
    data_quality: str,
    comparability: str,
    alignment: float,
    spill_label: str,
    spill_score: float,
    event_ctx: str,
    contract_audit: dict[str, Any],
    feature_warnings: list[str],
    substitutability: dict[str, Any],
) -> dict[str, Any]:
    dislocation_score = _safe_float(dislocation.get("score"))
    dislocation_score = float(np.clip(dislocation_score if np.isfinite(dislocation_score) else 0.0, 0.0, 1.0))
    sub_score = _safe_float(substitutability.get("score"))
    sub_score = float(np.clip(sub_score if np.isfinite(sub_score) else 0.0, 0.0, 1.0))
    spill_component = float(np.clip(spill_score if np.isfinite(spill_score) else 0.25, 0.0, 1.0))
    quality_component = _data_quality_component(data_quality)
    warning_penalty = min(0.20, 0.08 * len(feature_warnings))

    missing_z = not np.isfinite(_safe_float(dislocation.get("z_abs")))
    near_substitute = sub_score >= 0.78
    if missing_z and near_substitute and np.isfinite(_safe_float(dislocation.get("spread_abs"))):
        threshold = _safe_float(dislocation.get("spread_threshold"))
        spread_abs = _safe_float(dislocation.get("spread_abs"))
        if np.isfinite(threshold) and threshold > 0 and spread_abs >= threshold * 0.50:
            dislocation_score = max(dislocation_score, 0.70)

    trade_score = (
        0.35 * dislocation_score
        + 0.30 * sub_score
        + 0.18 * spill_component
        + 0.17 * quality_component
        - warning_penalty
    )
    trade_score = float(np.clip(trade_score, 0.0, 1.0))

    if trade_score >= 0.72:
        classification = "trade"
    elif trade_score >= 0.50:
        classification = "conditional"
    else:
        classification = "anomaly"

    reasons: list[str] = [
        _trade_interpretation(
            classification=classification,
            trade_score=trade_score,
            dislocation=dislocation,
            substitutability=substitutability,
            spill_label=spill_label,
            data_quality=data_quality,
            feature_warnings=feature_warnings,
            event_ctx=event_ctx,
        )
    ]
    if missing_z and near_substitute:
        reasons.append("Missing z-score did not block the signal because the pair behaves as near substitutes.")
    if not bool(contract_audit.get("auditable")):
        reasons.append("Contract liquidity still needs manual audit before execution.")

    data_ok = quality_component >= 0.70
    structure_ok = sub_score >= 0.58
    spillover_ok = spill_component >= 0.55 and "Unavailable" not in str(spill_label)
    meaningful = dislocation_score >= 0.50

    if event_ctx == "Systemic":
        anomaly_type = "Systemic Volatility Move"
    elif event_ctx == "Cluster" or str(signal_type).lower() == "event bump" or metric_family == "timing":
        anomaly_type = "Event or Cluster Volatility"
    elif sub_score < 0.45:
        anomaly_type = "Peer Comparability Breakdown"
    elif quality_component < 0.50:
        anomaly_type = "Data or Model Quality"
    elif spill_component < 0.45 or "Unavailable" in str(spill_label):
        anomaly_type = "Weak Spillover Support"
    elif not meaningful:
        anomaly_type = "Low-Conviction Dislocation"
    else:
        anomaly_type = "Market Structure Anomaly"

    return {
        "classification": classification,
        "judgment": _trade_judgment(classification),
        "trade_score": trade_score,
        "reasons": list(dict.fromkeys(r for r in reasons if r)),
        "anomaly_type": anomaly_type,
        "score_components": {
            "dislocation_magnitude": dislocation_score,
            "substitutability": sub_score,
            "spillover_strength": spill_component,
            "data_quality": quality_component,
            "feature_warning_penalty": warning_penalty,
        },
        "soft_checks": {
            "dislocation_strength": meaningful,
            "data_quality": data_ok,
            "structural_validity": structure_ok,
            "spillover_support": spillover_ok,
            "contract_auditability": bool(contract_audit.get("auditable")),
        },
    }


def _trade_type(metric_family: str) -> str:
    return {
        "level": "Delta-neutral vol RV",
        "slope": "Term-structure RV",
        "asymmetry": "Skew-transfer RV",
        "convexity": "Tail-risk RV",
        "timing": "Event-vol timing RV",
    }.get(str(metric_family), "Surface RV")


def _trade_horizon(maturity: str) -> str:
    days = _safe_float(str(maturity).rstrip("d"))
    if not np.isfinite(days) or days <= 0:
        return "Surface/term-structure horizon"
    if days <= 14:
        return "Days to two weeks"
    if days <= 60:
        return "Two to eight weeks"
    return "One to three months"


def _contract_price(contract: Mapping[str, Any], side: str) -> float:
    bid = _safe_float(contract.get("bid"))
    ask = _safe_float(contract.get("ask"))
    mid = _safe_float(contract.get("mid"))
    px = _safe_float(contract.get("price"))
    if side == "buy":
        if np.isfinite(ask) and ask > 0:
            return ask
    else:
        if np.isfinite(bid) and bid >= 0:
            return bid
    if np.isfinite(mid) and mid > 0:
        return mid
    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid and ask > 0:
        return 0.5 * (bid + ask)
    if np.isfinite(px) and px > 0:
        return px
    return np.nan


def _contract_label(contract: Mapping[str, Any]) -> str:
    expiry = str(contract.get("expiry", "-"))
    cp = str(contract.get("call_put", "-"))
    strike = _safe_float(contract.get("strike"))
    mny = _safe_float(contract.get("moneyness"))
    strike_text = f"{strike:.2f}" if np.isfinite(strike) else "-"
    mny_text = f"{mny:.2f}" if np.isfinite(mny) else "-"
    return f"{expiry} {cp} K={strike_text} K/S={mny_text}"


def _ensure_contract_greeks(contract: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(contract or {})
    needed = any(not np.isfinite(_safe_float(out.get(k))) for k in ("delta", "gamma", "vega", "theta"))
    if not needed:
        return out
    spot = _safe_float(out.get("spot"))
    strike = _safe_float(out.get("strike"))
    ttm_days = _safe_float(out.get("ttm_days"))
    iv = _safe_float(out.get("iv"))
    cp = str(out.get("call_put", "C"))
    if not (
        np.isfinite(spot)
        and spot > 0
        and np.isfinite(strike)
        and strike > 0
        and np.isfinite(ttm_days)
        and ttm_days > 0
        and np.isfinite(iv)
        and iv > 0
    ):
        return out
    try:
        from data.greeks import compute_all_greeks

        greeks = compute_all_greeks(
            S=float(spot),
            K=float(strike),
            T=float(ttm_days) / 365.25,
            sigma=float(iv),
            cp="P" if cp.upper().startswith("P") else "C",
        )
    except Exception:
        return out
    for key in ("price", "delta", "gamma", "vega", "theta"):
        if not np.isfinite(_safe_float(out.get(key))):
            out[key] = _safe_float(greeks.get(key))
    return out


def _select_contract_structure(contracts: list[dict[str, Any]], metric_family: str) -> list[dict[str, Any]]:
    clean = [_ensure_contract_greeks(c) for c in contracts or []]
    clean = [c for c in clean if np.isfinite(_safe_float(c.get("iv")))]
    if not clean:
        return []
    family = str(metric_family)
    if family in {"level", "slope", "timing"}:
        legs: list[dict[str, Any]] = []
        for cp in ("C", "P"):
            subset = [c for c in clean if str(c.get("call_put", "")).upper().startswith(cp)]
            if subset:
                legs.append(min(subset, key=lambda c: abs(_safe_float(c.get("moneyness")) - 1.0)))
        return legs or [min(clean, key=lambda c: abs(_safe_float(c.get("moneyness")) - 1.0))]
    if family == "asymmetry":
        puts = [c for c in clean if str(c.get("call_put", "")).upper().startswith("P")]
        return sorted(puts or clean, key=lambda c: abs(_safe_float(c.get("moneyness")) - 1.0))[:2]
    if family == "convexity":
        return sorted(clean, key=lambda c: abs(_safe_float(c.get("moneyness")) - 1.0), reverse=True)[:2]
    return clean[:2]


def _choose_trade_peer(peer: str, peer_list: list[str], feature_health: Mapping[str, Any] | None) -> str:
    if peer:
        return str(peer).upper()
    pairs = (feature_health or {}).get("pairs", []) if isinstance(feature_health, Mapping) else []
    best_peer = ""
    best_score = -np.inf
    for item in pairs:
        if not isinstance(item, Mapping):
            continue
        ticker = str(item.get("ticker", "")).upper()
        if ticker not in {p.upper() for p in peer_list}:
            continue
        score = _mean_finite(
            [
                _score_from_corr(item.get("correlation")),
                item.get("sign_consistency"),
            ],
            default=0.0,
        )
        if score > best_score:
            best_score = score
            best_peer = ticker
    return best_peer or (peer_list[0].upper() if peer_list else "")


def _spillover_beta(spill_meta: Mapping[str, Any] | None, substitutability: Mapping[str, Any] | None) -> dict[str, Any]:
    meta = dict(spill_meta or {})
    beta = abs(_safe_float(meta.get("median_response")))
    if np.isfinite(beta) and 0.05 <= beta <= 3.0:
        return {"beta": float(beta), "source": "spillover median response"}
    sub = _safe_float((substitutability or {}).get("score"))
    if np.isfinite(sub):
        return {"beta": float(np.clip(0.50 + 0.50 * sub, 0.50, 1.25)), "source": "substitutability-implied beta proxy"}
    return {"beta": 1.0, "source": "neutral fallback"}


def _integer_hedge_package(
    continuous_ratio: float,
    *,
    tolerance: float = HEDGE_RATIO_REL_TOLERANCE,
    max_contracts: int = MAX_HEDGE_PACKAGE_CONTRACTS,
) -> dict[str, Any]:
    ratio = _safe_float(continuous_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        ratio = 1.0
    max_contracts = max(1, int(max_contracts))
    ratio = float(np.clip(ratio, 1.0 / max_contracts, float(max_contracts)))

    best_any: tuple[float, int, int, int, float] | None = None
    best_within: tuple[int, float, int, int, float] | None = None
    for target_qty in range(1, max_contracts + 1):
        ideal_peer_qty = ratio * target_qty
        candidates = {
            int(np.floor(ideal_peer_qty)),
            int(np.ceil(ideal_peer_qty)),
            int(round(ideal_peer_qty)),
        }
        for peer_qty in candidates:
            if peer_qty < 1 or peer_qty > max_contracts:
                continue
            executable_ratio = peer_qty / target_qty
            abs_error = abs(executable_ratio - ratio)
            rel_error = abs_error / max(abs(ratio), 1e-8)
            total_contracts = target_qty + peer_qty
            any_score = (rel_error, total_contracts, target_qty, peer_qty, abs_error)
            if best_any is None or any_score < best_any:
                best_any = any_score
            if rel_error <= float(tolerance):
                within_score = (total_contracts, rel_error, target_qty, peer_qty, abs_error)
                if best_within is None or within_score < best_within:
                    best_within = within_score

    if best_within is not None:
        _total_contracts, rel_error, target_qty, peer_qty, abs_error = best_within
        executable_ratio = peer_qty / target_qty
    elif best_any is not None:
        rel_error, _total_contracts, target_qty, peer_qty, abs_error = best_any
        executable_ratio = peer_qty / target_qty
    else:
        peer_qty = int(np.clip(round(ratio), 1, max_contracts))
        target_qty = 1
        executable_ratio = float(peer_qty)
        abs_error = abs(executable_ratio - ratio)
        rel_error = abs_error / max(abs(ratio), 1e-8)

    within_tolerance = bool(rel_error <= float(tolerance))
    if abs_error <= 1e-12:
        status = "exact"
    elif within_tolerance:
        status = "within_tolerance"
    else:
        status = "outside_tolerance"
    return {
        "continuous_ratio": ratio,
        "executable_ratio": float(executable_ratio),
        "target_contracts": int(target_qty),
        "peer_contracts": int(peer_qty),
        "absolute_error": float(abs_error),
        "relative_error": float(rel_error),
        "tolerance": float(tolerance),
        "within_tolerance": within_tolerance,
        "status": status,
        "max_contracts": int(max_contracts),
    }


def _leg_exposure(leg: Mapping[str, Any]) -> dict[str, float]:
    contract = _ensure_contract_greeks(leg.get("contract", {}) if isinstance(leg.get("contract"), Mapping) else {})
    qty = _safe_float(leg.get("quantity"))
    sign = 1.0 if str(leg.get("action")) == "Buy" else -1.0
    spot = _safe_float(contract.get("spot"))
    if not np.isfinite(spot):
        strike = _safe_float(contract.get("strike"))
        mny = _safe_float(contract.get("moneyness"))
        spot = strike / mny if np.isfinite(strike) and np.isfinite(mny) and mny else np.nan
    multiplier = 100.0
    delta = _safe_float(contract.get("delta"))
    gamma = _safe_float(contract.get("gamma"))
    vega = _safe_float(contract.get("vega"))
    theta = _safe_float(contract.get("theta"))
    return {
        "delta_per_1pct": sign
        * qty
        * (delta if np.isfinite(delta) else 0.0)
        * (spot if np.isfinite(spot) else 0.0)
        * multiplier
        * 0.01,
        "vega": sign * qty * (vega if np.isfinite(vega) else 0.0) * multiplier,
        "gamma": sign * qty * (gamma if np.isfinite(gamma) else 0.0) * multiplier,
        "theta_per_day": sign * qty * (theta if np.isfinite(theta) else 0.0) * multiplier / 365.25,
    }


def _make_leg(action: str, ticker: str, contract: Mapping[str, Any], quantity: float = 1.0) -> dict[str, Any]:
    side = "buy" if action == "Buy" else "sell"
    px = _contract_price(contract, side)
    premium = px * quantity * 100.0 if np.isfinite(px) else np.nan
    return {
        "action": action,
        "ticker": ticker,
        "quantity": float(quantity),
        "contract": _ensure_contract_greeks(contract),
        "contract_description": _contract_label(contract),
        "price": px,
        "premium": premium,
    }


def _format_contract_quantity(quantity: Any) -> str:
    qty = _safe_float(quantity)
    if not np.isfinite(qty):
        return "?"
    nearest = round(qty)
    if abs(qty - nearest) <= 1e-9:
        return f"{int(nearest)}x"
    return f"{qty:.2f}x"


def _format_leg_list(legs: list[Mapping[str, Any]]) -> str:
    if not legs:
        return "No selected option leg"
    parts = []
    for leg in legs:
        qty_text = _format_contract_quantity(leg.get("quantity"))
        parts.append(f"{leg.get('action')} {qty_text} {leg.get('ticker')} {leg.get('contract_description')}")
    return "; ".join(parts)


def _classify_trade_family(
    metric_family: str, direction: str, net_premium: float, legs: list[Mapping[str, Any]]
) -> str:
    base = _trade_type(metric_family)
    if (
        direction == "Rich"
        and metric_family in {"asymmetry", "convexity"}
        and np.isfinite(net_premium)
        and net_premium > 0
        and any(str(leg.get("action")) == "Sell" for leg in legs)
    ):
        return "Delta-neutral premium collection"
    return base


def _trade_risk_summary(
    trade_type: str, classification: str, event_ctx: str, contract_audit: Mapping[str, Any]
) -> list[str]:
    risks = [
        "Convergence may fail if the surface difference is justified by a new catalyst.",
        "Spillover beta can change, leaving residual directional exposure.",
    ]
    if trade_type == "Delta-neutral premium collection":
        risks.append("Short optionality can lose on gap moves or realized volatility above implied.")
    elif trade_type == "Tail-risk RV":
        risks.append("Tail legs are jump-sensitive and can become hard to hedge in stress.")
    elif trade_type == "Skew-transfer RV":
        risks.append("Skew trades can carry hidden directional exposure despite delta offset.")
    elif trade_type in {"Term-structure RV", "Event-vol timing RV"}:
        risks.append("Calendar marks can move against the trade through roll-down or event repricing.")
    if classification == "conditional":
        risks.insert(0, "Conditional setup: wait for cleaner confirmation or size as an audit candidate.")
    if str(event_ctx) != "Idiosyncratic":
        risks.append(f"Current event context is {event_ctx}.")
    risks.extend(str(r) for r in contract_audit.get("risks", []))
    return list(dict.fromkeys(risks))


def _compile_trade_thesis(
    *,
    target: str,
    peer: str,
    asof: str | None,
    metric_family: str,
    feature: str,
    maturity_days: Any,
    direction: str,
    target_contracts: list[dict[str, Any]],
    spill_meta: Mapping[str, Any] | None,
    substitutability: Mapping[str, Any] | None,
    contract_audit: Mapping[str, Any],
    classification: Mapping[str, Any],
    event_ctx: str,
) -> dict[str, Any]:
    target_structure = _select_contract_structure(target_contracts, metric_family)
    peer_contracts = _load_supporting_contracts(peer, asof, maturity_days, metric_family) if peer else []
    peer_structure = _select_contract_structure(peer_contracts, metric_family)

    target_action = "Sell" if direction == "Rich" else "Buy"
    peer_action = "Buy" if direction == "Rich" else "Sell"
    target_unit_legs = [_make_leg(target_action, target, c, 1.0) for c in target_structure]
    peer_unit_legs = [_make_leg(peer_action, peer, c, 1.0) for c in peer_structure] if peer else []

    beta_info = _spillover_beta(spill_meta, substitutability)
    beta = _safe_float(beta_info.get("beta"))
    unit_target_delta = sum(_leg_exposure(leg)["delta_per_1pct"] for leg in target_unit_legs)
    raw_peer_unit_delta = sum(_leg_exposure(leg)["delta_per_1pct"] for leg in peer_unit_legs)
    adjusted_peer_delta = raw_peer_unit_delta * beta if np.isfinite(beta) else np.nan
    if (
        np.isfinite(unit_target_delta)
        and np.isfinite(adjusted_peer_delta)
        and abs(adjusted_peer_delta) > 1e-8
        and abs(unit_target_delta) > 1e-8
    ):
        continuous_hedge_ratio = float(np.clip(-unit_target_delta / adjusted_peer_delta, 0.05, 20.0))
    else:
        continuous_hedge_ratio = 1.0
    hedge_package = (
        _integer_hedge_package(continuous_hedge_ratio)
        if peer_unit_legs
        else {
            "continuous_ratio": continuous_hedge_ratio,
            "executable_ratio": np.nan,
            "target_contracts": 1,
            "peer_contracts": 0,
            "absolute_error": np.nan,
            "relative_error": np.nan,
            "tolerance": HEDGE_RATIO_REL_TOLERANCE,
            "within_tolerance": False,
            "status": "no_peer_hedge",
            "max_contracts": MAX_HEDGE_PACKAGE_CONTRACTS,
        }
    )
    target_contract_qty = int(hedge_package.get("target_contracts") or 1)
    peer_contract_qty = int(hedge_package.get("peer_contracts") or 0)
    hedge_ratio = _safe_float(hedge_package.get("executable_ratio"))
    target_legs = [_make_leg(target_action, target, leg["contract"], target_contract_qty) for leg in target_unit_legs]
    peer_legs = [_make_leg(peer_action, peer, leg["contract"], peer_contract_qty) for leg in peer_unit_legs]

    legs = target_legs + peer_legs
    gross_paid = sum(
        _safe_float(leg.get("premium"))
        for leg in legs
        if str(leg.get("action")) == "Buy" and np.isfinite(_safe_float(leg.get("premium")))
    )
    gross_received = sum(
        _safe_float(leg.get("premium"))
        for leg in legs
        if str(leg.get("action")) == "Sell" and np.isfinite(_safe_float(leg.get("premium")))
    )
    net_premium = gross_received - gross_paid
    raw_target_delta = sum(_leg_exposure(leg)["delta_per_1pct"] for leg in target_legs)
    raw_peer_delta = sum(_leg_exposure(leg)["delta_per_1pct"] for leg in peer_legs)
    net_delta_after = raw_target_delta + raw_peer_delta * beta if np.isfinite(beta) else np.nan
    exposures = {
        "raw_delta_target_per_1pct": raw_target_delta,
        "raw_delta_peer_per_1pct": raw_peer_delta,
        "unit_delta_target_per_1pct": unit_target_delta,
        "unit_delta_peer_per_1pct": raw_peer_unit_delta,
        "spillover_beta": beta,
        "adjusted_peer_delta_per_1pct": raw_peer_delta * beta if np.isfinite(beta) else np.nan,
        "estimated_net_delta_after_hedge_per_1pct": net_delta_after,
        "net_vega": sum(_leg_exposure(leg)["vega"] for leg in legs),
        "net_gamma": sum(_leg_exposure(leg)["gamma"] for leg in legs),
        "net_theta_per_day": sum(_leg_exposure(leg)["theta_per_day"] for leg in legs),
    }
    trade_type = _classify_trade_family(metric_family, direction, net_premium, legs)
    signal_text = {
        "level": "relative ATM implied-volatility level",
        "asymmetry": "relative skew / crash-protection asymmetry",
        "convexity": "relative wing and tail optionality",
        "slope": "relative term-structure shape",
        "timing": "relative event-vol timing",
    }.get(metric_family, f"{feature} surface dislocation")
    if direction == "Rich":
        expression = f"Short {target} optionality (rich) vs long {peer or 'peer'} optionality as hedge."
    else:
        expression = f"Long {target} optionality (cheap) vs short {peer or 'peer'} optionality as hedge."
    why = (
        f"{expression} This expresses the {signal_text} "
        "while using an integer contract hedge package to reduce equity-directional exposure."
    )
    title = f"{trade_type}: {_format_leg_list(target_legs)} / {_format_leg_list(peer_legs)}"
    main_risks = _trade_risk_summary(
        trade_type, str(classification.get("classification")), event_ctx, contract_audit
    )
    if peer_unit_legs:
        ratio_error = _safe_float(hedge_package.get("relative_error"))
        if hedge_package.get("status") == "outside_tolerance":
            main_risks.append(
                "Continuous hedge ratio could not be approximated within whole-contract tolerance; "
                "residual delta may be material."
            )
        elif np.isfinite(ratio_error) and ratio_error > 0.0:
            main_risks.append(
                "Hedge ratio is rounded to whole option contracts; check residual delta before execution."
            )
    return {
        "trade_type": trade_type,
        "title": title,
        "direction": f"{target_action} {target} structure / {peer_action.lower()} {peer or 'peer'} hedge structure",
        "buy_legs": [leg for leg in legs if str(leg.get("action")) == "Buy"],
        "sell_legs": [leg for leg in legs if str(leg.get("action")) == "Sell"],
        "buy_legs_text": _format_leg_list([leg for leg in legs if str(leg.get("action")) == "Buy"]),
        "sell_legs_text": _format_leg_list([leg for leg in legs if str(leg.get("action")) == "Sell"]),
        "gross_premium_paid": gross_paid,
        "gross_premium_received": gross_received,
        "net_premium": net_premium,
        "net_premium_label": (
            "credit"
            if np.isfinite(net_premium) and net_premium > 0
            else "debit" if np.isfinite(net_premium) and net_premium < 0 else "flat/unknown"
        ),
        "hedge_ratio": hedge_ratio,
        "continuous_hedge_ratio": continuous_hedge_ratio,
        "hedge_package": hedge_package,
        "hedge_ratio_source": beta_info.get("source", "unknown"),
        "exposures": exposures,
        "why_expresses_signal": why,
        "main_risks": list(dict.fromkeys(main_risks)),
        "supporting_contracts": target_contracts + peer_contracts,
    }


def _likely_driver(
    *,
    event_ctx: str,
    feature: str,
    metric_family: str,
    classification: dict[str, Any],
) -> str:
    if event_ctx == "Systemic":
        return "Broad peer-group volatility repricing"
    if event_ctx == "Cluster":
        return "Cluster or catalyst-linked peer move"
    if metric_family == "timing" or feature == "Event Bump":
        return "Localized expiry/event volatility"
    checks = classification.get("soft_checks", {})
    if not checks.get("structural_validity", True):
        return "Target/peer surface-grid comparability breakdown"
    if not checks.get("data_quality", True):
        return "Model-fit or data-quality degradation"
    if not checks.get("spillover_support", True):
        return "Historical spillover support does not confirm convergence"
    return "Target-vs-peer implied-volatility dislocation"


def _maturity_region(days: Any) -> str:
    d = _safe_float(days)
    if not np.isfinite(d) or d <= 0:
        return "surface"
    if d <= 21:
        return "front"
    if d <= 75:
        return "middle"
    return "back"


def _unique_flat(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        values = item if isinstance(item, list) else [item]
        for value in values:
            if value is None or value == "":
                continue
            key = str(value)
            if key not in seen:
                out.append(value)
                seen.add(key)
    return out


def _group_anomaly_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("_pair_key", "")),
            str(row.get("_feature_type", "")),
            str(row.get("maturity_region", "")),
        )
        groups.setdefault(key, []).append(row)

    grouped: list[dict[str, Any]] = []
    for (pair_key, feature_type, region), members in groups.items():
        members_sorted = sorted(
            members,
            key=lambda r: _safe_float(r.get("severity_score")),
            reverse=True,
        )
        rep = dict(members_sorted[0])
        group_size = len(members_sorted)
        if group_size > 1:
            rep["title"] = f"{group_size} {feature_type} anomalies for {pair_key} in the {region} maturity region"
            rep["why_it_matters"] = (
                f"{group_size} related signals cluster by pair, feature, and maturity. "
                f"{rep.get('why_it_matters', '')}"
            )
            rep["impact_on_trade_confidence"] = (
                f"Grouped anomaly, not a standalone trade. {rep.get('impact_on_trade_confidence', '')}"
            )
        rep["group_size"] = group_size
        rep["affected_names"] = _unique_flat(m.get("affected_names", []) for m in members_sorted)
        rep["supporting_contracts"] = _unique_flat(m.get("supporting_contracts", []) for m in members_sorted)[:12]
        rep["classification_reasons"] = _unique_flat(m.get("classification_reasons", []) for m in members_sorted)
        rep["member_signals"] = [m.get("source_signal", {}) for m in members_sorted]
        rep["severity_score"] = _mean_finite([m.get("severity_score") for m in members_sorted], default=0.0)
        rep["trade_score"] = _mean_finite([m.get("trade_score") for m in members_sorted], default=0.0)
        grouped.append(rep)
    return grouped


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
            "confidence_input": (
                "RV trade scoring combines dislocation magnitude, substitutability, "
                "spillover, data quality, and feature warnings."
            ),
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
        contract_audit = _contract_auditability(supporting_contracts)
        dislocation = _dislocation_evidence(
            z=z,
            percentile=pct,
            spread=spread,
            metric_family=metric_family,
        )
        substitutability = _substitutability_evidence(
            peer=peer,
            peer_list=peer_list,
            alignment=alignment,
            surface_meta=surface_meta,
            feature_health=feature_health,
            spill_score=spill_score,
        )
        classification = _classify_signal(
            signal_type=signal_type,
            metric_family=metric_family,
            dislocation=dislocation,
            data_quality=data_quality,
            comparability=comparability,
            alignment=alignment,
            spill_label=spill_label,
            spill_score=spill_score,
            event_ctx=event_ctx,
            contract_audit=contract_audit,
            feature_warnings=feature_warnings,
            substitutability=substitutability,
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

        tradeability = _safe_float(classification.get("trade_score"))
        if not np.isfinite(tradeability):
            tradeability = confidence_score
        if direction == "Neutral":
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
                "peer_dispersion": (
                    _safe_float(event_meta.get("peer_abs_median_move")) if isinstance(event_meta, dict) else np.nan
                ),
                "details": event_meta,
            },
            "dynamics": {
                "spillover_strength": (
                    spill_meta.get("strength", spill_label) if isinstance(spill_meta, dict) else spill_label
                ),
                "same_direction_probability": (
                    _safe_float(spill_meta.get("same_direction_probability"))
                    if isinstance(spill_meta, dict)
                    else np.nan
                ),
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
            "contract_audit": contract_audit,
            "dislocation_strength": dislocation,
            "substitutability": substitutability,
            "classification": classification,
            "narrative": narrative,
            "confidence_score": confidence_score,
        }

        rows.append(
            {
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
                "signal_classification": classification["classification"],
                "judgment": classification["judgment"],
                "trade_score": classification["trade_score"],
                "classification_reasons": classification["reasons"],
                "contract_auditability": contract_audit["label"],
                "signal": signal_obj,
            }
        )

    opp = pd.DataFrame(rows)
    opp = opp.sort_values(
        ["tradeability_score", "z_score", "spread"],
        key=lambda s: s.abs() if s.name in {"z_score", "spread"} else s,
        ascending=False,
    ).reset_index(drop=True)
    opp["rank"] = np.arange(1, len(opp) + 1)
    dashboard["opportunities"] = opp

    trade_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    for _, r in opp.iterrows():
        signal = r.get("signal", {}) if isinstance(r.get("signal"), dict) else {}
        magnitude = signal.get("magnitude", {}) if isinstance(signal.get("magnitude"), dict) else {}
        location = signal.get("location", {}) if isinstance(signal.get("location"), dict) else {}
        classification = signal.get("classification", {}) if isinstance(signal.get("classification"), dict) else {}
        dislocation = (
            signal.get("dislocation_strength", {}) if isinstance(signal.get("dislocation_strength"), dict) else {}
        )
        contract_audit = signal.get("contract_audit", {}) if isinstance(signal.get("contract_audit"), dict) else {}
        substitutability = (
            signal.get("substitutability", {}) if isinstance(signal.get("substitutability"), dict) else {}
        )
        metric_family = str(location.get("metric_family") or _metric_family(r.get("feature", "")))
        peer = str(magnitude.get("peer") or "").upper()
        reference_label = str(magnitude.get("reference_label") or "Weighted peer synthetic")
        trade_peer = _choose_trade_peer(
            peer,
            peer_list,
            signal.get("structure", {}).get("feature_health", {}) if isinstance(signal.get("structure"), dict) else {},
        )
        hedge_or_peer = trade_peer or reference_label.replace("Weighted peer synthetic", "weighted peer basket")

        if r.get("signal_classification") in {"trade", "conditional"}:
            trade = _compile_trade_thesis(
                target=target,
                peer=trade_peer,
                asof=asof,
                metric_family=metric_family,
                feature=str(r.get("feature", "")),
                maturity_days=location.get("T_days"),
                direction=str(r.get("direction")),
                target_contracts=signal.get("supporting_contracts", []),
                spill_meta=spill_meta,
                substitutability=substitutability,
                contract_audit=contract_audit,
                classification=classification,
                event_ctx=str(r.get("event_context", "Unknown")),
            )
            warning_text = str(r.get("warnings") or "")
            risks = list(trade.get("main_risks", []))
            if warning_text:
                risks.append(warning_text)
            judgment = str(r.get("judgment", _trade_judgment(str(r.get("signal_classification")))))
            signal["trade"] = trade
            source_signal = r.to_dict()
            trade_rows.append(
                {
                    "rank": 0,
                    "title": trade.get("title") or str(r.get("opportunity", "")).rstrip("."),
                    "judgment": judgment,
                    "trade_type": trade.get("trade_type", _trade_type(metric_family)),
                    "direction": trade.get("direction", ""),
                    "target": target,
                    "hedge_or_peer": hedge_or_peer,
                    "maturity": r.get("maturity", "All"),
                    "confidence": r.get("confidence", "Low"),
                    "horizon": _trade_horizon(str(r.get("maturity", ""))),
                    "buy_legs": trade.get("buy_legs_text", ""),
                    "sell_legs": trade.get("sell_legs_text", ""),
                    "net_premium": trade.get("net_premium", np.nan),
                    "estimated_delta_after_hedge": trade.get("exposures", {}).get(
                        "estimated_net_delta_after_hedge_per_1pct", np.nan
                    ),
                    "rationale": (
                        f"{trade.get('why_expresses_signal', '')} "
                        f"{' '.join(classification.get('reasons', [])[:1])}"
                    ),
                    "supporting_contracts": trade.get(
                        "supporting_contracts", signal.get("supporting_contracts", [])
                    ),
                    "risks": list(dict.fromkeys(risks)),
                    "trade_score": r.get("trade_score", r.get("tradeability_score", np.nan)),
                    "substitutability": substitutability.get("label", ""),
                    "tradeability_score": r.get("tradeability_score", np.nan),
                    "trade": trade,
                    "source_signal": source_signal,
                }
            )
            continue

        source_signal = r.to_dict()
        checks = classification.get("soft_checks", {}) if isinstance(classification.get("soft_checks"), dict) else {}
        failed = [name.replace("_", " ") for name, ok in checks.items() if not ok]
        if failed:
            impact = "Weakens trade confidence: " + ", ".join(failed) + "."
        else:
            impact = "Explains the surface move but does not create a clean standalone RV trade."
        if bool(dislocation.get("meaningful")):
            why_it_matters = (
                f"{r.get('why', '')} It is meaningful market context, but the trade gates did not all pass."
            )
        else:
            why_it_matters = f"{r.get('why', '')} It is unusual enough to monitor, but not strong enough to trade."
        spill_relevance = str(r.get("spillover_support", "Unavailable"))
        if not checks.get("spillover_support", False):
            spill_relevance = f"{spill_relevance}; does not currently support convergence."
        affected = [target] + ([peer] if peer else peer_list)
        region = _maturity_region(location.get("T_days"))
        pair_key = f"{target}/{peer}" if peer else f"{target}/peer basket"
        anomaly_rows.append(
            {
                "rank": 0,
                "title": str(r.get("opportunity", "")).rstrip("."),
                "judgment": "Not tradeable",
                "anomaly_type": classification.get("anomaly_type", "Market Structure Anomaly"),
                "affected_names": list(dict.fromkeys(affected)),
                "likely_driver": _likely_driver(
                    event_ctx=str(r.get("event_context", "Unknown")),
                    feature=str(r.get("feature", "")),
                    metric_family=metric_family,
                    classification=classification,
                ),
                "systemic_or_idiosyncratic": r.get("event_context", "Unknown"),
                "spillover_relevance": spill_relevance,
                "why_it_matters": why_it_matters,
                "impact_on_trade_confidence": impact,
                "supporting_contracts": signal.get("supporting_contracts", []),
                "severity_score": _mean_finite(
                    [
                        dislocation.get("score"),
                        r.get("tradeability_score"),
                        contract_audit.get("score", 0.0) if isinstance(contract_audit, dict) else 0.0,
                    ],
                    default=0.0,
                ),
                "trade_score": r.get("trade_score", r.get("tradeability_score", np.nan)),
                "substitutability": substitutability.get("label", ""),
                "maturity_region": region,
                "group_size": 1,
                "classification_reasons": classification.get("reasons", []),
                "source_signal": source_signal,
                "member_signals": [source_signal],
                "_pair_key": pair_key,
                "_feature_type": str(r.get("feature", "")),
            }
        )

    trade_df = pd.DataFrame(trade_rows)
    if trade_df.empty:
        trade_df = dashboard["trade_opportunities"]
    else:
        trade_df = trade_df.sort_values("trade_score", ascending=False).reset_index(drop=True)
        trade_df["rank"] = np.arange(1, len(trade_df) + 1)
    anomaly_df = pd.DataFrame(_group_anomaly_rows(anomaly_rows))
    if anomaly_df.empty:
        anomaly_df = dashboard["market_anomalies"]
    else:
        anomaly_df = anomaly_df.sort_values("severity_score", ascending=False).reset_index(drop=True)
        anomaly_df["rank"] = np.arange(1, len(anomaly_df) + 1)
    dashboard["trade_opportunities"] = trade_df
    dashboard["market_anomalies"] = anomaly_df

    strongest = opp.iloc[opp["spread"].abs().argmax()] if not opp.empty else None
    tradeable = trade_df.iloc[0] if trade_df is not None and not trade_df.empty else None
    systemic = (
        anomaly_df[anomaly_df["systemic_or_idiosyncratic"] == "Systemic"].head(1)
        if not anomaly_df.empty
        else pd.DataFrame()
    )
    weakest = opp.sort_values("tradeability_score", ascending=True).head(1)
    dashboard["context_cards"] = {
        "strongest_dislocation": strongest["opportunity"] if strongest is not None else "Unavailable",
        "most_tradeable": tradeable["title"] if tradeable is not None else "No classified trade opportunity",
        "most_systemic": systemic.iloc[0]["title"] if not systemic.empty else "No systemic signal",
        "weakest_signal": weakest.iloc[0]["opportunity"] if not weakest.empty else "Unavailable",
        "data_quality_warnings": len(warnings),
    }

    summary: list[str] = []
    for _, r in trade_df.head(2).iterrows():
        summary.append(f"Trade candidate: {r['title']}. Confidence is {str(r['confidence']).lower()}.")
    for _, r in anomaly_df.head(2).iterrows():
        summary.append(f"Market anomaly: {r['title']} ({r['anomaly_type']}).")
    if not summary:
        summary.append(f"No classified RV signals passed the current threshold for {target}.")
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
    from analysis.services.data_availability_service import get_daily_iv_for_spillover  # delayed

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
