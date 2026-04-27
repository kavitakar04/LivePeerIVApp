"""Term-structure data preparation service.

Owns the ATM term-structure extraction and composite-curve computation.
``analysis.analysis_pipeline`` re-exports ``prepare_term_data`` for backward
compatibility while the pipeline is being split.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import logging
import numpy as np
import pandas as pd

from analysis.atm_extraction import compute_atm_by_expiry
from analysis.confidence_bands import confidence_z_score, normalize_confidence_level, peer_composite_pillar_bands
from analysis.smile_data_service import get_smile_slice
from analysis.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CI,
    DEFAULT_MAX_EXPIRIES,
)

logger = logging.getLogger(__name__)


def _compute_term_atm_curve(
    ticker: str,
    asof: str,
    *,
    atm_band: float,
    min_boot: int,
    ci: float,
    max_expiries: int,
    method: str = "fit",
) -> pd.DataFrame:
    df = get_smile_slice(ticker, asof, T_target_years=None, max_expiries=max_expiries)
    if df is None or df.empty:
        return pd.DataFrame()
    return compute_atm_by_expiry(
        df,
        atm_band=atm_band,
        method=method,
        model="auto",
        vega_weighted=True,
        n_boot=min_boot,
        ci_level=ci,
    )


def _apply_term_feature_band_policy(curve: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    """Mark term-point provenance and keep only estimator CIs as plot bands."""
    if curve is None or curve.empty:
        return curve
    out = curve.copy()
    mode = str(feature_mode or "iv_atm").lower()
    out["term_feature_mode"] = mode
    out["n_obs"] = pd.to_numeric(out.get("count", np.nan), errors="coerce")
    disp = pd.to_numeric(out.get("atm_dispersion", np.nan), errors="coerce")
    y = pd.to_numeric(out.get("atm_vol", np.nan), errors="coerce")
    valid_disp = np.isfinite(y) & np.isfinite(disp) & (disp >= 0.0)
    out["quote_dispersion"] = np.where(valid_disp, disp, np.nan)
    if mode == "iv_atm":
        out["term_point_source"] = "single_atm_observation"
        out["band_source"] = "none_single_atm"
        out["atm_lo"] = np.nan
        out["atm_hi"] = np.nan
        return out

    out["term_point_source"] = "surface_aggregate"
    lo = pd.to_numeric(out.get("atm_lo", np.nan), errors="coerce")
    hi = pd.to_numeric(out.get("atm_hi", np.nan), errors="coerce")
    valid_ci = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & (lo <= y) & (hi >= y)
    out["band_source"] = np.where(valid_ci, "bootstrap_model_fit", "none_ci_unavailable")
    out["atm_lo"] = np.where(valid_ci, lo, np.nan)
    out["atm_hi"] = np.where(valid_ci, hi, np.nan)
    out["atm_dispersion"] = np.where(valid_disp, disp, np.nan)
    return out


def _log_term_atm_curve(
    *,
    ticker: str,
    role: str,
    curve: pd.DataFrame,
    weight: float | None = None,
    alignment: str = "raw",
) -> None:
    if curve is None or curve.empty:
        logger.warning(
            "term ATM extraction ticker=%s role=%s status=empty assigned_weight=%s alignment=%s",
            ticker,
            role,
            weight,
            alignment,
        )
        return
    for _, row in curve.iterrows():
        logger.info(
            "term ATM extraction ticker=%s role=%s expiry=%s T=%.6f spot=%s atm_strike=%s "
            "atm_iv=%s iv_source=%s valid_options=%s model=%s extraction_status=%s "
            "assigned_weight=%s alignment=%s",
            ticker,
            role,
            row.get("expiry", ""),
            float(row.get("T", np.nan)),
            row.get("spot", np.nan),
            row.get("atm_strike", np.nan),
            row.get("atm_vol", np.nan),
            row.get("iv_source", "sigma"),
            row.get("count", np.nan),
            row.get("model", "unknown"),
            row.get("extraction_status", row.get("model", "unknown")),
            weight,
            alignment,
        )


def _align_curve_to_terms(curve: pd.DataFrame, terms: np.ndarray, tol_years: float) -> pd.DataFrame:
    """Return one nearest curve row per aligned term, preserving term order."""
    if curve is None or curve.empty or "T" not in curve:
        return pd.DataFrame()
    arr_T = pd.to_numeric(curve["T"], errors="coerce").to_numpy(float)
    rows = []
    used: set[int] = set()
    for term in np.asarray(terms, dtype=float):
        if not np.isfinite(term) or arr_T.size == 0:
            continue
        diffs = np.abs(arr_T - term)
        if not np.isfinite(diffs).any():
            continue
        j = int(np.nanargmin(diffs))
        if j in used or not np.isfinite(diffs[j]) or diffs[j] > tol_years:
            continue
        used.add(j)
        rows.append(curve.iloc[j])
    if not rows:
        return pd.DataFrame(columns=curve.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def _intersect_terms_one_to_one(base_terms: np.ndarray, peer_terms: np.ndarray, tol_years: float) -> np.ndarray:
    """Keep base terms that can be matched to distinct peer terms within tolerance."""
    base = np.asarray(base_terms, dtype=float)
    peer = np.asarray(peer_terms, dtype=float)
    base = base[np.isfinite(base)]
    peer = peer[np.isfinite(peer)]
    if base.size == 0 or peer.size == 0:
        return np.array([], dtype=float)

    candidates: list[tuple[float, int, int]] = []
    for i, t in enumerate(base):
        for j, p in enumerate(peer):
            diff = abs(t - p)
            if diff <= tol_years:
                candidates.append((float(diff), i, j))
    candidates.sort(key=lambda item: item[0])

    used_base: set[int] = set()
    used_peer: set[int] = set()
    kept: list[float] = []
    for _, i, j in candidates:
        if i in used_base or j in used_peer:
            continue
        used_base.add(i)
        used_peer.add(j)
        kept.append(float(base[i]))
    return np.array(sorted(kept), dtype=float)


def prepare_term_data(
    target: str,
    asof: str,
    ci: float = DEFAULT_CI * 100.0,
    overlay_synth: bool = False,
    peers: Iterable[str] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    atm_band: float = DEFAULT_ATM_BAND,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    feature_mode: str = "iv_atm",
) -> Dict[str, Any]:
    """Precompute ATM term structure and synthetic overlay data."""
    feature_mode = str(feature_mode or "iv_atm").lower()
    ci_level = normalize_confidence_level(ci) if ci and float(ci) > 0 else 0.0
    min_boot = 100 if ci_level > 0 and feature_mode in {"surface", "surface_grid"} else 0
    extraction_method = "single" if feature_mode == "iv_atm" else "fit"
    atm_curve = _compute_term_atm_curve(
        target,
        asof,
        atm_band=atm_band,
        min_boot=min_boot,
        ci=ci,
        max_expiries=max_expiries,
        method=extraction_method,
    )
    if atm_curve is None or atm_curve.empty:
        return {}
    atm_curve = _apply_term_feature_band_policy(atm_curve, feature_mode)
    original_target_expiry_count = int(len(atm_curve))
    _log_term_atm_curve(ticker=target, role="target", curve=atm_curve, alignment="raw")

    synth_curve = None
    synth_bands = None
    peer_curves: Dict[str, pd.DataFrame] = {}
    weight_series = pd.Series(dtype=float)
    term_warnings: list[str] = []
    alignment_status = "raw"
    composite_status = "not_requested"

    if peers:
        w = pd.Series(weights if weights else {p: 1.0 for p in peers}, dtype=float)
        if w.sum() <= 0:
            w = pd.Series({p: 1.0 for p in peers}, dtype=float)
        w = (w / w.sum()).astype(float)
        weight_series = w.copy()
        peers = [p for p in w.index if p in peers]

        curves: Dict[str, pd.DataFrame] = {}
        for p in peers:
            c = _compute_term_atm_curve(
                p,
                asof,
                atm_band=atm_band,
                min_boot=min_boot,
                ci=ci,
                max_expiries=max_expiries,
                method=extraction_method,
            )
            if not c.empty:
                c = _apply_term_feature_band_policy(c, feature_mode)
                curves[p] = c
                _log_term_atm_curve(ticker=p, role="peer", curve=c, weight=float(w.get(p, np.nan)), alignment="raw")
            else:
                _log_term_atm_curve(ticker=p, role="peer", curve=c, weight=float(w.get(p, np.nan)), alignment="raw")
        peer_curves = curves

        if curves:
            tol_years = 10.0 / 365.25
            arrays = [atm_curve["T"].to_numpy(float)] + [c["T"].to_numpy(float) for c in curves.values()]
            common_T = arrays[0]
            for arr in arrays[1:]:
                common_T = _intersect_terms_one_to_one(common_T, arr, tol_years)
                if common_T.size == 0:
                    break

            if common_T.size > 0:
                alignment_status = "aligned"
                common_T = np.sort(common_T)
                aligned_target_curve = _align_curve_to_terms(atm_curve, common_T, tol_years)

                atm_data: Dict[str, np.ndarray] = {}
                aligned_peer_curves: Dict[str, pd.DataFrame] = {}
                for p, c in curves.items():
                    aligned_peer = _align_curve_to_terms(c, common_T, tol_years)
                    if len(aligned_peer) == len(common_T):
                        vals = pd.to_numeric(aligned_peer["atm_vol"], errors="coerce").to_numpy(float)
                        if np.isfinite(vals).all():
                            atm_data[p] = vals
                            aligned_peer_curves[p] = aligned_peer

                if atm_data and len(aligned_target_curve) == len(common_T):
                    atm_curve = aligned_target_curve
                    peer_curves = aligned_peer_curves
                    if len(atm_curve) < original_target_expiry_count:
                        term_warnings.append(
                            f"Expiry alignment uses {len(atm_curve)} of {original_target_expiry_count} target expiries;"
                            " dropped maturities without a one-to-one peer match."
                        )
                    pillar_days = common_T * 365.25
                    level = ci_level or 0.68
                    n_boot = max(min_boot, 1)
                    aligned_weights = w.reindex(list(atm_data)).dropna()
                    if aligned_weights.sum() <= 0:
                        aligned_weights = pd.Series(1.0, index=list(atm_data), dtype=float)
                    aligned_weights = aligned_weights / aligned_weights.sum()
                    weight_series = aligned_weights.copy()
                    synth_bands = peer_composite_pillar_bands(
                        atm_data,
                        aligned_weights.to_dict(),
                        pillar_days,
                        level=level,
                        n_boot=n_boot,
                    )
                    synth_curve = pd.DataFrame(
                        {
                            "T": common_T,
                            "atm_vol": synth_bands.mean,
                        }
                    )
                    synth_curve["term_feature_mode"] = feature_mode
                    synth_curve["term_point_source"] = "weighted_peer_composite"
                    synth_curve["n_obs"] = np.nan
                    if feature_mode in {"surface", "surface_grid"}:
                        z_score = confidence_z_score(level)
                        comp_lo = []
                        comp_hi = []
                        comp_disp = []
                        for k, _t in enumerate(common_T):
                            band_rows = []
                            for p in aligned_weights.index:
                                c = peer_curves.get(p)
                                if c is None or c.empty:
                                    continue
                                row = c.iloc[k]
                                lo = float(row.get("atm_lo", np.nan))
                                hi = float(row.get("atm_hi", np.nan))
                                mid = float(row.get("atm_vol", np.nan))
                                wt = float(aligned_weights.get(p, np.nan))
                                if (
                                    np.isfinite(lo)
                                    and np.isfinite(hi)
                                    and np.isfinite(mid)
                                    and lo <= mid <= hi
                                    and np.isfinite(wt)
                                    and wt > 0
                                ):
                                    band_rows.append((lo, hi, mid, wt))
                            if band_rows:
                                w_arr = np.array([row[3] for row in band_rows], dtype=float)
                                w_arr = w_arr / float(w_arr.sum())
                                lo_err = np.array([max(row[2] - row[0], 0.0) for row in band_rows], dtype=float)
                                hi_err = np.array([max(row[1] - row[2], 0.0) for row in band_rows], dtype=float)
                                lo_half = float(np.sqrt(np.sum((w_arr * lo_err) ** 2)))
                                hi_half = float(np.sqrt(np.sum((w_arr * hi_err) ** 2)))
                                center = float(synth_bands.mean[k])
                                comp_lo.append(max(center - lo_half, 0.0))
                                comp_hi.append(center + hi_half)
                                comp_disp.append(float(max(lo_half, hi_half) / z_score))
                            else:
                                comp_lo.append(np.nan)
                                comp_hi.append(np.nan)
                                comp_disp.append(np.nan)
                        synth_curve["atm_lo"] = comp_lo
                        synth_curve["atm_hi"] = comp_hi
                        synth_curve["atm_dispersion"] = comp_disp
                        synth_curve["band_source"] = "weighted_peer_bootstrap_model_fit"
                    else:
                        synth_curve["atm_lo"] = np.nan
                        synth_curve["atm_hi"] = np.nan
                        synth_curve["atm_dispersion"] = np.nan
                        synth_curve["band_source"] = "none_single_atm"
                    composite_status = "aligned_weighted"
                    logger.info(
                        "term peer composite built target=%s asof=%s alignment=%s common_expiries=%s weights=%s",
                        target,
                        asof,
                        alignment_status,
                        [float(x) for x in common_T],
                        aligned_weights.to_dict(),
                    )
                else:
                    composite_status = "invalid_no_aligned_peer_values"
                    term_warnings.append(
                        "Peer maturities overlap target, but no peer has a complete aligned ATM curve."
                    )
            else:
                alignment_status = "raw_no_overlap"
                composite_status = "invalid_no_maturity_overlap"
                term_warnings.append(
                    "Target and peer maturities do not overlap within 10 calendar days; "
                    "peer lines are raw and no weighted composite was built."
                )
        elif peers:
            composite_status = "invalid_no_peer_curves"
            term_warnings.append("No peer ATM curves were available for the selected date.")

        if peer_curves:
            target_vals = atm_curve["atm_vol"].to_numpy(float)
            target_med = float(np.nanmedian(target_vals)) if target_vals.size else np.nan
            for p, c in peer_curves.items():
                peer_med = float(np.nanmedian(c["atm_vol"].to_numpy(float))) if not c.empty else np.nan
                if np.isfinite(target_med) and np.isfinite(peer_med) and abs(peer_med - target_med) > 0.30:
                    msg = (
                        f"Extreme peer ATM level difference: {p} median {peer_med:.1%} "
                        f"vs {target} median {target_med:.1%}."
                    )
                    term_warnings.append(msg)
                    logger.warning("term peer overlay warning target=%s peer=%s reason=%s", target, p, msg)

    return {
        "atm_curve": atm_curve,
        "synth_curve": synth_curve,
        "synth_bands": synth_bands,
        "peer_curves": peer_curves,
        "weights": weight_series,
        "alignment_status": alignment_status,
        "composite_status": composite_status,
        "term_warnings": term_warnings,
    }


__all__ = ["prepare_term_data"]
