"""RV heatmap data preparation service.

Owns the surface-residual computation for the GUI RV Heatmap plot.
``analysis.analysis_pipeline`` re-exports ``prepare_rv_heatmap_data`` for
backward compatibility while the pipeline is being split.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from analysis.surfaces.peer_composite_builder import build_surface_grids, combine_surfaces
from analysis.weights.weight_service import compute_peer_weights


def prepare_rv_heatmap_data(
    target: str,
    peers: Iterable[str],
    asof: str,
    weight_mode: str = "corr_iv_atm",
    max_expiries: int = 6,
    lookback: int = 60,
) -> Dict[str, Any]:
    """Precompute per-cell surface-residual data for the GUI RV Heatmap plot."""
    from analysis.rv.rv_analysis import compute_surface_residual, compute_weight_stability  # delayed

    target = target.upper()
    peers = [p.upper() for p in peers]

    w = compute_peer_weights(target, peers, weight_mode=weight_mode)
    w_dict = w.to_dict()

    all_tickers = list(set([target] + list(w.index)))
    surfaces = build_surface_grids(tickers=all_tickers, max_expiries=max_expiries)

    target_surfaces = surfaces.get(target, {})
    peer_surfaces = {t: surfaces[t] for t in w.index if t in surfaces and t != target}
    synth_surfaces = combine_surfaces(peer_surfaces, w_dict) if peer_surfaces else {}

    residuals = compute_surface_residual(target_surfaces, synth_surfaces, lookback=lookback)

    asof_ts = pd.Timestamp(asof).normalize()
    latest_residual = residuals.get(asof_ts)
    if latest_residual is None and residuals:
        latest_residual = residuals[max(residuals.keys())]

    stability = compute_weight_stability(target, peers)

    return {
        "weights": w,
        "latest_residual": latest_residual,
        "weight_stability": stability,
        "asof": asof,
        "target": target,
    }


__all__ = ["prepare_rv_heatmap_data"]
