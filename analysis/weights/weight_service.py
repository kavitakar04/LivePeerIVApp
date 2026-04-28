"""Focused peer-weight service backed by unified weights."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from analysis.surfaces.peer_composite_builder import DEFAULT_MNY_BINS, DEFAULT_TENORS
from analysis.surfaces.pillar_selection import DEFAULT_PILLARS_DAYS
from analysis.weights.unified_weights import compute_unified_weights


def _canonical_weight_mode(mode: str) -> str:
    text = (mode or "corr_iv_atm").lower()
    if text in {"ul", "iv_atm", "surface", "surface_grid"}:
        return f"corr_{text}"
    return text


def compute_peer_weights(
    target: str,
    peers: Iterable[str],
    weight_mode: str = "corr_iv_atm",
    asof: str | None = None,
    pillar_days: Iterable[int] = DEFAULT_PILLARS_DAYS,
    tenor_days: Iterable[int] = DEFAULT_TENORS,
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS,
    surface_missing_policy: str = "median_impute",
    surface_min_coverage: float = 0.70,
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Compute peer-composite weights through the unified weight engine."""
    return compute_unified_weights(
        target=(target or "").upper(),
        peers=[p.upper() for p in peers],
        mode=_canonical_weight_mode(weight_mode),
        asof=asof,
        pillars_days=pillar_days,
        tenors=tenor_days,
        mny_bins=mny_bins,
        surface_missing_policy=surface_missing_policy,
        surface_min_coverage=surface_min_coverage,
        clip_negative=clip_negative,
        power=power,
    )
