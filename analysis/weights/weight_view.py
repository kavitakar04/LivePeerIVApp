"""Analysis-side peer weight resolution for GUI views."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import logging

from analysis.weights.correlation_utils import corr_weights
from analysis.weights import unified_weights
from analysis.config.settings import (
    DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_WEIGHT_POWER,
)

logger = logging.getLogger(__name__)


def _normalize_weights(weights: Optional[pd.Series], peers: list[str]) -> Optional[pd.Series]:
    if weights is None or weights.empty:
        return None
    w = weights.dropna().astype(float)
    w = w[w.index.isin(peers)]
    if w.empty or not np.isfinite(w.to_numpy(dtype=float)).any():
        return None
    total = float(w.sum())
    if total <= 0 or not np.isfinite(total):
        return None
    out = (w / total).reindex(peers).fillna(0.0).astype(float)
    arr = out.to_numpy(float)
    if not np.isfinite(arr).all() or (arr < -1e-12).any():
        return None
    if len(arr) > 1 and float(np.max(np.abs(arr))) > 0.98:
        return None
    out.attrs.update(getattr(weights, "attrs", {}))
    return out


def resolve_peer_weights(
    target: str,
    peers: Iterable[str],
    weight_mode: str,
    *,
    asof=None,
    pillars=None,
    settings: Optional[dict] = None,
    last_corr_df: Optional[pd.DataFrame] = None,
    last_corr_meta: Optional[dict] = None,
) -> pd.Series:
    """Resolve peer weights using unified weights, cached correlation, or equal fallback."""
    target = (target or "").upper()
    peers = [str(p).upper() for p in (peers or [])]
    pillars = list(pillars or DEFAULT_PILLAR_DAYS)
    settings = dict(settings or {})
    last_corr_meta = dict(last_corr_meta or {})
    weight_kwargs = {
        "pillars_days": pillars,
        "atm_tol_days": settings.get("pillar_tolerance_days"),
        "mny_bins": settings.get("mny_bins"),
        "tenors": settings.get("surface_tenors") or settings.get("pillars"),
        "surface_source": settings.get("surface_source", "fit"),
        "surface_model": settings.get("model", "svi"),
        "surface_missing_policy": settings.get("surface_missing_policy"),
        "surface_min_coverage": settings.get("surface_min_coverage"),
        "max_expiries": settings.get("max_expiries"),
        "power": settings.get("weight_power", DEFAULT_WEIGHT_POWER),
        "clip_negative": settings.get("clip_negative", DEFAULT_CLIP_NEGATIVE_WEIGHTS),
    }
    weight_kwargs = {k: v for k, v in weight_kwargs.items() if v is not None}

    attempts = (
        lambda: unified_weights.compute_unified_weights(
            target=target, peers=peers, mode=weight_mode, asof=asof, **weight_kwargs
        ),
        lambda: unified_weights.compute_unified_weights(
            target=target, peers=peers, mode=weight_mode, asof=asof, pillar_days=pillars
        ),
        lambda: unified_weights.compute_unified_weights(
            target=target, peers=peers, weight_mode=weight_mode, asof=asof, pillar_days=pillars
        ),
        lambda: unified_weights.compute_unified_weights(target, peers, weight_mode, asof, pillars),
    )
    for fn in attempts:
        try:
            normalized = _normalize_weights(fn(), peers)
            if normalized is not None:
                if normalized.attrs.get("weight_warning"):
                    logger.warning(normalized.attrs["weight_warning"])
                return normalized
        except TypeError:
            continue
        except Exception:
            break

    try:
        if (
            str(weight_mode or "").startswith("corr_")
            and isinstance(last_corr_df, pd.DataFrame)
            and not last_corr_df.empty
            and last_corr_meta.get("weight_mode") == weight_mode
            and last_corr_meta.get("clip_negative") == settings.get("clip_negative", DEFAULT_CLIP_NEGATIVE_WEIGHTS)
            and last_corr_meta.get("weight_power") == settings.get("weight_power", DEFAULT_WEIGHT_POWER)
            and last_corr_meta.get("pillars", []) == pillars
            and last_corr_meta.get("asof") == asof
            and set(last_corr_meta.get("tickers", [])) >= set([target] + peers)
        ):
            normalized = _normalize_weights(
                corr_weights(
                    last_corr_df,
                    target,
                    peers,
                    clip_negative=settings.get("clip_negative", DEFAULT_CLIP_NEGATIVE_WEIGHTS),
                    power=settings.get("weight_power", DEFAULT_WEIGHT_POWER),
                ),
                peers,
            )
            if normalized is not None:
                return normalized
    except Exception:
        pass

    eq = 1.0 / max(len(peers), 1)
    out = pd.Series(eq, index=peers, dtype=float)
    out.attrs["weight_warning"] = f"{weight_mode} weights unavailable or invalid; using equal weights"
    logger.warning(out.attrs["weight_warning"])
    return out
