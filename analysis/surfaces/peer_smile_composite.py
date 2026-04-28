"""Peer smile composite construction on a common moneyness grid."""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from analysis.surfaces.model_fit_service import fit_valid_model_params, predict_model_iv
from analysis.config.settings import DEFAULT_SMILE_GRID_POINTS, DEFAULT_SMILE_MONEYNESS_RANGE

LOGGER = logging.getLogger(__name__)
DEFAULT_SMILE_GRID = (*DEFAULT_SMILE_MONEYNESS_RANGE, DEFAULT_SMILE_GRID_POINTS)


def _slice_to_arrays(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        return (
            pd.to_numeric(data["T"], errors="coerce").to_numpy(float),
            pd.to_numeric(data["K"], errors="coerce").to_numpy(float),
            pd.to_numeric(data["sigma"], errors="coerce").to_numpy(float),
            pd.to_numeric(data["S"], errors="coerce").to_numpy(float),
        )
    return (
        np.asarray(data.get("T_arr", []), dtype=float),
        np.asarray(data.get("K_arr", []), dtype=float),
        np.asarray(data.get("sigma_arr", []), dtype=float),
        np.asarray(data.get("S_arr", []), dtype=float),
    )


def _filter_peer_quotes(
    S: float, K: np.ndarray, IV: np.ndarray, mny_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    mny = K / float(S)
    finite = np.isfinite(K) & np.isfinite(IV) & np.isfinite(mny)
    lo, hi = mny_range
    in_band = finite & (mny >= lo) & (mny <= hi)
    if int(in_band.sum()) >= 3:
        return K[in_band], IV[in_band]
    return K[finite], IV[finite]


def build_peer_smile_composite(
    peer_slices: Mapping[str, object],
    weights: Mapping[str, float] | pd.Series | None,
    *,
    model: str,
    target_T: float,
    moneyness_grid: tuple[float, float, int] = DEFAULT_SMILE_GRID,
) -> dict:
    """Evaluate each peer smile on the same grid and average pointwise."""
    lo, hi, n = moneyness_grid
    grid = np.linspace(float(lo), float(hi), int(n))
    weight_series = pd.Series(weights, dtype=float) if weights is not None and len(weights) else pd.Series(dtype=float)

    curves: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    raw_weights: dict[str, float] = {}

    for peer, data in peer_slices.items():
        peer_key = str(peer).upper()
        try:
            T_arr, K_arr, IV_arr, S_arr = _slice_to_arrays(data)
            finite_T = np.isfinite(T_arr)
            if not finite_T.any():
                skipped[peer_key] = "no finite expiries"
                continue
            T_used = float(T_arr[finite_T][np.argmin(np.abs(T_arr[finite_T] - float(target_T)))])
            mask = np.isclose(T_arr, T_used)
            if not mask.any():
                skipped[peer_key] = "no matching expiry slice"
                continue
            S = float(np.nanmedian(S_arr[mask]))
            if not np.isfinite(S) or S <= 0:
                skipped[peer_key] = "invalid spot"
                continue
            K_fit, IV_fit = _filter_peer_quotes(S, K_arr[mask], IV_arr[mask], (float(lo), float(hi)))
            if len(K_fit) < 3:
                skipped[peer_key] = "fewer than 3 valid quotes"
                continue
            params = fit_valid_model_params(model, S, K_fit, T_used, IV_fit)
            if not params:
                skipped[peer_key] = f"{model} fit rejected"
                continue
            curve = np.asarray(predict_model_iv(model, S, grid * S, T_used, params), dtype=float)
            if curve.shape != grid.shape or not np.isfinite(curve).all():
                skipped[peer_key] = "non-finite fitted grid"
                continue
            curves[peer_key] = curve
            raw_weights[peer_key] = float(weight_series.get(peer_key, weight_series.get(peer, 1.0)))
        except Exception as exc:
            skipped[peer_key] = str(exc)

    if not curves:
        LOGGER.warning(
            "peer smile composite failed: grid=%s n=%s target_T=%.6f skipped=%s",
            (float(lo), float(hi)),
            int(n),
            float(target_T),
            skipped,
        )
        return {
            "moneyness": np.array([], dtype=float),
            "iv": np.array([], dtype=float),
            "requested_moneyness": grid,
            "peer_curves": curves,
            "weights": {},
            "skipped": skipped,
            "degraded": True,
            "reason": "no valid peer smile curves",
        }

    w = pd.Series(raw_weights, dtype=float).reindex(curves.keys()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if (w < 0).any() or float(w.sum()) <= 0:
        w = pd.Series(1.0, index=list(curves.keys()), dtype=float)
    w = w / float(w.sum())

    curve_matrix = np.vstack([curves[p] for p in w.index])
    composite = np.average(curve_matrix, axis=0, weights=w.to_numpy(float))
    envelope_ok = bool(
        np.all(composite >= np.nanmin(curve_matrix, axis=0) - 1e-10)
        and np.all(composite <= np.nanmax(curve_matrix, axis=0) + 1e-10)
    )

    LOGGER.info(
        "peer smile composite built: grid=[%.3f, %.3f] points=%d model=%s "
        "peers_used=%s skipped=%s envelope_ok=%s weights=%s",
        float(lo),
        float(hi),
        int(n),
        str(model).lower(),
        list(w.index),
        skipped,
        envelope_ok,
        {k: float(v) for k, v in w.items()},
    )
    return {
        "moneyness": grid,
        "iv": composite,
        "peer_curves": curves,
        "weights": {k: float(v) for k, v in w.items()},
        "skipped": skipped,
        "envelope_ok": envelope_ok,
        "degraded": False,
    }


__all__ = ["DEFAULT_SMILE_GRID", "build_peer_smile_composite"]
