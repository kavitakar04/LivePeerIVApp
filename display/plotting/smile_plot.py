# display/plotting/smile_plot.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from volModel.sviFit import svi_smile_iv
from volModel.sabrFit import sabr_smile_iv
from volModel.polyFit import tps_smile_iv
from analysis.confidence_bands import Bands
from display.plotting.legend_utils import add_legend_toggles

ModelName = Literal["svi", "sabr", "tps"]


# ------------------
# helpers
# ------------------
def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return a.ravel() if a.ndim > 1 else a


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    mask = None
    for a in arrs:
        a = np.asarray(a, float)
        m = np.isfinite(a)
        mask = m if mask is None else (mask & m)
    return mask if mask is not None else np.array([], dtype=bool)


# ------------------
# main
# ------------------
def fit_and_plot_smile(
    ax: plt.Axes,
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    *,
    model: ModelName = "svi",
    params: Dict,
    bands: Optional[Bands] = None,
    moneyness_grid: Tuple[float, float, int] = (0.8, 1.2, 121),
    show_points: bool = True,
    call_put: Optional[np.ndarray] = None,   # array of 'C'/'P' per point
    beta: float = 0.5,              # SABR beta
    label: Optional[str] = None,
    line_kwargs: Optional[Dict] = None,
    enable_toggles: bool = True,       # legend/keyboard toggles (all models)
    use_checkboxes: bool = False,      # keep False by default; legend is primary
) -> Dict:
    """
    Plot observed points, model fit, and optional CI on moneyness (K/S).
    
    Supports SVI, SABR, and TPS models with interactive legend toggles.
    All operations are logged to .txt file via db_logger.
    Returns dict: {params, rmse, T, S, series_map or None}
    """

    # ---- safety check: ensure axes has valid figure
    if ax is None or ax.figure is None:
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": float(S), "series_map": None}

    # ---- sanitize
    S = float(S)
    K = _as_1d(K)
    iv = _as_1d(iv)

    m = _finite_mask(K, iv)
    if not np.any(m):
        # Ensure axes has a valid figure before adding text
        if ax.figure is not None:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return {"params": {}, "rmse": np.nan, "T": float(T), "S": S, "series_map": None}

    K = K[m]
    iv = iv[m]
    cp = np.asarray(call_put)[m] if call_put is not None and len(call_put) == len(m) else None

    # ---- grid in strike space via moneyness
    mlo, mhi, n = moneyness_grid
    m_grid = np.linspace(float(mlo), float(mhi), int(n))
    K_grid = m_grid * S

    # ---- artists map for legend toggles
    series_map: Dict[str, list] = {}

    # ---- observed points (split by call/put when available)
    if show_points:
        if cp is not None:
            call_mask = cp == "C"
            put_mask  = cp == "P"
            if call_mask.any():
                pts_c = ax.scatter(K[call_mask] / S, iv[call_mask],
                                   s=20, alpha=0.85, color="#1f77b4", label="Calls")
                if enable_toggles:
                    series_map["Calls"] = [pts_c]
            if put_mask.any():
                pts_p = ax.scatter(K[put_mask] / S, iv[put_mask],
                                   s=20, alpha=0.85, color="#d62728", label="Puts")
                if enable_toggles:
                    series_map["Puts"] = [pts_p]
            # fallback for any unlabelled rows
            other_mask = ~call_mask & ~put_mask
            if other_mask.any():
                pts_o = ax.scatter(K[other_mask] / S, iv[other_mask],
                                   s=20, alpha=0.85, color="grey", label="Other")
                if enable_toggles:
                    series_map["Other"] = [pts_o]
        else:
            pts = ax.scatter(K / S, iv, s=20, alpha=0.85, label="Observed")
            if enable_toggles:
                series_map["Observed"] = [pts]

    # ---- fit + optional CI
    if not params:
        raise ValueError("fit parameters must be provided")
    fit_params = params
    if model == "svi":
        y_fit = svi_smile_iv(S, K_grid, T, fit_params)
    elif model == "sabr":
        y_fit = sabr_smile_iv(S, K_grid, T, fit_params)
    else:
        y_fit = tps_smile_iv(S, K_grid, T, fit_params)

    # ---- fit line
    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 2 if show_points else 1.6)
    fit_lbl = label or (f"{model.upper()} fit")
    fit_line = ax.plot(K_grid / S, y_fit, label=fit_lbl, **line_kwargs)
    if enable_toggles:
        series_map[f"{model.upper()} Fit"] = list(fit_line)

    # ---- confidence bands
    if bands is not None:
        ci_fill = ax.fill_between(bands.x / S, bands.lo, bands.hi, alpha=0.20, label=f"{int(bands.level*100)}% CI")
        ci_mean = ax.plot(bands.x / S, bands.mean, lw=1, alpha=0.6, linestyle="--")
        if enable_toggles:
            series_map[f"{model.upper()} Confidence Interval"] = [ci_fill, *ci_mean]

    # ---- ATM marker (not part of toggles / legend)
    ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.85, label="_nolegend_")

    # ---- axes / legend
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Implied Vol")
    if not ax.get_legend():
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            if len(labels) >= 5:
                if len(ax.figure.axes) == 1:
                    ax.figure.subplots_adjust(right=0.78)
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                    fontsize=8,
                    framealpha=0.92,
                )
            else:
                if len(ax.figure.axes) == 1:
                    ax.figure.subplots_adjust(right=0.94)
                ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # ---- legend-first toggle system (primary), keyboard helpers
    if enable_toggles and series_map and ax.figure is not None:
        add_legend_toggles(ax, series_map)  # your improved legend system
        # checkboxes are optional; keep off unless explicitly asked


    # ---- fit quality
    rmse = float(fit_params.get("rmse", np.nan)) if isinstance(fit_params, dict) else np.nan

    return {
        "params": fit_params,
        "rmse": rmse,
        "T": float(T),
        "S": float(S),
        "series_map": series_map if enable_toggles else None,
    }


def plot_peer_composite_smile(
    ax: plt.Axes,
    bands: Bands,
    *,
    label: Optional[str] = "Peer composite",
    line_kwargs: Optional[Dict] = None,
) -> Bands:
    """Plot peer-composite smile using pre-computed confidence bands."""

    ax.fill_between(bands.x, bands.lo, bands.hi, alpha=0.20, label=f"CI ({int(bands.level*100)}%)")

    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 1.8)
    ax.plot(bands.x, bands.mean, label=label, **line_kwargs)

    ax.set_xlabel("Strike / Moneyness")
    ax.set_ylabel("Implied Vol")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc="best", fontsize=8)

    return bands
