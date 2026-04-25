"""Analysis helpers for term-structure plot data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from volModel.termFit import fit_term_structure, term_structure_iv


def term_x_values(df: pd.DataFrame, x_units: str = "years") -> np.ndarray:
    """Return term x-values in either years or days."""
    x = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
    return x * 365.25 if x_units == "days" else x


def compute_term_fit_curve(
    atm_df: pd.DataFrame,
    *,
    x_units: str = "years",
    degree: int = 2,
    points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the ATM term structure and return display x-values plus IV fit."""
    if atm_df is None or atm_df.empty or not {"T", "atm_vol"}.issubset(atm_df.columns):
        return np.array([]), np.array([])

    x = pd.to_numeric(atm_df["T"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(atm_df["atm_vol"], errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() <= degree:
        return np.array([]), np.array([])

    try:
        x_fit = x[mask]
        y_fit_obs = y[mask]
        params = fit_term_structure(x_fit, y_fit_obs, degree=degree)
        grid = np.linspace(float(x_fit.min()), float(x_fit.max()), int(points))
        fit_y = term_structure_iv(grid, params)
        grid_plot = grid * 365.25 if x_units == "days" else grid
        return grid_plot, fit_y
    except Exception:
        return np.array([]), np.array([])


def compute_term_spread_curve(
    target_curve: pd.DataFrame,
    synth_curve: pd.DataFrame,
    *,
    x_units: str = "years",
    points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Align target and synthetic term curves and return target minus synthetic."""
    if (
        target_curve is None
        or synth_curve is None
        or target_curve.empty
        or synth_curve.empty
        or not {"T", "atm_vol"}.issubset(target_curve.columns)
        or not {"T", "atm_vol"}.issubset(synth_curve.columns)
    ):
        return np.array([]), np.array([])

    x_tgt = term_x_values(target_curve, x_units=x_units)
    y_tgt = pd.to_numeric(target_curve["atm_vol"], errors="coerce").to_numpy(float)
    x_syn = term_x_values(synth_curve, x_units=x_units)
    y_syn = pd.to_numeric(synth_curve["atm_vol"], errors="coerce").to_numpy(float)

    valid_tgt = np.isfinite(x_tgt) & np.isfinite(y_tgt)
    valid_syn = np.isfinite(x_syn) & np.isfinite(y_syn)
    if valid_tgt.sum() < 2 or valid_syn.sum() < 2:
        return np.array([]), np.array([])

    lo_x = max(float(np.nanmin(x_tgt[valid_tgt])), float(np.nanmin(x_syn[valid_syn])))
    hi_x = min(float(np.nanmax(x_tgt[valid_tgt])), float(np.nanmax(x_syn[valid_syn])))
    if hi_x <= lo_x:
        return np.array([]), np.array([])

    grid = np.linspace(lo_x, hi_x, int(points))
    tgt_i = np.interp(grid, x_tgt[valid_tgt], y_tgt[valid_tgt])
    syn_i = np.interp(grid, x_syn[valid_syn], y_syn[valid_syn])
    return grid, tgt_i - syn_i


def term_ci_error(atm_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return y-error array for ATM confidence intervals when available."""
    if atm_df is None or atm_df.empty or not {"atm_vol", "atm_lo", "atm_hi"}.issubset(atm_df.columns):
        return None
    y = pd.to_numeric(atm_df["atm_vol"], errors="coerce").to_numpy(float)
    y_lo = pd.to_numeric(atm_df["atm_lo"], errors="coerce").to_numpy(float)
    y_hi = pd.to_numeric(atm_df["atm_hi"], errors="coerce").to_numpy(float)
    if not (np.isfinite(y_lo).any() and np.isfinite(y_hi).any()):
        return None
    return np.vstack([np.clip(y - y_lo, 0, None), np.clip(y_hi - y, 0, None)])
