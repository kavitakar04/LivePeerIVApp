from __future__ import annotations

"""Simple ATM term structure fitting utilities."""

from typing import Dict
import numpy as np

__all__ = ["fit_term_structure", "term_structure_iv"]


def fit_term_structure(T: np.ndarray, iv: np.ndarray, degree: int = 2) -> Dict[str, np.ndarray]:
    """Fit a polynomial term structure ``iv = f(T)``.

    Parameters
    ----------
    T : array-like
        Time to expiry in **years**.
    iv : array-like
        Implied volatilities at each ``T``.
    degree : int, optional
        Polynomial degree, default 2 (quadratic).

    Returns
    -------
    dict
        Dictionary with keys ``coeff`` (highest order first), ``rmse`` and ``degree``.
        If there are insufficient data points the coefficient array will be empty
        and ``rmse`` will be ``nan``.
    """
    T = np.asarray(T, dtype=float)
    iv = np.asarray(iv, dtype=float)
    mask = np.isfinite(T) & np.isfinite(iv)
    T = T[mask]
    iv = iv[mask]
    if T.size <= degree:
        return {"coeff": np.array([]), "rmse": float("nan"), "degree": int(degree)}
    coeff = np.polyfit(T, iv, int(degree))
    fit = np.polyval(coeff, T)
    rmse = float(np.sqrt(np.mean((fit - iv) ** 2)))
    return {"coeff": coeff, "rmse": rmse, "degree": int(degree)}


def term_structure_iv(T: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate fitted polynomial term structure at ``T`` values."""
    coeff = np.asarray(params.get("coeff", []), dtype=float)
    if coeff.size == 0:
        return np.full_like(np.asarray(T, dtype=float), np.nan, dtype=float)
    return np.polyval(coeff, np.asarray(T, dtype=float))
