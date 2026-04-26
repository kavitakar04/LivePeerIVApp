"""
Polynomial and Thin Plate Spline fitting for implied volatility surfaces.

This module provides two fitting approaches:
1. Simple polynomial fitting (quadratic)
2. Thin Plate Spline (TPS) fitting using RBF interpolation
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable
import numpy as np
import math

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

__all__ = [
    "fit_simple_poly",
    "fit_tps",
    "fit_poly",
    "fit_tps_slice",
    "tps_smile_iv",
    "TPSPredictor",
]


class TPSPredictor:
    """Pickle-safe callable wrapper for a fitted TPS interpolator."""

    def __init__(self, rbf):
        self.rbf = rbf

    def __call__(self, k_new):
        k_new = np.asarray(k_new, dtype=float)
        if k_new.ndim == 0:
            k_new = k_new.reshape(1)
        return self.rbf(k_new.reshape(-1, 1))


def fit_simple_poly(k: np.ndarray, iv: np.ndarray, weights: Optional[np.ndarray] = None,
                   band: float = 0.25) -> Dict[str, float]:
    """
    Simple quadratic polynomial fit around ATM (k=0).
    
    Returns f(0), f'(0), f''(0) and rmse where f(k) = a + b*k + c*k^2
    
    Parameters:
    -----------
    k : np.ndarray
        Log-moneyness values
    iv : np.ndarray  
        Implied volatility values
    weights : Optional[np.ndarray]
        Optional weights for fitting
    band : float
        Band around ATM for focusing the fit
        
    Returns:
    --------
    Dict with atm_vol, skew, curv, rmse, model
    """
    # Focus near-ATM
    mask = np.abs(k) <= band
    if mask.sum() < 3:
        # Widen if too sparse
        mask = np.argsort(np.abs(k))[:max(3, min(7, k.size))]
    
    x = k[mask]
    y = iv[mask]
    
    # Set up design matrix
    X = np.column_stack([np.ones_like(x), x, x**2])
    
    if weights is not None:
        w = weights[mask]
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # f(k) = a + b*k + c*k^2
    a, b, c = beta
    yhat = a + b*x + c*x**2
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    
    return {
        "atm_vol": float(a), 
        "skew": float(b), 
        "curv": float(2*c), 
        "rmse": rmse,
        "model": "simple_poly"
    }


def fit_tps(
    k: np.ndarray,
    iv: np.ndarray,
    weights: Optional[np.ndarray] = None,
    smoothing: float = 1e-2,  # non-zero default prevents overfitting
) -> Dict[str, float]:
  
    """
    Thin Plate Spline (TPS) fit using RBF interpolation.
    
    Parameters:
    -----------
    k : np.ndarray
        Log-moneyness values
    iv : np.ndarray
        Implied volatility values  
    weights : Optional[np.ndarray]
        Optional weights (used as inverse variances for smoothing)
    smoothing : float
        Smoothing parameter for TPS
        
    Returns:
    --------
    Dict with atm_vol, skew, curv, rmse, model and interpolator function
    """
    if not _HAS_SCIPY:
        # Fallback to simple polynomial if scipy not available
        return fit_simple_poly(k, iv, weights)
    
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    
    # Handle weights by converting to smoothing parameter per point
    # Convert weights to per-point smoothing – higher weight → less smoothing
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        point_smoothing = smoothing + (1.0 / np.clip(weights, 1e-8, None))
    else:
        point_smoothing = smoothing

    
    try:
        # Create TPS interpolator  
        rbf = RBFInterpolator(
            y=k.reshape(-1, 1),
            d=iv,
            kernel='thin_plate_spline',
            smoothing=point_smoothing
        )
        
        predict_iv = TPSPredictor(rbf)
        
        # Get ATM values using finite differences
        atm_vol = float(predict_iv(np.array([0.0]))[0])
        skew, curv = _finite_diff(lambda x: float(predict_iv(np.array([x]))[0]), 0.0)
        
        # Compute RMSE on original points
        iv_pred = predict_iv(k)
        rmse = float(np.sqrt(np.mean((iv_pred - iv)**2)))
        
        return {
            "atm_vol": atm_vol,
            "skew": skew, 
            "curv": curv,
            "rmse": rmse,
            "model": "tps",
            "interpolator": predict_iv
        }
        
    except Exception:
        # Fallback to simple polynomial on any error
        return fit_simple_poly(k, iv, weights)


def fit_poly(k: np.ndarray, iv: np.ndarray, weights: Optional[np.ndarray] = None,
            method: str = "simple", **kwargs) -> Dict[str, float]:
    """
    Main polynomial fitting function that dispatches to simple or TPS method.
    
    Parameters:
    -----------
    k : np.ndarray
        Log-moneyness values
    iv : np.ndarray
        Implied volatility values
    weights : Optional[np.ndarray]
        Optional weights for fitting
    method : str
        Either "simple" for quadratic polynomial or "tps" for thin plate spline
    **kwargs : 
        Additional arguments passed to specific fitting methods
        
    Returns:
    --------
    Dict with fitting results
    """
    if method.lower() == "tps":
        return fit_tps(k, iv, weights, **kwargs)
    else:
        return fit_simple_poly(k, iv, weights, **kwargs)


def fit_tps_slice(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    weights: Optional[np.ndarray] = None,
    smoothing: float = 1e-2,
) -> Dict[str, float]:
    """Fit a TPS-based smile slice in strike space.

    Parameters
    ----------
    S : float
        Spot price used to normalize strikes.
    K : np.ndarray
        Strike prices for the slice.
    T : float
        Expiry in years (unused but kept for API compatibility).
    iv : np.ndarray
        Observed implied volatilities.
    weights : Optional[np.ndarray]
        Optional weights for the fit.
    smoothing : float
        Smoothing parameter for TPS fitting.

    Returns
    -------
    dict
        Output of :func:`fit_poly` using the ``tps`` method.
    """

    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)
    k = np.log(np.clip(K, 1e-12, None) / float(S))
    return fit_poly(k, iv, weights=weights, method="tps", smoothing=smoothing)


def tps_smile_iv(
    S: float,
    K: np.ndarray,
    T: float,
    params: Dict[str, float],
) -> np.ndarray:
    """Evaluate a TPS/polynomial smile on strike grid ``K``.

    Parameters
    ----------
    S : float
        Spot price for log-moneyness conversion.
    K : np.ndarray
        Strike prices to evaluate.
    T : float
        Expiry in years (unused, for API symmetry).
    params : dict
        Parameters returned by :func:`fit_tps_slice`.

    Returns
    -------
    np.ndarray
        Implied volatilities at the strikes ``K``.
    """

    K = np.asarray(K, dtype=float)
    k = np.log(np.clip(K, 1e-12, None) / float(S))
    if params.get("model") == "tps" and "interpolator" in params:
        try:
            return np.asarray(params["interpolator"](k), dtype=float)
        except Exception:
            pass
    a = params.get("atm_vol", np.nan)
    b = params.get("skew", 0.0)
    c2 = params.get("curv", 0.0) / 2.0
    return a + b * k + c2 * k * k


def _finite_diff(f: Callable[[float], float], x0: float, h: float = 1e-3) -> Tuple[float, float]:
    """
    Compute first and second derivatives using finite differences.
    
    Returns f'(x0) and f''(x0)
    """
    f_p = f(x0 + h)
    f_m = f(x0 - h) 
    f0 = f(x0)
    first = (f_p - f_m) / (2*h)
    second = (f_p - 2*f0 + f_m) / (h*h)
    return float(first), float(second)


def fit_tps_slice(S: float, K: np.ndarray, T: float, iv: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 smoothing: float = 1e-2) -> Dict[str, float]:
    """
    Fit TPS to a single expiry slice for smile plotting.
    
    This is a wrapper around fit_tps that matches the interface used by
    SVI and SABR slice fitting functions.
    
    Parameters:
    -----------
    S : float
        Spot price (used for log-moneyness calculation)
    K : np.ndarray
        Strike prices
    T : float
        Time to expiry (not used in TPS but kept for interface compatibility)
    iv : np.ndarray
        Implied volatility values
    weights : Optional[np.ndarray]
        Optional weights for fitting
    smoothing : float
        Smoothing parameter for TPS
        
    Returns:
    --------
    Dict with TPS parameters and interpolator function
    """
    # Convert strikes to log-moneyness
    K = np.asarray(K, dtype=float)
    k = np.log(K / S)
    
    # Fit TPS using existing function
    result = fit_tps(k, iv, weights, smoothing)
    
    # Add the original spot price for use in prediction
    result["S"] = float(S)
    result["T"] = float(T)
    
    return result


def tps_smile_iv(S: float, K_grid: np.ndarray, T: float, fit_params: Dict) -> np.ndarray:
    """
    Evaluate TPS model at given strikes to generate smile curve.
    
    Parameters:
    -----------
    S : float
        Spot price
    K_grid : np.ndarray
        Strike prices where to evaluate the model
    T : float
        Time to expiry (not used but kept for interface compatibility)
    fit_params : Dict
        Parameters from fit_tps_slice containing the interpolator
        
    Returns:
    --------
    np.ndarray
        Implied volatility values at the grid points
    """
    if "interpolator" not in fit_params:
        # Fallback: return constant ATM vol if interpolator not available
        atm_vol = fit_params.get("atm_vol", 0.2)
        return np.full_like(K_grid, atm_vol, dtype=float)
    
    # Convert strikes to log-moneyness
    K_grid = np.asarray(K_grid, dtype=float)
    k_grid = np.log(K_grid / S)
    
    # Use the interpolator from fit_params
    interpolator = fit_params["interpolator"]
    iv_grid = interpolator(k_grid)
    
    return np.asarray(iv_grid, dtype=float)
# Backward compatibility aliases
_local_poly_fit_atm = fit_simple_poly
