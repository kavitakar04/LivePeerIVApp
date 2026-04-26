# analysis/confidence_bands.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from statistics import NormalDist
import numpy as np

# We reuse your smile fitters
from volModel.sviFit import fit_svi_slice, svi_smile_iv
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv
from volModel.polyFit import fit_tps_slice, tps_smile_iv

__all__ = [
    "Bands",
    "bootstrap_bands",
    "residual_bootstrap_bands",
    "svi_confidence_bands",
    "sabr_confidence_bands",
    "tps_confidence_bands",
    "generate_term_structure_confidence_bands",
    "peer_composite_confidence_bands",
    "peer_composite_weight_bands",
    "peer_composite_pillar_bands",
    "normalize_confidence_level",
    "confidence_z_score",
]

@dataclass
class Bands:
    x: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    level: float


def normalize_confidence_level(level: float) -> float:
    """Return a confidence level as a decimal strictly between 0 and 1."""
    level = float(level)
    if level > 1.0:
        level /= 100.0
    if not np.isfinite(level) or level <= 0.0 or level >= 1.0:
        raise ValueError(f"confidence level must be in (0, 1), got {level!r}")
    return level


def confidence_z_score(level: float) -> float:
    """Two-sided normal z-score for a confidence level."""
    level = normalize_confidence_level(level)
    return float(NormalDist().inv_cdf(0.5 + level / 2.0))

# -----------------------------
# Generic nonparametric bootstrap bands
# -----------------------------
def bootstrap_bands(
    x: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], Dict],
    pred_fn: Callable[[Dict, np.ndarray], np.ndarray],
    grid: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
    random_state: Optional[int] = 42,
) -> Bands:
    """
    x, y: raw points (e.g., K or moneyness, iv)
    fit_fn: returns params dict from (x, y)
    pred_fn: (params, grid) -> yhat on grid
    grid: where to compute bands
    level: 0.68 ~ 1 sigma-ish; 0.95 for wide bands
    """
    level = normalize_confidence_level(level)
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)

    # fit once on all data for the center line
    p0 = fit_fn(x, y)
    center = pred_fn(p0, grid)

    # bootstrap
    draws = np.empty((n_boot, grid.size), dtype=float)
    n = len(x)
    idx = np.arange(n)
    for b in range(n_boot):
        resample = rng.choice(idx, size=n, replace=True)
        xb = x[resample]
        yb = y[resample]
        try:
            pb = fit_fn(xb, yb)
            draws[b] = pred_fn(pb, grid)
        except Exception:
            draws[b] = np.nan

    # compute quantiles
    alpha = 1.0 - level
    lo = np.nanquantile(draws, alpha / 2.0, axis=0)
    hi = np.nanquantile(draws, 1.0 - alpha / 2.0, axis=0)

    return Bands(x=grid, mean=center, lo=lo, hi=hi, level=level)


def residual_bootstrap_bands(
    x: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray], Dict],
    pred_fn: Callable[[Dict, np.ndarray], np.ndarray],
    grid: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
    random_state: Optional[int] = 42,
) -> Bands:
    """
    Fixed-design residual bootstrap bands for fitted smile curves.

    Strikes/moneyness are the design points for a displayed expiry slice.  The
    bootstrap therefore resamples centered residuals at the original x-values
    instead of resampling x/y pairs and changing the strike design.
    """
    level = normalize_confidence_level(level)
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)

    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 3:
        nan = np.full(grid.shape, np.nan, dtype=float)
        return Bands(x=grid, mean=nan.copy(), lo=nan.copy(), hi=nan.copy(), level=level)

    p0 = fit_fn(x_fit, y_fit)
    center = np.asarray(pred_fn(p0, grid), dtype=float)
    fitted_at_x = np.asarray(pred_fn(p0, x_fit), dtype=float)
    residuals = y_fit - fitted_at_x
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 2:
        return Bands(x=grid, mean=center, lo=center.copy(), hi=center.copy(), level=level)
    residuals = residuals - float(np.nanmean(residuals))

    draws = np.empty((n_boot, grid.size), dtype=float)
    for b in range(n_boot):
        yb = fitted_at_x + rng.choice(residuals, size=x_fit.size, replace=True)
        try:
            pb = fit_fn(x_fit, yb)
            draws[b] = pred_fn(pb, grid)
        except Exception:
            draws[b] = np.nan

    alpha = 1.0 - level
    lo = np.nanquantile(draws, alpha / 2.0, axis=0)
    hi = np.nanquantile(draws, 1.0 - alpha / 2.0, axis=0)
    return Bands(x=grid, mean=center, lo=lo, hi=hi, level=level)

# -----------------------------
# SVI helper bands (expects S,K,T)
# -----------------------------
def svi_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        out = fit_svi_slice(S, K_, T, iv_)
        return out

    def _pred(p, Kq):
        return svi_smile_iv(S, np.asarray(Kq, float), T, p)

    return residual_bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)

# -----------------------------
# SABR helper bands (expects S,K,T)
# -----------------------------
def sabr_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    beta: float = 0.5,
    level: float = 0.68,
    n_boot: int = 200,
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        out = fit_sabr_slice(S, K_, T, iv_, beta=beta)
        return out

    def _pred(p, Kq):
        return sabr_smile_iv(S, np.asarray(Kq, float), T, p)

    return residual_bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)

# -----------------------------
# TPS helper bands (expects S,K,T)
# -----------------------------
def tps_confidence_bands(
    S: float,
    K: np.ndarray,
    T: float,
    iv: np.ndarray,
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
) -> Bands:
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    grid_K = np.asarray(grid_K, float)

    def _fit(K_, iv_):
        return fit_tps_slice(S, K_, T, iv_)

    def _pred(p, Kq):
        return tps_smile_iv(S, np.asarray(Kq, float), T, p)

    return residual_bootstrap_bands(K, iv, _fit, _pred, grid_K, level=level, n_boot=n_boot)

# -----------------------------
# Term structure bootstrap helper
# -----------------------------
def _polynomial_fit_fn(x: np.ndarray, y: np.ndarray, degree: int = 2) -> dict:
    """Fit polynomial to term structure data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < degree + 1:
        degree = max(1, np.sum(mask) - 1)

    if np.sum(mask) < 2:
        return {"coeffs": [np.nanmean(y)], "degree": 0}

    try:
        coeffs = np.polyfit(x[mask], y[mask], degree)
        return {"coeffs": coeffs, "degree": degree}
    except Exception:
        try:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            return {"coeffs": coeffs, "degree": 1}
        except Exception:
            return {"coeffs": [np.nanmean(y)], "degree": 0}


def _polynomial_pred_fn(params: dict, x_grid: np.ndarray) -> np.ndarray:
    """Predict using polynomial fit."""
    return np.polyval(params["coeffs"], x_grid)


def generate_term_structure_confidence_bands(
    T: np.ndarray,
    atm_vol: np.ndarray,
    level: float = 0.68,
    n_boot: int = 100,
    fit_degree: int = 2,
    grid_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate confidence bands for ATM term structure using bootstrap."""
    T = np.asarray(T, dtype=float)
    atm_vol = np.asarray(atm_vol, dtype=float)

    mask = np.isfinite(T) & np.isfinite(atm_vol)
    if np.sum(mask) < 3:
        return np.array([]), np.array([]), np.array([])

    T_clean = T[mask]
    vol_clean = atm_vol[mask]

    T_min, T_max = T_clean.min(), T_clean.max()
    T_grid = np.linspace(T_min, T_max, grid_points)

    try:
        bands = bootstrap_bands(
            x=T_clean,
            y=vol_clean,
            fit_fn=lambda x, y: _polynomial_fit_fn(x, y, degree=fit_degree),
            pred_fn=_polynomial_pred_fn,
            grid=T_grid,
            level=level,
            n_boot=n_boot,
        )
        return T_grid, bands.lo, bands.hi
    except Exception:
        return np.array([]), np.array([]), np.array([])

# -----------------------------
# Peer-composite specific confidence bands
# -----------------------------
def peer_composite_confidence_bands(
    surfaces: Dict[str, np.ndarray],
    weights: Dict[str, float],
    grid_K: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
    weight_uncertainty: bool = True,
    surface_uncertainty: bool = True,
) -> Bands:
    """
    Create confidence bands for peer-composite surfaces considering both
    weight uncertainty and individual surface uncertainty.
    
    Parameters
    ----------
    surfaces : dict
        {ticker -> iv_array} where iv_array corresponds to grid_K
    weights : dict
        {ticker -> weight} for the peer-composite combination
    grid_K : np.ndarray
        Strike/moneyness grid for evaluation
    level : float
        Confidence level (0.68 ≈ 1-sigma, 0.95 ≈ 2-sigma)
    n_boot : int
        Number of bootstrap samples
    weight_uncertainty : bool
        Whether to include uncertainty in the weights
    surface_uncertainty : bool
        Whether to include uncertainty in individual surfaces
        
    Returns
    -------
    Bands
        Confidence bands for the peer-composite surface
    """
    level = normalize_confidence_level(level)
    grid_K = np.asarray(grid_K, float)
    tickers = list(surfaces.keys())
    n_points = len(grid_K)
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {t: 1.0/len(tickers) for t in tickers}
    else:
        weights = {t: w/total_weight for t, w in weights.items()}
    
    # Compute baseline synthetic surface
    baseline_synth = np.zeros(n_points)
    for ticker in tickers:
        if ticker in surfaces and ticker in weights:
            baseline_synth += weights[ticker] * surfaces[ticker]
    
    # Bootstrap sampling
    rng = np.random.default_rng(42)
    draws = np.empty((n_boot, n_points), dtype=float)
    
    for b in range(n_boot):
        synth_iv = np.zeros(n_points)
        
        # Resample weights if weight_uncertainty is True
        if weight_uncertainty:
            # Add noise to weights (Dirichlet-like resampling)
            weight_noise = rng.gamma(10, size=len(tickers))  # concentration parameter
            weight_noise = weight_noise / weight_noise.sum()
            boot_weights = {tickers[i]: weight_noise[i] for i in range(len(tickers))}
        else:
            boot_weights = weights.copy()
        
        # Combine surfaces with potential surface uncertainty
        for ticker in tickers:
            if ticker in surfaces and ticker in boot_weights:
                surface_iv = surfaces[ticker].copy()
                
                # Add surface uncertainty via residual resampling
                if surface_uncertainty:
                    # Simple residual bootstrap: add random noise
                    residuals = rng.normal(0, 0.01, size=n_points)  # 1% vol noise
                    surface_iv += residuals
                
                synth_iv += boot_weights[ticker] * surface_iv
        
        draws[b] = synth_iv
    
    # Compute quantiles
    alpha = 1.0 - level
    lo = np.nanquantile(draws, alpha / 2.0, axis=0)
    hi = np.nanquantile(draws, 1.0 - alpha / 2.0, axis=0)
    
    return Bands(x=grid_K, mean=baseline_synth, lo=lo, hi=hi, level=level)


def peer_composite_weight_bands(
    correlation_matrix: np.ndarray,
    target_idx: int,
    peer_indices: list,
    level: float = 0.68,
    n_boot: int = 200,
) -> Dict[int, Bands]:
    """
    Create confidence bands for peer-composite weights based on correlation uncertainty.
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix for all tickers
    target_idx : int
        Index of target ticker in correlation matrix
    peer_indices : list
        Indices of peer tickers for peer-composite
    level : float
        Confidence level
    n_boot : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        {peer_idx -> Bands} where Bands.mean contains the weight values
    """
    level = normalize_confidence_level(level)
    n_peers = len(peer_indices)
    rng = np.random.default_rng(42)
    
    # Extract relevant correlation submatrix
    all_indices = [target_idx] + peer_indices
    corr_sub = correlation_matrix[np.ix_(all_indices, all_indices)]
    
    # Baseline weights (correlation-based)
    target_corrs = corr_sub[0, 1:]  # correlations with target
    baseline_weights = np.abs(target_corrs)
    baseline_weights = baseline_weights / baseline_weights.sum()
    
    # Bootstrap correlation matrix
    weight_draws = np.empty((n_boot, n_peers), dtype=float)
    
    for b in range(n_boot):
        # Add noise to correlation matrix
        noise = rng.normal(0, 0.05, size=corr_sub.shape)  # 5% correlation noise
        noise = (noise + noise.T) / 2  # Keep symmetric
        np.fill_diagonal(noise, 0)  # Keep diagonal as 1
        
        boot_corr = corr_sub + noise
        # Ensure valid correlation matrix
        boot_corr = np.clip(boot_corr, -0.99, 0.99)
        np.fill_diagonal(boot_corr, 1.0)
        
        # Compute weights from bootstrapped correlations
        boot_target_corrs = boot_corr[0, 1:]
        boot_weights = np.abs(boot_target_corrs)
        if boot_weights.sum() > 0:
            boot_weights = boot_weights / boot_weights.sum()
        else:
            boot_weights = np.ones(n_peers) / n_peers
            
        weight_draws[b] = boot_weights
    
    # Create bands for each peer
    result = {}
    alpha = 1.0 - level
    
    for i, peer_idx in enumerate(peer_indices):
        weights_i = weight_draws[:, i]
        lo = np.nanquantile(weights_i, alpha / 2.0)
        hi = np.nanquantile(weights_i, 1.0 - alpha / 2.0)
        
        # Create single-point bands for weights
        result[peer_idx] = Bands(
            x=np.array([peer_idx]),
            mean=np.array([baseline_weights[i]]),
            lo=np.array([lo]),
            hi=np.array([hi]),
            level=level
        )
    
    return result


def peer_composite_pillar_bands(
    atm_data: Dict[str, np.ndarray],
    weights: Dict[str, float], 
    pillar_days: np.ndarray,
    level: float = 0.68,
    n_boot: int = 200,
) -> Bands:
    """
    Create confidence bands for peer-composite ATM pillar curves.
    
    Parameters
    ----------
    atm_data : dict
        {ticker -> atm_iv_array} where arrays correspond to pillar_days
    weights : dict
        {ticker -> weight} for peer-composite combination
    pillar_days : np.ndarray
        Pillar tenor points (in days)
    level : float
        Confidence level
    n_boot : int
        Number of bootstrap samples
        
    Returns
    -------
    Bands
        Confidence bands for peer-composite ATM curve
    """
    level = normalize_confidence_level(level)
    pillar_days = np.asarray(pillar_days, float)
    tickers = list(atm_data.keys())
    n_pillars = len(pillar_days)
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {t: 1.0/len(tickers) for t in tickers}
    else:
        weights = {t: w/total_weight for t, w in weights.items()}
    
    # Baseline synthetic ATM curve
    baseline_atm = np.zeros(n_pillars)
    for ticker in tickers:
        if ticker in atm_data and ticker in weights:
            baseline_atm += weights[ticker] * atm_data[ticker]
    
    # Bootstrap
    rng = np.random.default_rng(42)
    draws = np.empty((n_boot, n_pillars), dtype=float)
    
    for b in range(n_boot):
        # Resample weights with uncertainty
        weight_noise = rng.gamma(10, size=len(tickers))
        weight_noise = weight_noise / weight_noise.sum()
        boot_weights = {tickers[i]: weight_noise[i] for i in range(len(tickers))}
        
        # Combine ATM curves with noise
        synth_atm = np.zeros(n_pillars)
        for ticker in tickers:
            if ticker in atm_data and ticker in boot_weights:
                atm_curve = atm_data[ticker].copy()
                # Add ATM-specific noise
                residuals = rng.normal(0, 0.005, size=n_pillars)  # 0.5% vol noise
                atm_curve += residuals
                synth_atm += boot_weights[ticker] * atm_curve
        
        draws[b] = synth_atm
    
    # Compute quantiles
    alpha = 1.0 - level
    lo = np.nanquantile(draws, alpha / 2.0, axis=0)
    hi = np.nanquantile(draws, 1.0 - alpha / 2.0, axis=0)
    
    return Bands(x=pillar_days, mean=baseline_atm, lo=lo, hi=hi, level=level)
