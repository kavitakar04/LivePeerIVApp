# volModel/sviFit.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple
import numpy as np
import math

# Optional SciPy; fall back to a tiny Nelder–Mead if missing
try:
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============================================================
# Raw-SVI total variance:
#   w(k) = a + b * [ rho*(k - m) + sqrt( (k - m)^2 + sigma^2 ) ]
# Implied vol:  iv(k) = sqrt( w(k) / T )
# ============================================================


def _safe_pos(x: float, lo: float = 1e-12) -> float:
    return float(x) if x > lo else float(lo)


def svi_total_variance_raw(
    k: np.ndarray | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray | float:
    """Raw-SVI total variance w(k)."""
    k = np.asarray(k, dtype=float)
    x = k - float(m)
    b = float(max(b, 1e-12))
    sigma = float(max(sigma, 1e-12))
    rho = float(np.clip(rho, -0.999, 0.999))
    a = float(max(a, 1e-12))
    w = a + b * (rho * x + np.sqrt(x * x + sigma * sigma))
    return np.asarray(w, dtype=float) if w.shape != () else float(w)


def svi_implied_vol(
    k: np.ndarray | float,
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray | float:
    """Black–Scholes implied vol from raw-SVI."""
    T = _safe_pos(float(T))
    w = svi_total_variance_raw(k, a, b, rho, m, sigma)
    return np.sqrt(np.asarray(w, dtype=float) / T)


# ============================================================
# Derivatives and decomposition (for diagnostics)
# ============================================================


def svi_w_prime_w_dprime(
    k: np.ndarray | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return w'(k) and w''(k) for raw-SVI."""
    k = np.asarray(k, dtype=float)
    x = k - float(m)
    b = float(max(b, 1e-12))
    sigma = float(max(sigma, 1e-12))
    denom = np.sqrt(x * x + sigma * sigma)
    w1 = b * (float(rho) + x / denom)
    w2 = b * (sigma * sigma) / np.power(denom, 3)
    return w1, w2


def svi_decompose_table(
    k: Iterable[float],
    params: Dict[str, float],
    T: float,
) -> np.ndarray:
    """
    Return array [len(k) x 6]: [k, base_a, linear, sqrt_term, w, iv].
    This decomposes w(k) into: a + (b*rho*(k-m)) + (b*sqrt((k-m)^2 + sigma^2)).
    """
    a = float(params["a"])
    b = float(params["b"])
    rho = float(params["rho"])
    m = float(params["m"])
    sigma = float(params["sigma"])
    k = np.asarray(k, dtype=float)
    x = k - m
    base = np.full_like(k, a, dtype=float)
    linear = b * rho * x
    sqrt_t = b * np.sqrt(x * x + sigma * sigma)
    w = base + linear + sqrt_t
    iv = np.sqrt(np.clip(w, 1e-12, None) / _safe_pos(T))
    return np.column_stack([k, base, linear, sqrt_t, w, iv])


def svi_iv_and_derivs_at_k(
    k0: float,
    T: float,
    params: Dict[str, float],
) -> Dict[str, float]:
    """
    Give iv(k0), iv'(k0), iv''(k0) using chain rule:
      iv = sqrt(w/T),  iv' = w'/(2 sqrt(w T)),  iv'' = w''/(2 sqrt(w T)) - (w'^2)/(4 (w T)^(3/2))
    """
    a, b, rho, m, sigma = (params[k] for k in ("a", "b", "rho", "m", "sigma"))
    T = _safe_pos(T)
    w = svi_total_variance_raw(k0, a, b, rho, m, sigma)
    w1, w2 = svi_w_prime_w_dprime(k0, a, b, rho, m, sigma)
    root = math.sqrt(_safe_pos(w * T))
    iv = math.sqrt(_safe_pos(w) / T)
    iv1 = float(w1) / (2.0 * root)
    iv2 = float(w2) / (2.0 * root) - (float(w1) ** 2) / (4.0 * root**3)
    return {"iv": float(iv), "iv_prime": float(iv1), "iv_second": float(iv2)}


# ============================================================
# Calibration (single expiry): fit a,b,rho,m,sigma
# ============================================================


def _nelder_mead(func, x0, maxiter=3000, tol=1e-12):
    """Tiny Nelder–Mead fallback."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        y = x0.copy()
        y[i] = y[i] + (0.05 if x0[i] == 0 else 0.05 * abs(x0[i]))
        simplex.append(y)
    simplex = np.array(simplex)
    fvals = np.array([func(s) for s in simplex])
    alpha, gamma, rho_c, sigma_c = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < maxiter:
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if np.std(fvals) < tol:
            break
        x_best, x_worst = simplex[0], simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - x_worst)
        fr = func(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            simplex[-1] = xe if fe < fr else xr
            fvals[-1] = min(fe, fr)
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + rho_c * (x_worst - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = x_best + sigma_c * (simplex[i] - x_best)
                    fvals[i] = func(simplex[i])
        it += 1
    return {"x": simplex[0], "fun": fvals[0], "nit": it}


def _clip_params(p: np.ndarray) -> np.ndarray:
    """Box constraints for raw-SVI parameters [a, b, rho, m, sigma]."""
    lb = np.array([1e-12, 1e-8, -0.999, -2.0, 1e-8], dtype=float)
    ub = np.array([5.0, 5.0, 0.999, 2.0, 5.0], dtype=float)
    p = np.asarray(p, dtype=float)
    return np.minimum(np.maximum(p, lb), ub)


def fit_svi_slice(S, K, T, iv_obs, x0=None):
    """
    Fit raw-parameter SVI to a single expiry slice.

    Parameters
    ----------
    S : float               spot (use forward if you have it)
    K : array-like          strikes
    T : float               time to expiry (years)
    iv_obs : array-like     observed implied vols for those strikes
    x0 : optional initial guess [a,b,rho,m,sigma]

    Returns dict: {a,b,rho,m,sigma,rmse,n}
    """
    K = np.asarray(K, dtype=float).reshape(-1)
    iv_obs = np.asarray(iv_obs, dtype=float).reshape(-1)

    mask = np.isfinite(K) & np.isfinite(iv_obs)
    K = K[mask]
    iv_obs = iv_obs[mask]
    if K.size < 3:
        return {"a": np.nan, "b": np.nan, "rho": np.nan, "m": np.nan, "sigma": np.nan, "rmse": np.nan, "n": int(K.size)}

    S = float(max(S, 1e-12))
    T = float(max(T, 1e-10))
    k = np.log(np.clip(K, 1e-12, None) / S)

    # --- initial guess (robust)
    atm = float(np.nanmedian(iv_obs))
    a0 = max((atm**2) * T * 0.5, 1e-8)
    b0 = 0.3
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.2
    p0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    p0 = _clip_params(p0)

    def obj(p):
        a, b, rho, m, sigma = _clip_params(p)
        x = k - m
        w = a + b * (rho * x + np.sqrt(x * x + sigma * sigma))
        iv_fit = np.sqrt(np.clip(w, 1e-12, None) / T)
        err = iv_fit - iv_obs
        return float(np.mean(err * err))

    # try SciPy if present, otherwise custom Nelder–Mead
    try:
        from scipy.optimize import minimize

        res = minimize(obj, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-9})
        p = _clip_params(res.x)
        rmse = float(math.sqrt(max(res.fun, 0.0)))
    except Exception:
        res = _nelder_mead(obj, p0, maxiter=4000, tol=1e-9)
        p = _clip_params(res["x"])
        rmse = float(math.sqrt(max(res["fun"], 0.0)))

    a, b, rho, m, sigma = [float(x) for x in p]
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "rmse": rmse, "n": int(K.size)}


def fit_svi_slice_from_moneyness(mny, T, iv_obs, x0=None):
    """If you already have moneyness M=K/S, use this."""
    mny = np.asarray(mny, dtype=float).reshape(-1)
    k = np.log(np.clip(mny, 1e-12, None))
    iv_obs = np.asarray(iv_obs, dtype=float).reshape(-1)
    mask = np.isfinite(k) & np.isfinite(iv_obs)
    k = k[mask]
    iv_obs = iv_obs[mask]
    if k.size < 3:
        return {"a": np.nan, "b": np.nan, "rho": np.nan, "m": np.nan, "sigma": np.nan, "rmse": np.nan, "n": int(k.size)}

    T = float(max(T, 1e-10))
    atm = float(np.nanmedian(iv_obs))
    a0 = max((atm**2) * T * 0.5, 1e-8)
    b0 = 0.3
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.2
    p0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    p0 = _clip_params(p0)

    def obj(p):
        a, b, rho, m, sigma = _clip_params(p)
        x = k - m
        w = a + b * (rho * x + np.sqrt(x * x + sigma * sigma))
        iv_fit = np.sqrt(np.clip(w, 1e-12, None) / T)
        err = iv_fit - iv_obs
        return float(np.mean(err * err))

    try:
        from scipy.optimize import minimize

        res = minimize(obj, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-9})
        p = _clip_params(res.x)
        rmse = float(math.sqrt(max(res.fun, 0.0)))
    except Exception:
        res = _nelder_mead(obj, p0, maxiter=4000, tol=1e-9)
        p = _clip_params(res["x"])
        rmse = float(math.sqrt(max(res["fun"], 0.0)))

    a, b, rho, m, sigma = [float(x) for x in p]
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "rmse": rmse, "n": int(k.size)}


def svi_smile_iv(S, K, T, params):
    """
    Vectorized: IV(K) for a single expiry using SVI params.
    params: dict {a,b,rho,m,sigma} or iterable (a,b,rho,m,sigma)
    """
    if isinstance(params, dict):
        a = float(params["a"])
        b = float(params["b"])
        rho = float(params["rho"])
        m = float(params["m"])
        sigma = float(params["sigma"])
    else:
        a, b, rho, m, sigma = [float(x) for x in params]

    S = float(max(S, 1e-12))
    K = np.asarray(K, dtype=float)
    T = float(max(T, 1e-10))

    k = np.log(np.clip(K, 1e-12, None) / S)
    x = k - m
    w = a + b * (rho * x + np.sqrt(x * x + sigma * sigma))
    return np.sqrt(np.clip(w, 1e-12, None) / T)


# ============================================================
# Convenience & compatibility aliases
# ============================================================

# Your plotting code expects these names:
fit_svi_smile = fit_svi_slice
svi_total_variance = svi_total_variance_raw
