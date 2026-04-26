# volModel/sabrFit.py
from __future__ import annotations
from typing import Optional, Dict, Iterable, Tuple
import math
import numpy as np
from functools import lru_cache

from .quality import validate_model_fit

# Optional SciPy; we fall back to a tiny Nelder–Mead if missing
try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============================================================
# Hagan (2002) lognormal SABR implied vol  = term1 * term2 * term3
#   term1: level factor   (alpha / (F K)^((1-β)/2)) / D(L)
#   term2: moneyness corr z/x(z)
#   term3: time correction 1 + T * [ ... ]
# Where L = ln(F/K), z = (ν/α) (F K)^((1-β)/2) L, and
# x(z) = ln( ( √(1 - 2ρ z + z^2) + z - ρ ) / (1 - ρ) )
# D(L) = 1 + ((1-β)^2/24) L^2 + ((1-β)^4/1920) L^4
# ============================================================

def _safe(val: float, lo: float = 1e-16) -> float:
    return float(val) if val > lo else float(lo)


def _signed_safe_den(val: float, lo: float = 1e-12) -> float:
    val = float(val)
    if abs(val) >= lo:
        return val
    return lo if val >= 0.0 else -lo


@lru_cache(maxsize=2048)
def _hagan_logn_terms_cached(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> Tuple[float, float, float, float, float]:
    """Internal cached core of Hagan's lognormal SABR approximation."""
    F = _safe(float(F))
    K = _safe(float(K))
    T = max(float(T), 1e-10)
    alpha = max(float(alpha), 1e-10)
    beta = float(np.clip(beta, 0.0, 1.0))
    rho = float(np.clip(rho, -0.999, 0.999))
    nu = max(float(nu), 1e-10)

    L = math.log(F / K)  # log-moneyness
    one_minus_b = 1.0 - beta
    FK_pow = (F * K) ** (0.5 * one_minus_b)
    L2 = L * L
    L4 = L2 * L2
    D = 1.0 + (one_minus_b**2 / 24.0) * L2 + (one_minus_b**4 / 1920.0) * L4

    term1 = (alpha / (_safe(FK_pow))) / D

    if abs(L) < 1e-14:
        zx = 1.0
    else:
        z = (nu / alpha) * FK_pow * L
        sqrt_arg = max(1.0 - 2.0 * rho * z + z * z, 1e-16)
        num = math.sqrt(sqrt_arg) + z - rho
        den = 1.0 - rho
        ratio = num / _safe(den)
        if ratio <= 0.0 or not math.isfinite(ratio):
            return (term1, float("nan"), 1.0, float("nan"), float(L))
        xz = math.log(ratio)
        if abs(z) < 1e-8:
            zx = 1.0 - 0.5 * rho * z
        else:
            zx = z / _signed_safe_den(xz)

    term2 = zx
    A = (one_minus_b**2 / 24.0) * (alpha * alpha) / (_safe((F * K) ** (one_minus_b)))
    B = (rho * beta * nu * alpha) / (4.0 * _safe(FK_pow))
    C = ((2.0 - 3.0 * rho * rho) / 24.0) * (nu * nu)
    term3 = 1.0 + T * (A + B + C)

    iv = term1 * term2 * term3
    if not math.isfinite(iv) or iv <= 0.0:
        iv = float("nan")
    return (
        float(term1),
        float(term2),
        float(term3),
        float(iv),
        float(L),
    )


def hagan_logn_terms(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> Dict[str, float]:
    """Return the three multiplicative components (term1, term2, term3) + iv.

    A small ``lru_cache`` accelerates repeated evaluations for identical
    parameters which show up during grid searches and plotting.  The cached
    core returns plain tuples to avoid accidental mutation of cached results;
    this thin wrapper materialises them as a dictionary for callers."""
    term1, term2, term3, iv, L = _hagan_logn_terms_cached(
        float(F), float(K), float(T), float(alpha), float(beta), float(rho), float(nu)
    )
    return {
        "term1": term1,
        "term2": term2,
        "term3": term3,
        "iv": iv,
        "L": L,
    }


@lru_cache(maxsize=4096)
def hagan_logn_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Convenience: only implied vol (product of terms).

    The function itself is cached as SABR evaluations are often repeated with
    identical arguments when building smiles or during optimisation."""
    return _hagan_logn_terms_cached(
        float(F), float(K), float(T), float(alpha), float(beta), float(rho), float(nu)
    )[3]


def sabr_smile_iv(
    S: float,
    K: np.ndarray,
    T: float,
    params: Dict[str, float],
) -> np.ndarray:
    """Vectorized SABR smile for one expiry slice."""
    F = _safe(float(S))   # use forward if available; spot as proxy is ok for equities
    alpha = float(params["alpha"]); beta = float(params["beta"])
    rho = float(params["rho"]); nu = float(params["nu"])
    K = np.asarray(K, dtype=float)
    out = np.empty_like(K, dtype=float)
    for i, k in enumerate(K):
        out[i] = hagan_logn_vol(F, float(k), float(T), alpha, beta, rho, nu)
    return out


# ============================================================
# Calibration (single slice): fit alpha, rho, nu (β fixed)
# ============================================================

def _nelder_mead(func, x0, maxiter=2000, tol=1e-10):
    """Tiny Nelder–Mead fallback (very basic)."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    # initial simplex
    simplex = [x0]
    for i in range(n):
        y = x0.copy()
        y[i] = y[i] + (0.05 if x0[i] == 0 else 0.05 * abs(x0[i]))
        simplex.append(y)
    simplex = np.array(simplex)
    fvals = np.array([func(s) for s in simplex])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < maxiter:
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        if np.std(fvals) < tol:
            break

        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second = simplex[-2]
        centroid = np.mean(simplex[:-1], axis=0)

        # reflect
        xr = centroid + alpha * (centroid - x_worst)
        fr = func(xr)

        if fr < fvals[0]:
            # expand
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            # contract
            xc = centroid + rho * (x_worst - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                # shrink
                for i in range(1, len(simplex)):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    fvals[i] = func(simplex[i])
        it += 1
    return {"x": simplex[0], "fun": fvals[0], "nit": it}


def fit_sabr_slice(
    S: float,
    K: np.ndarray,
    T: float,
    iv_obs: np.ndarray,
    beta: float = 0.5,
    x0: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    l2_reg: float = 0.0,          # tiny ridge on params for stability (0 = off)
    vega_weights: Optional[np.ndarray] = None,  # optional extra weighting
) -> Dict[str, float]:
    """
    Calibrate SABR (lognormal) to a single expiry slice.
    We treat F≈S; swap in forward if you have it.
    Fit params: alpha>0, -0.999<rho<0.999, nu>0  (β fixed).
    """
    K = np.asarray(K, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)
    mask = np.isfinite(K) & np.isfinite(iv_obs)
    K = K[mask]; iv_obs = iv_obs[mask]
    n = len(K)
    if n < 3:
        return {"alpha": np.nan, "beta": float(beta), "rho": np.nan, "nu": np.nan,
                "rmse": np.nan, "n": int(n)}

    F = _safe(float(S))
    T = max(float(T), 1e-10)
    beta = float(np.clip(beta, 0.0, 1.0))

    # initial guess
    atm_iv = float(np.median(iv_obs))
    alpha0 = max(atm_iv * (F ** (1 - beta)), 1e-4)
    rho0 = 0.0
    nu0 = 0.5
    x0 = np.array([alpha0, rho0, nu0], dtype=float) if x0 is None else np.asarray(x0, dtype=float)

    # bounds & clip helper
    lb = np.array([1e-6, -0.999, 1e-6], dtype=float)
    ub = np.array([5.0,    0.999, 5.0  ], dtype=float)
    def _clip(p): return np.minimum(np.maximum(p, lb), ub)

    # combined weights (user + vega)
    W = None
    if weights is not None:
        W = np.asarray(weights, dtype=float)[mask]
    if vega_weights is not None:
        vw = np.asarray(vega_weights, dtype=float)[mask]
        if W is None:
            W = vw
        else:
            W = W * vw
    if W is not None:
        W = np.where(np.isfinite(W) & (W > 0), W, np.nan)
        if np.isfinite(W).sum() >= 3:
            W = W / (np.nansum(W) + 1e-12)
        else:
            W = None

    def obj(p):
        a, r, n_ = _clip(np.asarray(p, dtype=float))
        iv_fit = np.array([hagan_logn_vol(F, float(k), T, a, beta, r, n_) for k in K], dtype=float)
        if not np.isfinite(iv_fit).all() or np.any(iv_fit <= 0):
            return 1e12
        err = iv_fit - iv_obs
        if W is not None:
            se = np.nansum(W * (err * err))
        else:
            se = float(np.nanmean(err * err))
        if l2_reg > 0.0:
            se += float(l2_reg) * (a*a + r*r + n_*n_)
        return se

    if _HAVE_SCIPY:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-12})
        p = _clip(res.x); rmse = math.sqrt(max(res.fun, 0.0))
    else:
        res = _nelder_mead(obj, x0, maxiter=3000, tol=1e-12)
        p = _clip(res["x"]); rmse = math.sqrt(max(res["fun"], 0.0))

    alpha, rho, nu = [float(v) for v in p]
    out = {
        "alpha": alpha,
        "beta": float(beta),
        "rho": rho,
        "nu": nu,
        "rmse": rmse,
        "n": int(n),
    }
    quality = validate_model_fit(
        "sabr",
        out,
        lambda p_: sabr_smile_iv(F, K, T, p_),
        iv_obs=iv_obs,
    )
    out["quality_ok"] = bool(quality.ok)
    if not quality.ok:
        out["quality_reason"] = quality.reason
    return out


# ============================================================
# Diagnostics helpers
# ============================================================

def sabr_slice_terms_table(
    S: float,
    K: Iterable[float],
    T: float,
    params: Dict[str, float],
) -> np.ndarray:
    """
    Return a table [len(K) x 5]: [K, term1, term2, term3, iv] for diagnostics/plots.
    """
    F = _safe(float(S))
    alpha = float(params["alpha"]); beta = float(params["beta"])
    rho = float(params["rho"]); nu = float(params["nu"])
    K = np.asarray(K, dtype=float)
    out = np.zeros((len(K), 5), dtype=float)
    for i, k in enumerate(K):
        t = hagan_logn_terms(F, float(k), float(T), alpha, beta, rho, nu)
        out[i] = [k, t["term1"], t["term2"], t["term3"], t["iv"]]
    return out


# Back-compat aliases (some of your plotting code imports these names)
fit_sabr_smile = fit_sabr_slice
hagan_lognormal_vol = hagan_logn_vol
