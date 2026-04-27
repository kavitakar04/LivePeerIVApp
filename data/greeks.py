"""Black–Scholes prices and Greeks (no external deps).
- Works for calls and puts with risk-free rate r and dividend/borrow yield q.
- Provides: price, delta, gamma, vega, theta, rho.
- Includes safe guards for edge cases (very small T or sigma).

All functions accept scalar floats and return floats. For vectorized usage on
DataFrames, see `compute_all_greeks_df` at the bottom.
"""

from __future__ import annotations
import math
from typing import Literal, Dict

from .interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD, get_ticker_interest_rate

OptionCP = Literal["C", "P"]

# ----------------------
# Normal PDF / CDF (no SciPy)
# ----------------------
SQRT2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def norm_cdf(x: float) -> float:
    # 0.5 * [1 + erf(x/sqrt(2))]
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ----------------------
# Core BS helpers
# ----------------------


def _safe_positive(x: float, eps: float = 1e-12) -> float:
    return x if x > eps else eps


def bs_d1_d2(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> tuple[float, float]:
    """Compute Black–Scholes d1 and d2 with safety for tiny values."""
    S = _safe_positive(S)
    K = _safe_positive(K)
    T = _safe_positive(T)
    sigma = _safe_positive(sigma)

    vol_sqrt_t = sigma * math.sqrt(T)
    m = math.log(S / K)
    d1 = (m + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


# ----------------------
# Prices
# ----------------------


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    cp: OptionCP = "C",
) -> float:
    """Black–Scholes price for call/put with continuous rates r and dividend yield q."""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)

    if cp == "C":
        return df_q * S * norm_cdf(d1) - df_r * K * norm_cdf(d2)
    else:  # Put via parity
        return df_r * K * norm_cdf(-d2) - df_q * S * norm_cdf(-d1)


# ----------------------
# Greeks (spot-based, not forward)
# ----------------------


def bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    cp: OptionCP = "C",
) -> float:
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    df_q = math.exp(-q * T)
    return df_q * norm_cdf(d1) if cp == "C" else df_q * (norm_cdf(d1) - 1.0)


def bs_gamma(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> float:
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    df_q = math.exp(-q * T)
    return df_q * norm_pdf(d1) / (_safe_positive(S) * _safe_positive(sigma) * math.sqrt(_safe_positive(T)))


def bs_vega(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> float:
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    df_q = math.exp(-q * T)
    # Vega per 1 vol unit (not %). Multiply by 0.01 if you want per vol point (1%)
    return df_q * _safe_positive(S) * norm_pdf(d1) * math.sqrt(_safe_positive(T))


def bs_theta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    cp: OptionCP = "C",
) -> float:
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    term1 = -(S * df_q * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(_safe_positive(T)))
    if cp == "C":
        return term1 - r * K * df_r * norm_cdf(d2) + q * S * df_q * norm_cdf(d1)
    else:
        return term1 + r * K * df_r * norm_cdf(-d2) - q * S * df_q * norm_cdf(-d1)


def bs_rho(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    cp: OptionCP = "C",
) -> float:
    _, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    df_r = math.exp(-r * T)
    if cp == "C":
        return _safe_positive(T) * K * df_r * norm_cdf(d2)
    else:
        return -_safe_positive(T) * K * df_r * norm_cdf(-d2)


# ----------------------
# Convenience: compute all at once
# ----------------------


def compute_all_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    cp: OptionCP = "C",
) -> Dict[str, float]:
    """Return price and Greeks in a dict for easy assignment/DB insert."""
    try:
        d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
        price = bs_price(S, K, T, sigma, r, q, cp)
        delta = bs_delta(S, K, T, sigma, r, q, cp)
        gamma = bs_gamma(S, K, T, sigma, r, q)
        vega = bs_vega(S, K, T, sigma, r, q)
        theta = bs_theta(S, K, T, sigma, r, q, cp)
        rho = bs_rho(S, K, T, sigma, r, q, cp)
        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
            "d1": float(d1),
            "d2": float(d2),
        }
    except Exception:
        return {
            "price": float("nan"),
            "delta": float("nan"),
            "gamma": float("nan"),
            "vega": float("nan"),
            "theta": float("nan"),
            "rho": float("nan"),
            "d1": float("nan"),
            "d2": float("nan"),
        }


# ----------------------
# Vectorized helper for pandas DataFrames
# ----------------------


def compute_all_greeks_df(
    df,
    r: float = None,  # Changed to None to allow ticker-specific rates
    q: float = STANDARD_DIVIDEND_YIELD,
    use_ticker_rates: bool = True,  # New parameter
    rate_date: str = None,  # Optional specific date for rates
):
    """Given a DataFrame with columns S, K, T, sigma, call_put -> add price & Greeks.

    If use_ticker_rates=True and 'ticker' column exists, will use ticker-specific rates.
    Otherwise falls back to the provided r or STANDARD_RISK_FREE_RATE.

    Modifies a copy and returns it (does not mutate in-place).
    """
    import pandas as pd

    out = df.copy()

    # Ensure required columns exist
    required = ["S", "K", "T", "sigma", "call_put"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Determine if we can use ticker-specific rates
    has_ticker = "ticker" in out.columns

    def row_fn(rw):
        # Determine the interest rate to use
        if use_ticker_rates and has_ticker:
            ticker = str(rw.get("ticker", ""))
            if ticker and ticker != "nan":
                # Convert percentage to decimal (ML rates are in percentage form)
                ticker_rate = get_ticker_interest_rate(ticker, rate_date)
                if ticker_rate is not None:
                    # ML rates are in percentage form, convert to decimal
                    effective_r = ticker_rate / 100.0
                else:
                    effective_r = r if r is not None else STANDARD_RISK_FREE_RATE
            else:
                effective_r = r if r is not None else STANDARD_RISK_FREE_RATE
        else:
            effective_r = r if r is not None else STANDARD_RISK_FREE_RATE

        g = compute_all_greeks(
            S=float(rw["S"]),
            K=float(rw["K"]),
            T=float(rw["T"]),
            sigma=float(rw["sigma"]),
            r=effective_r,
            q=q,
            cp=str(rw["call_put"]).upper(),
        )
        # Add the actual rate used to the output
        g["r"] = effective_r
        return pd.Series(g)

    greeks_df = out.apply(row_fn, axis=1)
    for col in greeks_df.columns:
        out[col] = greeks_df[col]
    return out
