# volModel/volModel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal
import numpy as np

try:
    import matplotlib.pyplot as plt  # for plot()

    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

from .sviFit import fit_svi_slice, svi_smile_iv
from .sabrFit import fit_sabr_slice, sabr_smile_iv
from .polyFit import fit_poly
from .models import SUPPORTED_MODELS

ModelName = Literal["svi", "sabr", "tps", "poly"]


@dataclass
class SliceParams:
    T: float
    n: int
    rmse: float
    params: Dict[str, float]


class VolModel:
    """
    Fit a per-day, per-ticker volatility smile model across expiries.
    - fit(model='svi'|'sabr'|'tps'|'poly'): calibrates each expiry slice independently
    - predict_iv(K, T): get IV at a (K, T)
    - smile(Ks, T): full smile at a given expiry
    - plot(T): quick visualization (if matplotlib available)
    """

    def __init__(self, model: ModelName = "svi", poly_method: str = "tps"):
        model = model.lower()
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported volatility model: {model}")
        self.model: ModelName = model
        self.poly_method: str = poly_method  # for polynomial fit: 'simple' or 'tps'
        self.S: Optional[float] = None
        self.slices: Dict[float, SliceParams] = {}  # keyed by T (years)
        self.beta_fixed: float = 0.5  # for SABR if used

    def fit(
        self,
        S: float,
        Ks: np.ndarray,
        Ts: np.ndarray,
        IVs: np.ndarray,
        weights: Optional[np.ndarray] = None,
        beta: float = 0.5,
    ) -> "VolModel":
        """
        Inputs are vectors (same length N):
            S: spot scalar
            Ks[N], Ts[N], IVs[N]
        Groups by unique T and fits per-slice.
        """
        self.model = self.model.lower()  # normalize
        self.S = float(S)
        self.slices.clear()
        self.beta_fixed = float(beta)

        # group by T
        arr = np.column_stack([Ks, Ts, IVs])
        # keep finite only
        finite_mask = np.isfinite(arr).all(axis=1) & np.isfinite(S)
        Ks = np.asarray(Ks)[finite_mask]
        Ts = np.asarray(Ts)[finite_mask]
        IVs = np.asarray(IVs)[finite_mask]
        W = np.asarray(weights)[finite_mask] if weights is not None else None

        if len(Ks) < 3:
            return self

        for T in np.unique(np.round(Ts, 8)):  # stable group key
            m = np.isclose(Ts, T)
            K_slice, iv_slice = Ks[m], IVs[m]
            if len(K_slice) < 3:
                continue
            if self.model == "svi":
                out = fit_svi_slice(S, K_slice, float(T), iv_slice)
            elif self.model == "sabr":
                out = fit_sabr_slice(S, K_slice, float(T), iv_slice, beta=self.beta_fixed)
            elif self.model in ("tps", "poly"):
                k_slice = np.log(np.clip(K_slice, 1e-12, None) / self.S)
                W_slice = W[m] if W is not None else None
                out = fit_poly(k_slice, iv_slice, weights=W_slice, method=self.poly_method)
            else:
                raise ValueError(f"unsupported volatility model: {self.model}")
            self.slices[float(T)] = SliceParams(
                T=float(T), n=int(out.get("n", len(K_slice))), rmse=float(out.get("rmse", np.nan)), params=out
            )
        return self

    def available_expiries(self):
        return sorted(self.slices.keys())

    def _poly_iv(self, Ks: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate polynomial/TPS slice."""
        Ks = np.asarray(Ks, dtype=float)
        k = np.log(np.clip(Ks, 1e-12, None) / float(self.S))
        if params.get("model") == "tps" and "interpolator" in params:
            return np.asarray(params["interpolator"](k), dtype=float)
        a = params.get("atm_vol", np.nan)
        b = params.get("skew", 0.0)
        c2 = params.get("curv", 0.0) / 2.0
        return a + b * k + c2 * k * k

    def predict_iv(self, K: float, T: float) -> float:
        """Evaluate IV at (K, T) using nearest fitted slice in T if exact T not present."""
        if not self.slices or self.S is None:
            return float("nan")
        # pick nearest T
        Ts = np.array(self.available_expiries(), dtype=float)
        Tq = float(T)
        Tn = float(Ts[np.argmin(np.abs(Ts - Tq))])
        p = self.slices[Tn].params
        if self.model == "svi":
            return float(svi_smile_iv(self.S, np.array([K]), Tn, p)[0])
        if self.model == "sabr":
            return float(sabr_smile_iv(self.S, np.array([K]), Tn, p)[0])
        return float(self._poly_iv(np.array([K], dtype=float), p)[0])

    def smile(self, Ks: np.ndarray, T: float) -> np.ndarray:
        """Vectorized IV smile at nearest fitted expiry."""
        if not self.slices or self.S is None:
            return np.full_like(np.asarray(Ks, dtype=float), np.nan, dtype=float)
        Ts = np.array(self.available_expiries(), dtype=float)
        Tq = float(T)
        Tn = float(Ts[np.argmin(np.abs(Ts - Tq))])
        p = self.slices[Tn].params
        if self.model == "svi":
            return svi_smile_iv(self.S, np.asarray(Ks, dtype=float), Tn, p)
        if self.model == "sabr":
            return sabr_smile_iv(self.S, np.asarray(Ks, dtype=float), Tn, p)
        return self._poly_iv(np.asarray(Ks, dtype=float), p)

    def plot(self, T: float, Ks: Optional[np.ndarray] = None) -> None:
        """Quick plot of the fitted smile at expiry nearest to T."""
        if not _HAVE_MPL:
            print("matplotlib not available; cannot plot.")
            return
        if not self.slices or self.S is None:
            print("No fitted slices.")
            return
        Ts = np.array(self.available_expiries(), dtype=float)
        Tn = float(Ts[np.argmin(np.abs(Ts - float(T)))])
        self.slices[Tn].params
        if Ks is None:
            # build a nice moneyness grid around ATM
            m = np.linspace(0.6, 1.4, 81)  # K/S range
            Ks = m * self.S
        iv = self.smile(np.asarray(Ks, dtype=float), Tn)

        plt.figure(figsize=(6, 4))
        plt.plot(Ks / self.S, iv, lw=2, label=f"{self.model.upper()} T≈{Tn:.3f}y")
        plt.axvline(1.0, ls="--", lw=1, color="grey")
        plt.xlabel("Moneyness K/S")
        plt.ylabel("Implied Vol")
        plt.title(f"Fitted smile ({self.model.upper()})")
        plt.legend()
        plt.tight_layout()
        plt.show()
