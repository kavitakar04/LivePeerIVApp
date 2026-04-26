"""Shared model fit and quality helpers.

This module is the analysis-layer boundary for GUI/pipeline code that needs
model params plus quality metadata.  ``volModel`` remains responsible for the
actual numerical models and validators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from volModel.polyFit import fit_tps_slice, tps_smile_iv
from volModel.quality import validate_model_fit, warn_model_fallback
from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv
from volModel.sviFit import fit_svi_slice, svi_smile_iv


def fit_model_params(model: str, S: float, K: np.ndarray, T: float, IV: np.ndarray) -> dict:
    model = str(model).lower()
    if model == "svi":
        return fit_svi_slice(S, K, T, IV)
    if model == "sabr":
        return fit_sabr_slice(S, K, T, IV)
    if model == "tps":
        return fit_tps_slice(S, K, T, IV)
    raise ValueError(f"unsupported model: {model}")


def predict_model_iv(model: str, S: float, K: np.ndarray, T: float, params: dict) -> np.ndarray:
    model = str(model).lower()
    if model == "svi":
        return svi_smile_iv(S, K, T, params)
    if model == "sabr":
        return sabr_smile_iv(S, K, T, params)
    if model == "tps":
        return tps_smile_iv(S, K, T, params)
    raise ValueError(f"unsupported model: {model}")


def quality_to_dict(quality: Any) -> dict:
    return {
        "ok": bool(getattr(quality, "ok", False)),
        "reason": getattr(quality, "reason", "") or "",
        "rmse": getattr(quality, "rmse", np.nan),
        "min_iv": getattr(quality, "min_iv", np.nan),
        "max_iv": getattr(quality, "max_iv", np.nan),
        "n": getattr(quality, "n", 0),
    }


@dataclass(frozen=True)
class ModelFitResult:
    """Typed analysis-layer contract for a fitted volatility model."""

    requested_model: str
    actual_model: str
    params: dict
    quality: dict
    ok: bool
    degraded: bool
    reason: str
    rmse: float
    n: int

    @classmethod
    def from_quality(cls, model: str, params: dict, quality: Any) -> "ModelFitResult":
        quality_meta = quality_to_dict(quality)
        ok = bool(quality_meta["ok"])
        return cls(
            requested_model=str(model).lower(),
            actual_model=str(model).lower() if ok else "",
            params=params if ok else {},
            quality=quality_meta,
            ok=ok,
            degraded=not ok,
            reason=str(quality_meta.get("reason") or ""),
            rmse=float(quality_meta.get("rmse", np.nan)),
            n=int(quality_meta.get("n", 0) or 0),
        )

    def as_legacy_tuple(self) -> tuple[dict, dict]:
        return self.params, self.quality


def quality_checked_contract(
    model: str,
    params: dict,
    S: float,
    K: np.ndarray,
    T: float,
    IV: np.ndarray,
) -> ModelFitResult:
    model = str(model).lower()
    quality = validate_model_fit(
        model,
        params,
        lambda p: predict_model_iv(model, S, K, T, p),
        iv_obs=IV,
    )
    result = ModelFitResult.from_quality(model, params, quality)
    if not result.ok:
        warn_model_fallback(
            requested_model=model,
            failed_model=model,
            fallback_model="none",
            message=result.reason,
            quality=quality,
        )
    return result


def quality_checked_result(
    model: str,
    params: dict,
    S: float,
    K: np.ndarray,
    T: float,
    IV: np.ndarray,
) -> tuple[dict, dict]:
    return quality_checked_contract(model, params, S, K, T, IV).as_legacy_tuple()


def quality_checked_params(
    model: str,
    params: dict,
    S: float,
    K: np.ndarray,
    T: float,
    IV: np.ndarray,
) -> dict:
    checked, _quality = quality_checked_result(model, params, S, K, T, IV)
    return checked


def fit_valid_model_result(
    model: str,
    S: float,
    K: np.ndarray,
    T: float,
    IV: np.ndarray,
) -> tuple[dict, dict]:
    return fit_valid_model_contract(model, S, K, T, IV).as_legacy_tuple()


def fit_valid_model_contract(
    model: str,
    S: float,
    K: np.ndarray,
    T: float,
    IV: np.ndarray,
) -> ModelFitResult:
    return quality_checked_contract(model, fit_model_params(model, S, K, T, IV), S, K, T, IV)


def fit_valid_model_params(model: str, S: float, K: np.ndarray, T: float, IV: np.ndarray) -> dict:
    params, _quality = fit_valid_model_result(model, S, K, T, IV)
    return params
