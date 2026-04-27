from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
_FALLBACK_WARNING_KEYS: set[tuple[str, str, str, str, int, str, str, str]] = set()

DEFAULT_MAX_RMSE = 0.25
DEFAULT_MAX_IV = 5.0
DEFAULT_MAX_ABS_SKEW = 10.0
DEFAULT_MAX_ABS_CURV = 250.0
DEFAULT_MIN_OBS = 3


@dataclass(frozen=True)
class ModelQuality:
    ok: bool
    reason: str = ""
    rmse: float = float("nan")
    min_iv: float = float("nan")
    max_iv: float = float("nan")
    n: int = 0


def validate_model_fit(
    model: str,
    params: Optional[dict[str, Any]],
    pred_fn: Callable[[dict[str, Any]], np.ndarray],
    *,
    iv_obs: Optional[np.ndarray] = None,
    min_obs: int = DEFAULT_MIN_OBS,
    max_rmse: float = DEFAULT_MAX_RMSE,
    max_iv: float = DEFAULT_MAX_IV,
    max_abs_skew: float = DEFAULT_MAX_ABS_SKEW,
    max_abs_curv: float = DEFAULT_MAX_ABS_CURV,
) -> ModelQuality:
    if not params:
        return ModelQuality(False, "missing params")

    n = int(params.get("n", 0) or 0)
    if iv_obs is not None:
        iv_obs = np.asarray(iv_obs, dtype=float)
        n = max(n, int(np.isfinite(iv_obs).sum()))
    if n < int(min_obs):
        return ModelQuality(False, f"not enough observations: {n}", n=n)

    try:
        pred = np.asarray(pred_fn(params), dtype=float)
    except Exception as exc:
        return ModelQuality(False, f"prediction failed: {exc}", n=n)

    if pred.size == 0:
        return ModelQuality(False, "empty prediction", n=n)
    if not np.isfinite(pred).all():
        return ModelQuality(False, "non-finite IV prediction", n=n)
    if np.any(pred <= 0):
        return ModelQuality(
            False,
            "non-positive IV prediction",
            min_iv=float(np.nanmin(pred)),
            max_iv=float(np.nanmax(pred)),
            n=n,
        )
    pred_min = float(np.nanmin(pred))
    pred_max = float(np.nanmax(pred))
    if pred_max > float(max_iv):
        return ModelQuality(
            False,
            f"IV prediction too high: {pred_max:.6g}",
            min_iv=pred_min,
            max_iv=pred_max,
            n=n,
        )

    rmse = params.get("rmse", np.nan)
    try:
        rmse = float(rmse)
    except Exception:
        rmse = float("nan")
    if not np.isfinite(rmse) and iv_obs is not None and pred.shape == iv_obs.shape:
        rmse = float(np.sqrt(np.nanmean((pred - iv_obs) ** 2)))
    if not np.isfinite(rmse):
        return ModelQuality(False, "non-finite RMSE", n=n)
    if rmse > float(max_rmse):
        return ModelQuality(False, f"RMSE too high: {rmse:.6g}", rmse=rmse, n=n)

    for key, limit in (("skew", max_abs_skew), ("curv", max_abs_curv)):
        if key in params:
            try:
                value = float(params[key])
            except Exception:
                return ModelQuality(False, f"non-numeric {key}", rmse=rmse, n=n)
            if not np.isfinite(value) or abs(value) > float(limit):
                return ModelQuality(False, f"abs({key}) too high: {value:.6g}", rmse=rmse, n=n)

    return ModelQuality(
        True,
        rmse=rmse,
        min_iv=pred_min,
        max_iv=pred_max,
        n=n,
    )


def warn_model_fallback(
    *,
    requested_model: str,
    failed_model: str,
    fallback_model: str,
    message: str,
    quality: Optional[ModelQuality] = None,
) -> None:
    metrics = ""
    key_metrics = ("", "", "", 0)
    if quality is not None:
        metrics = f" rmse={quality.rmse} min_iv={quality.min_iv} " f"max_iv={quality.max_iv} n={quality.n}"
        key_metrics = (
            f"{quality.rmse:.8g}" if np.isfinite(quality.rmse) else "nan",
            f"{quality.min_iv:.8g}" if np.isfinite(quality.min_iv) else "nan",
            f"{quality.max_iv:.8g}" if np.isfinite(quality.max_iv) else "nan",
            int(quality.n),
        )
    key = (
        str(requested_model),
        str(failed_model),
        str(fallback_model),
        str(message),
        int(key_metrics[3]),
        str(key_metrics[0]),
        str(key_metrics[1]),
        str(key_metrics[2]),
    )
    log = LOGGER.debug if key in _FALLBACK_WARNING_KEYS else LOGGER.warning
    _FALLBACK_WARNING_KEYS.add(key)
    log(
        "model fallback requested=%s failed=%s fallback=%s degraded=true reason=%s%s",
        requested_model,
        failed_model,
        fallback_model,
        message,
        metrics,
    )
