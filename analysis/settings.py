"""Central defaults for analysis and GUI configuration.

These constants are the stable bridge for a future GUI settings panel. Keep
domain defaults here, then pass user-selected overrides through settings dicts
or config dataclasses instead of re-declaring literals in downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# Vol surface and ATM extraction
DEFAULT_SURFACE_TENORS: Tuple[int, ...] = (7, 30, 60, 90, 180, 365)
DEFAULT_MONEYNESS_BINS: Tuple[Tuple[float, float], ...] = (
    (0.80, 0.90),
    (0.90, 1.10),
    (1.10, 1.25),
)
DEFAULT_PILLAR_DAYS: Tuple[int, ...] = (7, 30, 60, 90, 180, 365)
DEFAULT_NEAR_TERM_PILLAR_DAYS: Tuple[int, ...] = (7, 14, 30)
DEFAULT_EXTENDED_PILLAR_DAYS: Tuple[int, ...] = (7, 14, 30, 60, 90, 180, 365)
DEFAULT_PILLAR_TOLERANCE_DAYS = 14.0
DEFAULT_ATM_BAND = 0.05
DEFAULT_WEIGHT_ATM_BAND = 0.08
DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS = 10.0
DEFAULT_MAX_EXPIRIES = 6

# Relative-value and spillover windows
DEFAULT_RV_LOOKBACK_DAYS = 60
DEFAULT_SPILLOVER_EVENT_THRESHOLD = 0.10
DEFAULT_SPILLOVER_LOOKBACK_DAYS = 60
DEFAULT_SPILLOVER_REGRESSION_WINDOW_DAYS = 90
DEFAULT_SPILLOVER_HORIZONS: Tuple[int, ...] = (1, 3, 5)
DEFAULT_BACKGROUND_SPILLOVER_HORIZONS: Tuple[int, ...] = (0, 1, 3, 5, 10)
DEFAULT_BACKGROUND_WINDOW_DAYS = 90
DEFAULT_BACKGROUND_RECOMPUTE_TAIL_DAYS = 7
DEFAULT_BACKGROUND_MIN_SAMPLES = 20

# GUI-facing defaults
DEFAULT_MODEL = "svi"
DEFAULT_CI = 0.68
DEFAULT_X_UNITS = "years"
DEFAULT_WEIGHT_METHOD = "corr"
DEFAULT_FEATURE_MODE = "iv_atm"
DEFAULT_WEIGHT_POWER = 1.0
DEFAULT_CLIP_NEGATIVE_WEIGHTS = True
DEFAULT_OVERLAY = False

# Cache defaults
DEFAULT_CALC_CACHE_TTL_SEC = 900


@dataclass(frozen=True)
class AnalysisDefaults:
    """Snapshot of defaults suitable for GUI settings initialization."""

    pillar_days: Tuple[int, ...] = DEFAULT_PILLAR_DAYS
    surface_tenors: Tuple[int, ...] = DEFAULT_SURFACE_TENORS
    moneyness_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MONEYNESS_BINS
    pillar_tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS
    atm_band: float = DEFAULT_ATM_BAND
    max_expiries: int = DEFAULT_MAX_EXPIRIES
    rv_lookback_days: int = DEFAULT_RV_LOOKBACK_DAYS
    spillover_threshold: float = DEFAULT_SPILLOVER_EVENT_THRESHOLD
    spillover_lookback_days: int = DEFAULT_SPILLOVER_LOOKBACK_DAYS
    spillover_horizons: Tuple[int, ...] = DEFAULT_SPILLOVER_HORIZONS
    weight_method: str = DEFAULT_WEIGHT_METHOD
    feature_mode: str = DEFAULT_FEATURE_MODE
    weight_power: float = DEFAULT_WEIGHT_POWER
    clip_negative_weights: bool = DEFAULT_CLIP_NEGATIVE_WEIGHTS


DEFAULT_ANALYSIS_SETTINGS = AnalysisDefaults()


def parse_int_list(text: str, fallback: Tuple[int, ...] = DEFAULT_PILLAR_DAYS) -> list[int]:
    """Parse comma-separated integer settings with a safe fallback."""
    try:
        vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
        return vals or list(fallback)
    except Exception:
        return list(fallback)


def format_moneyness_bins(bins: Tuple[Tuple[float, float], ...] = DEFAULT_MONEYNESS_BINS) -> str:
    """Return compact text representation like ``0.80-0.90,0.90-1.10``."""
    return ",".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in bins)


def parse_moneyness_bins(
    text: str,
    fallback: Tuple[Tuple[float, float], ...] = DEFAULT_MONEYNESS_BINS,
) -> Tuple[Tuple[float, float], ...]:
    """Parse comma-separated moneyness bins such as ``0.80-0.90,0.90-1.10``."""
    try:
        bins: list[tuple[float, float]] = []
        for part in str(text).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" not in part:
                raise ValueError(part)
            lo_s, hi_s = part.split("-", 1)
            lo = float(lo_s.strip())
            hi = float(hi_s.strip())
            if hi <= lo:
                raise ValueError(part)
            bins.append((lo, hi))
        return tuple(bins) or fallback
    except Exception:
        return fallback
