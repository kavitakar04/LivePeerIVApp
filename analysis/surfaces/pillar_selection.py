"""Pillar selection facade.

This module gives new callers a focused import path while ``analysis.surfaces.pillars``
continues to preserve the historical public API.
"""

from __future__ import annotations

from analysis.surfaces.pillars import (
    DEFAULT_PILLARS_DAYS,
    EXTENDED_PILLARS_DAYS,
    build_atm_matrix,
    detect_available_pillars,
    load_atm,
    nearest_pillars,
)

__all__ = [
    "DEFAULT_PILLARS_DAYS",
    "EXTENDED_PILLARS_DAYS",
    "build_atm_matrix",
    "detect_available_pillars",
    "load_atm",
    "nearest_pillars",
]
