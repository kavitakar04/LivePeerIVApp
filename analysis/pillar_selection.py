"""Pillar selection facade.

This module gives new callers a focused import path while ``analysis.pillars``
continues to preserve the historical public API.
"""

from __future__ import annotations

from analysis.pillars import DEFAULT_PILLARS_DAYS, load_atm, nearest_pillars

__all__ = ["DEFAULT_PILLARS_DAYS", "load_atm", "nearest_pillars"]
