"""Compatibility wrapper for the canonical calculation cache.

New callers should import :func:`analysis.cache_io.compute_or_load` directly.
This module preserves the older import path while routing all artifacts to the
same TTL/versioned SQLite backend in ``data/calculations.db`` by default.
"""

from __future__ import annotations

from typing import Any, Callable

from analysis.cache_io import compute_or_load as _canonical_compute_or_load


def compute_or_load(
    name: str,
    payload: dict,
    builder: Callable[[], Any],
    *,
    db_path: str | None = None,
) -> Any:
    return _canonical_compute_or_load(
        name,
        payload,
        builder,
        **({"db_path": db_path} if db_path is not None else {}),
    )
