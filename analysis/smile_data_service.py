"""Smile data preparation facade.

Active GUI callers should import smile-slice and smile-plot payload helpers
from here.  Implementations remain in ``analysis.analysis_pipeline`` while the
large pipeline module is being split.
"""

from __future__ import annotations

from analysis.analysis_pipeline import get_smile_slice, get_smile_slices_batch, prepare_smile_data

__all__ = ["get_smile_slice", "get_smile_slices_batch", "prepare_smile_data"]
