from __future__ import annotations


SUPPORTED_MODELS = ("svi", "sabr", "tps", "poly")
GUI_MODELS = tuple(m for m in SUPPORTED_MODELS if m != "poly")
