import numpy as np

from analysis.config.settings import (
    DEFAULT_SMILE_MONEYNESS_RANGE,
    format_moneyness_range,
    parse_moneyness_range,
)
from display.gui.controllers.gui_plot_manager import _filter_smile_quotes, _smile_grid_from_settings


def test_parse_moneyness_range_accepts_valid_bounds():
    assert parse_moneyness_range("0.75-1.20") == (0.75, 1.20)
    assert format_moneyness_range((0.75, 1.2)) == "0.75-1.20"


def test_parse_moneyness_range_rejects_invalid_bounds():
    assert parse_moneyness_range("1.20-0.75") == DEFAULT_SMILE_MONEYNESS_RANGE
    assert parse_moneyness_range("bad") == DEFAULT_SMILE_MONEYNESS_RANGE


def test_smile_grid_uses_configured_range_with_default_point_count():
    assert _smile_grid_from_settings({"smile_moneyness_range": (0.75, 1.20)}) == (0.75, 1.20, 121)


def test_smile_quote_filter_uses_configured_range():
    S = 100.0
    K = np.array([70.0, 80.0, 100.0, 120.0, 130.0])
    IV = np.array([0.30, 0.25, 0.20, 0.25, 0.30])

    K_used, IV_used, _cp, excluded = _filter_smile_quotes(S, K, IV, mny_range=(0.80, 1.20))

    assert K_used.tolist() == [80.0, 100.0, 120.0]
    assert IV_used.tolist() == [0.25, 0.20, 0.25]
    assert excluded == 2
