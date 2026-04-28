import pandas as pd

from display.gui.controllers.gui_plot_manager import _value_for_asof


def test_value_for_asof_matches_string_to_timestamp_key():
    marker = object()
    mapping = {pd.Timestamp('2026-04-24'): marker}

    assert _value_for_asof(mapping, '2026-04-24') is marker


def test_value_for_asof_matches_timezone_input_to_naive_key():
    marker = object()
    mapping = {pd.Timestamp('2026-04-24'): marker}

    assert _value_for_asof(mapping, '2026-04-24T15:30:00+00:00') is marker
