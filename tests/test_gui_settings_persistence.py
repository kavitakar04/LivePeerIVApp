import json

from display.gui.controls.gui_input import _persistable_settings, _save_gui_preferences
from display.gui.controls.gui_input import (
    generate_even_mny_bins,
    generate_even_pillars,
    mny_bins_for_axis_settings,
    pillars_for_axis_settings,
    resolved_grid_preview,
)


def test_persistable_settings_keeps_only_user_preferences():
    settings = {
        "target": "WULF",
        "peers": ["IREN", "CORZ"],
        "model": "svi",
        "feature_mode": "iv_atm",
        "weight_method": "corr",
        "smile_moneyness_range": (0.75, 1.25),
        "term_axis_mode": "even",
        "term_axis_start": 7,
        "term_axis_end": 77,
        "term_axis_step": 7,
        "mny_axis_mode": "even",
        "mny_axis_min": 0.5,
        "mny_axis_max": 1.5,
        "mny_axis_step": 0.1,
        "mny_relative_to_atm": True,
        "max_expiries": 12,
        "underlying_lookback_days": 800,
        "transient_status": "do not persist",
    }

    saved = _persistable_settings(settings)

    assert saved["target"] == "WULF"
    assert saved["peers"] == ["IREN", "CORZ"]
    assert saved["smile_moneyness_range"] == [0.75, 1.25]
    assert saved["term_axis_mode"] == "even"
    assert saved["term_axis_start"] == 7
    assert saved["term_axis_end"] == 77
    assert saved["term_axis_step"] == 7
    assert saved["mny_axis_mode"] == "even"
    assert saved["mny_axis_min"] == 0.5
    assert saved["mny_axis_max"] == 1.5
    assert saved["mny_axis_step"] == 0.1
    assert saved["mny_relative_to_atm"] is True
    assert saved["max_expiries"] == 12
    assert saved["underlying_lookback_days"] == 800
    assert "transient_status" not in saved


def test_save_gui_preferences_writes_json_atomically(tmp_path):
    path = tmp_path / "gui_settings.json"

    _save_gui_preferences(
        {
            "target": "WULF",
            "peers": ["IREN"],
            "mny_bins": ((0.8, 0.9), (0.9, 1.1)),
            "smile_moneyness_range": (0.75, 1.20),
            "term_axis_mode": "custom",
            "term_axis_custom": "7,30",
            "mny_axis_mode": "custom",
            "mny_axis_custom": "0.80-0.90,0.90-1.10",
            "underlying_lookback_days": 800,
            "unknown": "ignored",
        },
        path=path,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "mny_axis_custom": "0.80-0.90,0.90-1.10",
        "mny_axis_mode": "custom",
        "mny_bins": [[0.8, 0.9], [0.9, 1.1]],
        "peers": ["IREN"],
        "smile_moneyness_range": [0.75, 1.2],
        "target": "WULF",
        "term_axis_custom": "7,30",
        "term_axis_mode": "custom",
        "underlying_lookback_days": 800,
    }


def test_even_axis_generators_create_resolved_numeric_grid():
    pillars = generate_even_pillars(7, 77, 7)
    bins = generate_even_mny_bins(0.5, 1.5, 0.1)

    assert pillars == [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
    assert bins[0] == (0.50, 0.60)
    assert bins[-1] == (1.40, 1.50)
    assert len(bins) == 10
    preview = resolved_grid_preview("even", "even", pillars, bins, relative_to_atm=True)
    assert "110 cells/ticker" in preview
    assert "K/S relative to ATM" in preview


def test_custom_axis_modes_use_raw_numeric_text():
    assert pillars_for_axis_settings("Custom", 7, 77, 7, "12,24") == [12, 24]
    assert mny_bins_for_axis_settings("Custom bins", 0.5, 1.5, 0.1, "0.85-0.95,0.95-1.05") == (
        (0.85, 0.95),
        (0.95, 1.05),
    )
