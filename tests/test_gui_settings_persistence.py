import json

from display.gui.gui_input import _persistable_settings, _save_gui_preferences


def test_persistable_settings_keeps_only_user_preferences():
    settings = {
        "target": "WULF",
        "peers": ["IREN", "CORZ"],
        "model": "svi",
        "feature_mode": "iv_atm",
        "weight_method": "corr",
        "max_expiries": 12,
        "transient_status": "do not persist",
    }

    saved = _persistable_settings(settings)

    assert saved["target"] == "WULF"
    assert saved["peers"] == ["IREN", "CORZ"]
    assert saved["max_expiries"] == 12
    assert "transient_status" not in saved


def test_save_gui_preferences_writes_json_atomically(tmp_path):
    path = tmp_path / "gui_settings.json"

    _save_gui_preferences(
        {
            "target": "WULF",
            "peers": ["IREN"],
            "mny_bins": ((0.8, 0.9), (0.9, 1.1)),
            "unknown": "ignored",
        },
        path=path,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "mny_bins": [[0.8, 0.9], [0.9, 1.1]],
        "peers": ["IREN"],
        "target": "WULF",
    }
