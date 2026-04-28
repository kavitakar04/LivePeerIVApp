import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from display.gui.controllers.gui_plot_manager import PlotManager


def test_term_replots_when_weight_method_changes(monkeypatch):
    pm = PlotManager()

    # Ensure weights function returns constant weights regardless of mode
    monkeypatch.setattr(
        pm,
        "_weights_from_ui_or_matrix",
        lambda target, peers, weight_mode, **kwargs: pd.Series({peers[0]: 1.0})
    )

    payloads = []

    def fake_compute_or_load(name, payload, builder, **kwargs):
        payloads.append(payload)
        # minimal valid return structure for _plot_term
        return {"atm_curve": pd.DataFrame({"T": [0.1], "atm_vol": [0.2]}), "synth_bands": None}

    monkeypatch.setattr(
        "display.gui.controllers.gui_plot_manager.compute_or_load", fake_compute_or_load
    )

    fig, ax = plt.subplots()
    settings = {
        "plot_type": "Term",
        "target": "TGT",
        "asof": "2024-01-01",
        "model": "svi",
        "T_days": 30,
        "ci": 68,
        "x_units": "years",
        "overlay_synth": False,
        "peers": ["AAA"],
        "pillars": [30],
        "weight_method": "corr",
        "feature_mode": "iv_atm",
        "max_expiries": 6,
    }

    pm.plot(ax, settings)
    settings["weight_method"] = "equal"
    pm.plot(ax, settings)

    assert len(payloads) == 2
    modes = [p.get("weight_mode") for p in payloads]
    assert modes[0] != modes[1]
