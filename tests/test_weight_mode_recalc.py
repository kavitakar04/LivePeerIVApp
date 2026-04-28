import pandas as pd
import matplotlib
matplotlib.use('Agg')

from display.gui.controllers.gui_plot_manager import PlotManager


def test_weights_recomputed_on_weight_mode_change(monkeypatch):
    pm = PlotManager()
    pm.last_corr_df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["PEER", "TARGET"],
        columns=["PEER", "TARGET"],
    )
    pm.last_corr_meta = {"weight_mode": "corr_iv_atm"}

    calls = {"compute": 0}

    def fake_compute_unified_weights(
        target, peers, mode, power=1.0, clip_negative=True, pillars_days=None, asof=None
    ):
        calls["compute"] += 1
        return pd.Series({peers[0]: 1.0})

    monkeypatch.setattr(
        "analysis.weights.unified_weights.compute_unified_weights", fake_compute_unified_weights
    )

    pm._weights_from_ui_or_matrix("TARGET", ["PEER"], "corr_iv_atm")
    pm._weights_from_ui_or_matrix("TARGET", ["PEER"], "corr_ul")

    assert calls["compute"] == 2
