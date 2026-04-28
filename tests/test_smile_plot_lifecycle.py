import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from display.gui.controllers.gui_plot_manager import PlotManager


def _manager_with_smile_context(ax):
    S = 100.0
    T = 30.0 / 365.25
    K = np.array([50.0, 80.0, 95.0, 100.0, 105.0, 120.0, 140.0])
    k = np.log(K / S)
    iv = 0.22 - 0.04 * k + 0.25 * k * k
    cp = np.array(["P", "P", "P", "C", "C", "C", "C"])

    mgr = PlotManager()
    mgr._current_plot_type = "smile"
    mgr._current_max_expiries = 1
    mgr._smile_ctx = {
        "ax": ax,
        "T_arr": np.full_like(K, T, dtype=float),
        "K_arr": K,
        "sigma_arr": iv,
        "S_arr": np.full_like(K, S, dtype=float),
        "cp_arr": cp,
        "Ts": np.array([T]),
        "idx": 0,
        "settings": {
            "target": "TGT",
            "asof": "2024-01-02",
            "model": "svi",
            "ci": 0.0,
            "weight_mode": "equal_surface",
            "feature_mode": "surface",
            "weight_method": "equal",
            "overlay_synth": False,
            "overlay_peers": False,
            "peers": [],
        },
        "weights": None,
        "tgt_surface": None,
        "syn_surface": None,
        "peer_slices": {},
        "expiry_arr": None,
        "fit_by_expiry": {},
    }
    return mgr


def test_repeated_smile_render_dedupes_legend_and_hides_extreme_quotes():
    fig, ax = plt.subplots()
    mgr = _manager_with_smile_context(ax)

    mgr._render_smile_at_index()
    mgr._render_smile_at_index()

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == list(dict.fromkeys(labels))
    assert "Calls" in labels
    assert "Puts" in labels
    assert "TGT SVI" in labels

    assert "Hidden" not in ax.get_title()
    events = mgr.last_fit_info["status_events"]
    assert any(
        event["category"] == "data_filter"
        and event["status"] == "warning"
        and "2 quotes outside 0.7-1.3 K/S" in event["message"]
        for event in events
    )
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets):
            x = np.asarray(offsets[:, 0], dtype=float)
            assert np.nanmin(x) >= 0.7
            assert np.nanmax(x) <= 1.3
    plt.close(fig)


def test_smile_render_skips_malformed_peer_composite_overlay(monkeypatch):
    fig, ax = plt.subplots()
    mgr = _manager_with_smile_context(ax)
    mgr._smile_ctx["settings"]["overlay_synth"] = True
    mgr._smile_ctx["settings"]["peers"] = ["P1"]
    mgr._smile_ctx["peer_slices"] = {
        "P1": {
            "T_arr": np.array([]),
            "K_arr": np.array([]),
            "sigma_arr": np.array([]),
            "S_arr": np.array([]),
        }
    }

    def malformed_composite(*_args, **_kwargs):
        return {
            "moneyness": np.linspace(0.7, 1.3, 121),
            "iv": np.array([]),
            "skipped": {"P1": "svi fit rejected"},
        }

    monkeypatch.setattr("display.gui.controllers.gui_plot_manager.build_peer_smile_composite", malformed_composite)

    mgr._render_smile_at_index()

    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert "Peer composite smile (equal)" not in labels
    assert any(
        event["category"] == "peer_composite"
        and event["status"] == "warning"
        and "mismatched shapes" in event["message"]
        for event in mgr.last_fit_info["status_events"]
    )
    plt.close(fig)
