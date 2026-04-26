import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from display.gui.gui_plot_manager import PlotManager
from display.plotting.term_plot import plot_atm_term_structure, plot_term_structure_comparison


def test_plot_atm_term_structure_handles_string_vols():
    atm_curve = pd.DataFrame({
        'T': [0.5, 1.0, 1.5],
        'atm_vol': ['0.2', '0.3', '0.4'],
    })
    fig, ax = plt.subplots()
    plot_atm_term_structure(ax, atm_curve)
    assert np.allclose(ax.lines[0].get_ydata(), [0.2, 0.3, 0.4])
    plt.close(fig)


def test_term_comparison_does_not_create_inset_axes():
    atm_curve = pd.DataFrame({
        "T": [0.1, 0.2, 0.3],
        "atm_vol": [0.2, 0.22, 0.24],
    })
    synth_curve = pd.DataFrame({
        "T": [0.1, 0.2, 0.3],
        "atm_vol": [0.19, 0.21, 0.23],
    })
    fig, ax = plt.subplots()

    plot_term_structure_comparison(ax, atm_curve, synth_curve=synth_curve, x_units="days")

    assert fig.axes == [ax]
    assert ax.get_xlabel() == "Time to Expiry (days)"
    plt.close(fig)


def test_term_comparison_default_layers_are_target_and_composite():
    atm_curve = pd.DataFrame({
        "T": [0.1, 0.2, 0.3],
        "atm_vol": [0.2, 0.22, 0.24],
        "atm_lo": [0.19, 0.21, 0.23],
        "atm_hi": [0.21, 0.23, 0.25],
    })
    synth_curve = pd.DataFrame({
        "T": [0.05, 0.1, 0.2, 0.3, 0.4],
        "atm_vol": [0.18, 0.19, 0.21, 0.23, 0.26],
        "atm_lo": [0.17, 0.18, 0.20, 0.22, 0.25],
        "atm_hi": [0.19, 0.20, 0.22, 0.24, 0.27],
    })
    peer_curve = pd.DataFrame({
        "T": [0.05, 0.2, 0.4],
        "atm_vol": [0.18, 0.215, 0.27],
    })
    fig, ax = plt.subplots()

    plot_term_structure_comparison(
        ax,
        atm_curve,
        peer_curves={"P1": peer_curve},
        synth_curve=synth_curve,
        x_units="days",
        fit=False,
        show_ci=True,
        title="ATM Term Structure: TGT vs Peer Composite",
    )

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["Target ATM", "Peer composite", "P1"]
    assert ax.get_title() == "ATM Term Structure: TGT vs Peer Composite"
    assert ax.get_ylabel() == "ATM IV"
    target_line = next(line for line in ax.lines if line.get_label() == "Target ATM")
    composite_line = next(line for line in ax.lines if line.get_label() == "Peer composite")
    assert target_line.get_linewidth() > composite_line.get_linewidth()
    assert target_line.get_zorder() > composite_line.get_zorder()

    target_min = 0.1 * 365.25
    target_max = 0.3 * 365.25
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        assert np.nanmin(x) >= target_min
        assert np.nanmax(x) <= target_max
    plt.close(fig)


def test_plot_manager_term_defaults_hide_peer_lines_and_fit():
    pm = PlotManager()
    data = {
        "atm_curve": pd.DataFrame({"T": [0.1, 0.2, 0.3], "atm_vol": [0.2, 0.22, 0.24]}),
        "synth_curve": pd.DataFrame({"T": [0.1, 0.2, 0.3], "atm_vol": [0.19, 0.21, 0.23]}),
        "peer_curves": {
            "P1": pd.DataFrame({"T": [0.1, 0.2, 0.3], "atm_vol": [0.18, 0.2, 0.22]})
        },
        "weights": pd.Series({"P1": 1.0}),
    }
    fig, ax = plt.subplots()

    pm._plot_term(ax, data, "TGT", "2024-01-01", "days", 0.68)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["Target ATM", "Peer composite"]
    assert "P1" not in labels
    assert "Term fit" not in labels
    assert ax.get_title() == "ATM Term Structure: TGT vs Peer Composite"
    plt.close(fig)


def test_plot_manager_term_shows_peer_legend_when_overlay_enabled():
    pm = PlotManager()
    data = {
        "atm_curve": pd.DataFrame({"T": [0.1, 0.2, 0.3], "atm_vol": [0.2, 0.22, 0.24]}),
        "synth_curve": pd.DataFrame({"T": [0.1, 0.2, 0.3], "atm_vol": [0.19, 0.21, 0.23]}),
        "peer_curves": {
            "P1": pd.DataFrame({"T": [0.1, 0.2, 0.3], "atm_vol": [0.18, 0.2, 0.22]})
        },
        "weights": pd.Series({"P1": 1.0}),
    }
    fig, ax = plt.subplots()

    pm._plot_term(ax, data, "TGT", "2024-01-01", "days", 0.68, overlay_peers=True)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["Target ATM", "Peer composite", "P1 (100.0%)"]
    assert "Individual peers shown: P1." in pm.last_description
    plt.close(fig)


def test_prepare_term_data_uses_same_atm_path_and_aligned_composite(monkeypatch):
    from analysis.analysis_pipeline import prepare_term_data
    import analysis.analysis_pipeline as pipeline

    def make_slice(ticker: str) -> pd.DataFrame:
        spot = 100.0
        levels = {"TGT": 0.22, "P1": 0.24, "P2": 0.20}
        rows = []
        for T in [30 / 365.25, 60 / 365.25]:
            for mny in [0.9, 0.98, 1.0, 1.02, 1.1]:
                rows.append(
                    {
                        "T": T,
                        "moneyness": mny,
                        "sigma": levels[ticker] + 0.02 * (mny - 1.0) ** 2,
                        "K": spot * mny,
                        "S": spot,
                        "expiry": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(round(T * 365.25))),
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        pipeline,
        "get_smile_slice",
        lambda ticker, asof, T_target_years=None, max_expiries=None: make_slice(ticker.upper()),
    )

    data = prepare_term_data(
        target="TGT",
        asof="2024-01-01",
        ci=0,
        overlay_synth=True,
        peers=["P1", "P2"],
        weights={"P1": 0.75, "P2": 0.25},
        max_expiries=2,
    )

    assert data["alignment_status"] == "aligned"
    assert data["composite_status"] == "aligned_weighted"
    assert not data["synth_curve"].empty
    assert set(data["peer_curves"]) == {"P1", "P2"}
    assert data["term_warnings"] == []
    assert set(data["atm_curve"]["model"]) != {"median"}
    for curve in data["peer_curves"].values():
        assert set(curve["model"]) != {"median"}
        assert {"spot", "atm_strike", "iv_source", "count"}.issubset(curve.columns)


def test_plot_manager_clear_child_axes_removes_stale_insets():
    fig, ax = plt.subplots()
    stale = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
    stale.set_xlabel("Log-moneyness")
    assert list(ax.child_axes) == [stale]

    PlotManager()._clear_child_axes(ax)

    assert fig.axes == [ax]
    assert list(ax.child_axes) == []
    plt.close(fig)
