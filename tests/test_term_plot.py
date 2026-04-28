import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from display.gui.controllers.gui_plot_manager import PlotManager
from display.plotting.charts.term_plot import plot_atm_term_structure, plot_term_structure_comparison


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


def test_term_plot_draws_quote_dispersion_as_error_bars():
    atm_curve = pd.DataFrame({
        "T": [0.1, 0.2, 0.3],
        "atm_vol": [0.20, 0.22, 0.24],
        "quote_dispersion": [0.01, 0.02, np.nan],
    })
    peer_curve = pd.DataFrame({
        "T": [0.1, 0.2, 0.3],
        "atm_vol": [0.19, 0.21, 0.23],
        "quote_dispersion": [0.015, 0.010, 0.012],
    })
    fig, ax = plt.subplots()

    plot_term_structure_comparison(
        ax,
        atm_curve,
        peer_curves={"P1": peer_curve},
        x_units="years",
        show_quote_dispersion=True,
    )

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Target ATM quote dispersion" in labels
    assert "P1 quote dispersion" in labels
    assert not any(line.get_label() == "Target ATM quote dispersion" for line in ax.lines)
    assert len(ax.collections) >= 2
    plt.close(fig)


def test_term_data_preserves_quote_dispersion_for_single_atm(monkeypatch):
    import analysis.services.term_data_service as term_data_service
    from analysis.services.term_data_service import prepare_term_data

    def make_slice(_ticker: str) -> pd.DataFrame:
        spot = 100.0
        return pd.DataFrame(
            [
                {
                    "T": 30 / 365.25,
                    "moneyness": mny,
                    "sigma": sigma,
                    "K": spot * mny,
                    "S": spot,
                    "expiry": pd.Timestamp("2024-01-31"),
                }
                for mny, sigma in [(0.96, 0.20), (0.99, 0.22), (1.01, 0.24), (1.04, 0.26)]
            ]
        )

    monkeypatch.setattr(
        term_data_service,
        "get_smile_slice",
        lambda ticker, asof, T_target_years=None, max_expiries=None: make_slice(ticker.upper()),
    )

    data = prepare_term_data(target="TGT", asof="2024-01-01", ci=0, max_expiries=1)

    assert np.isfinite(data["atm_curve"].loc[0, "quote_dispersion"])
    assert data["atm_curve"].loc[0, "band_source"] == "none_single_atm"


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
    import analysis.services.term_data_service as term_data_service
    from analysis.services.term_data_service import prepare_term_data

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
        term_data_service,
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


def test_prepare_term_data_does_not_expand_target_beyond_common_grid(monkeypatch):
    import analysis.services.term_data_service as term_data_service
    from analysis.services.term_data_service import prepare_term_data

    def make_curve(ticker: str) -> pd.DataFrame:
        days_by_ticker = {
            "TGT": [5, 12, 19, 26, 33, 53],
            "P1": [5, 12, 19, 26, 33, 53],
            "P2": [19, 53],
        }
        rows = []
        for days in days_by_ticker[ticker]:
            rows.append({
                "T": days / 365.25,
                "atm_vol": 0.20 + days / 10000.0,
                "expiry": pd.Timestamp("2024-01-01") + pd.Timedelta(days=days),
                "model": "svi",
                "spot": 100.0,
                "atm_strike": 100.0,
                "iv_source": "sigma",
                "count": 5,
            })
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        term_data_service,
        "_compute_term_atm_curve",
        lambda ticker, asof, **kwargs: make_curve(ticker.upper()),
    )

    data = prepare_term_data(
        target="TGT",
        asof="2024-01-01",
        ci=0,
        overlay_synth=True,
        peers=["P1", "P2"],
        weights={"P1": 0.5, "P2": 0.5},
        max_expiries=10,
    )

    target_days = list(np.round(data["atm_curve"]["T"].to_numpy(float) * 365.25).astype(int))
    synth_days = list(np.round(data["synth_curve"]["T"].to_numpy(float) * 365.25).astype(int))

    assert target_days == [19, 53]
    assert synth_days == [19, 53]
    assert all(len(curve) == 2 for curve in data["peer_curves"].values())


def test_plot_manager_clear_child_axes_removes_stale_insets():
    fig, ax = plt.subplots()
    stale = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
    stale.set_xlabel("Log-moneyness")
    assert list(ax.child_axes) == [stale]

    PlotManager()._clear_child_axes(ax)

    assert fig.axes == [ax]
    assert list(ax.child_axes) == []
    plt.close(fig)
