import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.feature_health import FeatureConstructionResult
from analysis.correlation_view import prepare_correlation_view
from display.gui.gui_plot_manager import PlotManager


def test_correlation_view_uses_selected_weight_feature_matrix(monkeypatch):
    feature_df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [3.0, 2.0, 1.0]],
        index=["TGT", "P1", "P2"],
        columns=["grid_1", "grid_2", "grid_3"],
    )
    feature_df.attrs["feature_diagnostics"] = {
        "feature_set": "surface_grid",
        "coordinate_system": "standardized_tenor_grid_x_moneyness_bin",
        "shape": feature_df.shape,
    }

    feature_result = FeatureConstructionResult(
        feature_matrix=feature_df,
        alignment_metadata={},
        coverage={"TGT": 1.0, "P1": 1.0, "P2": 1.0},
        warnings=[],
        feature_health={"available": True, "summary": feature_df.attrs["feature_diagnostics"]},
    )
    monkeypatch.setattr("analysis.correlation_view.build_feature_construction_result", lambda **_kwargs: feature_result)
    monkeypatch.setattr(
        "analysis.correlation_view.maybe_compute_weights",
        lambda **_kwargs: pd.Series({"P1": 0.7, "P2": 0.3}),
    )

    view = prepare_correlation_view(
        get_smile_slice=lambda *args, **kwargs: None,
        tickers=["TGT", "P1", "P2"],
        asof="2024-01-01",
        target="TGT",
        peers=["P1", "P2"],
        weight_mode="cosine_surface_grid",
        use_cache=False,
    )

    assert view.atm_df is feature_df
    assert view.context["feature_diagnostics"]["feature_set"] == "surface_grid"
    assert view.context["similarity_display_method"] == "cosine"
    assert view.corr_df.index.tolist() == ["TGT", "P1", "P2"]
    assert view.weights.to_dict() == {"P1": 0.7, "P2": 0.3}


def test_correlation_view_does_not_forward_show_values_to_weight_config(monkeypatch):
    feature_df = pd.DataFrame(
        [[1.0, 2.0], [1.0, 2.5]],
        index=["TGT", "P1"],
        columns=["rank0_atm", "rank1_atm"],
    )

    def fake_build_feature_construction_result(**kwargs):
        assert "show_values" not in kwargs
        return FeatureConstructionResult(
            feature_matrix=feature_df,
            alignment_metadata={},
            coverage={"TGT": 1.0, "P1": 1.0},
            warnings=[],
            feature_health={"available": True},
        )

    def fake_maybe_compute_weights(**kwargs):
        assert "show_values" not in kwargs
        return pd.Series({"P1": 1.0})

    monkeypatch.setattr(
        "analysis.correlation_view.build_feature_construction_result",
        fake_build_feature_construction_result,
    )
    monkeypatch.setattr(
        "analysis.correlation_view.maybe_compute_weights",
        fake_maybe_compute_weights,
    )

    view = prepare_correlation_view(
        get_smile_slice=lambda *args, **kwargs: None,
        tickers=["TGT", "P1"],
        asof="2024-01-01",
        target="TGT",
        peers=["P1"],
        weight_mode="corr_iv_atm",
        show_values=True,
        use_cache=False,
    )

    assert view.weights.to_dict() == {"P1": 1.0}


def test_relative_weight_matrix_uses_expiry_rank_term_structure(monkeypatch):
    pm = PlotManager()
    captured = {}

    def fake_prepare_correlation_view(**kwargs):
        captured["weight_mode"] = kwargs["weight_mode"]
        feature_df = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.1, 3.1, 4.1]],
            index=["TGT", "P1"],
        )
        feature_df.attrs["feature_diagnostics"] = {
            "coordinate_system": "native_expiry_ranks",
            "feature_set": "iv_atm_ranks",
        }
        return type(
            "View",
            (),
            {
                "corr_df": pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=["TGT", "P1"], columns=["TGT", "P1"]),
                "weights": pd.Series({"P1": 1.0}),
                "atm_df": feature_df,
                "context": {"feature_health": {}, "feature_diagnostics": feature_df.attrs["feature_diagnostics"]},
                "coverage": pd.Series({"TGT": 4, "P1": 4}),
                "overlap": pd.DataFrame([[4, 4], [4, 4]], index=["TGT", "P1"], columns=["TGT", "P1"]),
                "finite_count": 4,
                "total_cells": 4,
                "finite_ratio": 1.0,
                "method_label": "correlation",
                "basis_label": "ATM IV expiry ranks",
            },
        )()

    def fake_plot_correlation_details(*args, **kwargs):
        return None

    monkeypatch.setattr("display.gui.gui_plot_manager.prepare_correlation_view", fake_prepare_correlation_view)
    monkeypatch.setattr("display.gui.gui_plot_manager.plot_correlation_details", fake_plot_correlation_details)

    pm.last_settings = {
        "weight_method": "corr",
        "feature_mode": "iv_atm",
        "weight_power": 1.0,
        "clip_negative": True,
        "max_expiries": 12,
    }
    pm._current_max_expiries = 12
    fig, ax = plt.subplots()

    pm._plot_corr_matrix(ax, "TGT", ["P1"], "2024-01-01", [7, 30, 60], "corr_iv_atm", 0.05)

    assert captured["weight_mode"] == "corr_iv_atm_ranks"
    assert pm.last_corr_meta["weight_mode"] == "corr_iv_atm_ranks"
    plt.close(fig)
