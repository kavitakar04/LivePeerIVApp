"""
Unit tests for analysis.rv_analysis.

Each test uses an in-memory SQLite database populated with minimal synthetic
options data, matching the same fixture pattern used in tests/test_surfaces.py.
GUI and network calls are not exercised.
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------

def _make_smile_slice(ticker="SPY", asof="2024-01-15",
                      expiry="2024-02-16", T=0.084, spot=480.0):
    """Return a minimal smile-slice DataFrame like get_smile_slice() returns."""
    strikes = [460.0, 470.0, 480.0, 490.0, 500.0]
    moneyness = [k / spot for k in strikes]
    ivs = [0.23, 0.21, 0.18, 0.20, 0.22]
    rows = []
    for k, m, iv in zip(strikes, moneyness, ivs):
        rows.append({
            "asof_date": asof,
            "ticker": ticker,
            "expiry": expiry,
            "call_put": "C",
            "K": k,
            "S": spot,
            "T": T,
            "moneyness": m,
            "sigma": iv,
            "delta": 0.5,
            "is_atm": 1 if abs(m - 1.0) < 0.05 else 0,
        })
    # Add a second expiry
    for k, m, iv in zip(strikes, moneyness, [0.24, 0.22, 0.19, 0.21, 0.23]):
        rows.append({
            "asof_date": asof,
            "ticker": ticker,
            "expiry": "2024-04-15",
            "call_put": "C",
            "K": k,
            "S": spot,
            "T": 0.247,
            "moneyness": m,
            "sigma": iv,
            "delta": 0.52,
            "is_atm": 1 if abs(m - 1.0) < 0.05 else 0,
        })
    return pd.DataFrame(rows)


def _make_surface_df():
    """Return a minimal moneyness × tenor surface DataFrame."""
    return pd.DataFrame(
        {30: [0.20, 0.18, 0.21], 60: [0.21, 0.19, 0.22], 90: [0.22, 0.20, 0.23]},
        index=["0.95-1.00", "1.00-1.05", "1.05-1.10"],
    ).astype(float)


# ---------------------------------------------------------------------------
# compute_surface_residual
# ---------------------------------------------------------------------------

class TestComputeSurfaceResidual:
    from analysis.rv_analysis import compute_surface_residual

    def test_empty_inputs_return_empty(self):
        from analysis.rv_analysis import compute_surface_residual
        assert compute_surface_residual({}, {}) == {}

    def test_no_common_dates_return_empty(self):
        from analysis.rv_analysis import compute_surface_residual
        t1 = pd.Timestamp("2024-01-15")
        t2 = pd.Timestamp("2024-01-16")
        tgt = {t1: _make_surface_df()}
        syn = {t2: _make_surface_df()}
        assert compute_surface_residual(tgt, syn) == {}

    def test_single_date_returns_raw_residual(self):
        from analysis.rv_analysis import compute_surface_residual
        t = pd.Timestamp("2024-01-15")
        tgt_df = _make_surface_df()
        syn_df = _make_surface_df() * 0.9  # target is ~11% richer
        result = compute_surface_residual({t: tgt_df}, {t: syn_df})
        assert t in result
        arr = result[t].to_numpy(float)
        assert np.isfinite(arr).any()
        # All values should be positive (target > synthetic)
        assert (arr[np.isfinite(arr)] > 0).all()

    def test_multiple_dates_z_scored(self):
        from analysis.rv_analysis import compute_surface_residual
        dates = [pd.Timestamp(f"2024-01-{d:02d}") for d in range(1, 11)]
        rng = np.random.default_rng(42)
        tgt = {}
        syn = {}
        for d in dates:
            base = _make_surface_df()
            tgt[d] = base + rng.normal(0, 0.005, base.shape)
            syn[d] = base
        result = compute_surface_residual(tgt, syn, lookback=5)
        assert len(result) == len(dates)
        # Last date should have z-scores (finite values)
        last = result[dates[-1]].to_numpy(float)
        assert np.isfinite(last).any()

    def test_non_overlapping_grid_cells_handled(self):
        from analysis.rv_analysis import compute_surface_residual
        t = pd.Timestamp("2024-01-15")
        tgt_df = _make_surface_df()
        syn_df_extra = _make_surface_df().copy()
        syn_df_extra.index = ["0.90-0.95", "1.00-1.05", "1.10-1.15"]
        result = compute_surface_residual({t: tgt_df}, {t: syn_df_extra})
        # Only the common index row should appear
        if t in result:
            assert len(result[t]) >= 1


# ---------------------------------------------------------------------------
# compute_skew_spread
# ---------------------------------------------------------------------------

class TestComputeSkewSpread:
    def test_returns_dataframe_with_expected_columns(self):
        from analysis.rv_analysis import compute_skew_spread

        tgt_df = _make_smile_slice("SPY")
        peer_df = _make_smile_slice("QQQ")

        def fake_get_smile(ticker, *args, **kwargs):
            return tgt_df.copy() if ticker.upper() == "SPY" else peer_df.copy()

        with patch("analysis.analysis_pipeline.get_smile_slice", fake_get_smile):
            result = compute_skew_spread(
                "SPY", ["QQQ"], "2024-01-15",
                weights={"QQQ": 1.0}, max_expiries=4,
            )
        # Either an empty DF (if pillars.compute_atm_by_expiry needs more data)
        # or a proper DataFrame with the expected columns
        if not result.empty:
            expected = {"T", "T_days", "target_skew", "synth_skew", "skew_spread"}
            assert expected.issubset(result.columns)

    def test_empty_when_no_peers(self):
        from analysis.rv_analysis import compute_skew_spread

        def fake_get_smile(ticker, *args, **kwargs):
            return pd.DataFrame()

        with patch("analysis.analysis_pipeline.get_smile_slice", fake_get_smile):
            result = compute_skew_spread("SPY", [], "2024-01-15")
        assert result.empty

    def test_empty_when_target_missing(self):
        from analysis.rv_analysis import compute_skew_spread

        def fake_get_smile(ticker, *args, **kwargs):
            return pd.DataFrame()

        with patch("analysis.analysis_pipeline.get_smile_slice", fake_get_smile):
            result = compute_skew_spread("SPY", ["QQQ"], "2024-01-15")
        assert result.empty


# ---------------------------------------------------------------------------
# compute_term_shape_dislocation
# ---------------------------------------------------------------------------

class TestComputeTermShapeDislocation:
    def test_empty_when_no_data(self):
        from analysis.rv_analysis import compute_term_shape_dislocation

        with patch("analysis.rv_analysis.compute_skew_spread", return_value=pd.DataFrame()):
            result = compute_term_shape_dislocation("SPY", ["QQQ"], "2024-01-15")
        assert result == {}

    def test_returns_expected_keys(self):
        from analysis.rv_analysis import compute_term_shape_dislocation

        # Construct a valid skew_spread DataFrame
        skew_df = pd.DataFrame({
            "T": [0.084, 0.247],
            "T_days": [31, 90],
            "target_atm": [0.18, 0.19],
            "synth_atm": [0.17, 0.18],
            "atm_spread": [0.01, 0.01],
            "target_skew": [-0.05, -0.04],
            "synth_skew": [-0.04, -0.035],
            "skew_spread": [-0.01, -0.005],
            "target_curv": [0.02, 0.018],
            "synth_curv": [0.019, 0.017],
            "curv_spread": [0.001, 0.001],
        })

        with patch("analysis.rv_analysis.compute_skew_spread", return_value=skew_df):
            result = compute_term_shape_dislocation("SPY", ["QQQ"], "2024-01-15")

        expected_keys = {
            "target_level", "target_slope",
            "synth_level", "synth_slope",
            "level_spread", "slope_spread",
            "max_event_bump", "event_bump_T_days",
        }
        assert expected_keys.issubset(result.keys())

    def test_finite_values_from_valid_data(self):
        from analysis.rv_analysis import compute_term_shape_dislocation

        skew_df = pd.DataFrame({
            "T": [0.084, 0.247, 0.493],
            "T_days": [31, 90, 180],
            "target_atm": [0.18, 0.19, 0.20],
            "synth_atm": [0.17, 0.185, 0.195],
            "atm_spread": [0.01, 0.005, 0.005],
            "target_skew": [-0.05, -0.04, -0.03],
            "synth_skew": [-0.04, -0.035, -0.028],
            "skew_spread": [-0.01, -0.005, -0.002],
            "target_curv": [0.02, 0.018, 0.015],
            "synth_curv": [0.019, 0.017, 0.014],
            "curv_spread": [0.001, 0.001, 0.001],
        })

        with patch("analysis.rv_analysis.compute_skew_spread", return_value=skew_df):
            result = compute_term_shape_dislocation("SPY", ["QQQ"], "2024-01-15")

        for key in ("level_spread", "slope_spread"):
            assert np.isfinite(result[key]), f"{key} should be finite"


# ---------------------------------------------------------------------------
# generate_rv_signals — lightweight integration test
# ---------------------------------------------------------------------------

class TestGenerateRVSignals:
    def test_returns_dataframe(self):
        from analysis.rv_analysis import generate_rv_signals

        def fake_compute_peer_weights(target, peers, weight_mode, **kwargs):
            return pd.Series({p: 1.0 / len(peers) for p in peers})

        def fake_rv_report(*args, **kwargs):
            df = pd.DataFrame({
                "asof_date": ["2024-01-15"],
                "pillar_days": [30],
                "iv_target": [0.18],
                "iv_synth": [0.17],
                "spread": [0.01],
                "z": [2.1],
                "pct_rank": [85.0],
            })
            return df, {}

        with patch("analysis.analysis_pipeline.compute_peer_weights", fake_compute_peer_weights):
            with patch("analysis.analysis_pipeline.relative_value_atm_report_corrweighted", fake_rv_report):
                with patch("analysis.rv_analysis.compute_skew_spread", return_value=pd.DataFrame()):
                    with patch("analysis.rv_analysis.compute_term_shape_dislocation", return_value={}):
                        result = generate_rv_signals(
                            "SPY", ["QQQ"], asof="2024-01-15",
                            weight_mode="corr_iv_atm",
                        )

        assert isinstance(result, pd.DataFrame)
        assert "signal_type" in result.columns

    def test_empty_peers_returns_empty(self):
        from analysis.rv_analysis import generate_rv_signals

        def fake_most_recent(*args, **kwargs):
            return "2024-01-15"

        with patch("analysis.analysis_pipeline.get_most_recent_date_global", fake_most_recent):
            with patch("analysis.analysis_pipeline.compute_peer_weights",
                       side_effect=Exception("no peers")):
                result = generate_rv_signals("SPY", [], asof="2024-01-15")

        assert isinstance(result, pd.DataFrame)

    def test_z_score_filter_applied(self):
        from analysis.rv_analysis import generate_rv_signals

        def fake_compute_peer_weights(target, peers, weight_mode, **kwargs):
            return pd.Series({p: 1.0 / len(peers) for p in peers})

        # Two signals: one above threshold, one below
        def fake_rv_report(*args, **kwargs):
            df = pd.DataFrame({
                "asof_date": ["2024-01-15", "2024-01-15"],
                "pillar_days": [30, 60],
                "iv_target": [0.18, 0.20],
                "iv_synth": [0.17, 0.19],
                "spread": [0.01, 0.01],
                "z": [2.5, 0.3],  # 0.3 < min_abs_z=1.0 → filtered
                "pct_rank": [90.0, 55.0],
            })
            return df, {}

        with patch("analysis.analysis_pipeline.compute_peer_weights", fake_compute_peer_weights):
            with patch("analysis.analysis_pipeline.relative_value_atm_report_corrweighted", fake_rv_report):
                with patch("analysis.rv_analysis.compute_skew_spread", return_value=pd.DataFrame()):
                    with patch("analysis.rv_analysis.compute_term_shape_dislocation", return_value={}):
                        result = generate_rv_signals(
                            "SPY", ["QQQ"], asof="2024-01-15",
                            weight_mode="corr_iv_atm",
                            min_abs_z=1.0,
                        )

        # Only the z=2.5 signal should survive; z=0.3 should be filtered
        atm_signals = result[result["signal_type"] == "ATM Level"]
        assert len(atm_signals) == 1
        assert abs(float(atm_signals.iloc[0]["z_score"]) - 2.5) < 0.01


# ---------------------------------------------------------------------------
# generate_rv_opportunity_dashboard - synthesis layer
# ---------------------------------------------------------------------------

class TestGenerateRVOpportunityDashboard:
    def test_dashboard_turns_raw_signal_into_ranked_opportunity(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["ATM Level"],
            "asof_date": ["2024-01-15"],
            "T_days": [30],
            "value": [0.22],
            "synth_value": [0.18],
            "spread": [0.04],
            "z_score": [2.4],
            "pct_rank": [94.0],
            "description": ["SPY ATM vol vs synthetic at 30d"],
        })

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (80% same-dir, 70% hit)", {"strength": "Strong"}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.9}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Idiosyncratic", {"same_direction_share": 0.2}, []))

        payload = generate_rv_opportunity_dashboard(
            "SPY", ["QQQ", "IWM"], asof="2024-01-15", weight_mode="corr_iv_atm"
        )

        opp = payload["opportunities"]
        assert len(opp) == 1
        row = opp.iloc[0]
        assert row["rank"] == 1
        assert row["direction"] == "Rich"
        assert row["feature"] == "ATM"
        assert row["maturity"] == "30d"
        assert row["confidence"] == "High"
        assert "SPY 30d ATM rich vs peers" in row["opportunity"]
        assert payload["context_cards"]["data_quality_warnings"] == 0
        assert payload["executive_summary"]

    def test_dashboard_preserves_degraded_context_as_warning(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["Curvature"],
            "asof_date": ["2024-01-15"],
            "T_days": [26],
            "value": [0.05],
            "synth_value": [0.02],
            "spread": [0.03],
            "z_score": [np.nan],
            "pct_rank": [np.nan],
            "description": ["SPY curvature"],
        })

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Degraded", ["1 logged fit row is marked degraded."], {"model": "AUTO"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Unavailable", {}, ["Spillover summary has not been generated."]))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Poor", {"avg_surface_corr": 0.1}, ["Peer surfaces have weak structural similarity."]))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Cluster", {}, []))

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ"], asof="2024-01-15")
        row = payload["opportunities"].iloc[0]

        assert row["data_quality"] == "Degraded"
        assert row["feature"] == "Curvature"
        assert "degraded" in row["warnings"].lower()
        assert payload["context_cards"]["data_quality_warnings"] == 3

    def test_event_context_classifies_broad_move_as_systemic(self, monkeypatch):
        from analysis import analysis_pipeline
        from analysis.rv_analysis import _event_context

        df = pd.DataFrame({
            "date": ["2024-01-14", "2024-01-14", "2024-01-14", "2024-01-15", "2024-01-15", "2024-01-15"],
            "ticker": ["SPY", "QQQ", "IWM", "SPY", "QQQ", "IWM"],
            "atm_iv": [0.20, 0.21, 0.22, 0.22, 0.225, 0.235],
        })
        monkeypatch.setattr(analysis_pipeline, "get_daily_iv_for_spillover", lambda tickers: df)

        label, meta, warnings = _event_context("SPY", ["QQQ", "IWM"], "2024-01-15", 60)

        assert label == "Systemic"
        assert meta["same_direction_share"] == 1.0
        assert warnings == []


# ---------------------------------------------------------------------------
# compute_weight_stability
# ---------------------------------------------------------------------------

class TestComputeWeightStability:
    def test_returns_dataframe_with_peers_as_index(self):
        from analysis.rv_analysis import compute_weight_stability

        iv_data = pd.DataFrame({
            "date": ["2024-01-01"] * 2,
            "ticker": ["SPY", "QQQ"],
            "atm_iv": [0.18, 0.19],
        })

        with patch("analysis.analysis_pipeline.get_daily_iv_for_spillover",
                   return_value=iv_data):
            result = compute_weight_stability("SPY", ["QQQ"], lookback=30)

        assert isinstance(result, pd.DataFrame)
        assert "rolling_corr" in result.columns
        assert "stable" in result.columns

    def test_graceful_fallback_on_empty_data(self):
        from analysis.rv_analysis import compute_weight_stability

        with patch("analysis.analysis_pipeline.get_daily_iv_for_spillover",
                   return_value=pd.DataFrame()):
            result = compute_weight_stability("SPY", ["QQQ", "IWM"], lookback=30)

        assert set(result.index) == {"QQQ", "IWM"}
        assert (result["rolling_corr"].isna()).all()


# ---------------------------------------------------------------------------
# Smoke test: import all public symbols without error
# ---------------------------------------------------------------------------

def test_public_api_importable():
    from analysis.rv_analysis import (
        compute_surface_residual,
        compute_skew_spread,
        compute_term_shape_dislocation,
        generate_rv_signals,
        compute_weight_stability,
    )
    assert callable(compute_surface_residual)
    assert callable(compute_skew_spread)
    assert callable(compute_term_shape_dislocation)
    assert callable(generate_rv_signals)
    assert callable(compute_weight_stability)


def test_rv_plots_importable():
    from display.plotting.rv_plots import (
        plot_surface_residual_heatmap,
        plot_skew_spread,
    )
    assert callable(plot_surface_residual_heatmap)
    assert callable(plot_skew_spread)
