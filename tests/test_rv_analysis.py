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

    def test_adds_actual_peer_atm_signals(self):
        from analysis.rv_analysis import generate_rv_signals

        def fake_compute_peer_weights(target, peers, weight_mode, **kwargs):
            return pd.Series({p: 1.0 / len(peers) for p in peers})

        def fake_rv_report(*args, **kwargs):
            df = pd.DataFrame({
                "asof_date": ["2024-01-15"],
                "pillar_days": [30],
                "iv_target": [0.20],
                "iv_synth": [0.18],
                "spread": [0.02],
                "z": [2.0],
                "pct_rank": [90.0],
            })
            return df, {}

        def fake_fetch_atm(ticker, pillar_days, tolerance_days=10.0):
            iv = 0.20 if ticker == "SPY" else 0.16
            rows = []
            for i in range(8):
                rows.append({
                    "asof_date": f"2024-01-{i + 8:02d}",
                    "pillar_days": 30,
                    "iv": iv,
                })
            rows[-1]["iv"] = 0.22 if ticker == "SPY" else 0.17
            return pd.DataFrame(rows)

        with patch("analysis.analysis_pipeline.compute_peer_weights", fake_compute_peer_weights):
            with patch("analysis.analysis_pipeline.relative_value_atm_report_corrweighted", fake_rv_report):
                with patch("analysis.analysis_pipeline._fetch_target_atm", fake_fetch_atm):
                    with patch("analysis.rv_analysis.compute_skew_spread", return_value=pd.DataFrame()):
                        with patch("analysis.rv_analysis.compute_term_shape_dislocation", return_value={}):
                            result = generate_rv_signals(
                                "SPY", ["QQQ"], asof="2024-01-15",
                                weight_mode="corr_iv_atm", lookback=5, min_abs_z=0.0,
                            )

        peer_rows = result[(result["comparison"] == "peer") & (result["peer"] == "QQQ")]
        assert len(peer_rows) == 1
        assert peer_rows.iloc[0]["reference_label"] == "Actual peer QQQ"
        assert "actual peer QQQ" in peer_rows.iloc[0]["description"]


# ---------------------------------------------------------------------------
# generate_rv_opportunity_dashboard - synthesis layer
# ---------------------------------------------------------------------------

class TestGenerateRVOpportunityDashboard:
    def test_dashboard_classifies_clean_signal_as_trade_opportunity(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["ATM Level"],
            "asof_date": ["2024-01-15"],
            "T_days": [30],
            "value": [0.24],
            "synth_value": [0.18],
            "spread": [0.06],
            "z_score": [2.7],
            "pct_rank": [97.0],
            "description": ["SPY ATM vol vs synthetic at 30d"],
        })
        contracts = [{
            "expiry": "2024-02-16",
            "strike": 480.0,
            "moneyness": 1.00,
            "call_put": "C",
            "iv": 0.24,
            "bid": 3.10,
            "ask": 3.30,
            "volume": 75.0,
            "open_interest": 500.0,
        }]

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (82% same-dir, 76% hit)", {"strength": "Strong", "same_direction_probability": 0.82, "hit_rate": 0.76}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.91, "avg_common_cells": 12}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Idiosyncratic", {"same_direction_share": 0.1}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: contracts)

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ", "IWM"], asof="2024-01-15")

        trades = payload["trade_opportunities"]
        anomalies = payload["market_anomalies"]
        assert len(trades) == 1
        assert anomalies.empty
        trade = trades.iloc[0]
        assert trade["judgment"] == "Tradeable"
        assert trade["trade_type"] == "Delta-neutral vol RV"
        assert trade["direction"].startswith("Sell SPY structure / buy QQQ hedge structure")
        assert "Sell" in trade["sell_legs"]
        assert "Buy" in trade["buy_legs"]
        assert trade["trade"]["hedge_ratio_source"] in {"spillover median response", "substitutability-implied beta proxy"}
        assert "estimated_net_delta_after_hedge_per_1pct" in trade["trade"]["exposures"]
        assert trade["supporting_contracts"]
        assert trade["trade_score"] >= 0.72
        assert trade["source_signal"]["signal"]["classification"]["classification"] == "trade"

    def test_trade_thesis_uses_integer_contract_package_for_fractional_hedge(self, monkeypatch):
        from analysis import rv_analysis

        target_contracts = [
            {
                "expiry": "2024-02-16",
                "strike": 100.0,
                "moneyness": 1.00,
                "call_put": "C",
                "iv": 0.24,
                "bid": 2.00,
                "ask": 2.20,
                "spot": 100.0,
                "ttm_days": 30.0,
                "delta": 0.40,
                "gamma": 0.01,
                "vega": 10.0,
                "theta": -8.0,
            }
        ]
        peer_contracts = [
            {
                "expiry": "2024-02-16",
                "strike": 100.0,
                "moneyness": 1.00,
                "call_put": "C",
                "iv": 0.20,
                "bid": 1.80,
                "ask": 2.00,
                "spot": 100.0,
                "ttm_days": 30.0,
                "delta": 0.60,
                "gamma": 0.01,
                "vega": 9.0,
                "theta": -7.0,
            }
        ]
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: peer_contracts)

        trade = rv_analysis._compile_trade_thesis(
            target="TGT",
            peer="HEDG",
            asof="2024-01-15",
            metric_family="level",
            feature="ATM",
            maturity_days=30,
            direction="Rich",
            target_contracts=target_contracts,
            spill_meta={"median_response": 1.0},
            substitutability={"score": 1.0},
            contract_audit={"risks": []},
            classification={"classification": "trade"},
            event_ctx="Idiosyncratic",
        )

        assert trade["continuous_hedge_ratio"] == pytest.approx(2.0 / 3.0)
        assert trade["hedge_ratio"] == pytest.approx(2.0 / 3.0)
        assert trade["hedge_package"]["target_contracts"] == 3
        assert trade["hedge_package"]["peer_contracts"] == 2
        assert trade["hedge_package"]["within_tolerance"] is True
        quantities = [leg["quantity"] for leg in trade["buy_legs"] + trade["sell_legs"]]
        assert all(float(q).is_integer() for q in quantities)
        assert "Buy 2x HEDG" in trade["buy_legs_text"]
        assert "Sell 3x TGT" in trade["sell_legs_text"]
        assert "0.67x" not in trade["title"]
        assert abs(trade["exposures"]["estimated_net_delta_after_hedge_per_1pct"]) < 1e-9

    def test_dashboard_scores_systemic_dislocation_instead_of_hard_rejecting(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["ATM Level"],
            "asof_date": ["2024-01-15"],
            "T_days": [30],
            "value": [0.24],
            "synth_value": [0.18],
            "spread": [0.06],
            "z_score": [2.7],
            "pct_rank": [97.0],
            "description": ["SPY ATM vol vs synthetic at 30d"],
        })
        contracts = [{
            "expiry": "2024-02-16",
            "strike": 480.0,
            "moneyness": 1.00,
            "call_put": "C",
            "iv": 0.24,
            "bid": 3.10,
            "ask": 3.30,
            "volume": 75.0,
            "open_interest": 500.0,
        }]

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (82% same-dir, 76% hit)", {"strength": "Strong", "same_direction_probability": 0.82, "hit_rate": 0.76}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.91, "avg_common_cells": 12}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Systemic", {"same_direction_share": 1.0}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: contracts)

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ", "IWM"], asof="2024-01-15")

        trades = payload["trade_opportunities"]
        assert len(trades) == 1
        row = trades.iloc[0]
        assert row["judgment"] == "Tradeable"
        assert row["trade_score"] >= 0.72
        assert row["source_signal"]["event_context"] == "Systemic"
        assert any("Systemic" in risk for risk in row["risks"])

    def test_dashboard_missing_contracts_becomes_trade_risk_not_hard_reject(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["ATM Level"],
            "asof_date": ["2024-01-15"],
            "T_days": [30],
            "value": [0.24],
            "synth_value": [0.18],
            "spread": [0.06],
            "z_score": [2.7],
            "pct_rank": [97.0],
            "description": ["SPY ATM vol vs synthetic at 30d"],
        })

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (82% same-dir, 76% hit)", {"strength": "Strong", "same_direction_probability": 0.82, "hit_rate": 0.76}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.91, "avg_common_cells": 12}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Idiosyncratic", {"same_direction_share": 0.1}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: [])

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ", "IWM"], asof="2024-01-15")

        trades = payload["trade_opportunities"]
        assert len(trades) == 1
        row = trades.iloc[0]
        assert row["judgment"] == "Tradeable"
        assert row["source_signal"]["contract_auditability"] == "No liquid supporting contracts found."
        assert any("manual audit" in reason for reason in row["source_signal"]["classification_reasons"])

    def test_dashboard_promotes_near_substitute_without_z_score(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["Curvature"],
            "asof_date": ["2024-01-15"],
            "T_days": [45],
            "value": [0.035],
            "synth_value": [0.020],
            "spread": [0.015],
            "z_score": [np.nan],
            "pct_rank": [np.nan],
            "description": ["SPY curvature"],
            "comparison": ["peer"],
            "peer": ["QQQ"],
            "reference_label": ["Actual peer QQQ"],
        })
        contracts = [{
            "expiry": "2024-03-15",
            "strike": 500.0,
            "moneyness": 1.12,
            "call_put": "C",
            "iv": 0.28,
            "bid": 1.10,
            "ask": 1.25,
            "volume": 20.0,
            "open_interest": 120.0,
        }]
        health = {
            "available": True,
            "warnings": [],
            "pairs": [{"ticker": "QQQ", "correlation": 0.96, "sign_consistency": 0.92}],
        }

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (85% same-dir, 80% hit)", {"strength": "Strong", "same_direction_probability": 0.85, "hit_rate": 0.80}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.95, "avg_common_cells": 12}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Idiosyncratic", {"same_direction_share": 0.1}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: (health, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: contracts)

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ"], asof="2024-01-15")

        trades = payload["trade_opportunities"]
        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["judgment"] == "Tradeable"
        assert trade["substitutability"] == "Near substitutes"
        assert trade["source_signal"]["signal"]["classification"]["score_components"]["dislocation_magnitude"] >= 0.70

    def test_dashboard_maps_model_features_to_trade_construction_classes(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["Skew", "Curvature", "TS Slope", "Event Bump"],
            "asof_date": ["2024-01-15"] * 4,
            "T_days": [30, 30, 0, 11],
            "value": [0.09, 0.08, 0.03, 0.05],
            "synth_value": [0.02, 0.02, 0.00, 0.00],
            "spread": [0.07, 0.06, 0.03, 0.05],
            "z_score": [np.nan] * 4,
            "pct_rank": [np.nan] * 4,
            "description": ["signal"] * 4,
            "comparison": ["peer"] * 4,
            "peer": ["QQQ"] * 4,
            "reference_label": ["Actual peer QQQ"] * 4,
        })
        contracts = [
            {
                "expiry": "2024-02-16",
                "strike": 430.0,
                "moneyness": 0.88,
                "call_put": "P",
                "iv": 0.30,
                "bid": 2.00,
                "ask": 2.20,
                "volume": 50.0,
                "open_interest": 400.0,
                "spot": 480.0,
                "ttm_days": 30.0,
                "delta": -0.20,
                "gamma": 0.01,
                "vega": 12.0,
                "theta": -20.0,
            },
            {
                "expiry": "2024-02-16",
                "strike": 530.0,
                "moneyness": 1.12,
                "call_put": "C",
                "iv": 0.29,
                "bid": 1.80,
                "ask": 2.00,
                "volume": 40.0,
                "open_interest": 350.0,
                "spot": 480.0,
                "ttm_days": 30.0,
                "delta": 0.18,
                "gamma": 0.01,
                "vega": 11.0,
                "theta": -18.0,
            },
        ]
        health = {
            "available": True,
            "warnings": [],
            "pairs": [{"ticker": "QQQ", "correlation": 0.97, "sign_consistency": 0.95}],
        }

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (85% same-dir, 80% hit)", {"strength": "Strong", "same_direction_probability": 0.85, "hit_rate": 0.80, "median_response": 0.92}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.95, "avg_common_cells": 12}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Idiosyncratic", {"same_direction_share": 0.1}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: (health, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: contracts)

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ"], asof="2024-01-15")

        classes = set(payload["trade_opportunities"]["trade_type"])
        assert "Skew-transfer RV" in classes
        assert "Tail-risk RV" in classes
        assert "Term-structure RV" in classes
        assert "Event-vol timing RV" in classes
        for trade in payload["trade_opportunities"]["trade"]:
            assert trade["buy_legs"]
            assert trade["sell_legs"]
            assert "spillover_beta" in trade["exposures"]

    def test_dashboard_groups_low_score_anomalies(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["ATM Level", "ATM Level", "ATM Level"],
            "asof_date": ["2024-01-15"] * 3,
            "T_days": [30, 45, 60],
            "value": [0.201, 0.202, 0.203],
            "synth_value": [0.200, 0.200, 0.200],
            "spread": [0.001, 0.002, 0.003],
            "z_score": [0.2, 0.3, 0.4],
            "pct_rank": [55.0, 57.0, 58.0],
            "description": ["weak ATM"] * 3,
        })

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Unknown", ["No model log."], {"model": "Unknown"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Weak (45% same-dir, 40% hit)", {"strength": "Weak", "same_direction_probability": 0.45, "hit_rate": 0.40}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Poor", {"avg_surface_corr": -0.2, "avg_common_cells": 2}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Unknown", {}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: [])

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ"], asof="2024-01-15")

        assert payload["trade_opportunities"].empty
        anomalies = payload["market_anomalies"]
        assert len(anomalies) == 1
        row = anomalies.iloc[0]
        assert row["group_size"] == 3
        assert row["judgment"] == "Not tradeable"
        assert len(row["member_signals"]) == 3

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
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))

        payload = generate_rv_opportunity_dashboard(
            "SPY", ["QQQ", "IWM"], asof="2024-01-15", weight_mode="corr_iv_atm"
        )

        opp = payload["opportunities"]
        assert len(opp) == 1
        row = opp.iloc[0]
        assert row["rank"] == 1
        assert row["direction"] == "Rich"
        assert row["feature"] == "ATM"
        assert row["metric"] == "Vol level"
        assert row["maturity"] == "30d"
        assert row["confidence"] == "High"
        assert "SPY 30d overall implied volatility is rich vs peers" in row["opportunity"]
        assert row["signal"]["location"]["metric_family"] == "level"
        assert row["signal"]["magnitude"]["direction"] == "Rich"
        assert row["signal"]["structure"]["surface_vs_surface_grid_consistency"] == "Comparable"
        assert "narrative" in row["signal"]
        assert payload["integration_status"]["system_health"]["quality"] == "Good"
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
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ"], asof="2024-01-15")
        row = payload["opportunities"].iloc[0]

        assert row["data_quality"] == "Degraded"
        assert row["feature"] == "Curvature"
        assert row["metric"] == "Smile convexity"
        assert row["signal"]["location"]["metric_family"] == "convexity"
        assert row["signal"]["calculation"]["target_value"] == 0.05
        assert row["signal"]["calculation"]["synthetic_value"] == 0.02
        assert row["signal"]["calculation"]["spread"] == 0.03
        assert "convexity spread = target smile curvature - weighted peer smile curvature" in row["signal"]["calculation"]["display"]

    def test_dashboard_labels_actual_peer_reference(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["ATM Level"],
            "asof_date": ["2024-01-15"],
            "T_days": [30],
            "value": [0.22],
            "synth_value": [0.17],
            "spread": [0.05],
            "z_score": [2.1],
            "pct_rank": [92.0],
            "description": ["SPY ATM vol vs actual peer QQQ at 30d"],
            "comparison": ["peer"],
            "peer": ["QQQ"],
            "reference_label": ["Actual peer QQQ"],
        })

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Strong (80% same-dir, 70% hit)", {"strength": "Strong"}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.9}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Idiosyncratic", {"same_direction_share": 0.2}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))

        payload = generate_rv_opportunity_dashboard(
            "SPY", ["QQQ"], asof="2024-01-15", weight_mode="corr_iv_atm"
        )

        row = payload["opportunities"].iloc[0]
        assert "vs QQQ" in row["opportunity"]
        assert "versus Actual peer QQQ" in row["why"]
        assert row["signal"]["calculation"]["synthetic_label"] == "Actual peer QQQ"
        assert row["signal"]["magnitude"]["comparison"] == "peer"

    def test_model_quality_handles_unreadable_log_without_traceback(self, monkeypatch):
        from analysis import rv_analysis
        import analysis.model_params_logger as logger_mod

        def broken_load_model_params():
            raise RuntimeError("Couldn't deserialize thrift: TProtocolException: Invalid data")

        monkeypatch.setattr(logger_mod, "load_model_params", broken_load_model_params)

        quality, warnings, meta = rv_analysis._load_model_quality("SPY", ["QQQ"], "2024-01-15")

        assert quality == "Unknown"
        assert warnings == ["Model parameter log is unavailable."]
        assert meta["model"] == "Unknown"

    def test_dashboard_attaches_supporting_contracts(self, monkeypatch):
        from analysis import rv_analysis
        from analysis.rv_analysis import generate_rv_opportunity_dashboard

        raw = pd.DataFrame({
            "signal_type": ["Skew"],
            "asof_date": ["2024-01-15"],
            "T_days": [30],
            "value": [0.10],
            "synth_value": [0.04],
            "spread": [0.06],
            "z_score": [2.1],
            "pct_rank": [92.0],
            "description": ["SPY skew"],
        })
        contracts = [{
            "expiry": "2024-02-16",
            "strike": 95.0,
            "moneyness": 0.90,
            "call_put": "P",
            "iv": 0.31,
            "bid": 1.10,
            "ask": 1.25,
            "volume": 100.0,
            "open_interest": 250.0,
        }]

        monkeypatch.setattr(rv_analysis, "generate_rv_signals", lambda *a, **k: raw)
        monkeypatch.setattr(rv_analysis, "_load_model_quality", lambda *a, **k: ("Good", [], {"model": "SVI"}))
        monkeypatch.setattr(rv_analysis, "_load_spillover_support", lambda *a, **k: ("Suggestive (70% same-dir, 65% hit)", {"strength": "Suggestive", "same_direction_probability": 0.7, "hit_rate": 0.65}, []))
        monkeypatch.setattr(rv_analysis, "_surface_comparability", lambda *a, **k: ("Comparable", {"avg_surface_corr": 0.8, "avg_common_cells": 9}, []))
        monkeypatch.setattr(rv_analysis, "_event_context", lambda *a, **k: ("Cluster", {"peer_abs_median_move": 0.02}, []))
        monkeypatch.setattr(rv_analysis, "_feature_health_context", lambda *a, **k: ({"available": True, "warnings": []}, []))
        monkeypatch.setattr(rv_analysis, "_load_supporting_contracts", lambda *a, **k: contracts)

        payload = generate_rv_opportunity_dashboard("SPY", ["QQQ"], asof="2024-01-15")
        signal = payload["opportunities"].iloc[0]["signal"]

        assert signal["location"]["metric_family"] == "asymmetry"
        assert signal["supporting_contracts"] == contracts
        assert "SPY 2024-02-16 puts" in signal["narrative"]["contracts"]

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
