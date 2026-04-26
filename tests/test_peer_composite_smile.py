"""
Test for peer-composite smile plot functionality to prevent regression of flat line issue.
"""
import sqlite3
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.db_utils import ensure_initialized
from analysis.beta_builder import iv_surface_betas, peer_weights_from_correlations
from analysis.peer_composite_builder import build_surface_grids, combine_surfaces
from analysis.peer_smile_composite import build_peer_smile_composite


def test_peer_smile_composite_averages_fitted_peers_on_common_grid(monkeypatch):
    def make_peer(level, curvature):
        S = 100.0
        T = 30 / 365.25
        mny = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        iv = level + curvature * (mny - 1.0) ** 2
        return {
            "T_arr": np.full_like(mny, T, dtype=float),
            "K_arr": S * mny,
            "sigma_arr": iv,
            "S_arr": np.full_like(mny, S, dtype=float),
        }

    def fake_fit(_model, S, K, _T, IV):
        x = np.asarray(K, dtype=float) / float(S) - 1.0
        a, b, c = np.polyfit(x, np.asarray(IV, dtype=float), 2)
        return {"a": a, "b": b, "c": c}

    def fake_predict(_model, S, K, _T, params):
        x = np.asarray(K, dtype=float) / float(S) - 1.0
        return params["a"] * x**2 + params["b"] * x + params["c"]

    monkeypatch.setattr("analysis.peer_smile_composite.fit_valid_model_params", fake_fit)
    monkeypatch.setattr("analysis.peer_smile_composite.predict_model_iv", fake_predict)

    peers = {"P1": make_peer(0.20, 0.80), "P2": make_peer(0.30, 0.40)}
    out = build_peer_smile_composite(
        peers,
        {"P1": 0.25, "P2": 0.75},
        model="svi",
        target_T=30 / 365.25,
        moneyness_grid=(0.7, 1.3, 121),
    )

    grid = out["moneyness"]
    p1 = out["peer_curves"]["P1"]
    p2 = out["peer_curves"]["P2"]
    expected = 0.25 * p1 + 0.75 * p2

    assert len(grid) == 121
    assert np.allclose(out["iv"], expected)
    assert np.std(out["iv"]) > 1e-4
    assert np.all(out["iv"] >= np.minimum(p1, p2) - 1e-12)
    assert np.all(out["iv"] <= np.maximum(p1, p2) + 1e-12)
    assert out["envelope_ok"] is True
    assert out["skipped"] == {}


def create_smile_test_db():
    """Create test database with realistic volatility smile data across multiple dates."""
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    ensure_initialized(conn)
    
    # Create data for multiple dates to allow beta calculation
    dates = pd.date_range('2024-01-10', '2024-01-19', freq='D').strftime('%Y-%m-%d').tolist()
    
    rows = []
    for i, date in enumerate(dates):
        # Market volatility that varies over time
        market_vol = 0.20 + 0.05 * np.sin(i * 0.5)
        
        # SPY data with volatility smile
        spy_strikes = [450, 470, 480, 490, 510]
        spy_spot = 480.0
        for strike in spy_strikes:
            moneyness = strike / spy_spot
            log_mny = np.log(moneyness)
            # Create smile: higher vol at wings, lower at ATM
            smile_iv = market_vol + 0.04 * (log_mny ** 2) + 0.01 * np.random.normal()
            
            rows.append((
                date, 'SPY', '2024-02-16', float(strike), 'C',
                max(0.1, smile_iv), spy_spot, 0.084, moneyness, log_mny,
                0.5, 1 if strike == spy_spot else 0, 1000,
                5.0, 5.5, 5.25, None, None, None, None, None, None, None, None, None, "test"
            ))
        
        # QQQ data with different smile characteristics
        qqq_strikes = [370, 390, 400, 410, 430]
        qqq_spot = 400.0
        for strike in qqq_strikes:
            moneyness = strike / qqq_spot
            log_mny = np.log(moneyness)
            smile_iv = market_vol * 1.1 + 0.035 * (log_mny ** 2) + 0.008 * np.random.normal()
            
            rows.append((
                date, 'QQQ', '2024-02-16', float(strike), 'C',
                max(0.1, smile_iv), qqq_spot, 0.084, moneyness, log_mny,
                0.5, 1 if strike == qqq_spot else 0, 800,
                4.0, 4.5, 4.25, None, None, None, None, None, None, None, None, None, "test"
            ))
    
    conn.executemany(
        """
        INSERT INTO options_quotes (
            asof_date, ticker, expiry, strike, call_put,
            iv, spot, ttm_years, moneyness, log_moneyness, delta, is_atm,
            volume, bid, ask, mid,
            r, q, price, gamma, vega, theta, rho, d1, d2,
            vendor
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn


@pytest.fixture
def smile_test_db():
    """Pytest fixture for smile test database."""
    return create_smile_test_db()


def test_iv_surface_betas_no_crash(smile_test_db):
    """Test that iv_surface_betas doesn't crash with missing moneyness bins."""
    def mock_get_conn():
        return smile_test_db
    
    # Test with moneyness bins that may not exist in data
    surface_betas = iv_surface_betas(
        benchmark='SPY',
        tenors=(30,),
        mny_bins=((0.85, 0.95), (0.95, 1.05), (1.05, 1.15), (1.2, 1.3)),  # Last bin may be empty
        conn_fn=mock_get_conn
    )
    
    # Should not crash and should return some results
    assert isinstance(surface_betas, dict)
    # Should have at least some valid beta calculations
    valid_betas = sum(1 for series in surface_betas.values() if not series.empty and series.notna().any())
    assert valid_betas > 0, "Should have at least some valid surface betas"


def test_peer_composite_smile_variation(smile_test_db):
    """Test that peer-composite surfaces show smile variation, not flat lines."""
    def mock_get_conn():
        return smile_test_db
    
    with patch('data.db_utils.get_conn', mock_get_conn), \
         patch('analysis.peer_composite_builder.get_conn', mock_get_conn):
        
        # Test surface_grid mode (the one that was broken)
        weights = peer_weights_from_correlations(
            benchmark='SPY',
            peers=['QQQ'],
            mode='surface_grid',
            tenor_days=(30,),
            mny_bins=((0.85, 0.95), (0.95, 1.05), (1.05, 1.15)),
        )
        
        # Should get valid weights
        assert not weights.empty, "surface_grid mode should produce weights"
        assert abs(weights.sum() - 1.0) < 0.01, "Weights should sum to ~1.0"
        
        # Build surfaces
        surfaces = build_surface_grids(
            tickers=['SPY', 'QQQ'],
            tenors=(30,),
            mny_bins=((0.85, 0.95), (0.95, 1.05), (1.05, 1.15)),
            use_atm_only=False
        )
        
        # Create synthetic surface
        peer_surfaces = {'QQQ': surfaces['QQQ']}
        synthetic = combine_surfaces(peer_surfaces, weights.to_dict())
        
        assert synthetic, "Should create synthetic surfaces"
        
        # Get the latest synthetic surface
        latest_date = max(synthetic.keys())
        synth_surface = synthetic[latest_date]
        
        # Check that we have smile variation (not a flat line)
        for col in synth_surface.columns:
            col_values = synth_surface[col].dropna()
            if len(col_values) > 1:
                col_std = col_values.std()
                # Should have non-trivial variation (not flat line)
                assert col_std > 1e-6, f"Tenor {col} should show smile variation, not flat line (std={col_std})"
                
                # The range should be meaningful for volatility data
                col_range = col_values.max() - col_values.min()
                assert col_range > 1e-4, f"Tenor {col} should have meaningful vol range (range={col_range})"


def test_multiple_weight_modes_work(smile_test_db):
    """Test that all weight calculation modes work without producing flat lines."""
    def mock_get_conn():
        return smile_test_db
    
    modes = ['surface_grid', 'surface', 'iv_atm']
    
    with patch('data.db_utils.get_conn', mock_get_conn), \
         patch('analysis.peer_composite_builder.get_conn', mock_get_conn):
        
        for mode in modes:
            # Calculate weights
            weights = peer_weights_from_correlations(
                benchmark='SPY',
                peers=['QQQ'],
                mode=mode,
                tenor_days=(30,),
                mny_bins=((0.85, 0.95), (0.95, 1.05), (1.05, 1.15)),
            )
            
            assert not weights.empty, f"{mode} mode should produce weights"
            assert weights.sum() > 0.5, f"{mode} mode weights should be meaningful"
            
            # Test that synthetic surface has variation
            surfaces = build_surface_grids(
                tickers=['SPY', 'QQQ'],
                tenors=(30,),
                mny_bins=((0.85, 0.95), (0.95, 1.05), (1.05, 1.15)),
                use_atm_only=False
            )
            
            peer_surfaces = {'QQQ': surfaces['QQQ']}
            synthetic = combine_surfaces(peer_surfaces, weights.to_dict())
            
            if synthetic:
                latest_date = max(synthetic.keys())
                synth_surface = synthetic[latest_date]
                
                # Check for smile variation in each tenor
                has_variation = False
                for col in synth_surface.columns:
                    col_values = synth_surface[col].dropna()
                    if len(col_values) > 1:
                        col_std = col_values.std()
                        if col_std > 1e-6:
                            has_variation = True
                            break
                
                assert has_variation, f"{mode} mode should produce smile variation, not flat lines"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
