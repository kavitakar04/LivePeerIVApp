"""
Unit tests for surface building configuration values.

These tests exercise the same underlying functions the GUI calls to experiment
with how configuration values influence computations without using the GUI.
"""
import sqlite3
import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch
from typing import Dict
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.analysis_pipeline import PipelineConfig, build_surfaces
from analysis.peer_composite_builder import build_surface_grids
from data.db_utils import ensure_initialized


# Sample test data for options quotes
SAMPLE_OPTIONS_DATA = [
    # SPY data - mix of ATM and non-ATM options
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY', 
        'expiry': '2024-02-16',
        'strike': 480.0,
        'call_put': 'C',
        'iv': 0.18,
        'spot': 480.0,
        'ttm_years': 0.084,  # ~30 days
        'moneyness': 1.0,
        'log_moneyness': 0.0,
        'delta': 0.5,
        'is_atm': 1,
        'volume': 1000,
        'bid': 8.0,
        'ask': 8.5,
        'mid': 8.25
    },
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY',
        'expiry': '2024-02-16', 
        'strike': 460.0,
        'call_put': 'C',
        'iv': 0.22,
        'spot': 480.0,
        'ttm_years': 0.084,
        'moneyness': 0.958,  # OTM
        'log_moneyness': -0.042,
        'delta': 0.35,
        'is_atm': 0,
        'volume': 500,
        'bid': 22.0,
        'ask': 22.5,
        'mid': 22.25
    },
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY',
        'expiry': '2024-02-16',
        'strike': 500.0, 
        'call_put': 'C',
        'iv': 0.20,
        'spot': 480.0,
        'ttm_years': 0.084,
        'moneyness': 1.042,  # ITM
        'log_moneyness': 0.041,
        'delta': 0.65,
        'is_atm': 0,
        'volume': 750,
        'bid': 2.0,
        'ask': 2.5,
        'mid': 2.25
    },
    # Longer expiry data
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY',
        'expiry': '2024-04-15',
        'strike': 480.0,
        'call_put': 'C', 
        'iv': 0.19,
        'spot': 480.0,
        'ttm_years': 0.247,  # ~90 days
        'moneyness': 1.0,
        'log_moneyness': 0.0,
        'delta': 0.52,
        'is_atm': 1,
        'volume': 800,
        'bid': 15.0,
        'ask': 15.5,
        'mid': 15.25
    },
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY',
        'expiry': '2024-04-15',
        'strike': 460.0,
        'call_put': 'C',
        'iv': 0.23,
        'spot': 480.0, 
        'ttm_years': 0.247,
        'moneyness': 0.958,
        'log_moneyness': -0.042,
        'delta': 0.38,
        'is_atm': 0,
        'volume': 400,
        'bid': 26.0,
        'ask': 26.5,
        'mid': 26.25
    },
    # Additional expiry for max_expiries testing
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY',
        'expiry': '2024-07-15',
        'strike': 480.0,
        'call_put': 'C',
        'iv': 0.21,
        'spot': 480.0,
        'ttm_years': 0.493,  # ~180 days
        'moneyness': 1.0,
        'log_moneyness': 0.0,
        'delta': 0.54,
        'is_atm': 1,
        'volume': 600,
        'bid': 22.0,
        'ask': 22.5,
        'mid': 22.25
    }
]


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database with sample options data."""
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    
    # Initialize the database schema
    ensure_initialized(conn)
    
    # Insert sample data
    rows = []
    for q in SAMPLE_OPTIONS_DATA:
        rows.append((
            q["asof_date"], q["ticker"], q["expiry"], float(q["strike"]), q["call_put"],
            q.get("iv"), q.get("spot"), q.get("ttm_years"), q.get("moneyness"), 
            q.get("log_moneyness"), q.get("delta"),
            1 if q.get("is_atm") else 0,
            q.get("volume"), q.get("bid"), q.get("ask"), q.get("mid"),
            q.get("r"), q.get("q"), q.get("price"), q.get("gamma"), 
            q.get("vega"), q.get("theta"), q.get("rho"), q.get("d1"), q.get("d2"),
            q.get("vendor", "yfinance"),
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


def test_use_atm_only_filters_rows(test_db):
    """Test that use_atm_only=True filters to only ATM options."""
    # Patch the get_conn function to use our test database
    with patch('analysis.peer_composite_builder.get_conn', return_value=test_db):
        # Build surfaces with all options
        cfg_all = PipelineConfig(use_atm_only=False)
        surf_all = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_all.tenors,
            'mny_bins': cfg_all.mny_bins,  
            'use_atm_only': cfg_all.use_atm_only,
            'max_expiries': cfg_all.max_expiries
        })
        
        # Build surfaces with ATM-only filtering
        cfg_atm = PipelineConfig(use_atm_only=True)
        surf_atm = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_atm.tenors,
            'mny_bins': cfg_atm.mny_bins,
            'use_atm_only': cfg_atm.use_atm_only,
            'max_expiries': cfg_atm.max_expiries
        })
        
        # Check that we got results
        assert "SPY" in surf_all
        assert "SPY" in surf_atm
        
        # Get sample date present in both (should be our test date)
        date = pd.to_datetime('2024-01-15')
        assert date in surf_all["SPY"]
        assert date in surf_atm["SPY"]
        
        # ATM-only should have fewer or equal data points since it's more restrictive
        all_data_points = surf_all["SPY"][date].notna().sum().sum()
        atm_data_points = surf_atm["SPY"][date].notna().sum().sum()
        
        assert atm_data_points <= all_data_points
        print(f"All options data points: {all_data_points}")
        print(f"ATM-only data points: {atm_data_points}")


def test_fit_sampled_surface_grid_is_dense_on_requested_axes(test_db, monkeypatch):
    """Fit-sampled surface grids should populate requested tenors and K/S bins."""
    tenors = (7, 14, 21, 28)
    mny_bins = ((0.90, 1.00), (1.00, 1.10))

    def fake_fit(model, S, K, T, IV):
        return {"spot": S, "T": T}, {"ok": True, "n": len(K)}

    def fake_predict(model, S, K, T, params):
        return 0.20 + 0.01 * (np.asarray(K, dtype=float) / float(S)) + 0.001 * float(T)

    monkeypatch.setattr("analysis.model_fit_service.fit_valid_model_result", fake_fit)
    monkeypatch.setattr("analysis.model_fit_service.predict_model_iv", fake_predict)

    with patch("analysis.peer_composite_builder.get_conn", return_value=test_db):
        surf = build_surface_grids(
            tickers=["SPY"],
            tenors=tenors,
            mny_bins=mny_bins,
            max_expiries=2,
            surface_source="fit",
            model="svi",
        )

    date = pd.to_datetime("2024-01-15")
    grid = surf["SPY"][date]
    assert list(grid.columns) == list(tenors)
    assert list(grid.index) == ["0.90-1.00", "1.00-1.10"]
    assert grid.shape == (len(mny_bins), len(tenors))
    assert grid.notna().sum().sum() == len(mny_bins) * len(tenors)


def test_surface_feature_matrix_fit_mode_counts_full_grid(test_db, monkeypatch):
    """Surface-grid feature n should reflect all requested fit-sampled cells."""
    from analysis.analysis_pipeline import get_surface_grids_cached
    from analysis.unified_weights import surface_feature_matrix

    tenors = (7, 14, 21, 28)
    mny_bins = ((0.90, 1.00), (1.00, 1.10))

    def fake_fit(model, S, K, T, IV):
        return {"spot": S, "T": T}, {"ok": True, "n": len(K)}

    def fake_predict(model, S, K, T, params):
        return 0.25 + 0.02 * (np.asarray(K, dtype=float) / float(S)) + 0.001 * float(T)

    monkeypatch.setattr("analysis.model_fit_service.fit_valid_model_result", fake_fit)
    monkeypatch.setattr("analysis.model_fit_service.predict_model_iv", fake_predict)
    get_surface_grids_cached.cache_clear()

    with patch("analysis.peer_composite_builder.get_conn", return_value=test_db):
        _grids, X, names = surface_feature_matrix(
            ["SPY"],
            "2024-01-15",
            tenors=tenors,
            mny_bins=mny_bins,
            surface_source="fit",
            model="svi",
            max_expiries=2,
            standardize=False,
        )

    assert X.shape == (1, len(tenors) * len(mny_bins))
    assert len(names) == len(tenors) * len(mny_bins)


def test_tenor_bins_configuration(test_db):
    """Test that different tenor bin configurations affect results."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=test_db):
        # Default tenors
        cfg_default = PipelineConfig()
        surf_default = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_default.tenors,
            'mny_bins': cfg_default.mny_bins,
            'use_atm_only': cfg_default.use_atm_only,
            'max_expiries': cfg_default.max_expiries
        })
        
        # Custom tenors - fewer bins
        cfg_custom = PipelineConfig(tenors=(30, 90))
        surf_custom = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_custom.tenors,
            'mny_bins': cfg_custom.mny_bins,
            'use_atm_only': cfg_custom.use_atm_only,
            'max_expiries': cfg_custom.max_expiries
        })
        
        # Check that we got results
        assert "SPY" in surf_default
        assert "SPY" in surf_custom
        
        date = pd.to_datetime('2024-01-15')
        if date in surf_default["SPY"] and date in surf_custom["SPY"]:
            # Custom config with fewer tenors should have fewer columns
            default_cols = len(surf_default["SPY"][date].columns)
            custom_cols = len(surf_custom["SPY"][date].columns)
            
            print(f"Default tenor columns: {default_cols}")
            print(f"Custom tenor columns: {custom_cols}")
            
            # Custom should have at most the same number of columns (likely fewer)
            assert custom_cols <= default_cols


def test_moneyness_bins_configuration(test_db):
    """Test that different moneyness bin configurations affect results."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=test_db):
        # Default moneyness bins
        cfg_default = PipelineConfig()
        surf_default = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_default.tenors,
            'mny_bins': cfg_default.mny_bins,
            'use_atm_only': cfg_default.use_atm_only,
            'max_expiries': cfg_default.max_expiries
        })
        
        # Custom moneyness bins - fewer bins
        cfg_custom = PipelineConfig(mny_bins=((0.90, 1.10),))
        surf_custom = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_custom.tenors,
            'mny_bins': cfg_custom.mny_bins,
            'use_atm_only': cfg_custom.use_atm_only,
            'max_expiries': cfg_custom.max_expiries
        })
        
        # Check that we got results
        assert "SPY" in surf_default
        assert "SPY" in surf_custom
        
        date = pd.to_datetime('2024-01-15')
        if date in surf_default["SPY"] and date in surf_custom["SPY"]:
            # Custom config with fewer moneyness bins should have fewer rows
            default_rows = len(surf_default["SPY"][date])
            custom_rows = len(surf_custom["SPY"][date])
            
            print(f"Default moneyness rows: {default_rows}")
            print(f"Custom moneyness rows: {custom_rows}")
            
            # Custom should have fewer or equal rows
            assert custom_rows <= default_rows


def test_max_expiries_limits_data(test_db):
    """Test that max_expiries parameter limits the number of expiration dates."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=test_db):
        # No limit on expiries
        cfg_unlimited = PipelineConfig(max_expiries=None)
        surf_unlimited = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_unlimited.tenors,
            'mny_bins': cfg_unlimited.mny_bins,
            'use_atm_only': cfg_unlimited.use_atm_only,
            'max_expiries': cfg_unlimited.max_expiries
        })
        
        # Limit to 2 expiries
        cfg_limited = PipelineConfig(max_expiries=2)
        surf_limited = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_limited.tenors,
            'mny_bins': cfg_limited.mny_bins,
            'use_atm_only': cfg_limited.use_atm_only,
            'max_expiries': cfg_limited.max_expiries
        })
        
        # Check that we got results
        assert "SPY" in surf_unlimited
        assert "SPY" in surf_limited
        
        date = pd.to_datetime('2024-01-15')
        if date in surf_unlimited["SPY"] and date in surf_limited["SPY"]:
            # Limited should have fewer or equal data points
            unlimited_data = surf_unlimited["SPY"][date].notna().sum().sum()
            limited_data = surf_limited["SPY"][date].notna().sum().sum()
            
            print(f"Unlimited expiries data points: {unlimited_data}")
            print(f"Limited expiries data points: {limited_data}")
            
            assert limited_data <= unlimited_data


def test_build_surfaces_with_config(test_db):
    """Test the higher-level build_surfaces function with PipelineConfig."""
    with patch('analysis.analysis_pipeline._get_ro_conn', return_value=test_db):
        with patch('analysis.peer_composite_builder.get_conn', return_value=test_db):
            # Test with baseline config
            cfg_baseline = PipelineConfig(use_atm_only=False)
            surfaces_baseline = build_surfaces(tickers=["SPY"], cfg=cfg_baseline)
            
            # Test with altered config
            cfg_altered = PipelineConfig(use_atm_only=True, max_expiries=1)
            surfaces_altered = build_surfaces(tickers=["SPY"], cfg=cfg_altered)
            
            # Both should return results
            assert "SPY" in surfaces_baseline
            assert "SPY" in surfaces_altered
            
            # Check if we have data for our test date
            date = pd.to_datetime('2024-01-15')
            baseline_has_date = date in surfaces_baseline["SPY"]
            altered_has_date = date in surfaces_altered["SPY"]
            
            print(f"Baseline config has test date: {baseline_has_date}")
            print(f"Altered config has test date: {altered_has_date}")
            
            if baseline_has_date and altered_has_date:
                # Altered config should be more restrictive
                baseline_data = surfaces_baseline["SPY"][date].notna().sum().sum()
                altered_data = surfaces_altered["SPY"][date].notna().sum().sum()
                
                print(f"Baseline data points: {baseline_data}")
                print(f"Altered data points: {altered_data}")
                
                assert altered_data <= baseline_data


def test_configuration_isolation(test_db):
    """Test that different configurations produce isolated, predictable results."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=test_db):
        # Configuration 1: Very restrictive
        cfg_restrictive = PipelineConfig(
            tenors=(30, 90),
            mny_bins=((0.95, 1.05),),
            use_atm_only=True,
            max_expiries=1
        )
        
        # Configuration 2: More permissive  
        cfg_permissive = PipelineConfig(
            tenors=(7, 30, 60, 90, 180, 365),
            mny_bins=((0.80, 0.90), (0.95, 1.05), (1.10, 1.25)),
            use_atm_only=False,
            max_expiries=None
        )
        
        # Build surfaces with both configurations
        surf_restrictive = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_restrictive.tenors,
            'mny_bins': cfg_restrictive.mny_bins,
            'use_atm_only': cfg_restrictive.use_atm_only,
            'max_expiries': cfg_restrictive.max_expiries
        })
        
        surf_permissive = build_surface_grids(tickers=["SPY"], **{
            'tenors': cfg_permissive.tenors,
            'mny_bins': cfg_permissive.mny_bins,
            'use_atm_only': cfg_permissive.use_atm_only,
            'max_expiries': cfg_permissive.max_expiries
        })
        
        # Both should return results for SPY
        assert "SPY" in surf_restrictive
        assert "SPY" in surf_permissive
        
        date = pd.to_datetime('2024-01-15')
        if date in surf_restrictive["SPY"] and date in surf_permissive["SPY"]:
            # Restrictive should have fewer data points than permissive
            restrictive_data = surf_restrictive["SPY"][date].notna().sum().sum()
            permissive_data = surf_permissive["SPY"][date].notna().sum().sum()
            
            print(f"Restrictive config data points: {restrictive_data}")
            print(f"Permissive config data points: {permissive_data}")
            
            # The key assertion: restrictive config should yield fewer or equal data points
            assert restrictive_data <= permissive_data
            
            # Also check dimensions
            restrictive_shape = surf_restrictive["SPY"][date].shape
            permissive_shape = surf_permissive["SPY"][date].shape
            
            print(f"Restrictive shape: {restrictive_shape}")
            print(f"Permissive shape: {permissive_shape}")
            
            # Restrictive should have fewer or equal rows/cols due to fewer bins
            assert restrictive_shape[0] <= permissive_shape[0]  # rows (moneyness bins)
            assert restrictive_shape[1] <= permissive_shape[1]  # cols (tenor bins)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
