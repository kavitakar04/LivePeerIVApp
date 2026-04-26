"""
Additional tests for configuration edge cases and peer-composite functionality.

Tests edge cases and more complex scenarios for configuration-driven surface building.
"""
import sqlite3
import pandas as pd
import pytest
from unittest.mock import patch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.analysis_pipeline import PipelineConfig, build_synthetic_surface
from analysis.peer_composite_builder import build_surface_grids, combine_surfaces
from data.db_utils import ensure_initialized


# More comprehensive test data including multiple tickers
MULTI_TICKER_DATA = [
    # SPY data
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY', 
        'expiry': '2024-02-16',
        'strike': 480.0,
        'call_put': 'C',
        'iv': 0.18,
        'spot': 480.0,
        'ttm_years': 0.084,
        'moneyness': 1.0,
        'is_atm': 1,
        'volume': 1000
    },
    {
        'asof_date': '2024-01-15',
        'ticker': 'SPY',
        'expiry': '2024-04-15',
        'strike': 480.0,
        'call_put': 'C',
        'iv': 0.19,
        'spot': 480.0,
        'ttm_years': 0.247,
        'moneyness': 1.0,
        'is_atm': 1,
        'volume': 800
    },
    # QQQ data
    {
        'asof_date': '2024-01-15',
        'ticker': 'QQQ',
        'expiry': '2024-02-16', 
        'strike': 380.0,
        'call_put': 'C',
        'iv': 0.25,
        'spot': 380.0,
        'ttm_years': 0.084,
        'moneyness': 1.0,
        'is_atm': 1,
        'volume': 1500
    },
    {
        'asof_date': '2024-01-15',
        'ticker': 'QQQ',
        'expiry': '2024-04-15',
        'strike': 380.0,
        'call_put': 'C',
        'iv': 0.26,
        'spot': 380.0,
        'ttm_years': 0.247,
        'moneyness': 1.0,
        'is_atm': 1,
        'volume': 1200
    },
    # IWM data
    {
        'asof_date': '2024-01-15',
        'ticker': 'IWM',
        'expiry': '2024-02-16',
        'strike': 200.0,
        'call_put': 'C', 
        'iv': 0.28,
        'spot': 200.0,
        'ttm_years': 0.084,
        'moneyness': 1.0,
        'is_atm': 1,
        'volume': 900
    }
]


@pytest.fixture
def multi_ticker_db():
    """Create an in-memory SQLite database with multi-ticker sample data."""
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    
    # Initialize the database schema
    ensure_initialized(conn)
    
    # Insert sample data
    rows = []
    for q in MULTI_TICKER_DATA:
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


def test_multiple_tickers_configuration(multi_ticker_db):
    """Test configuration affects multiple tickers consistently."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=multi_ticker_db):
        # Test with multiple tickers
        tickers = ["SPY", "QQQ", "IWM"]
        
        # Baseline configuration
        cfg_baseline = PipelineConfig(use_atm_only=False)
        surf_baseline = build_surface_grids(tickers=tickers, **{
            'tenors': cfg_baseline.tenors,
            'mny_bins': cfg_baseline.mny_bins,
            'use_atm_only': cfg_baseline.use_atm_only,
            'max_expiries': cfg_baseline.max_expiries
        })
        
        # ATM-only configuration
        cfg_atm = PipelineConfig(use_atm_only=True)
        surf_atm = build_surface_grids(tickers=tickers, **{
            'tenors': cfg_atm.tenors,
            'mny_bins': cfg_atm.mny_bins,
            'use_atm_only': cfg_atm.use_atm_only,
            'max_expiries': cfg_atm.max_expiries
        })
        
        # Check that all tickers are present in both configurations
        for ticker in tickers:
            assert ticker in surf_baseline, f"Missing {ticker} in baseline"
            assert ticker in surf_atm, f"Missing {ticker} in ATM-only"
            
        print(f"Baseline tickers: {list(surf_baseline.keys())}")
        print(f"ATM-only tickers: {list(surf_atm.keys())}")
        
        # Verify configuration consistency across tickers
        date = pd.to_datetime('2024-01-15')
        for ticker in tickers:
            if ticker in surf_baseline and ticker in surf_atm:
                if date in surf_baseline[ticker] and date in surf_atm[ticker]:
                    baseline_data = surf_baseline[ticker][date].notna().sum().sum()
                    atm_data = surf_atm[ticker][date].notna().sum().sum()
                    
                    print(f"{ticker} - Baseline: {baseline_data}, ATM-only: {atm_data}")
                    # ATM filtering should be consistent across all tickers
                    assert atm_data <= baseline_data


def test_empty_configuration_results():
    """Test behavior with configurations that might return empty results."""
    # Create minimal database with just one data point
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    ensure_initialized(conn)
    
    # Add single data point
    conn.execute("""
        INSERT INTO options_quotes (
            asof_date, ticker, expiry, strike, call_put,
            iv, spot, ttm_years, moneyness, is_atm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ('2024-01-15', 'TEST', '2024-02-16', 100.0, 'C', 0.20, 100.0, 0.084, 1.0, 0))
    conn.commit()
    
    with patch('analysis.peer_composite_builder.get_conn', return_value=conn):
        # Configuration that should return no results (ATM-only but no ATM data)
        cfg_empty = PipelineConfig(use_atm_only=True)
        surf_empty = build_surface_grids(tickers=["TEST"], **{
            'tenors': cfg_empty.tenors,
            'mny_bins': cfg_empty.mny_bins,
            'use_atm_only': cfg_empty.use_atm_only,
            'max_expiries': cfg_empty.max_expiries
        })
        
        # Should handle empty results gracefully
        if "TEST" in surf_empty:
            # If ticker is present, it should have valid structure
            assert isinstance(surf_empty["TEST"], dict)
            print(f"Empty config result for TEST: {len(surf_empty['TEST'])} dates")


def test_synthetic_surface_with_config(multi_ticker_db):
    """Test peer-composite construction with different configurations."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=multi_ticker_db):
        with patch('analysis.analysis_pipeline._get_ro_conn', return_value=multi_ticker_db):
            # Create synthetic surface with equal weights
            weights = {"SPY": 0.5, "QQQ": 0.5}
            
            # Test with baseline config
            cfg_baseline = PipelineConfig(use_atm_only=False)
            synthetic_baseline = build_synthetic_surface(
                weights=weights,
                cfg=cfg_baseline,
                most_recent_only=True
            )
            
            # Test with ATM-only config
            cfg_atm = PipelineConfig(use_atm_only=True)
            synthetic_atm = build_synthetic_surface(
                weights=weights,
                cfg=cfg_atm,
                most_recent_only=True
            )
            
            # Both should return dict[timestamp, DataFrame] structure
            assert isinstance(synthetic_baseline, dict)
            assert isinstance(synthetic_atm, dict)
            
            print(f"Synthetic baseline dates: {len(synthetic_baseline)}")
            print(f"Peer composite ATM-only dates: {len(synthetic_atm)}")
            
            # If both have data, compare structure
            if synthetic_baseline and synthetic_atm:
                sample_date = next(iter(synthetic_baseline))
                if sample_date in synthetic_atm:
                    baseline_shape = synthetic_baseline[sample_date].shape
                    atm_shape = synthetic_atm[sample_date].shape
                    
                    print(f"Synthetic baseline shape: {baseline_shape}")  
                    print(f"Peer composite ATM shape: {atm_shape}")
                    
                    # Should have same structure but potentially different values
                    assert baseline_shape == atm_shape


def test_combine_surfaces_functionality(multi_ticker_db):
    """Test the combine_surfaces function with different configurations."""
    with patch('analysis.peer_composite_builder.get_conn', return_value=multi_ticker_db):
        # Build surfaces for individual tickers
        tickers = ["SPY", "QQQ"]
        cfg = PipelineConfig(use_atm_only=True)  # Use ATM-only for cleaner data
        
        surfaces = build_surface_grids(tickers=tickers, **{
            'tenors': cfg.tenors,
            'mny_bins': cfg.mny_bins,
            'use_atm_only': cfg.use_atm_only,
            'max_expiries': cfg.max_expiries
        })
        
        # Define weights for combination
        weights = {"SPY": 0.6, "QQQ": 0.4}
        
        # Combine surfaces
        combined = combine_surfaces(surfaces, weights)
        
        # Should return dict[timestamp, DataFrame]
        assert isinstance(combined, dict)
        
        if combined:
            sample_date = next(iter(combined))
            combined_surface = combined[sample_date]
            
            print(f"Combined surface shape: {combined_surface.shape}")
            print(f"Combined surface data points: {combined_surface.notna().sum().sum()}")
            
            # Combined surface should have same structure as individual surfaces
            if "SPY" in surfaces and sample_date in surfaces["SPY"]:
                spy_shape = surfaces["SPY"][sample_date].shape
                assert combined_surface.shape == spy_shape


def test_configuration_caching_behavior():
    """Test that configuration changes properly invalidate caches."""
    # This is mainly to ensure our cached functions work properly
    from analysis.analysis_pipeline import get_surface_grids_cached
    
    # Create test database
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys=ON;")
    ensure_initialized(conn)
    
    # Add test data
    conn.execute("""
        INSERT INTO options_quotes (
            asof_date, ticker, expiry, strike, call_put,
            iv, spot, ttm_years, moneyness, is_atm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ('2024-01-15', 'TEST', '2024-02-16', 100.0, 'C', 0.20, 100.0, 0.084, 1.0, 1))
    conn.commit()
    
    with patch('analysis.peer_composite_builder.get_conn', return_value=conn):
        # Test different configurations
        cfg1 = PipelineConfig(use_atm_only=True)
        cfg2 = PipelineConfig(use_atm_only=False)
        
        # These should be cached separately due to different configs
        result1 = get_surface_grids_cached(cfg1, "TEST")
        result2 = get_surface_grids_cached(cfg2, "TEST")
        
        # Both should be valid dict structures
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        
        print(f"Config1 (ATM-only) results: {len(result1)}")
        print(f"Config2 (all data) results: {len(result2)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])