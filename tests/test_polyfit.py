"""
Tests for the polyFit module - both simple polynomial and TPS fitting.
"""
import pytest
import numpy as np
import sys
import os
import pickle

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volModel.polyFit import fit_simple_poly, fit_tps, fit_poly


@pytest.fixture
def sample_iv_data():
    """Create sample implied volatility data for testing."""
    # Generate synthetic smile data - typical volatility smile shape
    k = np.linspace(-0.3, 0.3, 20)  # log-moneyness from -30% to +30%
    # Quadratic smile: higher vol for OTM puts and calls
    iv = 0.2 + 0.1 * k**2 + 0.02 * k  # Base vol 20%, skew, and convexity
    return k, iv


@pytest.fixture
def weighted_iv_data():
    """Create sample IV data with weights."""
    k = np.linspace(-0.2, 0.2, 15)
    iv = 0.18 + 0.08 * k**2 - 0.01 * k
    # Higher weights for ATM options
    weights = np.exp(-5 * k**2)  # Gaussian weights centered at ATM
    return k, iv, weights


class TestSimplePolyFit:
    """Test simple polynomial fitting functionality."""
    
    def test_simple_poly_basic_fit(self, sample_iv_data):
        """Test basic functionality of simple polynomial fit."""
        k, iv = sample_iv_data
        result = fit_simple_poly(k, iv)
        
        # Check that we get expected keys
        expected_keys = ["atm_vol", "skew", "curv", "rmse", "model"]
        assert all(key in result for key in expected_keys)
        
        # Check model type
        assert result["model"] == "simple_poly"
        
        # Check reasonable values
        assert 0.1 < result["atm_vol"] < 0.4  # ATM vol should be reasonable
        assert result["rmse"] >= 0  # RMSE should be non-negative
        assert np.isfinite(result["atm_vol"])
        assert np.isfinite(result["skew"])
        assert np.isfinite(result["curv"])
    
    def test_simple_poly_with_weights(self, weighted_iv_data):
        """Test simple polynomial fit with weights."""
        k, iv, weights = weighted_iv_data
        
        result_weighted = fit_simple_poly(k, iv, weights=weights)
        result_unweighted = fit_simple_poly(k, iv, weights=None)
        
        # Both should work
        assert result_weighted["model"] == "simple_poly"
        assert result_unweighted["model"] == "simple_poly"
        
        # Results should be different (weights should matter)
        assert result_weighted["atm_vol"] != result_unweighted["atm_vol"]
    
    def test_simple_poly_band_parameter(self, sample_iv_data):
        """Test different band parameters."""
        k, iv = sample_iv_data
        
        result_narrow = fit_simple_poly(k, iv, band=0.1)
        result_wide = fit_simple_poly(k, iv, band=0.5)
        
        # Both should work
        assert result_narrow["model"] == "simple_poly"
        assert result_wide["model"] == "simple_poly"
        
        # Results may be different depending on data distribution
        assert np.isfinite(result_narrow["rmse"])
        assert np.isfinite(result_wide["rmse"])
    
    def test_simple_poly_edge_cases(self):
        """Test edge cases for simple polynomial fit."""
        # Very few points
        k_few = np.array([-0.1, 0.0, 0.1])
        iv_few = np.array([0.22, 0.20, 0.22])
        
        result = fit_simple_poly(k_few, iv_few)
        assert result["model"] == "simple_poly"
        assert np.isfinite(result["atm_vol"])


class TestTPSFit:
    """Test Thin Plate Spline fitting functionality."""
    
    def test_tps_basic_fit(self, sample_iv_data):
        """Test basic TPS functionality."""
        k, iv = sample_iv_data
        result = fit_tps(k, iv)
        
        # Should have expected keys
        expected_keys = ["atm_vol", "skew", "curv", "rmse", "model"]
        assert all(key in result for key in expected_keys)
        
        # Model should be either 'tps' (if scipy available) or 'simple_poly' (fallback)
        assert result["model"] in ["tps", "simple_poly"]
        
        # Check reasonable values
        assert np.isfinite(result["atm_vol"])
        assert np.isfinite(result["skew"])
        assert np.isfinite(result["curv"])
        assert result["rmse"] >= 0
    
    def test_tps_with_interpolator(self, sample_iv_data):
        """Test TPS interpolator functionality if available."""
        k, iv = sample_iv_data
        result = fit_tps(k, iv)
        
        # If TPS worked, should have interpolator
        if result["model"] == "tps":
            assert "interpolator" in result
            
            # Test interpolation at a new point
            interp_func = result["interpolator"]
            test_k = np.array([0.05])  # Test near ATM
            iv_pred = interp_func(test_k)
            
            assert len(iv_pred) == 1
            assert np.isfinite(iv_pred[0])
            assert iv_pred[0] > 0  # Should be positive vol

    def test_tps_interpolator_is_pickle_safe(self, sample_iv_data):
        """TPS params are cached by the GUI, so the interpolator must pickle."""
        k, iv = sample_iv_data
        result = fit_tps(k, iv)

        if result["model"] == "tps":
            restored = pickle.loads(pickle.dumps(result))
            test_k = np.array([-0.05, 0.0, 0.05])

            assert np.all(np.isfinite(restored["interpolator"](test_k)))
    
    def test_tps_with_smoothing(self, sample_iv_data):
        """Test TPS with different smoothing parameters."""
        k, iv = sample_iv_data
        
        result_smooth = fit_tps(k, iv, smoothing=0.01)
        result_rough = fit_tps(k, iv, smoothing=0.0)
        
        # Both should work (or fallback to simple_poly)
        assert result_smooth["model"] in ["tps", "simple_poly"]
        assert result_rough["model"] in ["tps", "simple_poly"]
    
    def test_tps_with_weights(self, weighted_iv_data):
        """Test TPS with weights."""
        k, iv, weights = weighted_iv_data
        
        result = fit_tps(k, iv, weights=weights)
        assert result["model"] in ["tps", "simple_poly"]
        assert np.isfinite(result["atm_vol"])


class TestPolyFitDispatcher:
    """Test the main fit_poly dispatcher function."""
    
    def test_method_dispatch(self, sample_iv_data):
        """Test that method parameter correctly dispatches."""
        k, iv = sample_iv_data
        
        result_simple = fit_poly(k, iv, method="simple")
        result_tps = fit_poly(k, iv, method="tps")
        
        # Simple should always be simple_poly
        assert result_simple["model"] == "simple_poly"
        
        # TPS should be either tps or simple_poly (fallback)
        assert result_tps["model"] in ["tps", "simple_poly"]
    
    def test_default_method(self, sample_iv_data):
        """Test default method behavior."""
        k, iv = sample_iv_data
        
        # Default should be simple
        result_default = fit_poly(k, iv)
        result_explicit = fit_poly(k, iv, method="simple")
        
        assert result_default["model"] == result_explicit["model"]
        assert result_default["atm_vol"] == result_explicit["atm_vol"]
    
    def test_case_insensitive_method(self, sample_iv_data):
        """Test that method names are case insensitive."""
        k, iv = sample_iv_data
        
        result_upper = fit_poly(k, iv, method="TPS")
        result_lower = fit_poly(k, iv, method="tps")
        result_mixed = fit_poly(k, iv, method="Tps")
        
        # All should give same model type
        assert result_upper["model"] == result_lower["model"] == result_mixed["model"]


class TestIntegrationAndCompatibility:
    """Test integration with existing codebase."""
    
    def test_backward_compatibility_alias(self, sample_iv_data):
        """Test that backward compatibility alias works."""
        from volModel.polyFit import _local_poly_fit_atm
        
        k, iv = sample_iv_data
        result = _local_poly_fit_atm(k, iv)
        
        # Should work exactly like fit_simple_poly
        assert result["model"] == "simple_poly"
        assert np.isfinite(result["atm_vol"])
    
    def test_realistic_iv_surface_data(self):
        """Test with realistic implied volatility surface data."""
        # More realistic IV surface data
        np.random.seed(42)  # For reproducible tests
        
        # Create data that looks like real IV smile
        k = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
        # Classic volatility smile: higher vol for deep OTM, skew for puts
        iv = 0.2 + 0.15 * k**2 + 0.05 * k + 0.01 * np.random.randn(len(k))
        
        # Test both methods
        result_simple = fit_poly(k, iv, method="simple")
        result_tps = fit_poly(k, iv, method="tps")
        
        # Both should produce reasonable results
        assert 0.1 < result_simple["atm_vol"] < 0.4
        assert 0.1 < result_tps["atm_vol"] < 0.4
        
        # TPS should generally fit better (lower RMSE) unless it falls back
        if result_tps["model"] == "tps":
            assert result_tps["rmse"] <= result_simple["rmse"] * 1.1  # Allow some tolerance
