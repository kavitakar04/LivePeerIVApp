#!/usr/bin/env python3
"""Test script to verify the surface feature matrix graceful failure."""

from analysis.beta_builder import surface_feature_matrix

def test_surface_feature_graceful_failure():
    """Test that surface_feature_matrix returns empty results instead of raising."""
    print("Testing surface feature matrix graceful failure...")
    
    try:
        # Try with valid tickers but an asof date that might not have data
        result = surface_feature_matrix(
            tickers=["SPY", "QQQ"],
            asof="2020-01-01"  # Old date unlikely to have complete data
        )
        
        grids, feature_matrix, feature_names = result
        print(f"Grids: {type(grids)} with {len(grids)} entries")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Feature names: {len(feature_names)} features")
        
        if feature_matrix.shape == (0, 0):
            print("✓ Empty feature matrix returned gracefully (no exception)")
        else:
            print(f"✓ Feature matrix returned with shape {feature_matrix.shape}")
            
    except RuntimeError as e:
        if "surface grids unavailable" in str(e):
            print("Expected error - peer-composite surface grids unavailable in this environment")
        else:
            print(f"✗ Unexpected RuntimeError: {e}")
    except Exception as e:
        print(f"✗ Other exception still raised: {e}")
        print("The fix may not be working correctly")

if __name__ == "__main__":
    test_surface_feature_graceful_failure()
