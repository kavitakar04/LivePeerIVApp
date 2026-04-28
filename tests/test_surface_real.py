#!/usr/bin/env python3
"""Test the surface feature matrix with a realistic scenario."""

from analysis.weights.beta_builder import surface_feature_matrix
import pandas as pd

def test_with_real_dates():
    """Test surface_feature_matrix with actual dates that exist."""
    print("Testing surface_feature_matrix with existing dates...")
    
    try:
        # Test with a date that exists
        result = surface_feature_matrix(
            tickers=["SPY", "QQQ"],
            asof="2025-08-15"  # Date that exists in the grid
        )
        
        grids, feature_matrix, feature_names = result
        print(f"✓ Success with existing date!")
        print(f"  Grids: {len(grids)} entries")
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Feature names: {len(feature_names)} features")
        
    except Exception as e:
        print(f"✗ Error with existing date: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTesting with non-existent date...")
    try:
        # Test with a date that doesn't exist
        result = surface_feature_matrix(
            tickers=["SPY", "QQQ"],
            asof="2020-01-01"  # Date that doesn't exist
        )
        
        grids, feature_matrix, feature_names = result
        print(f"✓ Success with non-existent date!")
        print(f"  Grids: {len(grids)} entries")
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Feature names: {len(feature_names)} features")
        
        if feature_matrix.shape == (0, 0):
            print("  ✓ Empty matrix returned as expected!")
        
    except Exception as e:
        print(f"✗ Error with non-existent date: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_real_dates()
