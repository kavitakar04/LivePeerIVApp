#!/usr/bin/env python3
"""Test script to debug underlying price features."""

from analysis.weights.unified_weights import UnifiedWeightComputer, WeightConfig, WeightMethod, FeatureSet

def debug_underlying_features():
    """Debug why underlying price features are empty."""
    computer = UnifiedWeightComputer()
    
    print("Testing underlying price features...")
    
    # Test underlying price features directly
    try:
        feature_df = computer._build_underlying_px_features(["SPY", "QQQ", "IWM"])
        if feature_df is None:
            print("✗ _build_underlying_px_features returned None")
        elif feature_df.empty:
            print("✗ _build_underlying_px_features returned empty DataFrame")
        else:
            print(f"✓ _build_underlying_px_features returned shape: {feature_df.shape}")
            print(f"Columns: {list(feature_df.columns)}")
            print(f"Index: {list(feature_df.index)}")
            if feature_df.shape[0] > 0:
                print("Sample data:")
                print(feature_df.head())
    except Exception as e:
        print(f"✗ Exception in _build_underlying_px_features: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the full weight computation
    print("\nTesting full weight computation with UL mode...")
    try:
        config = WeightConfig(
            method=WeightMethod.CORRELATION,
            feature_set=FeatureSet.UNDERLYING_PX,
            asof=None
        )
        
        weights = computer.compute_weights("SPY", ["QQQ", "IWM"], config)
        print(f"✓ Weights computed successfully: {weights}")
        
    except Exception as e:
        print(f"✗ Weight computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_underlying_features()
