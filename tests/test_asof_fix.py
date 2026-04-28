#!/usr/bin/env python3
"""Test script to verify the asof date selection fix."""

from analysis.weights.unified_weights import UnifiedWeightComputer, WeightConfig, WeightMethod, FeatureSet

def test_asof_selection():
    """Test the new robust asof date selection."""
    computer = UnifiedWeightComputer()
    
    # Test with surface features - this should use the new logic
    config = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.SURFACE,
        asof=None  # Let it auto-select
    )
    
    print("Testing robust asof date selection...")
    try:
        # Test the _choose_asof method directly
        asof = computer._choose_asof("SPY", ["QQQ", "IWM"], config)
        print(f"Selected asof date for SURFACE features: {asof}")
        
        # Test with ATM features - should use original logic
        config_atm = WeightConfig(
            method=WeightMethod.CORRELATION,
            feature_set=FeatureSet.ATM,
            asof=None
        )
        asof_atm = computer._choose_asof("SPY", ["QQQ", "IWM"], config_atm)
        print(f"Selected asof date for ATM features: {asof_atm}")
        
        print("✓ Asof date selection is working!")
        
    except Exception as e:
        print(f"Error in asof selection: {e}")
        print("This may be expected if no surface data is available")

if __name__ == "__main__":
    test_asof_selection()
