#!/usr/bin/env python3
"""Test ul weight mode specifically."""

import pytest
from analysis.weights.unified_weights import UnifiedWeightComputer, WeightConfig


def test_ul_weight_mode():
    """Test that ul weight mode works correctly."""
    config = WeightConfig.from_mode("corr_ul")
    
    # Test with real data
    computer = UnifiedWeightComputer()
    
    # This should work with actual underlying data
    weights = computer.compute_weights(
        target="SPY",
        peers=["BMNR", "IWM"],
        config=config
    )
    
    assert not weights.empty
    assert len(weights) == 2
    assert abs(weights.sum() - 1.0) < 1e-10  # Should sum to 1
    print(f"UL weights: {weights}")


if __name__ == "__main__":
    test_ul_weight_mode()
