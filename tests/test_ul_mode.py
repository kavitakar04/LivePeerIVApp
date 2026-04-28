#!/usr/bin/env python3
"""Test script to debug ul weight mode conversion."""

from analysis.weights.unified_weights import WeightConfig

def test_ul_mode_conversion():
    """Test the canonical 'corr_ul' mode conversion."""
    print("Testing 'corr_ul' mode conversion...")

    config = WeightConfig.from_mode("corr_ul")
    print(f"Method: {config.method}")
    print(f"Feature set: {config.feature_set}")
    print(f"Expected: CORRELATION method with UNDERLYING_PX features")
    
    # Test through the actual peer-composite interface
    print("\nTesting through peer-composite interface...")
    from analysis.services.peer_composite_service import PeerCompositeBuilder, PeerCompositeConfig
    
    cfg = PeerCompositeConfig(
        target="SPY",
        peers=("QQQ", "IWM"),
        weight_mode="corr_ul"
    )
    
    builder = PeerCompositeBuilder(cfg)
    try:
        weights = builder.compute_weights()
        print(f"✓ Weights computed successfully: {weights}")
    except Exception as e:
        print(f"✗ Weight computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ul_mode_conversion()
