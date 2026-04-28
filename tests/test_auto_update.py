#!/usr/bin/env python3
"""Test auto-fetching of underlying price data."""

from analysis.weights.unified_weights import UnifiedWeightComputer, WeightConfig
from data.data_pipeline import ensure_underlying_price_data, check_and_update_underlying_prices
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_auto_update():
    """Test the auto-update functionality."""
    print("Testing auto-update of underlying price data...")
    
    # Test with a new ticker that might not have data
    test_tickers = ["SPY", "QQQ", "NEWTICKERTEST"]
    
    print(f"\n1. Testing ensure_underlying_price_data...")
    success = ensure_underlying_price_data(test_tickers)
    print(f"   Result: {success}")
    
    print(f"\n2. Testing check_and_update_underlying_prices...")
    updated = check_and_update_underlying_prices(set(test_tickers))
    print(f"   Updated: {updated} records")
    
    print(f"\n3. Testing 'ul' weight mode with auto-update...")
    try:
        config = WeightConfig.from_mode("corr_ul")
        computer = UnifiedWeightComputer()
        
        # This should trigger auto-update in underlying_returns_matrix
        weights = computer.compute_weights(
            target="SPY",
            peers=["QQQ", "IWM"],
            config=config
        )
        
        print(f"   ✓ Weights computed successfully: {weights}")
        
    except Exception as e:
        print(f"   ✗ Weight computation failed: {e}")

if __name__ == "__main__":
    test_auto_update()
