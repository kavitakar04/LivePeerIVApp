#!/usr/bin/env python3
"""
Debug script to investigate the unified weights ATM logging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
from analysis.weights.unified_weights import UnifiedWeightComputer, WeightConfig, WeightMethod, FeatureSet

# Set up logging to see the debug messages
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def debug_unified_weights():
    """Debug unified weights to see the atm_rows logging."""
    
    computer = UnifiedWeightComputer()
    
    config = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.ATM,
        clip_negative=True,
        power=1.0
    )
    
    target = "SPY"
    peers = ["QQQ", "IWM"]
    
    print(f"Computing weights for {target} vs {peers}")
    print(f"Config: {config}")
    
    try:
        weights = computer.compute_weights(target, peers, config)
        print(f"Result: {weights}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_unified_weights()
