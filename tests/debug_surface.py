#!/usr/bin/env python3
"""Debug the surface feature matrix issue."""

from analysis.peer_composite_builder import build_surface_grids

def debug_surface_grids():
    """Debug what build_surface_grids returns."""
    print("Testing build_surface_grids...")
    
    try:
        # Test with valid tickers
        result = build_surface_grids(tickers=["SPY", "QQQ"])
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        if result is None:
            print("build_surface_grids returned None")
        elif isinstance(result, dict):
            print(f"Returned dict with {len(result)} entries")
            for key, value in result.items():
                print(f"  {key}: {type(value)} -> {value}")
        else:
            print(f"Unexpected result type: {type(result)}")
            
    except Exception as e:
        print(f"Exception in build_surface_grids: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_surface_grids()
