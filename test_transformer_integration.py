#!/usr/bin/env python3
"""
ABOUTME: Test transformer integration with existing game system
ABOUTME: Uses the berghain core modules to run games with dual-head transformer
"""

import sys
import torch
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'colab'))

# Test imports first
try:
    from models.dual_head_transformer import DualHeadTransformer
    from berghain.core.api_client import BerghainAPIClient
    from berghain.core.game_state import GameState
    from berghain.core.constraints import Constraint
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This is expected - let's test with API directly")

def test_api_connection():
    """Test direct API connection."""
    
    print("🔌 Testing API connection...")
    
    try:
        client = BerghainAPIClient()
        print("✅ API client created")
        
        # Try to start a game
        response = client.start_game(scenario=1)
        if response and 'game_id' in response:
            print(f"✅ Game started: {response['game_id']}")
            return response['game_id']
        else:
            print(f"❌ Failed to start game: {response}")
            return None
            
    except Exception as e:
        print(f"❌ API error: {e}")
        return None

def test_with_ultimate3():
    """Test by running ultimate3 solver for comparison."""
    
    print("\n🎯 Testing with Ultimate3 solver (current best)")
    
    try:
        from berghain.solvers.ultimate3_solver import Ultimate3Solver
        
        solver = Ultimate3Solver()
        print(f"✅ Ultimate3 solver created: {solver.get_strategy_name()}")
        
        # This will help us compare with transformer performance
        return True
        
    except Exception as e:
        print(f"❌ Ultimate3 test failed: {e}")
        return False

def run_quick_test():
    """Quick test to see system status."""
    
    print("🚀 Transformer Integration Test")
    print("=" * 40)
    
    # Test 1: Model loading
    model_path = "berghain_transformer/models/berghain_transformer_deployment.pt"
    if Path(model_path).exists():
        print("✅ Transformer model exists")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"✅ Model loads: {checkpoint['training_stats']['total_parameters']} params")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
    else:
        print("❌ Transformer model not found")
    
    # Test 2: API connection
    game_id = test_api_connection()
    
    # Test 3: Existing solver
    ultimate_works = test_with_ultimate3()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"  Model ready: {Path(model_path).exists()}")
    print(f"  API working: {game_id is not None}")
    print(f"  Ultimate3 available: {ultimate_works}")
    
    if game_id and ultimate_works:
        print(f"\n🎯 System ready! Run comparison:")
        print(f"python main.py run --scenario 1 --strategy ultimate3 --count 1")
        print(f"(Once transformer integrated, we'll compare performance)")
    
    return game_id is not None

if __name__ == "__main__":
    run_quick_test()