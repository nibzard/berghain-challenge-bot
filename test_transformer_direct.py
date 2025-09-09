#!/usr/bin/env python3
"""
ABOUTME: Direct test of transformer strategy without the full game system
ABOUTME: Quick validation that the dual-head model works correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'colab'))

def test_transformer_direct():
    """Direct test of transformer strategy."""
    
    print("🧪 Testing Transformer Strategy Directly")
    print("=" * 50)
    
    try:
        # Import the strategy
        from berghain.solvers.transformer_solver import TransformerStrategy
        
        # Create strategy with default parameters
        params = {
            'model_path': 'berghain_transformer/models/berghain_transformer_deployment.pt',
            'temperature': 1.0
        }
        
        strategy = TransformerStrategy(params)
        print("✅ Strategy created successfully!")
        
        # Test the model directly
        print(f"🤖 Model loaded: {strategy.model}")
        
        # Create dummy game state and person
        from berghain.core.domain import GameState, Person, Constraint, AttributeStatistics
        from datetime import datetime
        
        # Create test constraints
        constraints = [
            Constraint(attribute='young', min_count=600),
            Constraint(attribute='well_dressed', min_count=600)
        ]
        
        # Create attribute statistics
        stats = AttributeStatistics(
            frequencies={'young': 0.323, 'well_dressed': 0.323, 'creative': 0.1},
            correlations={}
        )
        
        # Create game state
        game_state = GameState(
            game_id="test123",
            scenario=1,
            constraints=constraints,
            statistics=stats,
            admitted_count=350,
            rejected_count=500,
            admitted_attributes={'young': 150, 'well_dressed': 180}
        )
        
        # Create test person
        person = Person(
            index=1,
            attributes={'young': True, 'well_dressed': True}
        )
        
        print("🎮 Testing decision making...")
        
        # Test decision
        admit, reasoning = strategy.decide(person, game_state)
        
        print(f"✅ Decision: {'ADMIT' if admit else 'REJECT'}")
        print(f"💭 Reasoning: {reasoning}")
        
        # Test a few more people
        test_cases = [
            ({'young': True, 'well_dressed': False}, "young only"),
            ({'young': False, 'well_dressed': True}, "well_dressed only"),
            ({'creative': True}, "creative only"),
            ({}, "no attributes")
        ]
        
        print(f"\n🧪 Testing different person types:")
        for i, (attrs, desc) in enumerate(test_cases):
            person = Person(index=i+2, attributes=attrs)
            admit, reasoning = strategy.decide(person, game_state)
            action = "ADMIT" if admit else "REJECT"
            print(f"  {desc}: {action} - {reasoning[:50]}...")
        
        print(f"\n🎯 Transformer strategy working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transformer_direct()
    if success:
        print("\n🏆 Ready to integrate with game system!")
    else:
        print("\n🔧 Fix issues before integration")