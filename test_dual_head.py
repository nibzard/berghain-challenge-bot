#!/usr/bin/env python3
"""
ABOUTME: Test script for dual-head transformer deployment
ABOUTME: Quick test to verify the Colab-trained model works locally
"""

import sys
from pathlib import Path
import torch
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from berghain_transformer.dual_head_solver import DualHeadSolver
from berghain.core.game_state import GameState
from berghain.core.constraints import Constraint

logging.basicConfig(level=logging.INFO)

def test_dual_head_model():
    """Test the dual-head transformer model."""
    
    print("🚀 Testing Dual-Head Transformer Deployment")
    print("=" * 50)
    
    # Load the model
    model_path = "berghain_transformer/models/berghain_transformer_deployment.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please download from Google Drive and place in berghain_transformer/models/")
        return False
    
    try:
        # Initialize solver
        print("📥 Loading dual-head transformer...")
        solver = DualHeadSolver(model_path=model_path, temperature=1.0)
        print("✅ Model loaded successfully!")
        
        # Create a test game state
        constraints = [
            Constraint(attribute='young', min_count=600),
            Constraint(attribute='well_dressed', min_count=600)
        ]
        
        game_state = GameState(
            constraints=constraints,
            attribute_counts={'young': 150, 'well_dressed': 200},
            rejection_count=100,
            admit_count=350,
            people_seen=450,
            game_status='in_progress'
        )
        
        # Test a few decisions
        test_people = [
            {'young': True, 'well_dressed': True, 'id': 1},   # Perfect person
            {'young': True, 'well_dressed': False, 'id': 2},  # Partially good
            {'young': False, 'well_dressed': False, 'id': 3}, # Not ideal
            {'creative': True, 'id': 4},                      # Wrong attributes
        ]
        
        print("\n🧪 Testing decisions:")
        print("-" * 30)
        
        for i, person in enumerate(test_people):
            decision = solver.strategy.decide(person, game_state)
            
            attrs = []
            if person.get('young'): attrs.append('young')
            if person.get('well_dressed'): attrs.append('well_dressed')
            if person.get('creative'): attrs.append('creative')
            attr_str = ', '.join(attrs) if attrs else 'none'
            
            action = "ADMIT" if decision.admit else "REJECT"
            print(f"Person {i+1} [{attr_str}]: {action}")
            print(f"  Reasoning: {decision.reasoning}")
            print()
        
        print("🎯 Model deployment test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dual_head_model()
    if success:
        print("\n🏆 Ready for record attempt!")
        print("Next steps:")
        print("1. python main.py run --scenario 1 --strategy dual_head_transformer --count 10")
        print("2. If successful, run larger batches to find the best performance")
    else:
        print("\n🔧 Fix the issues above before proceeding")