#!/usr/bin/env python3
"""
ABOUTME: Simple test to load and verify the Colab-trained model
ABOUTME: Tests the deployment model file without full integration
"""

import torch
import sys
from pathlib import Path
import json

def test_model_loading():
    """Test loading the Colab deployment model."""
    
    print("🚀 Testing Colab Model Deployment")
    print("=" * 50)
    
    # Check if model exists
    model_path = "berghain_transformer/models/berghain_transformer_deployment.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please download from Google Drive and place in berghain_transformer/models/")
        return False
    
    try:
        # Load the deployment model
        print("📥 Loading deployment model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check contents
        print("✅ Model loaded successfully!")
        print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"📊 Model config: {config}")
            
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            print(f"🏆 Training stats:")
            for key, value in stats.items():
                print(f"  - {key}: {value}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"📈 Model has {len(state_dict)} parameters")
            
            # Check parameter shapes
            print("🔧 Key parameter shapes:")
            for name, param in list(state_dict.items())[:5]:
                print(f"  - {name}: {param.shape}")
            
        print("\n✅ Colab model deployment file is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_inference():
    """Test basic model inference."""
    
    model_path = "berghain_transformer/models/berghain_transformer_deployment.pt"
    
    if not Path(model_path).exists():
        print("⚠️ Skipping inference test - model not found")
        return False
    
    try:
        print("\n🧪 Testing Model Inference")
        print("-" * 30)
        
        # Add colab path to import the model
        sys.path.append(str(Path(__file__).parent / 'colab'))
        from models.dual_head_transformer import DualHeadTransformer
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Create model
        model = DualHeadTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test input (batch_size=1, seq_len=50, features=14)
        test_input = torch.randn(1, 50, 14)
        
        print(f"📊 Test input shape: {test_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
            
        print("✅ Forward pass successful!")
        print(f"🎯 Constraint logits shape: {output.constraint_logits.shape}")
        print(f"⚡ Efficiency logits shape: {output.efficiency_logits.shape}")
        print(f"🤝 Combined logits shape: {output.combined_logits.shape}")
        print(f"⚖️ Head weights: {output.head_weights[0, -1].cpu().numpy()}")
        
        # Test decision
        constraint_pred = output.constraint_logits[0, -1].argmax().item()
        efficiency_pred = output.efficiency_logits[0, -1].argmax().item()
        combined_pred = output.combined_logits[0, -1].argmax().item()
        
        print(f"🧠 Constraint prediction: {constraint_pred}")
        print(f"⚡ Efficiency prediction: {efficiency_pred}")
        print(f"🎯 Combined prediction: {combined_pred}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_model_loading()
    success2 = test_model_inference() if success1 else False
    
    if success1 and success2:
        print("\n🏆 Colab model is ready for deployment!")
        print("Next step: Integrate with game system")
    else:
        print("\n🔧 Fix the issues above before deployment")