#!/usr/bin/env python3
"""
ABOUTME: Direct runner for dual-head transformer without full system integration
ABOUTME: Quick deployment test for the Colab-trained model
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import requests
import json
import time
from pathlib import Path

# Add colab path for model import
sys.path.append(str(Path(__file__).parent / 'colab'))
from models.dual_head_transformer import DualHeadTransformer

class TransformerGameRunner:
    """Direct game runner for the dual-head transformer."""
    
    def __init__(self, model_path: str, api_url: str = "https://berghain.challenges.listenlabs.ai"):
        self.api_url = api_url
        self.model = self.load_model(model_path)
        self.decision_history = []
        
    def load_model(self, model_path: str):
        """Load the dual-head transformer model."""
        print(f"📥 Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        model = DualHeadTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded: {checkpoint['training_stats']['total_parameters']} parameters")
        return model
    
    def encode_game_state(self, game_state, person):
        """Encode game state and person into 14-dimensional feature vector."""
        
        # Get constraint requirements (scenario 1: young + well_dressed)
        young_current = game_state.get('young', 0)
        well_dressed_current = game_state.get('well_dressed', 0)
        young_needed = max(0, 600 - young_current)
        well_dressed_needed = max(0, 600 - well_dressed_current)
        
        # Calculate progress and pressure metrics
        total_admitted = sum(game_state.values())
        total_rejected = game_state.get('_rejection_count', 0)
        rejection_rate = total_rejected / max(1, total_admitted + total_rejected)
        
        game_progress = min(1.0, total_admitted / 1000.0)
        remaining_capacity = max(0, 1000 - total_admitted)
        time_pressure = 1.0 - (remaining_capacity / 1000.0)
        
        constraint_pressure = (young_needed + well_dressed_needed) / max(1, remaining_capacity)
        constraint_pressure = min(1.0, constraint_pressure)
        
        efficiency_trend = max(0.0, min(1.0, 1.0 - rejection_rate))
        
        # Person attributes
        person_young = person.get('young', False)
        person_well_dressed = person.get('well_dressed', False)
        
        # Person value for constraints
        person_value = 0.0
        if person_young and young_needed > 0:
            person_value += 0.5
        if person_well_dressed and well_dressed_needed > 0:
            person_value += 0.5
        
        # 14-dimensional feature vector
        features = np.array([
            young_current,
            young_needed,
            well_dressed_current,
            well_dressed_needed,
            constraint_pressure,
            total_admitted,
            total_rejected,
            rejection_rate,
            efficiency_trend,
            game_progress,
            time_pressure,
            float(person_young),
            float(person_well_dressed),
            person_value
        ], dtype=np.float32)
        
        return features
    
    def make_decision(self, game_state, person):
        """Make admission decision using dual-head transformer."""
        
        # Encode current state
        state_features = self.encode_game_state(game_state, person)
        
        # Create sequence input (50 timesteps)
        seq_len = 50
        context_length = min(len(self.decision_history), seq_len - 1)
        
        if context_length > 0:
            # Use recent history as context
            context_features = np.stack([
                hist['features'] for hist in self.decision_history[-context_length:]
            ])
            
            if context_length < seq_len - 1:
                # Pad with zeros
                padding = np.zeros((seq_len - 1 - context_length, 14), dtype=np.float32)
                sequence_features = np.vstack([padding, context_features, state_features.reshape(1, -1)])
            else:
                # Use last seq_len-1 decisions plus current
                sequence_features = np.vstack([context_features[-(seq_len-1):], state_features.reshape(1, -1)])
        else:
            # No history, pad with zeros
            padding = np.zeros((seq_len - 1, 14), dtype=np.float32)
            sequence_features = np.vstack([padding, state_features.reshape(1, -1)])
        
        # Convert to tensor and get prediction
        input_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Get probabilities from combined head
            combined_logits = output.combined_logits[0, -1, :]
            probs = F.softmax(combined_logits, dim=-1)
            action_prob = probs[1].item()  # Probability of admission
            
            # Get additional info
            constraint_conf = output.constraint_confidence[0, -1].item()
            efficiency_conf = output.efficiency_confidence[0, -1].item()
            head_weights = output.head_weights[0, -1].cpu().numpy()
        
        # Make decision
        admit = action_prob > 0.5
        
        # Store in history
        self.decision_history.append({
            'features': state_features,
            'admit': admit,
            'confidence': max(action_prob, 1 - action_prob),
            'action_prob': action_prob
        })
        
        # Reasoning
        reasoning = f"DH: p={action_prob:.3f}, conf=[{constraint_conf:.2f},{efficiency_conf:.2f}], w={head_weights:.2f}"
        
        return admit, reasoning
    
    def play_game(self, scenario: int = 1):
        """Play one game using the transformer."""
        
        print(f"🎮 Starting Scenario {scenario} with Dual-Head Transformer")
        print("=" * 60)
        
        # Reset history
        self.decision_history = []
        
        # Start game
        start_response = requests.post(f"{self.api_url}/start", json={"scenario": scenario})
        if start_response.status_code != 200:
            print(f"❌ Failed to start game: {start_response.text}")
            return None
        
        game_data = start_response.json()
        game_id = game_data["game_id"]
        print(f"🎯 Game ID: {game_id}")
        
        decisions_made = 0
        total_admitted = 0
        
        while True:
            # Get next person
            person_response = requests.get(f"{self.api_url}/game/{game_id}/person")
            
            if person_response.status_code == 404:
                print("🏁 Game completed!")
                break
            elif person_response.status_code != 200:
                print(f"❌ Error getting person: {person_response.text}")
                break
            
            person_data = person_response.json()
            person = person_data["person"]
            game_state = person_data["game_state"]
            
            # Make decision
            admit, reasoning = self.make_decision(game_state, person)
            
            # Submit decision
            decision_response = requests.post(
                f"{self.api_url}/game/{game_id}/decision",
                json={"admit": admit}
            )
            
            if decision_response.status_code != 200:
                print(f"❌ Error submitting decision: {decision_response.text}")
                break
            
            decisions_made += 1
            if admit:
                total_admitted += 1
            
            # Show progress
            if decisions_made % 1000 == 0:
                attrs = [k for k, v in person.items() if v and k != 'id']
                action = "ADMIT" if admit else "REJECT"
                print(f"Decision {decisions_made}: {action} [{', '.join(attrs)}] - {reasoning}")
                print(f"  Progress: {total_admitted} admitted, {game_state.get('_rejection_count', 0)} rejected")
        
        # Get final result
        result_response = requests.get(f"{self.api_url}/game/{game_id}/result")
        if result_response.status_code == 200:
            result = result_response.json()
            print(f"\n🏆 GAME RESULT:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Score: {result.get('score', 'N/A')}")
            print(f"  Decisions: {decisions_made}")
            print(f"  Admitted: {result.get('admitted', 'N/A')}")
            print(f"  Rejected: {result.get('rejected', 'N/A')}")
            print(f"  Constraints: {result.get('constraints_satisfied', 'N/A')}")
            
            rejections = result.get('rejected', 0)
            if rejections < 716:
                print(f"🎉 NEW RECORD! Beat 716 with {rejections} rejections!")
            elif rejections < 800:
                print(f"🔥 Excellent! {rejections} rejections (better than many strategies)")
            else:
                print(f"📈 Result: {rejections} rejections (current record: 716)")
            
            return result
        else:
            print(f"❌ Error getting result: {result_response.text}")
            return None

if __name__ == "__main__":
    # Run transformer game
    model_path = "berghain_transformer/models/berghain_transformer_deployment.pt"
    runner = TransformerGameRunner(model_path)
    
    print("🚀 Dual-Head Transformer Record Attempt!")
    print("Current record to beat: 716 rejections")
    print("-" * 50)
    
    result = runner.play_game(scenario=1)
    
    if result and result.get('rejected', float('inf')) < 716:
        print("\n🏆 CONGRATULATIONS! NEW WORLD RECORD!")
    else:
        print("\n🎯 Good run! Try again for the record.")