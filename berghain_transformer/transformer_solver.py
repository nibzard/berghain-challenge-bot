"""Transformer-based solver for the Berghain Challenge game."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import requests
import time
import json

from .models.transformer_model import BerghainTransformer, DecisionTransformer
from .data.preprocessor import GameStateEncoder, load_encoder


class TransformerSolver:
    """Solver that uses a trained transformer model for decision making."""
    
    def __init__(
        self,
        model_path: Path,
        encoder_path: Path,
        api_url: str = "https://berghain-trainer-production.up.railway.app",
        scenario: int = 1,
        temperature: float = 1.0,
        top_k: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        debug: bool = False
    ):
        self.api_url = api_url
        self.scenario = scenario
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.debug = debug
        
        # Load encoder
        self.encoder = load_encoder(encoder_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Game state tracking
        self.reset_game_state()
        
        # Sequence buffers for transformer input
        self.max_seq_length = 100
        self.state_buffer = deque(maxlen=self.max_seq_length)
        self.action_buffer = deque(maxlen=self.max_seq_length)
        self.reward_buffer = deque(maxlen=self.max_seq_length)
        self.rtg_buffer = deque(maxlen=self.max_seq_length)
    
    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load the trained transformer model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type from checkpoint
        state_dict = checkpoint['model_state_dict']
        has_rtg_embedding = any('rtg_embedding' in k for k in state_dict.keys())
        
        if has_rtg_embedding:
            model = DecisionTransformer(
                state_dim=self.encoder.feature_dim,
                action_dim=2
            )
        else:
            model = BerghainTransformer(
                state_dim=self.encoder.feature_dim,
                action_dim=2,
                use_value_head=False
            )
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def reset_game_state(self):
        """Reset internal game state tracking."""
        self.game_state = {
            'total_admitted': 0,
            'total_rejected': 0,
            'constraints': [],
            'attribute_counts': defaultdict(int),
            'recent_decisions': deque(maxlen=10),
            'game_id': None,
            'cumulative_reward': 0
        }
        
        # Clear sequence buffers
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.rtg_buffer.clear()
    
    def start_game(self) -> Dict[str, Any]:
        """Start a new game."""
        response = requests.post(
            f"{self.api_url}/api/games",
            json={"scenario": self.scenario}
        )
        response.raise_for_status()
        
        game_data = response.json()
        self.game_state['game_id'] = game_data['game_id']
        
        # Initialize constraints
        if 'constraints' in game_data:
            self.game_state['constraints'] = [
                {
                    'attribute': c['attribute'],
                    'required': c['required'],
                    'current': 0
                }
                for c in game_data['constraints']
            ]
        
        if self.debug:
            print(f"Started game {self.game_state['game_id']}")
            print(f"Constraints: {self.game_state['constraints']}")
        
        return game_data
    
    def get_next_person(self) -> Optional[Dict[str, Any]]:
        """Get the next person in queue."""
        response = requests.get(
            f"{self.api_url}/api/games/{self.game_state['game_id']}/next-person"
        )
        
        if response.status_code == 204:
            return None
        
        response.raise_for_status()
        return response.json()
    
    def make_decision(self, person: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Make admission decision using the transformer model."""
        # Create current state representation
        event = {
            'event_type': 'person_evaluated',
            'person': person
        }
        state_vector = self.encoder.encode_state(event, self.game_state)
        
        # Add to state buffer
        self.state_buffer.append(state_vector)
        
        # Prepare transformer input
        states = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            if isinstance(self.model, DecisionTransformer):
                # For Decision Transformer, we need returns-to-go
                # Estimate based on remaining constraints
                remaining_needed = sum(
                    max(0, c['required'] - c['current'])
                    for c in self.game_state['constraints']
                )
                estimated_rtg = remaining_needed * 2.0  # Heuristic multiplier
                
                if self.rtg_buffer:
                    rtg = torch.FloatTensor(list(self.rtg_buffer)).unsqueeze(0).to(self.device)
                else:
                    rtg = torch.full((1, len(self.state_buffer)), estimated_rtg).to(self.device)
                
                # Include previous actions if available
                if self.action_buffer:
                    actions = torch.LongTensor(list(self.action_buffer)).unsqueeze(0).to(self.device)
                    action_logits = self.model(
                        states=states,
                        actions=actions,
                        returns_to_go=rtg
                    )
                else:
                    action_logits = self.model(
                        states=states,
                        returns_to_go=rtg
                    )
            else:
                # Standard transformer
                if self.action_buffer and self.reward_buffer:
                    actions = torch.LongTensor(list(self.action_buffer)).unsqueeze(0).to(self.device)
                    rewards = torch.FloatTensor(list(self.reward_buffer)).unsqueeze(0).to(self.device)
                    action_logits, _ = self.model(
                        states=states,
                        actions=actions,
                        rewards=rewards
                    )
                else:
                    action_logits, _ = self.model(states=states)
        
        # Get the prediction for the current state (last in sequence)
        current_logits = action_logits[0, -1]
        
        # Apply temperature and top-k sampling
        if self.temperature != 1.0:
            current_logits = current_logits / self.temperature
        
        if self.top_k > 1:
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(current_logits, min(self.top_k, 2))
            probs = F.softmax(top_k_logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
            action = top_k_indices[action_idx].item()
        else:
            # Greedy selection
            action = torch.argmax(current_logits).item()
        
        admit = bool(action)
        
        # Calculate confidence
        probs = F.softmax(current_logits, dim=-1)
        confidence = probs[action].item()
        
        # Create decision info
        decision_info = {
            'admitted': admit,
            'confidence': confidence,
            'reasoning': self._generate_reasoning(person, admit, confidence),
            'model_output': {
                'logits': current_logits.cpu().numpy().tolist(),
                'probabilities': probs.cpu().numpy().tolist()
            }
        }
        
        # Update buffers
        self.action_buffer.append(action)
        
        # Update game state
        self._update_game_state(person, admit)
        
        return admit, decision_info
    
    def _generate_reasoning(self, person: Dict[str, Any], admit: bool, confidence: float) -> str:
        """Generate human-readable reasoning for the decision."""
        attributes = person.get('attributes', [])
        
        # Check constraint relevance
        relevant_constraints = []
        for constraint in self.game_state['constraints']:
            if constraint['attribute'] in attributes:
                deficit = constraint['required'] - constraint['current']
                if deficit > 0:
                    relevant_constraints.append((constraint['attribute'], deficit))
        
        if admit:
            if relevant_constraints:
                constraint_str = ', '.join([f"{attr} (need {deficit} more)" 
                                           for attr, deficit in relevant_constraints])
                reasoning = f"Admitting person with needed attributes: {constraint_str}"
            else:
                reasoning = f"Admitting based on model confidence ({confidence:.2%})"
        else:
            if relevant_constraints:
                reasoning = f"Rejecting despite having needed attributes (model confidence: {confidence:.2%})"
            else:
                reasoning = f"Rejecting - no critical attributes needed (confidence: {confidence:.2%})"
        
        # Add capacity warnings
        if self.game_state['total_admitted'] > 900:
            reasoning += " [WARNING: Approaching admission limit]"
        if self.game_state['total_rejected'] > 18000:
            reasoning += " [WARNING: Approaching rejection limit]"
        
        return reasoning
    
    def _update_game_state(self, person: Dict[str, Any], admitted: bool):
        """Update internal game state after a decision."""
        # Update counts
        if admitted:
            self.game_state['total_admitted'] += 1
            for constraint in self.game_state['constraints']:
                if constraint['attribute'] in person.get('attributes', []):
                    constraint['current'] += 1
        else:
            self.game_state['total_rejected'] += 1
        
        # Update attribute counts
        for attr in person.get('attributes', []):
            self.game_state['attribute_counts'][attr] += 1
        
        # Update recent decisions
        self.game_state['recent_decisions'].append(admitted)
        
        # Calculate reward for this decision
        reward = self._calculate_reward(person, admitted)
        self.reward_buffer.append(reward)
        self.game_state['cumulative_reward'] += reward
        
        # Update returns-to-go if using Decision Transformer
        if isinstance(self.model, DecisionTransformer):
            # Estimate remaining returns
            remaining_needed = sum(
                max(0, c['required'] - c['current'])
                for c in self.game_state['constraints']
            )
            estimated_rtg = remaining_needed * 2.0 - self.game_state['cumulative_reward']
            self.rtg_buffer.append(max(0, estimated_rtg))
    
    def _calculate_reward(self, person: Dict[str, Any], admitted: bool) -> float:
        """Calculate reward for a decision."""
        reward = 0.0
        
        if admitted:
            # Reward for admitting people with needed attributes
            for constraint in self.game_state['constraints']:
                if constraint['current'] < constraint['required']:
                    if constraint['attribute'] in person.get('attributes', []):
                        deficit_ratio = 1 - (constraint['current'] / constraint['required'])
                        reward += 1.0 + deficit_ratio
        
        # Penalties for approaching limits
        if self.game_state['total_admitted'] > 900:
            reward -= 0.5
        if self.game_state['total_rejected'] > 18000:
            reward -= 0.5
        
        # Small penalty for any action
        reward -= 0.01
        
        return reward
    
    def submit_decision(self, person_id: str, admit: bool) -> Dict[str, Any]:
        """Submit decision to the API."""
        response = requests.post(
            f"{self.api_url}/api/games/{self.game_state['game_id']}/evaluate-person",
            json={
                "person_id": person_id,
                "decision": "admit" if admit else "reject"
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_game_status(self) -> Dict[str, Any]:
        """Get current game status."""
        response = requests.get(
            f"{self.api_url}/api/games/{self.game_state['game_id']}/status"
        )
        response.raise_for_status()
        return response.json()
    
    def run_game(self) -> Dict[str, Any]:
        """Run a complete game."""
        # Start game
        self.reset_game_state()
        game_info = self.start_game()
        
        decisions_made = []
        start_time = time.time()
        
        while True:
            # Get next person
            person = self.get_next_person()
            if person is None:
                break
            
            # Make decision
            admit, decision_info = self.make_decision(person)
            
            # Submit decision
            result = self.submit_decision(person['person_id'], admit)
            
            # Record decision
            decisions_made.append({
                'person': person,
                'decision': decision_info,
                'result': result
            })
            
            if self.debug and len(decisions_made) % 100 == 0:
                status = self.get_game_status()
                print(f"Progress: {len(decisions_made)} decisions")
                print(f"Admitted: {status['total_admitted']}, Rejected: {status['total_rejected']}")
                print(f"Constraints: {status['constraint_status']}")
            
            # Check for game completion
            if result.get('game_status') == 'completed':
                break
        
        # Get final status
        final_status = self.get_game_status()
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        
        result_summary = {
            'game_id': self.game_state['game_id'],
            'scenario': self.scenario,
            'success': final_status['game_status'] == 'completed',
            'total_admitted': final_status['total_admitted'],
            'total_rejected': final_status['total_rejected'],
            'constraint_status': final_status['constraint_status'],
            'decisions_made': len(decisions_made),
            'elapsed_time': elapsed_time,
            'model_info': {
                'type': type(self.model).__name__,
                'temperature': self.temperature,
                'top_k': self.top_k
            }
        }
        
        if self.debug:
            print("\n" + "="*50)
            print("Game Complete!")
            print(f"Success: {result_summary['success']}")
            print(f"Total Admitted: {result_summary['total_admitted']}")
            print(f"Total Rejected: {result_summary['total_rejected']}")
            print(f"Constraints Met: {all(c['satisfied'] for c in final_status['constraint_status'])}")
            print("="*50)
        
        return result_summary