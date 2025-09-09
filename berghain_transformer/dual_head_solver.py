"""
ABOUTME: Dual-head transformer solver compatible with existing Berghain solver interface
ABOUTME: Wraps the Colab-trained dual-head model for deployment in the game system
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple

from berghain.core.game_state import GameState
from berghain.core.decision import Decision
from berghain.solvers.base_solver import BaseSolver, DecisionStrategy

logger = logging.getLogger(__name__)

class GameStateEncoder:
    """Simple encoder for game state features matching the training format."""
    
    def __init__(self):
        """Initialize the encoder to match ultra_elite_preprocessor format."""
        pass
    
    def encode_game_state(self, game_state: GameState, person: Dict[str, Any]) -> np.ndarray:
        """Encode game state and person into 14-dimensional feature vector."""
        
        # Get constraint requirements (scenario 1: young + well_dressed)
        young_needed = max(0, 600 - game_state.attribute_counts.get('young', 0))
        well_dressed_needed = max(0, 600 - game_state.attribute_counts.get('well_dressed', 0))
        
        # Calculate pressure and progress metrics
        total_admitted = sum(game_state.attribute_counts.values()) // 2  # Rough estimate
        total_rejected = game_state.rejection_count
        rejection_rate = total_rejected / max(1, total_admitted + total_rejected)
        
        # Game progress (based on admitted people)
        game_progress = min(1.0, total_admitted / 1000.0)
        
        # Time pressure (urgency based on remaining capacity)
        remaining_capacity = max(0, 1000 - total_admitted)
        time_pressure = 1.0 - (remaining_capacity / 1000.0)
        
        # Constraint pressure (how urgent constraint satisfaction is)
        constraint_pressure = (young_needed + well_dressed_needed) / max(1, remaining_capacity)
        constraint_pressure = min(1.0, constraint_pressure)
        
        # Efficiency trend (simplified)
        efficiency_trend = max(0.0, min(1.0, 1.0 - rejection_rate))
        
        # Person attributes
        person_young = person.get('young', False)
        person_well_dressed = person.get('well_dressed', False)
        
        # Person value (how valuable this person is for constraints)
        person_value = 0.0
        if person_young and young_needed > 0:
            person_value += 0.5
        if person_well_dressed and well_dressed_needed > 0:
            person_value += 0.5
        
        # Create 14-dimensional feature vector matching training format
        features = np.array([
            game_state.attribute_counts.get('young', 0),     # young_current
            young_needed,                                     # young_needed
            game_state.attribute_counts.get('well_dressed', 0),  # well_dressed_current
            well_dressed_needed,                             # well_dressed_needed
            constraint_pressure,                             # constraint_pressure
            total_admitted,                                  # total_admitted
            total_rejected,                                  # total_rejected
            rejection_rate,                                  # rejection_rate
            efficiency_trend,                                # efficiency_trend
            game_progress,                                   # game_progress
            time_pressure,                                   # time_pressure
            float(person_young),                             # person_young
            float(person_well_dressed),                      # person_well_dressed
            person_value                                     # person_value
        ], dtype=np.float32)
        
        return features

class DualHeadDecisionStrategy(DecisionStrategy):
    """Decision strategy using dual-head transformer model."""
    
    def __init__(
        self,
        model,
        encoder: GameStateEncoder,
        temperature: float = 1.0,
        confidence_threshold: float = 0.5
    ):
        self.model = model
        self.encoder = encoder
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.decision_history = []
        
    def decide(self, person: Dict[str, Any], game_state: GameState) -> Decision:
        """Make admission decision using dual-head transformer."""
        
        # Encode current game state and person
        state_features = self.encoder.encode_game_state(game_state, person)
        
        # Prepare input for model (create sequence of length 50, padding if needed)
        seq_len = 50
        context_length = min(len(self.decision_history), seq_len - 1)
        
        if context_length > 0:
            # Use recent decision history as context
            context_features = np.stack([
                self.encoder.encode_game_state(hist['game_state'], hist['person'])
                for hist in self.decision_history[-context_length:]
            ])
            
            # Pad to sequence length and add current state
            if context_length < seq_len - 1:
                # Pad with zeros
                padding = np.zeros((seq_len - 1 - context_length, 14), dtype=np.float32)
                sequence_features = np.vstack([padding, context_features, state_features.reshape(1, -1)])
            else:
                # Use last seq_len-1 decisions plus current
                sequence_features = np.vstack([context_features[-(seq_len-1):], state_features.reshape(1, -1)])
        else:
            # No history, pad with zeros and add current state
            padding = np.zeros((seq_len - 1, 14), dtype=np.float32)
            sequence_features = np.vstack([padding, state_features.reshape(1, -1)])
        
        # Convert to tensor [1, seq_len, features]
        input_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            
            # Use combined logits for final decision
            combined_logits = output.combined_logits[0, -1, :]  # Last timestep, both classes
            
            # Apply temperature
            if self.temperature != 1.0:
                combined_logits = combined_logits / self.temperature
            
            # Get probabilities
            probs = F.softmax(combined_logits, dim=-1)
            action_prob = probs[1].item()  # Probability of admission (class 1)
            
            # Get head confidences
            constraint_conf = output.constraint_confidence[0, -1].item()
            efficiency_conf = output.efficiency_confidence[0, -1].item()
            head_weights = output.head_weights[0, -1].cpu().numpy()
        
        # Make decision based on probability
        admit = action_prob > 0.5
        confidence = max(action_prob, 1 - action_prob)
        
        # Create reasoning with dual-head information
        reasoning = f"DualHead: p={action_prob:.3f} (constraint={constraint_conf:.2f}, efficiency={efficiency_conf:.2f})"
        reasoning += f" weights=[{head_weights[0]:.2f}, {head_weights[1]:.2f}]"
        
        if action_prob > 0.7:
            reasoning += " - Strong admission signal"
        elif action_prob < 0.3:
            reasoning += " - Strong rejection signal"
        else:
            reasoning += " - Uncertain decision"
        
        decision = Decision(admit=admit, reasoning=reasoning)
        
        # Store decision in history for context
        self.decision_history.append({
            'game_state': game_state,
            'person': person,
            'decision': decision
        })
        
        # Limit history size
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-50:]
        
        return decision

class DualHeadSolver(BaseSolver):
    """Solver using dual-head transformer trained in Google Colab."""
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 1.0,
        confidence_threshold: float = 0.5
    ):
        # Load model
        self.model = self.load_model(model_path)
        self.encoder = GameStateEncoder()
        
        self.strategy = DualHeadDecisionStrategy(
            model=self.model,
            encoder=self.encoder,
            temperature=temperature,
            confidence_threshold=confidence_threshold
        )
        
    def load_model(self, model_path: str):
        """Load dual-head transformer from Colab deployment file."""
        logger.info(f"Loading dual-head model from {model_path}")
        
        # Import the dual-head transformer
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        
        from models.dual_head_transformer import DualHeadTransformer
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Our Colab deployment format
            model_config = checkpoint.get('model_config', {})
            logger.info(f"Model config: {model_config}")
            
            # Create model with config
            model = DualHeadTransformer(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Log training stats if available
            if 'training_stats' in checkpoint:
                stats = checkpoint['training_stats']
                logger.info(f"Loaded model - Parameters: {stats.get('total_parameters', 'unknown')}, "
                          f"Best accuracy: {stats.get('best_accuracy', 'unknown'):.3f}")
        else:
            raise ValueError("Checkpoint format not recognized")
        
        model.eval()
        return model
    
    def get_strategy_name(self) -> str:
        return "dual_head_transformer"