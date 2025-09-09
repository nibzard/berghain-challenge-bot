# ABOUTME: Elite LSTM solver using model trained on elite games
# ABOUTME: Uses supervised learning approach trained on high-quality elite game data

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver
from ..training.enhanced_data_preprocessor import EnhancedGameDataPreprocessor
from .rbcr_solver import RBCRStrategy

logger = logging.getLogger(__name__)


class EnhancedLSTMPolicyNetwork(nn.Module):
    """Enhanced LSTM network for learning elite game strategies."""
    
    def __init__(self, input_dim=15, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Enhanced LSTM with more capacity
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-layer output with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, 2)  # Binary classification
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 4)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, self.hidden_state = self.lstm(x, getattr(self, 'hidden_state', None))
        
        # Apply fully connected layers to the last timestep
        out = lstm_out[:, -1, :]  # Take only the last timestep
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.layer_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out  # Shape: (batch_size, 2)
    
    def reset_hidden_state(self):
        """Reset the hidden state for a new game."""
        if hasattr(self, 'hidden_state'):
            delattr(self, 'hidden_state')


class EliteLSTMSolver(BaseSolver):
    """Solver using a trained LSTM model from elite games."""
    
    def __init__(
        self, 
        model_path: str,
        solver_id: str = "elite_lstm",
        config_manager=None,
        api_client=None,
        enable_high_score_check: bool = True,
        device: str = 'cpu'
    ):
        strategy = EliteLSTMStrategy(model_path, device=device)
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class EliteLSTMStrategy(BaseDecisionStrategy):
    """
    Elite LSTM strategy using supervised learning on high-quality games.
    
    This strategy uses a pre-trained LSTM network trained on elite game data
    collected from successful runs. The model learned from the best strategies
    and should make optimal admission decisions.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', fallback_strategy: str = 'greedy', sequence_length: int = 100):
        """
        Initialize Elite LSTM strategy with trained model.
        
        Args:
            model_path: Path to the trained model file
            device: Device to run inference on ('cpu' or 'cuda')
            fallback_strategy: Strategy to use if model fails
            sequence_length: Length of decision sequence for context
        """
        super().__init__({})
        
        self.model_path = model_path
        self.device = device
        self.fallback_strategy = fallback_strategy
        self.sequence_length = sequence_length
        self.model = None
        self.model_loaded = False
        
        # Decision history for sequential prediction
        self.decision_history = []
        self.feature_extractor = EnhancedGameDataPreprocessor(sequence_length=sequence_length)
        
        # RBCR strategy for early decisions (cold-start solution)
        rbcr_params = {
            'rate_floor_early': 0.62,  # Higher acceptance rate floor for elite games
            'rate_floor_mid': 0.60,
            'rate_floor_late': 0.58,
            'filler_max': 0.18,
            'resolve_every': 40
        }
        self.rbcr_strategy = RBCRStrategy(rbcr_params)
        self.rbcr_cutoff = 60  # Use RBCR for first 60 decisions to avoid cold-start
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained Elite LSTM model."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get('model_config', {
                'input_dim': 15,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout': 0.3
            })
            
            # Create model
            self.model = EnhancedLSTMPolicyNetwork(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Successfully loaded Elite LSTM model from {self.model_path}")
            logger.info(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to load Elite LSTM model from {self.model_path}: {e}")
            self.model_loaded = False
    
    @property
    def name(self) -> str:
        return "Elite_LSTM"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Make admission decision using the trained Elite LSTM model.
        
        Args:
            person: Person requesting admission
            game_state: Current game state
            
        Returns:
            Tuple of (accept_decision, reasoning)
        """
        # Reset for new games
        if not hasattr(game_state, '_elite_lstm_started'):
            self.reset_for_new_game()
            game_state._elite_lstm_started = True
        
        # Use RBCR for early decisions to avoid LSTM cold-start problem
        total_decisions = game_state.admitted_count + game_state.rejected_count
        if total_decisions < self.rbcr_cutoff:
            try:
                accept, rbcr_reasoning = self.rbcr_strategy.should_accept(person, game_state)
                
                # CRITICAL: Constraint safety override
                constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
                if constraint_override is not None:
                    accept = constraint_override
                    reasoning = f"Elite_LSTM_RBCR_CONSTRAINT_OVERRIDE: {constraint_reason}"
                else:
                    reasoning = f"Elite_LSTM_RBCR: {rbcr_reasoning}"
                
                # Track decision for LSTM transition
                decision_record = {
                    'attributes': {
                        'young': person.has_attribute('young'),
                        'well_dressed': person.has_attribute('well_dressed')
                    },
                    'decision': accept
                }
                self.decision_history.append(decision_record)
                
                return accept, reasoning
                
            except Exception as e:
                logger.warning(f"Elite LSTM RBCR decision failed: {e}, falling back to LSTM")
        
        # Use LSTM for later decisions once we have enough history
        if self.model_loaded and self.model is not None:
            try:
                # Create decision record for feature extraction
                decision_record = {
                    'attributes': {
                        'young': person.has_attribute('young'),
                        'well_dressed': person.has_attribute('well_dressed')
                    },
                    'decision': True  # Placeholder, will be updated
                }
                
                # Add to history
                self.decision_history.append(decision_record)
                
                # Create game data for feature extraction
                constraint_dict = {}
                for constraint in game_state.constraints:
                    constraint_dict[constraint.attribute] = constraint.min_count
                
                game_data = {
                    'constraints': [
                        {'attribute': 'young', 'min_count': constraint_dict.get('young', 600)},
                        {'attribute': 'well_dressed', 'min_count': constraint_dict.get('well_dressed', 600)}
                    ],
                    'decisions': self.decision_history.copy(),
                    'success': True,  # Placeholder
                    'admitted_count': game_state.admitted_count,
                    'rejected_count': game_state.rejected_count
                }
                
                # Extract features
                features, _ = self.feature_extractor.extract_features_and_labels(game_data)
                if len(features) == 0:
                    raise ValueError("No features extracted")
                
                # Take the last feature vector (current decision)
                current_features = features[-1:].reshape(1, 1, -1)  # (batch=1, seq=1, features)
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(current_features).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(input_tensor)  # (1, 2)
                    probabilities = torch.softmax(output, dim=1)
                    decision_prob = probabilities[0, 1].item()  # Probability of accepting
                    
                    # Decision: accept if probability > 0.5
                    accept = decision_prob > 0.5
                    confidence = max(decision_prob, 1 - decision_prob)
                
                # CRITICAL: Constraint safety override before returning
                constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
                if constraint_override is not None:
                    accept = constraint_override
                    reasoning = f"Elite_LSTM_CONSTRAINT_OVERRIDE: {constraint_reason}"
                else:
                    reasoning = f"Elite_LSTM: {decision_prob:.3f} confidence ({confidence:.3f})"
                
                # Update decision record with actual decision
                self.decision_history[-1]['decision'] = accept
                
                return accept, reasoning
                
            except Exception as e:
                logger.warning(f"Elite LSTM prediction failed: {e}, using fallback")
                # Remove the failed decision from history
                if self.decision_history:
                    self.decision_history.pop()
        
        # Fallback to greedy strategy
        return self._fallback_decision(person, game_state)
    
    def _fallback_decision(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Fallback decision when model is unavailable."""
        # Simple greedy: accept if we need the attributes and have capacity
        has_space = game_state.admitted_count < game_state.target_capacity
        
        # Check what constraints we need
        constraint_dict = {}
        for constraint in game_state.constraints:
            constraint_dict[constraint.attribute] = constraint.min_count
        
        young_needed = constraint_dict.get('young', 600)
        well_dressed_needed = constraint_dict.get('well_dressed', 600)
        
        young_admitted = game_state.admitted_attributes.get('young', 0)
        well_dressed_admitted = game_state.admitted_attributes.get('well_dressed', 0)
        
        needs_young = (young_admitted < young_needed) and person.has_attribute('young')
        needs_well_dressed = (well_dressed_admitted < well_dressed_needed) and person.has_attribute('well_dressed')
        
        accept = has_space and (needs_young or needs_well_dressed)
        reasoning = f"Fallback({self.fallback_strategy}): space={has_space}, needs={'young' if needs_young else ''}{'well_dressed' if needs_well_dressed else ''}"
        
        return accept, reasoning
    
    def _constraint_safety_check(self, person: Person, game_state: GameState) -> Tuple[Optional[bool], str]:
        """
        Critical constraint safety check - overrides all other logic.
        Returns (None, reason) if no override needed.
        Returns (True/False, reason) if override is required.
        """
        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')
        
        # Get current constraint status
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        
        # Calculate deficits
        young_deficit = max(0, 600 - young_current)
        well_dressed_deficit = max(0, 600 - well_dressed_current)
        
        # MANDATORY ACCEPT: Critical constraint situation
        if capacity_remaining <= max(young_deficit, well_dressed_deficit):
            # Running out of capacity and still need constraints
            if young_deficit > 0 and has_young:
                return True, f"MUST_ACCEPT_young_deficit={young_deficit}_cap={capacity_remaining}"
            if well_dressed_deficit > 0 and has_well_dressed:
                return True, f"MUST_ACCEPT_well_dressed_deficit={well_dressed_deficit}_cap={capacity_remaining}"
            # If we need both and person has both
            if young_deficit > 0 and well_dressed_deficit > 0 and has_young and has_well_dressed:
                return True, f"MUST_ACCEPT_dual_needed_y={young_deficit}_w={well_dressed_deficit}_cap={capacity_remaining}"
        
        # MANDATORY REJECT: Would make constraint satisfaction impossible
        if capacity_remaining > 0:
            # Check if accepting this person would use capacity we need for constraints
            remaining_after = capacity_remaining - 1
            if remaining_after < (young_deficit + well_dressed_deficit):
                # Only allow if this person helps with constraints
                if not ((young_deficit > 0 and has_young) or (well_dressed_deficit > 0 and has_well_dressed)):
                    return False, f"MUST_REJECT_constraint_safety_y_need={young_deficit}_w_need={well_dressed_deficit}_cap_after={remaining_after}"
        
        # CAPACITY FILL: If constraints are met, fill remaining capacity
        if young_deficit == 0 and well_dressed_deficit == 0 and capacity_remaining > 0:
            return True, f"FILL_CAPACITY_constraints_met_cap={capacity_remaining}"
        
        # No override needed
        return None, "no_constraint_override"
    
    def reset_for_new_game(self):
        """Reset state for a new game."""
        self.decision_history = []
        if self.model is not None:
            self.model.reset_hidden_state()
        # Reset RBCR strategy state
        if hasattr(self, 'rbcr_strategy'):
            # RBCR is stateless, no reset needed
            pass
        logger.debug("Elite LSTM strategy reset for new game")
