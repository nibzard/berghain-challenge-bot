# ABOUTME: Ultra-elite LSTM solver using enhanced model with 35 strategic features and attention mechanism
# ABOUTME: Advanced supervised learning approach trained on <800 rejection ultra-elite game data

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver
from ..training.ultra_elite_preprocessor import UltraElitePreprocessor
from ..training.enhanced_lstm_models import UltraEliteLSTMNetwork

logger = logging.getLogger(__name__)


class UltraEliteLSTMSolver(BaseSolver):
    """Solver using an ultra-elite LSTM model with 35 strategic features."""
    
    def __init__(
        self, 
        model_path: str = "models/ultra_elite_lstm_best.pth",
        solver_id: str = "ultra_elite_lstm",
        config_manager=None,
        api_client=None,
        enable_high_score_check: bool = True,
        device: str = 'cpu'
    ):
        strategy = UltraEliteLSTMStrategy(model_path, device=device)
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class UltraEliteLSTMStrategy(BaseDecisionStrategy):
    """
    Ultra-Elite LSTM strategy with 35 strategic features and attention mechanism.
    
    This strategy uses an enhanced LSTM network trained on ultra-elite games
    (<800 rejections) with advanced features including:
    - Lookahead constraints and risk assessment
    - Pattern recognition and streaks
    - Dynamic thresholds and adaptation
    - Attention mechanism for critical decisions
    """
    
    def __init__(
        self, 
        model_path: str = "models/ultra_elite_lstm_best.pth", 
        device: str = 'cpu', 
        fallback_strategy: str = 'intelligent_greedy', 
        sequence_length: int = 100,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize Ultra-Elite LSTM strategy with enhanced model.
        
        Args:
            model_path: Path to the trained ultra-elite model file
            device: Device to run inference on ('cpu' or 'cuda')
            fallback_strategy: Strategy to use if model fails
            sequence_length: Length of decision sequence for context
            confidence_threshold: Minimum confidence for model decisions
        """
        super().__init__({})
        
        self.model_path = model_path
        self.device = device
        self.fallback_strategy = fallback_strategy
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        
        # Enhanced decision history tracking
        self.decision_history = []
        self.game_metrics = {
            'total_decisions': 0,
            'accept_count': 0,
            'constraint_progress': {'young': 0.0, 'well_dressed': 0.0},
            'phase_changes': [],
            'critical_decisions': []
        }
        
        # Ultra-elite preprocessor with 35 features
        self.feature_extractor = UltraElitePreprocessor(sequence_length=sequence_length)
        
        # Load the enhanced model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained Ultra-Elite LSTM model."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.error(f"Ultra-elite model file not found: {self.model_path}")
                return
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration
            model_config = checkpoint.get('model_config', {})
            model_type = model_config.get('model_type', 'ultra_elite')
            
            if model_type == 'ultra_elite':
                self.model = UltraEliteLSTMNetwork(
                    input_dim=35,
                    hidden_dim=model_config.get('hidden_dim', 512),
                    num_layers=model_config.get('num_layers', 3),
                    num_heads=8,
                    dropout=0.2,
                    use_attention=True,
                    use_positional_encoding=True
                )
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            
            # Log model information
            val_acc = checkpoint.get('val_acc', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
            dataset_stats = checkpoint.get('dataset_stats', {})
            
            logger.info(f"✅ Successfully loaded Ultra-Elite LSTM model from {self.model_path}")
            logger.info(f"   Validation accuracy: {val_acc:.2f}%" if isinstance(val_acc, float) else f"   Validation accuracy: {val_acc}")
            logger.info(f"   Validation loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"   Validation loss: {val_loss}")
            
            if dataset_stats:
                logger.info(f"   Trained on {dataset_stats.get('total_games', 'N/A')} games")
                logger.info(f"   Features per timestep: {dataset_stats.get('feature_dim', 35)}")
            
        except Exception as e:
            logger.error(f"Failed to load Ultra-Elite LSTM model from {self.model_path}: {e}")
            self.model_loaded = False
    
    @property
    def name(self) -> str:
        return "Ultra_Elite_LSTM"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Make admission decision using the ultra-elite LSTM model with 35 strategic features.
        
        Args:
            person: Person requesting admission
            game_state: Current game state
            
        Returns:
            Tuple of (accept_decision, reasoning)
        """
        # Initialize for new games
        if not hasattr(game_state, '_ultra_elite_lstm_started'):
            self.reset_for_new_game()
            game_state._ultra_elite_lstm_started = True
        
        # Use ultra-elite model if available
        if self.model_loaded and self.model is not None:
            try:
                # Create decision record for feature extraction
                decision_record = {
                    'person': {
                        'attributes': {
                            'young': person.has_attribute('young'),
                            'well_dressed': person.has_attribute('well_dressed')
                        }
                    },
                    'accepted': True  # Placeholder, will be updated
                }
                
                # Add to history
                self.decision_history.append(decision_record)
                
                # Create comprehensive game data for 35-feature extraction
                constraint_dict = {}
                for constraint in game_state.constraints:
                    constraint_dict[constraint.attribute] = constraint.min_count
                
                game_data = {
                    'constraints': [
                        {'attribute': 'young', 'min_count': constraint_dict.get('young', 600)},
                        {'attribute': 'well_dressed', 'min_count': constraint_dict.get('well_dressed', 600)}
                    ],
                    'decisions': self.decision_history.copy(),
                    'success': True,
                    'admitted_count': game_state.admitted_count,
                    'rejected_count': game_state.rejected_count,
                    'game_id': 'live_prediction'
                }
                
                # Extract 35 strategic features
                features, _ = self.feature_extractor.extract_enhanced_features_and_labels(game_data)
                
                if len(features) == 0:
                    raise ValueError("No features extracted from current game state")
                
                # Use recent history for sequence context (up to sequence_length)
                history_length = min(len(features), self.sequence_length)
                sequence_features = features[-history_length:].reshape(1, history_length, -1)
                
                # Pad if necessary
                if history_length < self.sequence_length:
                    padding = np.zeros((1, self.sequence_length - history_length, features.shape[1]))
                    sequence_features = np.concatenate([padding, sequence_features], axis=1)
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(sequence_features).to(self.device)
                
                # Get prediction with attention
                with torch.no_grad():
                    if hasattr(self.model, 'forward') and 'return_attention' in str(self.model.forward.__code__.co_varnames):
                        output, attention_weights = self.model(input_tensor, return_attention=True)
                    else:
                        output = self.model(input_tensor)
                        attention_weights = None
                    
                    # Get prediction for the last timestep (current decision)
                    last_output = output[0, -1, :]  # (2,) - logits for reject/accept
                    probabilities = torch.softmax(last_output, dim=0)
                    
                    decision_prob = probabilities[1].item()  # Probability of accepting
                    confidence = max(decision_prob, 1 - decision_prob)
                    
                    # Enhanced decision logic with confidence threshold
                    accept = decision_prob > 0.5 and confidence >= self.confidence_threshold
                    
                    # CRITICAL: Constraint safety override before any other logic
                    constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
                    if constraint_override is not None:
                        accept = constraint_override
                        reasoning = f"Ultra_Elite_CONSTRAINT_OVERRIDE: {constraint_reason}"
                    # If low confidence, use strategic fallback
                    elif confidence < self.confidence_threshold:
                        fallback_decision, fallback_reason = self._strategic_fallback(person, game_state)
                        accept = fallback_decision
                        reasoning = f"Ultra_Elite_LSTM_Fallback: low_conf={confidence:.3f}, {fallback_reason}"
                    else:
                        reasoning = f"Ultra_Elite_LSTM: prob={decision_prob:.3f}, conf={confidence:.3f}"
                        
                        # Add attention insight if available
                        if attention_weights is not None:
                            # Find most attended position
                            attention_mean = attention_weights.mean(dim=1)  # Average across heads
                            most_attended_pos = torch.argmax(attention_mean[0, -1, :]).item()
                            attention_score = attention_mean[0, -1, most_attended_pos].item()
                            reasoning += f", att_pos={most_attended_pos}, att_score={attention_score:.3f}"
                
                # Update decision record with actual decision
                self.decision_history[-1]['accepted'] = accept
                
                # Update game metrics
                self._update_game_metrics(accept, person, game_state)
                
                return accept, reasoning
                
            except Exception as e:
                logger.warning(f"Ultra-Elite LSTM prediction failed: {e}, using strategic fallback")
                # Remove the failed decision from history
                if self.decision_history:
                    self.decision_history.pop()
        
        # Strategic fallback when model is unavailable
        # But still check constraints first
        constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
        if constraint_override is not None:
            return constraint_override, f"Ultra_Elite_FALLBACK_CONSTRAINT: {constraint_reason}"
        
        return self._strategic_fallback(person, game_state)
    
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
        
        # SPECIAL CASE: At 999 capacity, accept anyone with needed attributes to reach 1000
        if capacity_remaining == 1:
            if young_deficit > 0 and has_young:
                return True, f"FINAL_ACCEPT_young_deficit={young_deficit}_at_999"
            if well_dressed_deficit > 0 and has_well_dressed:
                return True, f"FINAL_ACCEPT_well_dressed_deficit={well_dressed_deficit}_at_999"
            # If both deficits exist and person has both, prioritize
            if young_deficit > 0 and well_dressed_deficit > 0 and has_young and has_well_dressed:
                return True, f"FINAL_ACCEPT_dual_needed_at_999"
        
        # CAPACITY FILL: If constraints are met, fill remaining capacity
        if young_deficit == 0 and well_dressed_deficit == 0 and capacity_remaining > 0:
            return True, f"FILL_CAPACITY_constraints_met_cap={capacity_remaining}"
        
        # ANTI-LOOP: If at 999 and very close to completing (deficit <= 1), accept any helpful person
        if capacity_remaining == 1 and (young_deficit <= 1 or well_dressed_deficit <= 1):
            if (young_deficit == 1 and has_young) or (well_dressed_deficit == 1 and has_well_dressed):
                return True, f"ANTI_LOOP_999_close_completion_y={young_deficit}_w={well_dressed_deficit}"
        
        # No override needed
        return None, "no_constraint_override"
    
    def _strategic_fallback(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Enhanced strategic fallback based on ultra-elite game patterns."""
        has_space = game_state.admitted_count < game_state.target_capacity
        
        # Get constraint information
        constraint_dict = {}
        for constraint in game_state.constraints:
            constraint_dict[constraint.attribute] = constraint.min_count
        
        young_target = constraint_dict.get('young', 600)
        well_dressed_target = constraint_dict.get('well_dressed', 600)
        
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        
        # Calculate urgency and strategic value
        young_progress = young_current / young_target
        well_dressed_progress = well_dressed_current / well_dressed_target
        
        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')
        
        # Enhanced decision logic based on ultra-elite patterns
        strategic_value = 0.0
        
        # High value for needed dual attributes
        if has_young and has_well_dressed:
            if young_progress < 0.9 or well_dressed_progress < 0.9:
                strategic_value = 1.0
        
        # Medium value for single needed attributes
        elif has_young and young_progress < 0.8:
            strategic_value = 0.6
        elif has_well_dressed and well_dressed_progress < 0.8:
            strategic_value = 0.6
        
        # Emergency acceptance if far behind
        elif young_progress < 0.5 and has_young:
            strategic_value = 0.8
        elif well_dressed_progress < 0.5 and has_well_dressed:
            strategic_value = 0.8
        
        # Capacity pressure adjustment
        capacity_pressure = game_state.admitted_count / game_state.target_capacity
        if capacity_pressure > 0.8:
            strategic_value *= 0.7  # Be more selective when capacity is tight
        
        accept = has_space and strategic_value > 0.5
        
        reasoning = f"Strategic(value={strategic_value:.2f}, young_prog={young_progress:.2f}, wd_prog={well_dressed_progress:.2f})"
        
        return accept, reasoning
    
    def _update_game_metrics(self, accepted: bool, person: Person, game_state: GameState):
        """Update internal game metrics for better decision making."""
        self.game_metrics['total_decisions'] += 1
        
        if accepted:
            self.game_metrics['accept_count'] += 1
            
            # Update constraint progress
            if person.has_attribute('young'):
                young_current = game_state.admitted_attributes.get('young', 0) + 1
                young_target = 600  # Default constraint
                self.game_metrics['constraint_progress']['young'] = young_current / young_target
            
            if person.has_attribute('well_dressed'):
                wd_current = game_state.admitted_attributes.get('well_dressed', 0) + 1
                wd_target = 600
                self.game_metrics['constraint_progress']['well_dressed'] = wd_current / wd_target
        
        # Detect phase changes (acceptance rate shifts)
        if len(self.decision_history) >= 20 and len(self.decision_history) % 10 == 0:
            recent_20 = self.decision_history[-20:]
            recent_10 = self.decision_history[-10:]
            
            rate_20 = np.mean([d['accepted'] for d in recent_20])
            rate_10 = np.mean([d['accepted'] for d in recent_10])
            
            if abs(rate_20 - rate_10) > 0.3:  # Significant phase change
                self.game_metrics['phase_changes'].append({
                    'position': len(self.decision_history),
                    'old_rate': rate_20,
                    'new_rate': rate_10
                })
    
    def reset_for_new_game(self):
        """Reset state for a new game."""
        self.decision_history = []
        self.game_metrics = {
            'total_decisions': 0,
            'accept_count': 0,
            'constraint_progress': {'young': 0.0, 'well_dressed': 0.0},
            'phase_changes': [],
            'critical_decisions': []
        }
        
        logger.debug("Ultra-Elite LSTM strategy reset for new game")


# Register the solver in the factory
def register_ultra_elite_lstm_solver():
    """Register the ultra-elite LSTM solver."""
    try:
        from ..runner.solver_factory import SolverFactory
        SolverFactory.register_solver('ultra_elite_lstm', UltraEliteLSTMSolver)
        logger.info("Ultra-Elite LSTM solver registered successfully")
    except ImportError:
        logger.warning("Could not register Ultra-Elite LSTM solver - factory not available")