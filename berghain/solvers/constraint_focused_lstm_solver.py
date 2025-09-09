# ABOUTME: Constraint-focused LSTM solver that prioritizes meeting all constraints exactly
# ABOUTME: Enhanced ultra-elite model with constraint satisfaction override logic

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
from .rbcr_solver import RBCRStrategy

logger = logging.getLogger(__name__)


class ConstraintFocusedLSTMSolver(BaseSolver):
    """Solver using constraint-focused LSTM with mandatory constraint satisfaction."""
    
    def __init__(
        self, 
        model_path: str = "models/fixed_ultra_elite_lstm_best.pth",
        solver_id: str = "constraint_focused_lstm",
        config_manager=None,
        api_client=None,
        enable_high_score_check: bool = True,
        device: str = 'cpu'
    ):
        strategy = ConstraintFocusedLSTMStrategy(model_path, device=device)
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class ConstraintFocusedLSTMStrategy(BaseDecisionStrategy):
    """
    Constraint-focused LSTM strategy that prioritizes meeting all constraints.
    
    This strategy uses the ultra-elite LSTM model but adds constraint-aware
    decision logic to ensure we meet all requirements:
    - 600+ young people
    - 600+ well_dressed people  
    - 1000 total admitted (capacity limit)
    """
    
    def __init__(
        self, 
        model_path: str = "models/fixed_ultra_elite_lstm_best.pth", 
        device: str = 'cpu', 
        sequence_length: int = 80,
        confidence_threshold: float = 0.5,
        adaptive_confidence: bool = True
    ):
        super().__init__({})
        
        self.model_path = model_path
        self.device = device
        self.sequence_length = sequence_length
        self.base_confidence_threshold = confidence_threshold
        self.adaptive_confidence = adaptive_confidence
        self.model = None
        self.model_loaded = False
        
        # Decision history and constraint tracking
        self.decision_history = []
        self.constraint_tracker = {
            'young_admitted': 0,
            'well_dressed_admitted': 0,
            'total_admitted': 0,
            'total_decisions': 0
        }
        
        # Feature extractor
        self.feature_extractor = UltraElitePreprocessor(sequence_length=sequence_length)
        
        # Hybrid optimal control parameters
        self.hybrid_mode = True
        self.p_young = 0.323
        self.p_well_dressed = 0.323  
        self.p_both = 0.144
        
        # RBCR strategy for early decisions
        # Use more accepting RBCR parameters for better early acceptance
        rbcr_params = {
            'rate_floor_early': 0.60,  # Higher acceptance rate floor (was 0.575)
            'rate_floor_mid': 0.58,    # Higher mid-game floor (was 0.56)
            'filler_max': 0.18,        # Allow more filler (was 0.14)
            'resolve_every': 40        # Recalculate more frequently (was 50)
        }
        self.rbcr_strategy = RBCRStrategy(rbcr_params)
        self.rbcr_cutoff = 60  # Use RBCR for more decisions (was 50)
        
        # Load model
        self._load_model()
    
    def _get_adaptive_confidence_threshold(self, game_state: GameState) -> float:
        """Calculate adaptive confidence threshold based on game progress."""
        if not self.adaptive_confidence:
            return self.base_confidence_threshold
        
        # Use decision count as proxy for game progress
        total_decisions = self.constraint_tracker['total_decisions']
        capacity_remaining = max(1000 - game_state.admitted_count, 0)
        
        # Start moderate, become more accepting as game progresses (optimized for higher acceptance)
        if total_decisions < 200:
            # Early game: moderate selectivity (reduced from 0.75)
            return 0.65
        elif capacity_remaining > 400:
            # Mid game: less selective (reduced from 0.65)
            return 0.55
        elif capacity_remaining > 200:
            # Late mid game: more accepting (reduced from 0.45)
            return 0.40
        else:
            # End game: very accepting (reduced from 0.3)
            return 0.25
    
    def _calculate_expected_value_heuristic(self, person: Person, game_state: GameState) -> float:
        """Calculate simplified expected value using optimal control insights."""
        if not self.hybrid_mode:
            return 0.0
        
        # Current state
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        capacity_remaining = max(1000 - game_state.admitted_count, 0)
        
        # Deficits
        young_deficit = max(600 - young_current, 0) 
        well_dressed_deficit = max(600 - well_dressed_current, 0)
        
        # Person attributes
        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')
        
        if capacity_remaining <= 0:
            return float('inf')  # No capacity
            
        # Calculate acceptance value (negative is good)
        accept_value = 0.0
        
        # Value of fulfilling constraints
        if young_deficit > 0 and has_young:
            accept_value -= min(young_deficit, 100) * 0.5  # Reward for young
        if well_dressed_deficit > 0 and has_well_dressed:
            accept_value -= min(well_dressed_deficit, 100) * 0.5  # Reward for well_dressed
            
        # Bonus for dual attributes when we need both
        if young_deficit > 0 and well_dressed_deficit > 0 and has_young and has_well_dressed:
            accept_value -= 20.0  # Strong preference for dual
            
        # Penalty for using capacity when constraints are met
        if young_deficit == 0 and well_dressed_deficit == 0:
            accept_value += 25.0  # Strong penalty for unnecessary admission
        elif young_deficit == 0 or well_dressed_deficit == 0:
            accept_value += 5.0  # Moderate penalty when one constraint is met
            
        # Expected future cost if we reject (simplified)
        reject_value = 1.0  # Cost of one rejection
        
        # Early game: be more selective (lower reject penalty to encourage rejections)
        if capacity_remaining > 600:
            reject_value *= 0.5  # Make rejections less costly early
        elif capacity_remaining > 400:
            reject_value *= 0.7
            
        return accept_value - reject_value  # Negative means prefer accept
    
    
    def _load_model(self) -> None:
        """Load the trained Ultra-Elite LSTM model."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get('model_config', {})
            
            # Create model with config from checkpoint
            self.model = UltraEliteLSTMNetwork(
                input_dim=model_config.get('input_dim', 35),
                hidden_dim=model_config.get('hidden_dim', 128),
                num_layers=model_config.get('num_layers', 2),
                num_heads=4,
                dropout=0.2,
                use_attention=True,
                use_positional_encoding=True
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"✅ Loaded Constraint-Focused LSTM model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    @property
    def name(self) -> str:
        return "Constraint_Focused_LSTM"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Make admission decision with mandatory constraint satisfaction logic.
        """
        # Initialize for new games
        if not hasattr(game_state, '_constraint_focused_started'):
            self.reset_for_new_game()
            game_state._constraint_focused_started = True
        
        # Update our constraint tracking
        self.constraint_tracker['total_decisions'] += 1
        total_decisions = self.constraint_tracker['total_decisions']
        
        # Get current constraint status
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        total_admitted = game_state.admitted_count
        
        # Calculate constraint deficits
        young_deficit = max(600 - young_current, 0)
        well_dressed_deficit = max(600 - well_dressed_current, 0)
        capacity_remaining = max(1000 - total_admitted, 0)
        
        # Person attributes
        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')
        
        # CONSTRAINT PRIORITY LOGIC: Override model if needed for constraint satisfaction
        
        # 1. EMERGENCY: Must accept if constraint is in danger (more aggressive)
        emergency_accept = False
        emergency_reason = ""
        
        # If we're running out of capacity and still need constraints (reduced threshold)
        if capacity_remaining <= max(young_deficit, well_dressed_deficit) * 0.8:
            if young_deficit > 0 and has_young:
                emergency_accept = True
                emergency_reason = f"EMERGENCY_young_deficit={young_deficit}_cap={capacity_remaining}"
            elif well_dressed_deficit > 0 and has_well_dressed:
                emergency_accept = True
                emergency_reason = f"EMERGENCY_well_dressed_deficit={well_dressed_deficit}_cap={capacity_remaining}"
        
        # 2. CRITICAL: Very low capacity left, need dual attributes (reduced threshold)
        elif capacity_remaining < 30 and young_deficit > 0 and well_dressed_deficit > 0:
            if has_young and has_well_dressed:
                emergency_accept = True
                emergency_reason = f"CRITICAL_dual_needed_cap={capacity_remaining}"
        
        # 3. HIGH PRIORITY: Large deficit with needed attribute (reduced threshold)
        elif young_deficit > 75 and has_young:
            emergency_accept = True
            emergency_reason = f"HIGH_PRIORITY_young_deficit={young_deficit}"
        elif well_dressed_deficit > 75 and has_well_dressed:
            emergency_accept = True
            emergency_reason = f"HIGH_PRIORITY_well_dressed_deficit={well_dressed_deficit}"
        
        # If emergency override, accept immediately
        if emergency_accept and capacity_remaining > 0:
            self._update_tracking(True, has_young, has_well_dressed)
            return True, emergency_reason
        
        # 4. MANDATORY REJECT: No capacity left (should never happen - game must use all 1000)
        if capacity_remaining <= 0:
            return False, "NO_CAPACITY_REMAINING"
        
        # 5. STRATEGIC REJECT: Don't need this person's attributes and capacity is tight (reduced threshold)
        if capacity_remaining < 50:
            if young_deficit == 0 and well_dressed_deficit == 0:
                # All constraints met, can be very selective
                if not (has_young or has_well_dressed):
                    return False, "CONSTRAINTS_MET_selective"
            elif young_deficit == 0 and not has_well_dressed:
                # Only need well_dressed, this person doesn't have it
                return False, f"ONLY_NEED_well_dressed_deficit={well_dressed_deficit}"
            elif well_dressed_deficit == 0 and not has_young:
                # Only need young, this person doesn't have it  
                return False, f"ONLY_NEED_young_deficit={young_deficit}"
        
        # 0. EARLY GAME RBCR: Use proven RBCR logic for first decisions (cold-start fix)
        if total_decisions <= self.rbcr_cutoff:
            rbcr_accept, rbcr_reason = self.rbcr_strategy.should_accept(person, game_state)
            if rbcr_accept:
                self._update_tracking(True, has_young, has_well_dressed)
                return True, f"RBCR_early_{rbcr_reason}_dec={total_decisions}"
            else:
                return False, f"RBCR_early_{rbcr_reason}_dec={total_decisions}"
        
        # 6. MANDATORY CAPACITY FILL: Must reach exactly 1000 admissions
        if capacity_remaining > 0:
            # If constraints are satisfied, accept everyone to fill capacity
            if young_deficit == 0 and well_dressed_deficit == 0:
                self._update_tracking(True, has_young, has_well_dressed)
                return True, f"MUST_FILL_CAPACITY_constraints_met_cap={capacity_remaining}"
            
            # If capacity is running low but constraints not met, be strategic
            elif capacity_remaining <= max(young_deficit, well_dressed_deficit) + 10:
                # Critical capacity situation - prioritize constraint satisfaction
                if (young_deficit > 0 and has_young) or (well_dressed_deficit > 0 and has_well_dressed):
                    self._update_tracking(True, has_young, has_well_dressed)
                    return True, f"CRITICAL_CAPACITY_need_constraints_cap={capacity_remaining}_y={young_deficit}_w={well_dressed_deficit}"
                elif has_young and has_well_dressed:
                    # Dual attributes are always valuable in critical situations
                    self._update_tracking(True, has_young, has_well_dressed)
                    return True, f"CRITICAL_CAPACITY_dual_attrs_cap={capacity_remaining}"
        
        # 7. HYBRID OPTIMAL-LSTM DECISION (after sufficient history)
        if self.model_loaded and self.model is not None:
            try:
                # Get both LSTM prediction and optimal control heuristic
                model_accept, model_reasoning = self._get_model_prediction(person, game_state)
                expected_value = self._calculate_expected_value_heuristic(person, game_state)
                
                # Hybrid decision: combine LSTM with optimal control insights
                optimal_suggests_accept = expected_value < 0  # Negative value means accept is good
                
                # Hybrid decision logic: combine LSTM and optimal control
                if model_accept and optimal_suggests_accept:
                    # Both agree to accept
                    if capacity_remaining > 0:
                        self._update_tracking(True, has_young, has_well_dressed)
                        return True, f"HYBRID_BOTH_ACCEPT: {model_reasoning} ev={expected_value:.2f}"
                elif model_accept and not optimal_suggests_accept:
                    # LSTM says accept, optimal says reject - be conservative in early game
                    total_decisions = self.constraint_tracker['total_decisions']
                    if total_decisions < 1200 and expected_value > 2:  # Strong reject signal (lowered threshold)
                        return False, f"HYBRID_OPTIMAL_OVERRIDE_REJECT: {model_reasoning} ev={expected_value:.2f}"
                    elif capacity_remaining > 0:
                        self._update_tracking(True, has_young, has_well_dressed)
                        return True, f"HYBRID_LSTM_OVERRIDE_ACCEPT: {model_reasoning} ev={expected_value:.2f}"
                elif not model_accept and optimal_suggests_accept:
                    # LSTM says reject, optimal says accept - constraint override logic
                    if young_deficit > 30 and has_young and capacity_remaining > young_deficit:
                        self._update_tracking(True, has_young, has_well_dressed)
                        return True, f"HYBRID_OPTIMAL_OVERRIDE_ACCEPT_y={young_deficit} ev={expected_value:.2f}"
                    elif well_dressed_deficit > 30 and has_well_dressed and capacity_remaining > well_dressed_deficit:
                        self._update_tracking(True, has_young, has_well_dressed)
                        return True, f"HYBRID_OPTIMAL_OVERRIDE_ACCEPT_w={well_dressed_deficit} ev={expected_value:.2f}"
                    elif young_deficit > 0 and well_dressed_deficit > 0 and has_young and has_well_dressed:
                        self._update_tracking(True, has_young, has_well_dressed)
                        return True, f"HYBRID_DUAL_OVERRIDE_y={young_deficit}_w={well_dressed_deficit} ev={expected_value:.2f}"
                    else:
                        return False, f"HYBRID_BOTH_LEAN_REJECT: {model_reasoning} ev={expected_value:.2f}"
                else:
                    # Both agree to reject
                    return False, f"HYBRID_BOTH_REJECT: {model_reasoning} ev={expected_value:.2f}"
                
                # Fallback
                if capacity_remaining <= 0:
                    return False, f"HYBRID_NO_CAPACITY: {model_reasoning}"
                        
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}, using fallback")
        
        # 8. FALLBACK: Ensure we NEVER leave capacity unused
        if capacity_remaining > 0:
            # ALWAYS accept if capacity remains - we must fill all 1000 slots
            if young_deficit == 0 and well_dressed_deficit == 0:
                # Constraints satisfied - accept anyone to fill remaining capacity
                self._update_tracking(True, has_young, has_well_dressed)
                return True, f"FALLBACK_FILL_REMAINING_CAPACITY_cap={capacity_remaining}"
            elif (young_deficit > 0 and has_young) or (well_dressed_deficit > 0 and has_well_dressed):
                # Need these attributes
                self._update_tracking(True, has_young, has_well_dressed)
                return True, f"FALLBACK_NEED_ATTRS_y={young_deficit}_w={well_dressed_deficit}"
            elif has_young and has_well_dressed:
                # Dual attributes are always valuable
                self._update_tracking(True, has_young, has_well_dressed)
                return True, f"FALLBACK_dual_valuable"
            else:
                # Even if we don't need their specific attributes, we must fill capacity
                # Only reject if capacity is very tight and we need specific constraints
                if capacity_remaining <= max(young_deficit, well_dressed_deficit):
                    return False, f"FALLBACK_capacity_too_tight_cap={capacity_remaining}_y={young_deficit}_w={well_dressed_deficit}"
                else:
                    self._update_tracking(True, has_young, has_well_dressed)
                    return True, f"FALLBACK_fill_capacity_anyway_cap={capacity_remaining}"
        else:
            return False, "FALLBACK_exactly_1000_reached"
    
    def _get_model_prediction(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Get prediction from the LSTM model."""
        # Create decision record for feature extraction
        decision_record = {
            'person': {
                'attributes': {
                    'young': person.has_attribute('young'),
                    'well_dressed': person.has_attribute('well_dressed')
                }
            },
            'accepted': True  # Placeholder
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
            'success': True,
            'admitted_count': game_state.admitted_count,
            'rejected_count': game_state.rejected_count,
            'game_id': 'constraint_focused_prediction'
        }
        
        # Extract features
        features, _ = self.feature_extractor.extract_enhanced_features_and_labels(game_data)
        
        if len(features) == 0:
            raise ValueError("No features extracted")
        
        # Use recent history for context
        history_length = min(len(features), self.sequence_length)
        sequence_features = features[-history_length:].reshape(1, history_length, -1)
        
        # Pad if necessary
        if history_length < self.sequence_length:
            padding = np.zeros((1, self.sequence_length - history_length, features.shape[1]))
            sequence_features = np.concatenate([padding, sequence_features], axis=1)
        
        # Convert to tensor and predict
        input_tensor = torch.FloatTensor(sequence_features).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            last_output = output[0, -1, :]  # Last timestep
            probabilities = torch.softmax(last_output, dim=0)
            
            decision_prob = probabilities[1].item()  # Accept probability
            confidence = max(decision_prob, 1 - decision_prob)
            
            # Model decision with adaptive confidence threshold
            current_threshold = self._get_adaptive_confidence_threshold(game_state)
            accept = decision_prob > 0.5 and confidence >= current_threshold
            
            # Update decision record
            self.decision_history[-1]['accepted'] = accept
            
            return accept, f"prob={decision_prob:.3f}_conf={confidence:.3f}_thresh={current_threshold:.2f}"
    
    def _update_tracking(self, accepted: bool, has_young: bool, has_well_dressed: bool):
        """Update internal constraint tracking."""
        if accepted:
            self.constraint_tracker['total_admitted'] += 1
            if has_young:
                self.constraint_tracker['young_admitted'] += 1
            if has_well_dressed:
                self.constraint_tracker['well_dressed_admitted'] += 1
    
    def reset_for_new_game(self):
        """Reset state for a new game."""
        self.decision_history = []
        self.constraint_tracker = {
            'young_admitted': 0,
            'well_dressed_admitted': 0,
            'total_admitted': 0,
            'total_decisions': 0
        }
        logger.debug("Constraint-Focused LSTM strategy reset for new game")