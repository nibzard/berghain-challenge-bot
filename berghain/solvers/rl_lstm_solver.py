# ABOUTME: LSTM-based reinforcement learning solver for Berghain game
# ABOUTME: Integrates trained RL policy with existing solver architecture

import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver
from ..training.lstm_policy import PolicyInference

logger = logging.getLogger(__name__)


class RLLSTMSolver(BaseSolver):
    """Solver using a trained LSTM-based reinforcement learning policy."""
    
    def __init__(
        self, 
        model_path: str,
        solver_id: str = "rl_lstm",
        config_manager=None,
        api_client=None,
        enable_high_score_check: bool = True,
        device: str = 'cpu'
    ):
        strategy = RLLSTMStrategy(model_path, device=device)
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class RLLSTMStrategy(BaseDecisionStrategy):
    """
    Reinforcement Learning strategy using LSTM policy network.
    
    This strategy uses a pre-trained LSTM policy network to make admission
    decisions. The network was trained using PPO on successful game trajectories
    and should have learned optimal sequential decision-making patterns.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', fallback_strategy: str = 'greedy'):
        """
        Initialize RL strategy with trained model.
        
        Args:
            model_path: Path to the trained model file
            device: Device to run inference on ('cpu' or 'cuda')
            fallback_strategy: Strategy to use if model fails
        """
        # Initialize with empty params - RL model handles its own parameters
        super().__init__({})
        
        self.model_path = model_path
        self.device = device
        self.fallback_strategy = fallback_strategy
        self.policy_inference = None
        self.model_loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained RL model."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return
            
            self.policy_inference = PolicyInference(self.model_path, device=self.device)
            self.model_loaded = True
            logger.info(f"Successfully loaded RL model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load RL model from {self.model_path}: {e}")
            self.model_loaded = False
    
    @property
    def name(self) -> str:
        return "RL_LSTM"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Make admission decision using the trained RL policy.
        
        Args:
            person: Person requesting admission
            game_state: Current game state
            
        Returns:
            Tuple of (accept_decision, reasoning)
        """
        # Reset hidden state for new games
        if hasattr(game_state, '_rl_game_started') and not game_state._rl_game_started:
            if self.policy_inference:
                self.policy_inference.reset()
            game_state._rl_game_started = True
        elif not hasattr(game_state, '_rl_game_started'):
            if self.policy_inference:
                self.policy_inference.reset()
            game_state._rl_game_started = True
        
        # Try to use RL model if available
        if self.model_loaded and self.policy_inference:
            try:
                accept, reasoning = self.policy_inference.get_action(person, game_state, deterministic=True)
                return accept, f"rl_lstm_{reasoning}"
                
            except Exception as e:
                logger.warning(f"RL model inference failed: {e}. Using fallback strategy.")
        
        # Fallback to simple greedy strategy if RL model unavailable
        return self._fallback_decision(person, game_state)
    
    def _fallback_decision(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Fallback decision logic when RL model is unavailable.
        
        Uses a simple greedy strategy similar to existing solvers.
        """
        constraint_shortage = game_state.constraint_shortage()
        
        # Accept if person has any attributes we need
        needed_attributes = []
        for attr in person.attributes:
            if person.has_attribute(attr) and constraint_shortage.get(attr, 0) > 0:
                needed_attributes.append(attr)
        
        if needed_attributes:
            reasoning = f"fallback_greedy_needs_{'_'.join(needed_attributes)}"
            return True, reasoning
        else:
            return False, "fallback_greedy_no_needed_attributes"
    
    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters for logging."""
        return {
            'strategy_type': 'rl_lstm',
            'model_path': self.model_path,
            'device': self.device,
            'model_loaded': self.model_loaded,
            'fallback_strategy': self.fallback_strategy
        }


class RLLSTMHybridStrategy(BaseDecisionStrategy):
    """
    Hybrid strategy that combines RL decisions with rule-based fallbacks.
    
    This strategy uses the RL model for most decisions but applies safety rules
    for edge cases or when the RL model's confidence is low.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cpu',
        confidence_threshold: float = 0.8,
        safety_rules: bool = True
    ):
        super().__init__({
            'confidence_threshold': confidence_threshold,
            'safety_rules': safety_rules
        })
        
        self.rl_strategy = RLLSTMStrategy(model_path, device)
        self.confidence_threshold = confidence_threshold
        self.safety_rules = safety_rules
    
    @property
    def name(self) -> str:
        return "RL_LSTM_Hybrid"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Make decision using hybrid RL + rule-based approach.
        """
        # CRITICAL: Constraint safety override - this overrides all other logic
        constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
        if constraint_override is not None:
            return constraint_override, f"RL_LSTM_HYBRID_CONSTRAINT_OVERRIDE: {constraint_reason}"
        
        # Safety rules - always accept dual attribute people early in game
        if self.safety_rules and game_state.admitted_count < 500:
            constraint_attrs = [c.attribute for c in game_state.constraints]
            if len(constraint_attrs) >= 2:
                has_both = all(person.has_attribute(attr) for attr in constraint_attrs[:2])
                if has_both:
                    return True, "hybrid_safety_dual_early"
        
        # Use RL decision
        rl_accept, rl_reasoning = self.rl_strategy.should_accept(person, game_state)
        
        # Apply additional safety checks
        if self.safety_rules:
            # Don't accept fillers if all constraints are met
            if rl_accept and game_state.are_all_constraints_satisfied():
                constraint_attrs = [c.attribute for c in game_state.constraints]
                has_needed_attr = any(person.has_attribute(attr) for attr in constraint_attrs)
                if not has_needed_attr:
                    return False, "hybrid_safety_reject_filler_constraints_met"
            
            # Emergency accept if very close to constraint failure
            if not rl_accept:
                shortage = game_state.constraint_shortage()
                remaining_capacity = game_state.remaining_capacity
                for attr, short in shortage.items():
                    if short > 0 and person.has_attribute(attr):
                        # If we have very few spots left and need this attribute
                        if remaining_capacity < short * 2:  # Less than 2x the shortage
                            return True, f"hybrid_safety_emergency_accept_{attr}"
        
        return rl_accept, f"hybrid_{rl_reasoning}"
    
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
    
    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        params = self.rl_strategy.get_params()
        params.update({
            'strategy_type': 'rl_lstm_hybrid',
            'confidence_threshold': self.confidence_threshold,
            'safety_rules': self.safety_rules
        })
        return params


# Factory functions for easy integration with config system
def create_rl_lstm_solver(
    model_path: str,
    solver_id: str = "rl_lstm",
    config_manager=None,
    **kwargs
) -> RLLSTMSolver:
    """
    Factory function to create RL LSTM solver.
    
    Args:
        model_path: Path to trained model
        solver_id: Unique solver identifier
        config_manager: Configuration manager (optional)
        **kwargs: Additional arguments passed to solver
    
    Returns:
        RLLSTMSolver instance
    """
    return RLLSTMSolver(
        model_path=model_path,
        solver_id=solver_id,
        config_manager=config_manager,
        **kwargs
    )


def create_rl_lstm_hybrid_solver(
    model_path: str,
    solver_id: str = "rl_lstm_hybrid",
    config_manager=None,
    **kwargs
) -> BaseSolver:
    """
    Factory function to create hybrid RL LSTM solver.
    
    Args:
        model_path: Path to trained model
        solver_id: Unique solver identifier
        config_manager: Configuration manager (optional)
        **kwargs: Additional arguments passed to strategy
    
    Returns:
        BaseSolver with RLLSTMHybridStrategy
    """
    strategy = RLLSTMHybridStrategy(model_path, **kwargs)
    return BaseSolver(strategy, solver_id)