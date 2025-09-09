"""ABOUTME: Adaptive Random Walk Hybrid (ARWH) strategy combining best performing strategies with intelligent selection
ABOUTME: Uses Thompson Sampling to learn optimal strategy mixing ratios for each game phase"""

import math
import random
import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver

# Import strategies we'll use as components
from .rbcr2_solver import RBCR2Strategy
from .perfect_solver import PerfectStrategy
from .ultimate3_solver import Ultimate3Strategy
from .dual_deficit_solver import DualDeficitController


class RandomWalkHybridSolver(BaseSolver):
    """Adaptive Random Walk Hybrid solver using multiple strategy components."""
    
    def __init__(self, solver_id: str = "random_walk_hybrid", config_manager=None, api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("random_walk_hybrid")
        strategy = RandomWalkHybridStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class RandomWalkHybridStrategy(BaseDecisionStrategy):
    """Adaptive strategy that uses random walk to select between proven strategies."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Phase thresholds
            'early_phase_cutoff': 0.3,
            'mid_phase_cutoff': 0.7,
            
            # Strategy weights (initial probabilities)
            'early_weights': [0.5, 0.3, 0.1, 0.1],  # [RBCR2, Perfect, Ultimate3, DualDeficit]
            'mid_weights': [0.2, 0.5, 0.2, 0.1],
            'late_weights': [0.1, 0.2, 0.6, 0.1],
            
            # Adaptive learning (Thompson Sampling)
            'thompson_alpha': 1.0,  # Prior success count
            'thompson_beta': 1.0,   # Prior failure count
            'learning_window': 20,  # Number of decisions to track per strategy
            'adaptation_rate': 0.1, # How fast to adapt weights
            
            # Safety parameters
            'min_strategy_prob': 0.05,      # Minimum probability for any strategy
            'constraint_boost_factor': 0.3, # Boost for emergency strategy when constraints at risk
            'emergency_override_threshold': 100, # Deficit level that triggers emergency boost
            
            # Strategy mixing
            'enable_strategy_mixing': True,  # Allow blending decisions from multiple strategies
            'mixing_confidence_threshold': 0.7, # Only mix when top strategy confidence < this
            'max_strategies_in_mix': 2,     # Maximum strategies to blend
            
            # Random walk parameters
            'walk_temperature': 0.1,        # Temperature for probability sampling
            'walk_momentum': 0.9,           # Momentum for strategy transitions
            'walk_exploration': 0.05,       # Exploration rate for strategy selection
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # Initialize component strategies
        self._init_component_strategies()
        
        # Strategy tracking
        self._strategy_names = ['RBCR2', 'Perfect', 'Ultimate3', 'DualDeficit']
        self._current_strategy_idx = 0
        self._strategy_momentum = 0.0
        
        # Thompson Sampling state (Alpha/Beta parameters for each strategy in each phase)
        self._thompson_alphas = {
            'early': [float(self.params['thompson_alpha'])] * 4,
            'mid': [float(self.params['thompson_alpha'])] * 4,
            'late': [float(self.params['thompson_alpha'])] * 4,
        }
        self._thompson_betas = {
            'early': [float(self.params['thompson_beta'])] * 4,
            'mid': [float(self.params['thompson_beta'])] * 4,
            'late': [float(self.params['thompson_beta'])] * 4,
        }
        
        # Performance tracking
        self._decision_history: List[Dict] = []
        self._strategy_performance: Dict[str, List[bool]] = {
            phase: [[] for _ in range(4)] for phase in ['early', 'mid', 'late']
        }
        
        # Game state tracking
        self._game_decision_count = 0
        self._last_constraint_check = {'young': 0, 'well_dressed': 0}

    def _init_component_strategies(self):
        """Initialize the component strategies we'll use."""
        # Create instances of each strategy with their default parameters
        self._rbcr2_strategy = RBCR2Strategy()
        self._perfect_strategy = PerfectStrategy()
        self._ultimate3_strategy = Ultimate3Strategy()
        self._dual_deficit_strategy = DualDeficitController()
        
        self._component_strategies = [
            self._rbcr2_strategy,
            self._perfect_strategy,
            self._ultimate3_strategy,
            self._dual_deficit_strategy
        ]

    @property
    def name(self) -> str:
        return "RandomWalkHybrid"

    def _get_game_phase(self, game_state: GameState) -> str:
        """Determine current game phase."""
        capacity_ratio = game_state.capacity_ratio
        if capacity_ratio < float(self.params['early_phase_cutoff']):
            return 'early'
        elif capacity_ratio < float(self.params['mid_phase_cutoff']):
            return 'mid'
        else:
            return 'late'

    def _sample_strategy_probabilities(self, phase: str, game_state: GameState) -> List[float]:
        """Sample strategy probabilities using Thompson Sampling."""
        # Get base weights
        if phase == 'early':
            base_weights = self.params['early_weights'].copy()
        elif phase == 'mid':
            base_weights = self.params['mid_weights'].copy()
        else:
            base_weights = self.params['late_weights'].copy()
        
        # Sample from Beta distributions for each strategy
        sampled_weights = []
        for i in range(4):
            alpha = self._thompson_alphas[phase][i]
            beta = self._thompson_betas[phase][i]
            sampled_weight = np.random.beta(alpha, beta)
            sampled_weights.append(sampled_weight)
        
        # Combine with base weights
        combined_weights = []
        adaptation_rate = float(self.params['adaptation_rate'])
        for i in range(4):
            combined = (1 - adaptation_rate) * base_weights[i] + adaptation_rate * sampled_weights[i]
            combined_weights.append(combined)
        
        # Apply constraint boost if needed
        shortage = game_state.constraint_shortage()
        max_deficit = max(shortage.values()) if shortage else 0
        emergency_threshold = int(self.params['emergency_override_threshold'])
        
        if max_deficit > emergency_threshold:
            boost = float(self.params['constraint_boost_factor'])
            combined_weights[3] += boost  # Boost DualDeficit strategy
        
        # Ensure minimum probabilities
        min_prob = float(self.params['min_strategy_prob'])
        for i in range(4):
            combined_weights[i] = max(combined_weights[i], min_prob)
        
        # Normalize to probabilities
        total_weight = sum(combined_weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in combined_weights]
        else:
            probabilities = [0.25] * 4  # Fallback to uniform
        
        return probabilities

    def _select_strategy_with_random_walk(self, probabilities: List[float]) -> int:
        """Select strategy using random walk with momentum."""
        # Apply momentum to current strategy
        momentum = float(self.params['walk_momentum'])
        exploration = float(self.params['walk_exploration'])
        
        # Boost probability of current strategy based on momentum
        momentum_probabilities = probabilities.copy()
        momentum_probabilities[self._current_strategy_idx] *= (1 + momentum * self._strategy_momentum)
        
        # Renormalize
        total = sum(momentum_probabilities)
        if total > 0:
            momentum_probabilities = [p / total for p in momentum_probabilities]
        
        # Add exploration noise
        if random.random() < exploration:
            selected_idx = random.randint(0, 3)
        else:
            # Sample from the momentum-adjusted distribution
            selected_idx = np.random.choice(4, p=momentum_probabilities)
        
        # Update momentum
        if selected_idx == self._current_strategy_idx:
            self._strategy_momentum = min(1.0, self._strategy_momentum + 0.1)
        else:
            self._strategy_momentum = max(0.0, self._strategy_momentum - 0.2)
        
        self._current_strategy_idx = selected_idx
        return selected_idx

    def _get_strategy_decision(self, strategy_idx: int, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Get decision from a specific strategy."""
        strategy = self._component_strategies[strategy_idx]
        strategy_name = self._strategy_names[strategy_idx]
        
        try:
            # Get decision from the component strategy
            accept, reason = strategy.should_accept(person, game_state)
            return accept, f"rwh_{strategy_name.lower()}_{reason}"
        except Exception as e:
            # Fallback to safe decision if strategy fails
            return False, f"rwh_{strategy_name.lower()}_error_{str(e)[:50]}"

    def _maybe_mix_strategies(self, probabilities: List[float], person: Person, game_state: GameState) -> Tuple[bool, str, int]:
        """Optionally mix decisions from multiple strategies."""
        if not self.params['enable_strategy_mixing']:
            return None, "", -1
        
        # Only mix if the top strategy has low confidence
        max_prob = max(probabilities)
        confidence_threshold = float(self.params['mixing_confidence_threshold'])
        
        if max_prob > confidence_threshold:
            return None, "", -1  # High confidence, don't mix
        
        # Get top N strategies for mixing
        max_strategies = int(self.params['max_strategies_in_mix'])
        sorted_indices = sorted(range(4), key=lambda i: probabilities[i], reverse=True)
        top_strategies = sorted_indices[:max_strategies]
        
        # Get decisions from top strategies
        decisions = []
        reasons = []
        for idx in top_strategies:
            accept, reason = self._get_strategy_decision(idx, person, game_state)
            decisions.append(accept)
            reasons.append(f"{self._strategy_names[idx]}:{accept}")
        
        # Weighted voting
        weighted_vote = 0.0
        total_weight = 0.0
        for i, idx in enumerate(top_strategies):
            weight = probabilities[idx]
            weighted_vote += weight * (1.0 if decisions[i] else 0.0)
            total_weight += weight
        
        if total_weight > 0:
            acceptance_score = weighted_vote / total_weight
            final_accept = acceptance_score > 0.5
            
            mixed_reason = f"rwh_mixed_{';'.join(reasons)}_score={acceptance_score:.2f}"
            return final_accept, mixed_reason, -1  # -1 indicates mixed decision
        
        return None, "", -1

    def _update_thompson_sampling(self, phase: str, strategy_idx: int, success: bool):
        """Update Thompson Sampling parameters based on decision outcome."""
        if strategy_idx < 0:  # Mixed decision, update all strategies that participated
            return
        
        if success:
            self._thompson_alphas[phase][strategy_idx] += 1.0
        else:
            self._thompson_betas[phase][strategy_idx] += 1.0
        
        # Apply decay to prevent parameters from growing too large
        decay = 0.999
        self._thompson_alphas[phase][strategy_idx] *= decay
        self._thompson_betas[phase][strategy_idx] *= decay

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        self._game_decision_count += 1
        
        # CRITICAL: Constraint safety override - this overrides all other logic
        constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
        if constraint_override is not None:
            return constraint_override, f"RWH_CONSTRAINT_OVERRIDE: {constraint_reason}"
        
        # Emergency mode check
        if self.is_emergency_mode(game_state):
            return True, "rwh_emergency_mode"

        # Determine game phase
        phase = self._get_game_phase(game_state)
        
        # Sample strategy probabilities using Thompson Sampling
        probabilities = self._sample_strategy_probabilities(phase, game_state)
        
        # Try strategy mixing first
        mixed_accept, mixed_reason, mixed_idx = self._maybe_mix_strategies(probabilities, person, game_state)
        if mixed_idx == -1 and mixed_reason:  # Valid mixed decision
            return mixed_accept, mixed_reason
        
        # Select single strategy using random walk
        selected_strategy_idx = self._select_strategy_with_random_walk(probabilities)
        
        # Get decision from selected strategy
        accept, reason = self._get_strategy_decision(selected_strategy_idx, person, game_state)
        
        # Track decision for learning (we'll update Thompson sampling at game end)
        decision_info = {
            'phase': phase,
            'strategy_idx': selected_strategy_idx,
            'accept': accept,
            'person_attrs': len(self.get_person_constraint_attributes(person, game_state)),
            'game_decision_count': self._game_decision_count,
            'probabilities': probabilities.copy()
        }
        self._decision_history.append(decision_info)
        
        return accept, reason

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

    def on_game_end(self, result) -> None:
        """Update Thompson Sampling based on game outcome."""
        game_success = (result.status == "completed" and 
                       result.game_state.constraint_progress().get('young', 0) >= 1.0 and
                       result.game_state.constraint_progress().get('well_dressed', 0) >= 1.0)
        
        # Update Thompson Sampling for each decision
        for decision_info in self._decision_history:
            phase = decision_info['phase']
            strategy_idx = decision_info['strategy_idx']
            
            # Simple success metric: did this decision contribute to overall success?
            # More sophisticated metrics could be implemented here
            decision_success = game_success  # For now, use game success as proxy
            
            self._update_thompson_sampling(phase, strategy_idx, decision_success)
        
        # Clear decision history for next game
        self._decision_history.clear()
        self._game_decision_count = 0
        
        # Reset strategy state
        self._current_strategy_idx = 0
        self._strategy_momentum = 0.0

    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters including learned weights."""
        params = super().get_params()
        params.update({
            'strategy_type': 'random_walk_hybrid',
            'thompson_alphas': self._thompson_alphas,
            'thompson_betas': self._thompson_betas,
            'current_strategy': self._strategy_names[self._current_strategy_idx],
            'strategy_momentum': self._strategy_momentum,
        })
        return params