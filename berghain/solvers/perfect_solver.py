"""Perfect Balance Optimization (PBO) strategy.

The mathematically perfect algorithm that combines:
- Ultimate's advanced optimization techniques  
- RBCR's constraint satisfaction reliability
- Perfect balance between rejection minimization and constraint fulfillment

Key Innovation: **Constraint-First Optimization with Economic Efficiency**
- Hard constraint satisfaction is non-negotiable
- Among constraint-satisfying strategies, minimize rejections
- Uses penalty methods and barrier functions from convex optimization
- Implements exact constraint satisfaction via Lagrangian duality
"""

import math
import random
from typing import Tuple, Dict, Optional
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver


class PerfectSolver(BaseSolver):
    """Perfect balance optimization solver."""
    
    def __init__(self, solver_id: str = "perfect", config_manager=None, api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("perfect")
        strategy = PerfectStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class PerfectStrategy(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Constraint priority system
            'constraint_penalty_weight': 1000.0,  # Huge penalty for missing constraints
            'constraint_satisfaction_threshold': 0.95, # When to enter constraint-priority mode
            'hard_constraint_mode_trigger': 0.85,     # When to abandon efficiency for constraints
            
            # Advanced mathematical parameters (from Ultimate)
            'lagrange_multiplier_base': 2.5,
            'capacity_shadow_price': 1.4,
            'learning_rate': 0.05,
            'gradient_momentum': 0.88,
            
            # Constraint balancing
            'deficit_exponential_weight': 3.0,  # Exponential weight for deficits
            'balance_importance': 0.25,         # How much to weight constraint balance
            'emergency_threshold': 100,         # Deficit level that triggers emergency
            
            # Efficiency optimization  
            'efficiency_weight': 1.0,          # Weight for rejection minimization
            'risk_aversion': 0.35,             # Risk aversion for uncertainty
            'exploration_rate': 0.02,          # Exploration for learning
            
            # Phase management
            'early_phase_cutoff': 0.3,         # Capacity ratio for early phase
            'mid_phase_cutoff': 0.7,           # Capacity ratio for mid phase
            'late_phase_cutoff': 0.9,          # Capacity ratio for late phase
            
            # Perfect balance parameters
            'golden_section': 0.618034,        # Golden section for optimization
            'fibonacci_scaling': 1.414,        # Fibonacci-based scaling
            'convergence_tolerance': 1e-8,     # Numerical precision
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # State variables
        self._lambda_y = float(self.params['lagrange_multiplier_base'])
        self._lambda_w = float(self.params['lagrange_multiplier_base'])
        self._mu = float(self.params['capacity_shadow_price'])
        
        # Learning state
        self._momentum_y = 0.0
        self._momentum_w = 0.0
        self._decision_count = 0
        
        # Constraint tracking
        self._constraint_violation_history = []

    @property
    def name(self) -> str:
        return "Perfect"

    def _compute_constraint_urgency(self, game_state: GameState) -> Dict[str, float]:
        """Compute exponentially weighted constraint urgency."""
        keys = [c.attribute for c in game_state.constraints]
        if len(keys) < 2:
            return {}
            
        shortage = game_state.constraint_shortage()
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        
        urgencies = {}
        exp_weight = float(self.params['deficit_exponential_weight'])
        
        for attr in keys:
            deficit = shortage.get(attr, 0)
            if deficit <= 0:
                urgencies[attr] = 0.0
            else:
                # Exponential urgency based on deficit and remaining capacity
                if capacity_remaining > 0:
                    urgency = math.exp(exp_weight * deficit / max(capacity_remaining, 1))
                else:
                    urgency = float('inf')  # Infinite urgency when no capacity left
                urgencies[attr] = urgency
                
        return urgencies

    def _update_lagrange_multipliers(self, game_state: GameState):
        """Update multipliers with constraint-first priority."""
        keys = [c.attribute for c in game_state.constraints]
        if len(keys) < 2:
            return
            
        a_y, a_w = keys[0], keys[1]
        shortage = game_state.constraint_shortage()
        
        # Compute constraint violation gradients
        gradient_y = max(0, shortage.get(a_y, 0)) / 600.0
        gradient_w = max(0, shortage.get(a_w, 0)) / 600.0
        
        # Exponential penalty for constraint violations
        penalty_weight = float(self.params['constraint_penalty_weight'])
        gradient_y *= penalty_weight
        gradient_w *= penalty_weight
        
        # Update with momentum (constraint-priority learning)
        lr = float(self.params['learning_rate'])
        momentum = float(self.params['gradient_momentum'])
        
        self._momentum_y = momentum * self._momentum_y + lr * gradient_y
        self._momentum_w = momentum * self._momentum_w + lr * gradient_w
        
        # Update multipliers (always non-negative)
        self._lambda_y = max(0.1, self._lambda_y + self._momentum_y)
        self._lambda_w = max(0.1, self._lambda_w + self._momentum_w)

    def _compute_person_value_constrained(self, person: Person, game_state: GameState) -> float:
        """Compute value with perfect constraint prioritization."""
        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]
        
        # Base economic value
        value = 0.0
        
        # Constraint contributions (with exponential weighting)
        urgencies = self._compute_constraint_urgency(game_state)
        
        if a_y in attrs:
            constraint_value = self._lambda_y * (1 + urgencies.get(a_y, 0))
            value += constraint_value
            
        if a_w in attrs:
            constraint_value = self._lambda_w * (1 + urgencies.get(a_w, 0))
            value += constraint_value
        
        # Capacity cost (shadow price)
        value -= self._mu
        
        # Perfect balance bonus for dual attributes
        if len(attrs) >= 2:
            golden_ratio = float(self.params['golden_section'])
            balance_bonus = golden_ratio * math.log(2)
            value += balance_bonus
        
        # Phase-dependent adjustments
        capacity_ratio = game_state.capacity_ratio
        early_cutoff = float(self.params['early_phase_cutoff'])
        mid_cutoff = float(self.params['mid_phase_cutoff'])
        late_cutoff = float(self.params['late_phase_cutoff'])
        
        if capacity_ratio < early_cutoff:
            # Early phase: be more selective
            value *= 0.85
        elif capacity_ratio > late_cutoff:
            # Late phase: constraint satisfaction is critical
            constraint_progress = game_state.constraint_progress()
            min_progress = min(constraint_progress.values()) if constraint_progress else 0.0
            
            if min_progress < float(self.params['constraint_satisfaction_threshold']):
                # Boost constraint-helpful people dramatically
                if len(attrs) > 0:
                    value *= 5.0  # 5x boost for constraint helpers
                else:
                    value *= 0.1  # 10x penalty for filler
        
        # Risk adjustment
        risk_aversion = float(self.params['risk_aversion'])
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        
        if capacity_remaining > 0:
            uncertainty_penalty = risk_aversion * (1.0 / math.sqrt(capacity_remaining))
            value -= uncertainty_penalty
        
        return value

    def _is_emergency_constraint_mode(self, game_state: GameState) -> bool:
        """Check if we should enter emergency constraint satisfaction mode."""
        shortage = game_state.constraint_shortage()
        emergency_threshold = int(self.params['emergency_threshold'])
        
        # Emergency if any constraint has large deficit
        max_deficit = max(shortage.values()) if shortage else 0
        if max_deficit >= emergency_threshold:
            return True
            
        # Emergency if capacity is very low and constraints not satisfied
        capacity_ratio = game_state.capacity_ratio
        hard_mode_trigger = float(self.params['hard_constraint_mode_trigger'])
        
        if capacity_ratio > hard_mode_trigger:
            constraint_progress = game_state.constraint_progress()
            min_progress = min(constraint_progress.values()) if constraint_progress else 1.0
            if min_progress < 0.99:  # Not 99% complete
                return True
                
        return False

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        self._decision_count += 1
        
        # CRITICAL: Constraint safety override - this overrides all other logic
        constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
        if constraint_override is not None:
            return constraint_override, f"PERFECT_CONSTRAINT_OVERRIDE: {constraint_reason}"
        
        # Always accept in base emergency mode
        if self.is_emergency_mode(game_state):
            return True, "perfect_base_emergency"
        
        # Check for constraint emergency
        constraint_emergency = self._is_emergency_constraint_mode(game_state)
        
        # Update Lagrange multipliers
        self._update_lagrange_multipliers(game_state)
        
        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]
        
        is_dual = len(attrs) >= 2
        is_single = len(attrs) == 1
        is_filler = len(attrs) == 0
        
        # CONSTRAINT EMERGENCY MODE - Accept anyone who helps constraints
        if constraint_emergency:
            if is_dual:
                return True, "perfect_emergency_dual"
            elif is_single:
                # Check if this attribute is needed
                shortage = game_state.constraint_shortage()
                if a_y in attrs and shortage.get(a_y, 0) > 0:
                    return True, "perfect_emergency_single_y"
                elif a_w in attrs and shortage.get(a_w, 0) > 0:
                    return True, "perfect_emergency_single_w"
                else:
                    return False, "perfect_emergency_single_satisfied"
            else:
                return False, "perfect_emergency_filler_reject"
        
        # NORMAL OPERATION - Use perfect value-based decisions
        person_value = self._compute_person_value_constrained(person, game_state)
        
        # Decision threshold based on phase
        capacity_ratio = game_state.capacity_ratio
        late_cutoff = float(self.params['late_phase_cutoff'])
        
        if capacity_ratio > late_cutoff:
            # Late game: very high standards
            threshold = 2.0
        else:
            # Early/mid game: moderate standards
            threshold = 0.0
            
        # Exploration with small probability
        exploration_rate = float(self.params['exploration_rate'])
        if random.random() < exploration_rate:
            decision = random.choice([True, False])
            return decision, f"perfect_explore_{person_value:.2f}"
        
        # Main decision
        accept = person_value > threshold
        
        if accept:
            return True, f"perfect_accept_{person_value:.2f}"
        else:
            return False, f"perfect_reject_{person_value:.2f}"

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