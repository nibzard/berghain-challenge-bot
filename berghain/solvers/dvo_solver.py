"""Dynamic Value-Based Optimization (DVO) strategy.

Mathematical approach using dynamic programming principles:
- Computes value function for each person type based on marginal constraint utility
- Uses optimal stopping theory with capacity and deficit awareness  
- Employs multi-dimensional state space (capacity_remaining, young_deficit, well_dressed_deficit)
- Implements Bellman-style value iteration for optimal thresholds
"""

import math
import random
from typing import Tuple, Dict
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver


class DVOSolver(BaseSolver):
    """DVO algorithm solver."""
    
    def __init__(self, solver_id: str = "dvo", config_manager=None, api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("dvo")
        strategy = DVOStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class DVOStrategy(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Core value function parameters
            'constraint_weight': 1000.0,    # Weight for meeting constraints vs capacity usage
            'urgency_exponent': 1.8,        # Exponential urgency growth near capacity
            'deficit_scaling': 2.2,         # How much to scale deficit urgency
            
            # Dynamic thresholds
            'early_selectivity': 0.85,      # How selective to be early (higher = more selective)
            'late_panic_threshold': 0.92,   # When to switch to panic mode (capacity ratio)
            'critical_deficit_threshold': 50, # Remaining deficit that triggers emergency
            
            # Value function parameters
            'dual_bonus': 0.35,             # Extra value for people with both attributes
            'single_base_value': 0.12,      # Base acceptance probability for single-attribute
            'filler_max_rate': 0.08,        # Maximum filler acceptance rate
            
            # Advanced math parameters
            'bellman_iterations': 5,        # Value function update iterations
            'state_discretization': 20,     # Granularity for value function approximation
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # Precomputed value tables (will be computed on first use)
        self._value_table: Dict[Tuple[int, int, int], float] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        return "DVO"

    def _initialize_value_function(self, gs: GameState):
        """Initialize value function using dynamic programming approach."""
        if self._initialized:
            return
            
        # Extract scenario parameters
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            self._initialized = True
            return
            
        # Get probabilities from calibrated data
        p_y = float(gs.statistics.frequencies.get(keys[0], 0.323))
        p_w = float(gs.statistics.frequencies.get(keys[1], 0.323))
        rho = float(gs.statistics.correlations.get(keys[0], {}).get(keys[1], 0.183))
        
        # Compute joint distribution
        denom = math.sqrt(max(p_y*(1-p_y), 1e-9) * max(p_w*(1-p_w), 1e-9))
        self.p11 = max(0.0, min(min(p_y, p_w), p_y*p_w + rho*denom))  # both
        self.p10 = max(0.0, p_y - self.p11)  # young only
        self.p01 = max(0.0, p_w - self.p11)  # well_dressed only
        self.p00 = max(0.0, 1.0 - (self.p11 + self.p10 + self.p01))  # neither
        
        # Precompute value function via approximate dynamic programming
        self._compute_value_table(gs.target_capacity)
        self._initialized = True

    def _compute_value_table(self, capacity: int):
        """Compute value function table using Bellman iteration."""
        max_deficit = 150  # Reasonable upper bound for deficit states
        iterations = int(self.params.get('bellman_iterations', 5))
        
        # Initialize value table
        for s in range(0, capacity + 1, 10):  # Every 10 capacity units
            for dy in range(0, max_deficit + 1, 5):  # Every 5 deficit units
                for dw in range(0, max_deficit + 1, 5):
                    self._value_table[(s, dy, dw)] = self._terminal_value(s, dy, dw)
        
        # Bellman iteration
        for _ in range(iterations):
            new_table = {}
            for (s, dy, dw), old_val in self._value_table.items():
                if s <= 0:
                    new_table[(s, dy, dw)] = old_val
                    continue
                    
                # Compute expected value from optimal policy
                ev_accept_dual = self._expected_value_after_accept(s-1, max(0, dy-1), max(0, dw-1))
                ev_accept_y = self._expected_value_after_accept(s-1, max(0, dy-1), dw)
                ev_accept_w = self._expected_value_after_accept(s-1, dy, max(0, dw-1))
                ev_accept_none = self._expected_value_after_accept(s-1, dy, dw)
                ev_reject = self._expected_value_after_reject(s, dy, dw)
                
                # Optimal policy value
                value = (self.p11 * max(ev_accept_dual, ev_reject) +
                        self.p10 * max(ev_accept_y, ev_reject) +
                        self.p01 * max(ev_accept_w, ev_reject) +
                        self.p00 * max(ev_accept_none, ev_reject))
                        
                new_table[(s, dy, dw)] = value
            self._value_table.update(new_table)

    def _terminal_value(self, capacity_remaining: int, young_deficit: int, well_deficit: int) -> float:
        """Terminal value when game ends."""
        if young_deficit <= 0 and well_deficit <= 0:
            return 1000.0  # Success bonus
        if capacity_remaining <= 0:
            return -1000.0 * (young_deficit + well_deficit)  # Failure penalty
        return 0.0

    def _expected_value_after_accept(self, s: int, dy: int, dw: int) -> float:
        """Expected future value after accepting current person."""
        return self._get_value(s, dy, dw) + 100.0  # Immediate reward for admission

    def _expected_value_after_reject(self, s: int, dy: int, dw: int) -> float:
        """Expected future value after rejecting current person."""  
        return self._get_value(s, dy, dw) - 1.0  # Small penalty for rejection

    def _get_value(self, s: int, dy: int, dw: int) -> float:
        """Get value with interpolation for states not in table."""
        # Round to nearest tabulated state
        s_key = max(0, min(1000, (s // 10) * 10))
        dy_key = max(0, min(150, (dy // 5) * 5))
        dw_key = max(0, min(150, (dw // 5) * 5))
        
        return self._value_table.get((s_key, dy_key, dw_key), 0.0)

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "dvo_emergency"
            
        self._initialize_value_function(game_state)
        
        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]
        
        # Current state
        s = game_state.target_capacity - game_state.admitted_count
        shortage = game_state.constraint_shortage()
        dy = shortage.get(a_y, 0)
        dw = shortage.get(a_w, 0)
        
        # Determine person type and compute marginal value
        is_dual = len(attrs) >= 2
        is_y = (a_y in attrs)
        is_w = (a_w in attrs)
        is_filler = len(attrs) == 0
        
        if is_dual:
            # Always accept duals - highest value
            return True, "dvo_dual"
        
        # Compute acceptance threshold using value function
        if is_y:
            value_accept = self._expected_value_after_accept(s-1, max(0, dy-1), dw)
            value_reject = self._expected_value_after_reject(s, dy, dw)
            threshold = 0.7 if dy > 0 else 0.4
        elif is_w:
            value_accept = self._expected_value_after_accept(s-1, dy, max(0, dw-1))
            value_reject = self._expected_value_after_reject(s, dy, dw)
            threshold = 0.7 if dw > 0 else 0.4
        else:
            # Filler - use very restrictive policy
            cr = game_state.capacity_ratio
            if cr > float(self.params.get('late_panic_threshold', 0.92)):
                return False, "dvo_filler_late_reject"
            
            # Dynamic filler rate based on constraint satisfaction
            progress = game_state.constraint_progress()
            min_progress = min(progress.values()) if progress else 0.0
            
            # Allow some filler early, very little late
            base_rate = float(self.params.get('filler_max_rate', 0.08))
            filler_rate = base_rate * (1.0 - min_progress) * (1.0 - cr)**2
            
            if random.random() < filler_rate:
                return True, "dvo_filler_accept"
            return False, "dvo_filler_reject"
        
        # For single attributes, use value-based threshold with urgency scaling
        urgency_multiplier = 1.0
        if dy > 0 or dw > 0:
            # Increase urgency as we approach capacity
            cr = game_state.capacity_ratio
            urgency_multiplier = 1.0 + (cr ** float(self.params.get('urgency_exponent', 1.8)))
            
        # Critical deficit override
        critical_threshold = int(self.params.get('critical_deficit_threshold', 50))
        if (is_y and dy > 0 and dy <= critical_threshold) or (is_w and dw > 0 and dw <= critical_threshold):
            urgency_multiplier *= 2.0
            
        final_threshold = threshold / urgency_multiplier
        
        # Value-based decision
        value_ratio = (value_accept - value_reject) / max(abs(value_accept), abs(value_reject), 100.0)
        
        if value_ratio > final_threshold:
            return True, f"dvo_single_{'y' if is_y else 'w'}_value"
        else:
            return False, f"dvo_single_{'y' if is_y else 'w'}_reject"