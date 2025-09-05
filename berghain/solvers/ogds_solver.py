# ABOUTME: Oracle-Gated Deficit Strategy solver implementation
# ABOUTME: Uses feasibility oracle to maintain budget-aware rejection decisions

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from ..analysis.feasibility_table import FeasibilityOracle
from .base_solver import BaseSolver


class OGDSSolver(BaseSolver):
    """Solver for the Oracle-Gated Deficit Strategy."""
    
    def __init__(self, solver_id: str = "ogds", config_manager=None, api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("ogds")
        strategy = OGDSStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class OGDSStrategy(BaseDecisionStrategy):
    """
    Oracle-Gated Deficit Strategy (OGDS).

    This strategy makes decisions based on maintaining a high probability of
    successfully meeting all constraints within a target rejection count. It uses
    a pre-computed Feasibility Oracle to guide its decisions.

    Core Principles:
    1.  **Always Accept Duals:** People with both required attributes are always accepted
        as they are the most efficient way to meet constraints.
    2.  **Oracle-Gated Rejections:** Before any rejection (of a single-attribute or
        filler person), consult the Feasibility Oracle. If rejecting would make the
        game infeasible to win within the remaining "rejection budget", the person
        is accepted out of necessity.
    3.  **Phase-Based Single-Attribute Logic:**
        - In the early game, if feasibility is not at risk, the strategy may
          probabilistically reject single-attribute people to "gamble" for more
          efficient dual-attribute arrivals.
        - In the late game, any needed single-attribute person is accepted.
    4.  **Strict Filler Control:** People with no required attributes ("fillers") are
        rejected by default, unless doing so is infeasible.
    """

    def __init__(self, strategy_params: dict = None):
        defaults = {
            'target_rejections': 710,
            'early_game_cutoff': 0.5,  # Corresponds to 500 admitted
            'lag_bias_threshold': 1.05,  # Accept if one constraint is 5% more progressed
            'p_both': 0.1444,
            'p_young_only': 0.1786,
            'p_well_dressed_only': 0.1782,
            'oracle_delta': 0.005  # Corresponds to a 99.5% confidence requirement
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)

        # Initialize the Feasibility Oracle
        self.oracle = FeasibilityOracle(
            p11=self.params['p_both'],
            p10=self.params['p_young_only'],
            p01=self.params['p_well_dressed_only'],
            p00=1.0 - (self.params['p_both'] + self.params['p_young_only'] + self.params['p_well_dressed_only']),
            delta=self.params['oracle_delta'],
            cache_key="scenario_1"  # Use the pre-computed cache for speed
        )

    @property
    def name(self) -> str:
        return "OGDS"

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        constraint_keys = [c.attribute for c in game_state.constraints]
        
        # Fallback for non-scenario-1 games
        if 'young' not in constraint_keys or 'well_dressed' not in constraint_keys:
            shortage = game_state.constraint_shortage()
            needed_attrs = [attr for attr in self.get_person_constraint_attributes(person, game_state) 
                          if shortage.get(attr, 0) > 0]
            return bool(needed_attrs), "ogds_fallback_greedy"

        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')

        # Principle 1: Always accept duals (certainty-first)
        if has_young and has_well_dressed:
            return True, "ogds_dual"

        # Calculate current state
        admitted = game_state.admitted_count
        rejected = game_state.rejected_count

        D_y = max(0, 600 - game_state.admitted_attributes.get('young', 0))
        D_w = max(0, 600 - game_state.admitted_attributes.get('well_dressed', 0))

        rejections_rem = self.params['target_rejections'] - rejected
        admits_rem = 1000 - admitted
        
        # If we're over budget, only accept people who help with constraints
        if rejections_rem <= 0:
            return (has_young and D_y > 0) or (has_well_dressed and D_w > 0), "ogds_over_budget"
        
        people_to_see_rem = admits_rem + rejections_rem

        # Principle 4: Strict filler control (check this first before oracle)
        if not has_young and not has_well_dressed:
            # Only use oracle for fillers when we're truly close to budget
            if rejections_rem < 50:  # Very tight budget - safety net
                is_safe_to_reject = self.oracle.is_feasible(D_y, D_w, people_to_see_rem - 1)
                if not is_safe_to_reject:
                    return True, "ogds_forced_accept_infeasible_rejection"
            return False, "ogds_filler_reject"

        # Don't accept surplus attributes (people with attributes we don't need)
        is_surplus = (has_young and D_y == 0) or (has_well_dressed and D_w == 0)
        if is_surplus:
            # Only use oracle for surplus when extremely close to budget
            if rejections_rem < 25:  # Extremely tight budget
                is_safe_to_reject = self.oracle.is_feasible(D_y, D_w, people_to_see_rem - 1)
                if not is_safe_to_reject:
                    return True, "ogds_forced_accept_infeasible_rejection"
            return False, "ogds_single_surplus"

        # Principle 3: Constraint-first logic with budget awareness
        # Calculate constraint urgency - how badly do we need this attribute?
        constraint_urgency_y = D_y / max(1, admits_rem)  # How many young per remaining slot
        constraint_urgency_w = D_w / max(1, admits_rem)  # How many well_dressed per remaining slot
        
        # If we're getting close to capacity and still need a lot, be very aggressive
        capacity_pressure = admitted / 1000.0
        
        game_progress = admitted / 1000.0
        if game_progress >= self.params['early_game_cutoff']:
            # Late game: ALWAYS accept any needed single-attribute person
            # Priority is constraint satisfaction, not rejection minimization
            return True, "ogds_single_late_game_needed"
        
        # Early game: Be more selective, but still prioritize constraint satisfaction
        progress_y = game_state.admitted_attributes.get('young', 0) / 600.0
        progress_w = game_state.admitted_attributes.get('well_dressed', 0) / 600.0
        
        # Calculate how urgent this attribute is
        if has_young and has_well_dressed:
            return True, "ogds_dual"  # Already handled above, but safety
        elif has_young:
            urgency = constraint_urgency_y
            is_lagging = progress_y < progress_w * self.params['lag_bias_threshold']
        else:  # has_well_dressed
            urgency = constraint_urgency_w  
            is_lagging = progress_w < progress_y * self.params['lag_bias_threshold']
        
        # Acceptance probability based on urgency and game phase
        base_accept_prob = 0.6  # Base 60% acceptance for needed attributes
        
        # Increase acceptance probability if urgent or lagging
        if urgency > 0.8:  # Very urgent - need many of this attribute
            base_accept_prob = 0.9
        elif urgency > 0.5:  # Somewhat urgent
            base_accept_prob = 0.8
        elif is_lagging:  # This attribute is behind
            base_accept_prob = 0.75
        
        # Increase acceptance probability as we get closer to capacity
        if capacity_pressure > 0.7:  # Getting close to full
            base_accept_prob = min(1.0, base_accept_prob + 0.2)
        elif capacity_pressure > 0.5:
            base_accept_prob = min(1.0, base_accept_prob + 0.1)
        
        should_accept = random.random() < base_accept_prob
        
        # Safety check: only use oracle if we're about to reject and very close to budget
        if not should_accept and rejections_rem < 100:
            is_safe_to_reject = self.oracle.is_feasible(D_y, D_w, people_to_see_rem - 1)
            if not is_safe_to_reject:
                return True, "ogds_forced_accept_infeasible_rejection"
        
        reason = "ogds_single_early_urgent" if urgency > 0.5 else \
                 "ogds_single_early_lagging" if is_lagging else \
                 "ogds_single_early_balanced"
        
        return should_accept, reason