# ABOUTME: Simplified Oracle-Gated Deficit Strategy solver 
# ABOUTME: Focus on constraint satisfaction first, minimize rejections second

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver


class OGDSSimpleSolver(BaseSolver):
    """Simplified OGDS solver focused on constraint satisfaction."""
    
    def __init__(self, solver_id: str = "ogds_simple", config_manager=None, api_client=None, enable_high_score_check: bool = False):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("ogds")
        strategy = OGDSSimpleStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class OGDSSimpleStrategy(BaseDecisionStrategy):
    """
    Simplified Oracle-Gated Deficit Strategy.
    
    Focus purely on constraint satisfaction:
    1. Always accept duals
    2. Reject fillers
    3. Be very aggressive about accepting needed single attributes
    4. Only gamble on rejections early in game when constraints are balanced
    """

    def __init__(self, strategy_params: dict = None):
        defaults = {
            'early_game_cutoff': 0.6,  # Switch to late game at 600 admitted
            'lag_bias_threshold': 1.1,  # 10% lag before prioritizing
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)

    @property
    def name(self) -> str:
        return "OGDS_Simple"

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        constraint_keys = [c.attribute for c in game_state.constraints]
        
        # Fallback for non-scenario-1 games
        if 'young' not in constraint_keys or 'well_dressed' not in constraint_keys:
            shortage = game_state.constraint_shortage()
            needed_attrs = [attr for attr in self.get_person_constraint_attributes(person, game_state) 
                          if shortage.get(attr, 0) > 0]
            return bool(needed_attrs), "ogds_simple_fallback"

        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')

        # Rule 1: Always accept duals (most efficient)
        if has_young and has_well_dressed:
            return True, "ogds_simple_dual"

        # Calculate current needs
        D_y = max(0, 600 - game_state.admitted_attributes.get('young', 0))
        D_w = max(0, 600 - game_state.admitted_attributes.get('well_dressed', 0))
        
        admitted = game_state.admitted_count

        # Rule 2: Reject fillers (people with no needed attributes)
        if not has_young and not has_well_dressed:
            return False, "ogds_simple_filler_reject"

        # Rule 3: Don't accept surplus attributes
        is_surplus = (has_young and D_y == 0) or (has_well_dressed and D_w == 0)
        if is_surplus:
            return False, "ogds_simple_surplus_reject"

        # Rule 4: Phase-based logic for single attributes
        game_progress = admitted / 1000.0
        
        if game_progress >= self.params['early_game_cutoff']:
            # Late game: Accept ALL needed single attributes
            return True, "ogds_simple_late_accept"
        
        # Early game: Be selective but constraint-aware
        progress_y = game_state.admitted_attributes.get('young', 0) / 600.0
        progress_w = game_state.admitted_attributes.get('well_dressed', 0) / 600.0
        
        # Calculate constraint urgency
        admits_rem = 1000 - admitted
        urgency_y = D_y / max(1, admits_rem)
        urgency_w = D_w / max(1, admits_rem)
        
        if has_young:
            urgency = urgency_y
            is_lagging = progress_y < progress_w / self.params['lag_bias_threshold']
        else:  # has_well_dressed
            urgency = urgency_w
            is_lagging = progress_w < progress_y / self.params['lag_bias_threshold']
        
        # Acceptance probability based on urgency
        if urgency > 1.0:  # Very urgent - need more than 1 per remaining slot
            accept_prob = 0.98
        elif urgency > 0.8:  # Quite urgent
            accept_prob = 0.9
        elif urgency > 0.6:  # Somewhat urgent  
            accept_prob = 0.85
        elif is_lagging:  # This constraint is behind
            accept_prob = 0.8
        else:  # Constraints are balanced, can afford to be picky
            accept_prob = 0.7
        
        # Increase acceptance as we get closer to capacity
        if game_progress > 0.8:  # Very close to full
            accept_prob = 1.0
        elif game_progress > 0.7:  # Getting close
            accept_prob = min(1.0, accept_prob + 0.1)
        
        should_accept = random.random() < accept_prob
        
        reason = f"ogds_simple_early_urgent_{urgency:.1f}" if urgency > 0.6 else \
                 "ogds_simple_early_lagging" if is_lagging else \
                 "ogds_simple_early_balanced"
        
        return should_accept, reason