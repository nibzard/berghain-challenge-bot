# ABOUTME: Rarity-weighted strategy implementation
# ABOUTME: Uses empirically-derived selectivity based on attribute rarity

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class RarityWeightedStrategy(BaseDecisionStrategy):
    """Strategy based on attribute rarity and constraint urgency."""
    
    def __init__(self, strategy_params: dict = None):
        # Default parameters optimized from empirical analysis
        default_params = {
            'ultra_rare_threshold': 0.10,
            'rare_accept_rate': 0.98,
            'common_reject_rate': 0.05,
            'phase1_multi_attr_only': True,
            'deficit_panic_threshold': 0.8,
            'early_game_threshold': 0.3,
            'mid_game_threshold': 0.7,
        }
        
        if strategy_params:
            default_params.update(strategy_params)
            
        super().__init__(default_params)
    
    @property
    def name(self) -> str:
        return "RarityWeighted"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Make decision based on rarity-weighted scoring."""
        
        # Emergency mode - accept almost everyone
        if self.is_emergency_mode(game_state):
            return True, "emergency_rejection_limit"
        
        # Get analysis data
        constraint_attrs = self.get_constraint_attributes(game_state)
        person_constraint_attrs = self.get_person_constraint_attributes(person, game_state)
        rarity_scores = self.calculate_rarity_scores(game_state)
        phase = self.get_game_phase(game_state)
        
        # No constraint attributes
        if not person_constraint_attrs:
            if phase == "panic":
                accept_prob = 0.3
                return random.random() < accept_prob, "panic_mode_filler"
            else:
                accept_prob = self.params['common_reject_rate']
                return random.random() < accept_prob, "no_constraints_filler"
        
        # Has constraint attributes - calculate value
        person_rarity_sum = sum(rarity_scores[attr] for attr in person_constraint_attrs)
        has_ultra_rare = any(
            game_state.statistics.get_frequency(attr) < self.params['ultra_rare_threshold']
            for attr in person_constraint_attrs
        )
        
        # Check for critical constraints (very behind target)
        progress = game_state.constraint_progress()
        critical_constraints = {
            attr for attr, prog in progress.items() 
            if prog < self.params['deficit_panic_threshold']
        }
        helps_critical = bool(person_constraint_attrs & critical_constraints)
        
        # Phase-based decision logic
        if phase == "early":
            if self.params['phase1_multi_attr_only'] and len(person_constraint_attrs) < 2:
                return False, "early_phase_single_attr"
            if has_ultra_rare:
                return True, f"early_ultra_rare_{list(person_constraint_attrs)}"
            if len(person_constraint_attrs) >= 2:
                return random.random() < 0.9, f"early_multi_attr_{len(person_constraint_attrs)}"
            return False, "early_phase_common"
            
        elif phase == "mid":
            if helps_critical:
                return True, f"mid_critical_help_{list(person_constraint_attrs & critical_constraints)}"
            if has_ultra_rare:
                return random.random() < self.params['rare_accept_rate'], f"mid_ultra_rare_{list(person_constraint_attrs)}"
            if person_rarity_sum > 5.0:  # High combined rarity
                return random.random() < 0.8, f"mid_high_rarity_{person_rarity_sum:.1f}"
            return random.random() < 0.5, "mid_moderate_value"
            
        elif phase == "late":
            if helps_critical:
                return True, f"late_critical_help_{list(person_constraint_attrs & critical_constraints)}"
            
            # Check if we still need this person's attributes
            shortage = game_state.constraint_shortage()
            needed_attrs = [attr for attr in person_constraint_attrs if shortage[attr] > 0]
            
            if needed_attrs:
                return random.random() < 0.7, f"late_needed_attr_{needed_attrs}"
            return random.random() < 0.3, "late_surplus_attr"
            
        else:  # panic
            if person_constraint_attrs:
                return True, f"panic_any_constraint_{list(person_constraint_attrs)}"
            return random.random() < 0.5, "panic_filler"