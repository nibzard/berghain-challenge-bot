"""Diversity-first strategy that prioritizes underrepresented constraints."""

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class DiversityFirstStrategy(BaseDecisionStrategy):
    """Focus on balancing constraint representation, especially early in the game."""

    def __init__(self, strategy_params: dict = None):
        default_params = {
            # Shared keys for variation tooling
            'ultra_rare_threshold': 0.10,
            'rare_accept_rate': 0.95,
            'common_reject_rate': 0.05,
            'deficit_panic_threshold': 0.8,
            'early_game_threshold': 0.30,
            'mid_game_threshold': 0.70,

            # Diversity-specific
            'diversity_emphasis': 1.5,  # weight for underrepresented attributes
            'min_diversity_ratio': 0.6, # below this progress, considered underrepresented
            'multi_attr_bonus': 1.2,
        }
        if strategy_params:
            default_params.update(strategy_params)
        super().__init__(default_params)

    @property
    def name(self) -> str:
        return "DiversityFirst"

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        attrs = self.get_person_constraint_attributes(person, game_state)
        phase = self.get_game_phase(game_state)

        if not attrs:
            base = self.params['common_reject_rate']
            if phase == 'early':
                base *= 0.4
            return random.random() < max(0.01, min(0.5, base)), "diversity_no_constraints"

        progress = game_state.constraint_progress()
        shortage = game_state.constraint_shortage()
        threshold = self.params['min_diversity_ratio']

        underrep = [a for a in attrs if progress[a] < threshold]
        needed = [a for a in attrs if shortage[a] > 0]

        # Prioritize underrepresented constraints
        if underrep:
            prob = 0.8 * self.params['diversity_emphasis']
            if len(attrs) > 1:
                prob *= self.params['multi_attr_bonus']
            prob = max(0.3, min(0.99, prob))
            return random.random() < prob, f"diversity_underrep_{underrep}"

        # Then prioritize needed constraints
        if needed:
            prob = 0.7
            if len(attrs) > 1:
                prob *= self.params['multi_attr_bonus']
            return random.random() < min(0.95, prob), f"diversity_needed_{needed}"

        # Otherwise, selective depending on phase
        base = 0.2 if phase == 'early' else 0.4 if phase == 'mid' else 0.3
        return random.random() < base, "diversity_surplus"

