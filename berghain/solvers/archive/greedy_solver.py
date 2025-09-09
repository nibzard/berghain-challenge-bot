"""Greedy constraint-first strategy."""

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class GreedyConstraintStrategy(BaseDecisionStrategy):
    """Accepts if a person helps any unmet constraint; minimal nuance otherwise."""

    def __init__(self, strategy_params: dict = None):
        default_params = {
            # Shared keys for variation tooling
            'ultra_rare_threshold': 0.10,
            'rare_accept_rate': 0.98,
            'common_reject_rate': 0.03,
            'deficit_panic_threshold': 0.8,
            'early_game_threshold': 0.30,
            'mid_game_threshold': 0.70,

            # Greedy-specific
            'needed_accept_prob': 0.90,
            'filler_prob': 0.05,
        }
        if strategy_params:
            default_params.update(strategy_params)
        super().__init__(default_params)

    @property
    def name(self) -> str:
        return "Greedy"

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        attrs = self.get_person_constraint_attributes(person, game_state)
        if not attrs:
            # Minimal filler
            p = max(0.01, min(0.5, self.params.get('filler_prob', 0.05)))
            return random.random() < p, "greedy_no_constraints"

        shortage = game_state.constraint_shortage()
        needed = [a for a in attrs if shortage[a] > 0]

        if needed:
            # Ultra-rare auto accept
            if any(game_state.statistics.get_frequency(a) < self.params['ultra_rare_threshold'] for a in needed):
                return True, f"greedy_ultra_rare_{needed}"
            p = self.params['needed_accept_prob']
            return random.random() < p, f"greedy_needed_{needed}"
        else:
            # If not needed, small chance
            p = max(0.01, min(0.5, self.params.get('filler_prob', 0.05)))
            return random.random() < p, "greedy_surplus"

