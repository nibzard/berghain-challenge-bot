"""Balanced strategy that blends rarity and shortage signals."""

import random
from typing import Tuple, Dict
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class BalancedStrategy(BaseDecisionStrategy):
    """Combines attribute rarity and constraint shortage with phase-aware tuning."""

    def __init__(self, strategy_params: dict = None):
        default_params = {
            # Shared keys for variation tooling
            'ultra_rare_threshold': 0.10,
            'rare_accept_rate': 0.95,
            'common_reject_rate': 0.05,
            'deficit_panic_threshold': 0.8,
            'early_game_threshold': 0.30,
            'mid_game_threshold': 0.70,

            # Balanced-specific weights
            'rarity_weight': 1.0,
            'shortage_weight': 1.2,
            'multi_attr_bonus': 1.3,
            'needed_attr_boost': 1.2,
            'base_offset': 0.20,    # baseline probability floor
            'score_scale': 0.05,    # converts score to probability contribution
        }
        if strategy_params:
            default_params.update(strategy_params)
        super().__init__(default_params)

    @property
    def name(self) -> str:
        return "Balanced"

    def _min_counts(self, game_state: GameState) -> Dict[str, int]:
        return {c.attribute: c.min_count for c in game_state.constraints}

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        constraint_attrs = self.get_constraint_attributes(game_state)
        person_attrs = self.get_person_constraint_attributes(person, game_state)
        phase = self.get_game_phase(game_state)

        # No relevant attributes
        if not person_attrs:
            base = self.params['common_reject_rate']
            if phase == 'early':
                base *= 0.5
            elif phase == 'late':
                base *= 1.2
            elif phase == 'panic':
                base = max(base, 0.6)
            return random.random() < max(0.01, min(0.99, base)), "balanced_no_constraints"

        # Critical help shortcut
        critical = self.get_critical_constraints(game_state, threshold=self.params['deficit_panic_threshold'])
        if person_attrs & critical:
            return True, f"balanced_critical_help_{list(person_attrs & critical)}"

        # Score by rarity and shortage
        rarity_scores = self.calculate_rarity_scores(game_state)
        shortage = game_state.constraint_shortage()
        min_counts = self._min_counts(game_state)

        rarity_sum = sum(rarity_scores[a] for a in person_attrs)
        # Normalize shortage by min_count to get relative need
        shortage_sum = 0.0
        for a in person_attrs:
            mc = max(1, min_counts.get(a, 1))
            shortage_sum += shortage[a] / mc

        score = (
            rarity_sum * self.params['rarity_weight'] +
            shortage_sum * self.params['shortage_weight']
        )

        # Ultra-rare boost
        has_ultra_rare = any(
            game_state.statistics.get_frequency(a) < self.params['ultra_rare_threshold']
            for a in person_attrs
        )

        # Multi-attribute bonus
        if len(person_attrs) > 1:
            score *= self.params['multi_attr_bonus']

        # Needed attributes bonus
        needed_attrs = [a for a in person_attrs if shortage[a] > 0]
        if needed_attrs:
            score *= self.params['needed_attr_boost']

        # Map score to probability
        base_prob = self.params['base_offset'] + score * self.params['score_scale']

        # Phase adjustments
        if phase == 'early' and len(person_attrs) < 2:
            base_prob *= 0.8
        elif phase == 'late' and not needed_attrs:
            base_prob *= 0.7
        elif phase == 'panic':
            base_prob = max(base_prob, 0.9)

        if has_ultra_rare:
            base_prob = max(base_prob, self.params['rare_accept_rate'])

        final_prob = max(0.01, min(0.99, base_prob))
        accept = random.random() < final_prob
        return accept, f"balanced_{phase}_score{score:.2f}_p{final_prob:.2f}"

