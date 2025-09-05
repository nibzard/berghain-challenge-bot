"""Quota-tracking strategy for Scenario 1 style constraints.

Heuristic goals:
- Always accept dual-attribute people (both constraints).
- Prefer the single attribute that is currently behind (gap-aware).
- Keep filler (neither) minimal and mostly after high progress.

Parameters (YAML):
- filler_prob: Base acceptance probability for no-constraint (default 0.03)
- filler_unlock_progress: Start allowing filler once min(progress) >= this (default 0.9)
- lag_bias_threshold: Accept single when its progress is behind the other by at least this (default 0.02)
- always_accept_singles_until: Accept singles unconditionally until min(progress) reaches this (default 0.75)
"""

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class QuotaTrackerStrategy(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            'filler_prob': 0.03,
            'filler_unlock_progress': 0.90,
            'lag_bias_threshold': 0.02,
            'always_accept_singles_until': 0.75,
            # Acceptance-rate controller (helps avoid early termination)
            'acc_rate_target': 0.52,
            'acc_rate_margin': 0.02,
            'filler_prob_max': 0.12,
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)

    @property
    def name(self) -> str:
        return "QuotaTracker"

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        attrs = self.get_person_constraint_attributes(person, game_state)
        # Current acceptance dynamics
        total_dec = max(1, game_state.admitted_count + game_state.rejected_count)
        acc_rate = game_state.admitted_count / total_dec
        acc_target = float(self.params.get('acc_rate_target', 0.52))
        acc_margin = float(self.params.get('acc_rate_margin', 0.02))
        need_rate_boost = acc_rate < (acc_target - acc_margin)
        # No constraints on this person: consider small filler
        if not attrs:
            progress = game_state.constraint_progress()
            min_prog = min(progress.values()) if progress else 0.0
            if min_prog < self.params['filler_unlock_progress']:
                return False, "quota_no_constraints_locked"
            base_p = float(self.params.get('filler_prob', 0.03))
            max_p = float(self.params.get('filler_prob_max', 0.12))
            # Increase filler probability if acceptance rate is lagging
            p = base_p
            if need_rate_boost:
                shortfall = max(0.0, (acc_target - acc_rate))
                p = min(max_p, base_p + 1.5 * shortfall)
            return random.random() < max(0.0, min(0.5, p)), "quota_filler"

        # Dual-attribute: always accept
        if len(attrs) >= 2:
            return True, "quota_dual"

        # Single-attribute: accept, with bias toward lagging attribute
        progress = game_state.constraint_progress()
        # Expect two constraints (Scenario 1), but work for generic >=1
        # Pick the other attribute's progress for comparison if two exist
        if len(progress) >= 2:
            keys = list(progress.keys())
            # Identify the attribute this person contributes to
            a = next(iter(attrs))
            b = keys[0] if keys[1] == a else keys[1]
            gap = progress[b] - progress[a]
        else:
            a = next(iter(attrs))
            gap = 0.0

        min_prog = min(progress.values()) if progress else 0.0

        # Early to mid-game: accept singles broadly to secure quotas
        if min_prog < float(self.params.get('always_accept_singles_until', 0.75)):
            return True, f"quota_single_early_{a}"

        # Later: accept if this attribute is lagging by threshold or still needed
        shortage = game_state.constraint_shortage()
        if shortage.get(a, 0) > 0:
            # If behind by enough, strong accept; otherwise probabilistic accept
            if gap > float(self.params.get('lag_bias_threshold', 0.02)):
                return True, f"quota_single_lagging_{a}"
            # Soft probability to avoid overfilling one side
            return random.random() < 0.85, f"quota_single_needed_{a}"

        # Surplus single when both constraints close to done: be selective.
        # However, if acceptance rate is lagging, accept with a small probability
        # to prevent rate collapse while keeping balance.
        if need_rate_boost:
            return random.random() < 0.25, f"quota_single_surplus_acc_boost_{a}"
        return random.random() < 0.15, f"quota_single_surplus_{a}"
