"""Dual-deficit controller strategy.

Principles:
- Always accept dual-attribute.
- Early phase: accept singles broadly until strong base built.
- Mid/Late: accept singles according to urgency u_a = shortage_a / remaining_accepts.
- Maintain acceptance-rate with a small PI controller that modulates filler.
"""

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class DualDeficitController(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            'early_singles_until': 0.85,
            'lag_bias_threshold': 0.01,
            'single_accept_prob': 0.90,
            'surplus_single_p_base': 0.10,
            'surplus_single_p_boost': 0.25,
            # Acceptance rate targets by phase
            'r_target_early': 0.58,
            'r_target_mid': 0.545,
            'r_target_late': 0.53,
            # PI controller for filler prob
            'filler_base': 0.02,
            'filler_max': 0.15,
            'pi_kp': 0.8,
            'pi_ki': 0.1,
            # Feasibility oracle (normal approximation)
            'oracle_delta': 0.005
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        self._acc_int_err = 0.0

    @property
    def name(self) -> str:
        return "DualDeficit"

    def _rate_target(self, phase: str) -> float:
        if phase == 'early':
            return float(self.params.get('r_target_early', 0.56))
        if phase == 'mid':
            return float(self.params.get('r_target_mid', 0.53))
        return float(self.params.get('r_target_late', 0.52))

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        phase = self.get_game_phase(game_state)
        total_dec = max(1, game_state.admitted_count + game_state.rejected_count)
        acc_rate = game_state.admitted_count / total_dec
        r_target = self._rate_target(phase)
        err = r_target - acc_rate
        # integrate error (bounded)
        self._acc_int_err = max(-0.5, min(0.5, self._acc_int_err + err))

        attrs = self.get_person_constraint_attributes(person, game_state)

        # Filler decision via PI controller + feasibility oracle
        if not attrs:
            base = float(self.params.get('filler_base', 0.02))
            kp = float(self.params.get('pi_kp', 0.8))
            ki = float(self.params.get('pi_ki', 0.1))
            p = base + kp * err + ki * self._acc_int_err
            p = max(0.0, min(float(self.params.get('filler_max', 0.15)), p))
            # Only accept filler if feasibility remains high-probability safe
            if self._feasible_after(game_state, accept_y=False, accept_w=False):
                return random.random() < p, "dd_filler"
            return False, "dd_filler_block_oracle"

        # Always accept dual-attribute
        if len(attrs) >= 2:
            return True, "dd_dual"

        progress = game_state.constraint_progress()
        a = next(iter(attrs))
        keys = list(progress.keys())
        b = keys[0] if keys[1] == a else keys[1] if len(keys) > 1 else a
        gap = progress[b] - progress[a]

        min_prog = min(progress.values()) if progress else 0.0
        if min_prog < float(self.params.get('early_singles_until', 0.85)):
            # Always accept early singles; feasibility handled via filler only
            return True, f"dd_single_early_{a}"

        # Urgency-based acceptance
        remaining_accepts = max(1, game_state.target_capacity - game_state.admitted_count)
        shortage = game_state.constraint_shortage()
        u_a = shortage.get(a, 0) / remaining_accepts

        # Strong accept if urgency high or attribute is clearly lagging
        if u_a > 0.55 or gap > float(self.params.get('lag_bias_threshold', 0.01)):
            return True, f"dd_single_urgent_{a}"

        # Otherwise probabilistic acceptance if still needed
        if shortage.get(a, 0) > 0:
            p = float(self.params.get('single_accept_prob', 0.9)) * max(0.6, min(1.0, u_a / 0.5))
            p = max(0.2, min(0.95, p))
            return random.random() < p, f"dd_single_needed_{a}"

        # Surplus single: lightly accept, boost if rate is lagging
        base_p = float(self.params.get('surplus_single_p_base', 0.10))
        boost_p = float(self.params.get('surplus_single_p_boost', 0.25))
        p = boost_p if err > 0.0 else base_p
        return random.random() < p, f"dd_single_surplus_{a}"

    def _feasible_after(self, gs: GameState, accept_y: bool, accept_w: bool) -> bool:
        """Normal-approx feasibility: after hypothetically accepting this person, is meeting quotas likely?"""
        # Compute hypothetical deficits after acceptance
        # Determine attribute keys
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            return True
        a_y, a_w = keys[0], keys[1]
        # current admitted toward each
        cur_y = gs.admitted_attributes.get(a_y, 0)
        cur_w = gs.admitted_attributes.get(a_w, 0)
        # add hypothetical acceptance toward attributes
        if accept_y:
            cur_y += 1
        if accept_w:
            cur_w += 1
        req_y = next(c.min_count for c in gs.constraints if c.attribute == a_y)
        req_w = next(c.min_count for c in gs.constraints if c.attribute == a_w)
        dy = max(0, req_y - cur_y)
        dw = max(0, req_w - cur_w)
        s = max(0, gs.target_capacity - (gs.admitted_count + 1))
        if dy == 0 and dw == 0:
            return True
        if s == 0:
            return False
        # Helpful arrival probabilities (from statistics)
        pbb = 0.0
        p_y = float(gs.statistics.frequencies.get(a_y, 0.0))
        p_w = float(gs.statistics.frequencies.get(a_w, 0.0))
        # Approximate p(Both) via correlation if available
        rho = float(gs.statistics.correlations.get(a_y, {}).get(a_w, 0.0))
        import math
        denom = math.sqrt(max(p_y*(1-p_y), 1e-9) * max(p_w*(1-p_w), 1e-9))
        pbb = max(0.0, min(min(p_y, p_w), p_y*p_w + rho*denom))
        qy = min(1.0, pbb + (p_y - pbb))  # P(has young) = p_y
        qw = min(1.0, pbb + (p_w - pbb))  # P(has well)  = p_w
        # Normal tail check for both deficits
        return (self._binom_tail_ge(s, qy, dy) and self._binom_tail_ge(s, qw, dw))

    def _binom_tail_ge(self, n: int, p: float, k: int) -> bool:
        """Approximate P[Bin(n,p) >= k] >= 1-delta using normal approx with CC."""
        if k <= 0:
            return True
        if n <= 0:
            return False
        import math
        mu = n * p
        var = n * p * (1-p)
        if var < 1e-6:
            return k <= mu
        sigma = math.sqrt(var)
        # continuity correction
        z = (k - 0.5 - mu) / sigma
        # upper tail prob
        # 1 - Phi(z)
        # compare to delta
        delta = float(self.params.get('oracle_delta', 0.005))
        # quick approx for 1 - Phi(z)
        # use erf
        tail = 0.5 * (1 - math.erf(z / math.sqrt(2)))
        return tail <= delta
