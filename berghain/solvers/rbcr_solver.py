"""Re-solving Bid-Price with Confidence Reserves (RBCR) strategy.

Heuristic, near-optimal controller:
- Recompute dual-like urgency weights every K arrivals from deficits and
  expected helpful rates.
- Always accept duals; accept singles that reduce the larger Lagrange-like
  urgency; allow small filler only to maintain acceptance-rate floor and only
  when feasibility (normal-approx) looks safe.
"""

import random
from typing import Tuple
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from ..analysis.feasibility_table import FeasibilityOracle


class RBCRStrategy(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            'resolve_every': 50,
            'rate_floor': 0.54,
            'filler_base': 0.02,
            'filler_max': 0.12,
            'pi_kp': 0.6,
            'pi_ki': 0.08,
            'oracle_delta': 0.01,
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        self._since_resolve = 0
        self._lambda_y = 0.0
        self._lambda_w = 0.0
        self._acc_int_err = 0.0
        self._oracle: FeasibilityOracle | None = None

    @property
    def name(self) -> str:
        return "RBCR"

    def _ensure_oracle(self, gs: GameState):
        if self._oracle is not None:
            return
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            return
        a_y, a_w = keys[0], keys[1]
        p_y = float(gs.statistics.frequencies.get(a_y, 0.0))
        p_w = float(gs.statistics.frequencies.get(a_w, 0.0))
        # correlation -> p11
        import math
        rho = float(gs.statistics.correlations.get(a_y, {}).get(a_w, 0.0))
        denom = math.sqrt(max(p_y*(1-p_y), 1e-9) * max(p_w*(1-p_w), 1e-9))
        p11 = max(0.0, min(min(p_y, p_w), p_y*p_w + rho*denom))
        p10 = max(0.0, p_y - p11)
        p01 = max(0.0, p_w - p11)
        p00 = max(0.0, 1.0 - (p11 + p10 + p01))
        self._oracle = FeasibilityOracle(p11, p10, p01, p00,
                                         delta=float(self.params.get('oracle_delta', 0.01)),
                                         samples=4000,
                                         cache_key=f"scenario_{gs.scenario}")

    def _recompute_duals(self, gs: GameState):
        self._ensure_oracle(gs)
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            self._lambda_y = self._lambda_w = 0.0
            return
        a_y, a_w = keys[0], keys[1]
        req_y = next(c.min_count for c in gs.constraints if c.attribute == a_y)
        req_w = next(c.min_count for c in gs.constraints if c.attribute == a_w)
        cy = gs.admitted_attributes.get(a_y, 0)
        cw = gs.admitted_attributes.get(a_w, 0)
        dy = max(0, req_y - cy)
        dw = max(0, req_w - cw)
        s = max(1, gs.target_capacity - gs.admitted_count)
        # helpful rates
        p_y = float(gs.statistics.frequencies.get(a_y, 0.0))
        p_w = float(gs.statistics.frequencies.get(a_w, 0.0))
        # approximate duals as deficit divided by expected helpful arrivals per remaining slot
        # use helpful probabilities (including duals)
        # p(help_y)=p_y, p(help_w)=p_w
        qy = max(1e-6, p_y)
        qw = max(1e-6, p_w)
        self._lambda_y = min(10.0, dy / (s * qy))
        self._lambda_w = min(10.0, dw / (s * qw))
        self._since_resolve = 0

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        self._since_resolve += 1
        if self._since_resolve >= int(self.params.get('resolve_every', 50)):
            self._recompute_duals(game_state)

        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]

        # acceptance rate PI controller
        total_dec = max(1, game_state.admitted_count + game_state.rejected_count)
        acc_rate = game_state.admitted_count / total_dec
        floor_r = float(self.params.get('rate_floor', 0.54))
        err = floor_r - acc_rate
        self._acc_int_err = max(-0.5, min(0.5, self._acc_int_err + err))
        kp = float(self.params.get('pi_kp', 0.6))
        ki = float(self.params.get('pi_ki', 0.08))

        if not attrs:
            # Filler only to maintain acceptance-rate, and only if feasibility OK
            p = float(self.params.get('filler_base', 0.02)) + kp * err + ki * self._acc_int_err
            p = max(0.0, min(float(self.params.get('filler_max', 0.12)), p))
            if p <= 0.0:
                return False, "rbcr_filler_zero"
            # Allow filler freely until strong base exists; gate with oracle later
            progress = game_state.constraint_progress()
            min_prog = min(progress.values()) if progress else 0.0
            if min_prog < 0.85 or self._feasible_after(game_state, False, False):
                return random.random() < p, "rbcr_filler"
            return False, "rbcr_filler_block_oracle"

        # Always accept duals
        if len(attrs) >= 2:
            return True, "rbcr_dual"

        # Single: accept if needed (reduces any positive deficit); otherwise soft
        is_y = (a_y in attrs)
        is_w = (a_w in attrs)
        shortage = game_state.constraint_shortage()
        if is_y and shortage.get(a_y, 0) > 0:
            return True, "rbcr_single_need_y"
        if is_w and shortage.get(a_w, 0) > 0:
            return True, "rbcr_single_need_w"
        # urgency comparison if neither strictly needed (close to done)
        if is_y and (self._lambda_y >= self._lambda_w):
            return True, "rbcr_single_y"
        if is_w and (self._lambda_w > self._lambda_y):
            return True, "rbcr_single_w"

        # If tie or mild, allow with small probability if feasibility safe
        if self._feasible_after(game_state, is_y, is_w):
            return random.random() < 0.2, "rbcr_single_soft"
        return False, "rbcr_single_block_oracle"

    def _feasible_after(self, gs: GameState, accept_y: bool, accept_w: bool) -> bool:
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            return True
        a_y, a_w = keys[0], keys[1]
        cy = gs.admitted_attributes.get(a_y, 0) + (1 if accept_y else 0)
        cw = gs.admitted_attributes.get(a_w, 0) + (1 if accept_w else 0)
        ry = next(c.min_count for c in gs.constraints if c.attribute == a_y)
        rw = next(c.min_count for c in gs.constraints if c.attribute == a_w)
        dy = max(0, ry - cy)
        dw = max(0, rw - cw)
        s = max(0, gs.target_capacity - (gs.admitted_count + 1))
        if dy == 0 and dw == 0:
            return True
        if s <= 0:
            return False
        # Use oracle if available, otherwise fallback to normal approx
        if self._oracle is not None:
            return self._oracle.is_feasible(dy, dw, s)
        p_y = float(gs.statistics.frequencies.get(a_y, 0.0))
        p_w = float(gs.statistics.frequencies.get(a_w, 0.0))
        return (self._binom_tail_ge(s, p_y, dy) and self._binom_tail_ge(s, p_w, dw))

    def _binom_tail_ge(self, n: int, p: float, k: int) -> bool:
        if k <= 0:
            return True
        if n <= 0:
            return False
        import math
        mu = n*p
        var = n*p*(1-p)
        if var < 1e-6:
            return k <= mu
        sigma = math.sqrt(var)
        z = (k - 0.5 - mu) / sigma
        delta = float(self.params.get('oracle_delta', 0.01))
        tail = 0.5 * (1 - math.erf(z / math.sqrt(2)))
        return tail <= delta
