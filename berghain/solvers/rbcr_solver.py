"""Re-solving Bid-Price with Confidence Reserves (RBCR) strategy.

Heuristic, near-optimal controller:
- Recompute dual-like urgency weights every K arrivals from deficits and
  expected helpful rates.
- Always accept duals; accept singles that reduce the larger Lagrange-like
  urgency; allow small filler only to maintain acceptance-rate floor and only
  when feasibility (normal-approx) looks safe.
"""

import random
from typing import Tuple, Optional
from pathlib import Path
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
try:
    from ..analysis.feasibility_table import FeasibilityOracle  # optional
except Exception:
    FeasibilityOracle = None  # type: ignore


class RBCRStrategy(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            'resolve_every': 50,
            # Dynamic rate floor schedule
            'rate_floor_early': 0.575,
            'rate_floor_mid': 0.56,
            'rate_floor_late': 0.545,
            'rate_cut_early': 0.40,   # capacity ratio threshold
            'rate_cut_mid': 0.70,
            'filler_base': 0.02,
            'filler_max': 0.14,
            'pi_kp': 0.9,
            'pi_ki': 0.12,
            'oracle_delta': 0.012,
            # Dual-learning across runs
            'dual_eta': 0.05,
            'dual_decay': 0.995,
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        self._since_resolve = 0
        self._lambda_y = 0.0
        self._lambda_w = 0.0
        self._acc_int_err = 0.0
        self._oracle: FeasibilityOracle | None = None
        self._duals_path = Path('game_logs/meta/rbcr_duals.json')
        self._duals = self._load_duals()

    @property
    def name(self) -> str:
        return "RBCR"

    def _ensure_oracle(self, gs: GameState):
        if self._oracle is not None or FeasibilityOracle is None:
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
        # Initialize from learned duals if available
        key = f"scenario_{gs.scenario}"
        lam_y0 = float(self._duals.get(key, {}).get('lambda_y', 0.0))
        lam_w0 = float(self._duals.get(key, {}).get('lambda_w', 0.0))
        self._lambda_y = max(lam_y0, min(10.0, dy / (s * qy)))
        self._lambda_w = max(lam_w0, min(10.0, dw / (s * qw)))
        self._since_resolve = 0

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        # CRITICAL: Constraint safety override - this overrides all other logic
        constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
        if constraint_override is not None:
            return constraint_override, f"RBCR_CONSTRAINT_OVERRIDE: {constraint_reason}"
        
        if self.is_emergency_mode(game_state):
            return True, "emergency_mode"

        self._since_resolve += 1
        if self._since_resolve >= int(self.params.get('resolve_every', 50)):
            self._recompute_duals(game_state)

        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]

        # acceptance rate PI controller with dynamic schedule
        total_dec = max(1, game_state.admitted_count + game_state.rejected_count)
        acc_rate = game_state.admitted_count / total_dec
        cr = game_state.capacity_ratio
        r_e = float(self.params.get('rate_floor_early', 0.575))
        r_m = float(self.params.get('rate_floor_mid', 0.56))
        r_l = float(self.params.get('rate_floor_late', 0.545))
        c_e = float(self.params.get('rate_cut_early', 0.40))
        c_m = float(self.params.get('rate_cut_mid', 0.70))
        floor_r = r_e if cr < c_e else (r_m if cr < c_m else r_l)
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
            # Allow filler freely when rate is below target OR until strong base exists;
            # only gate with oracle when rate is healthy and progress is high.
            progress = game_state.constraint_progress()
            min_prog = min(progress.values()) if progress else 0.0
            if err > 0.0 or min_prog < 0.90 or self._feasible_after(game_state, False, False):
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
        # If acceptance is lagging, don't gate soft singles; otherwise use oracle
        if err > 0.0 or self._feasible_after(game_state, is_y, is_w):
            return random.random() < 0.35, "rbcr_single_soft"
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
        s = max(0, gs.target_capacity - gs.admitted_count)
        if dy == 0 and dw == 0:
            return True
        # CRITICAL FIX: Don't reject at 999 - allow the final admission
        if s <= 0:
            return False
        # Special case: At 999, accept anyone with needed attributes
        if s == 1 and (dy > 0 or dw > 0):
            return True
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

    # --- Dual learning persistence ---
    def _load_duals(self):
        try:
            if self._duals_path.exists():
                import json
                return json.load(open(self._duals_path, 'r'))
        except Exception:
            pass
        return {}

    def on_game_end(self, result) -> None:
        try:
            key = f"scenario_{result.game_state.scenario}"
            ca = result.game_state.admitted_attributes
            # assume two constraints
            attrs = [c.attribute for c in result.game_state.constraints]
            y, w = attrs[0], attrs[1]
            err_y = max(0, 600 - ca.get(y, 0))
            err_w = max(0, 600 - ca.get(w, 0))
            eta = float(self.params.get('dual_eta', 0.05))
            decay = float(self.params.get('dual_decay', 0.995))
            prev = self._duals.get(key, {'lambda_y': 0.0, 'lambda_w': 0.0, 'eta': eta})
            lam_y = max(0.0, float(prev['lambda_y']) + prev.get('eta', eta) * err_y / 600.0)
            lam_w = max(0.0, float(prev['lambda_w']) + prev.get('eta', eta) * err_w / 600.0)
            new_eta = max(0.001, prev.get('eta', eta) * decay)
            self._duals[key] = {'lambda_y': lam_y, 'lambda_w': lam_w, 'eta': new_eta}
            self._duals_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            json.dump(self._duals, open(self._duals_path, 'w'), indent=2)
        except Exception:
            pass

    def _constraint_safety_check(self, person: Person, game_state: GameState) -> Tuple[Optional[bool], str]:
        """
        Critical constraint safety check - overrides all other logic.
        Returns (None, reason) if no override needed.
        Returns (True/False, reason) if override is required.
        """
        has_young = person.has_attribute('young')
        has_well_dressed = person.has_attribute('well_dressed')
        
        # Get current constraint status
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        
        # Calculate deficits
        young_deficit = max(0, 600 - young_current)
        well_dressed_deficit = max(0, 600 - well_dressed_current)
        
        # MANDATORY ACCEPT: Critical constraint situation
        if capacity_remaining <= max(young_deficit, well_dressed_deficit):
            # Running out of capacity and still need constraints
            if young_deficit > 0 and has_young:
                return True, f"MUST_ACCEPT_young_deficit={young_deficit}_cap={capacity_remaining}"
            if well_dressed_deficit > 0 and has_well_dressed:
                return True, f"MUST_ACCEPT_well_dressed_deficit={well_dressed_deficit}_cap={capacity_remaining}"
            # If we need both and person has both
            if young_deficit > 0 and well_dressed_deficit > 0 and has_young and has_well_dressed:
                return True, f"MUST_ACCEPT_dual_needed_y={young_deficit}_w={well_dressed_deficit}_cap={capacity_remaining}"
        
        # MANDATORY REJECT: Would make constraint satisfaction impossible
        if capacity_remaining > 0:
            # Check if accepting this person would use capacity we need for constraints
            remaining_after = capacity_remaining - 1
            if remaining_after < (young_deficit + well_dressed_deficit):
                # Only allow if this person helps with constraints
                if not ((young_deficit > 0 and has_young) or (well_dressed_deficit > 0 and has_well_dressed)):
                    return False, f"MUST_REJECT_constraint_safety_y_need={young_deficit}_w_need={well_dressed_deficit}_cap_after={remaining_after}"
        
        # SPECIAL CASE: At 999 capacity, accept anyone with needed attributes to reach 1000
        if capacity_remaining == 1:
            if young_deficit > 0 and has_young:
                return True, f"FINAL_ACCEPT_young_deficit={young_deficit}_at_999"
            if well_dressed_deficit > 0 and has_well_dressed:
                return True, f"FINAL_ACCEPT_well_dressed_deficit={well_dressed_deficit}_at_999"
            # If both deficits exist and person has both, prioritize
            if young_deficit > 0 and well_dressed_deficit > 0 and has_young and has_well_dressed:
                return True, f"FINAL_ACCEPT_dual_needed_at_999"
        
        # CAPACITY FILL: If constraints are met, fill remaining capacity
        if young_deficit == 0 and well_dressed_deficit == 0 and capacity_remaining > 0:
            return True, f"FILL_CAPACITY_constraints_met_cap={capacity_remaining}"
        
        # ANTI-LOOP: If at 999 and very close to completing (deficit <= 1), accept any helpful person
        if capacity_remaining == 1 and (young_deficit <= 1 or well_dressed_deficit <= 1):
            if (young_deficit == 1 and has_young) or (well_dressed_deficit == 1 and has_well_dressed):
                return True, f"ANTI_LOOP_999_close_completion_y={young_deficit}_w={well_dressed_deficit}"
        
        # No override needed
        return None, "no_constraint_override"
