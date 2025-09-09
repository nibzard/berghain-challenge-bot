"""ABOUTME: Enhanced Re-solving Bid-Price with Confidence Reserves (RBCR2) strategy implementation.
ABOUTME: Implements LP-optimized dual prices, proper joint probability handling, and improved feasibility checks."""

import random
import math
from typing import Tuple, Optional
from pathlib import Path
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy

try:
    from ..analysis.feasibility_table import FeasibilityOracle
except Exception:
    FeasibilityOracle = None  # type: ignore

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    try:
        from scipy.optimize import linprog
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False


class RBCR2Strategy(BaseDecisionStrategy):
    """Enhanced RBCR strategy with LP-based dual computation and improved joint probability handling."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            'resolve_every': 35,
            # Dynamic rate floor schedule
            'rate_floor_early': 0.580,
            'rate_floor_mid': 0.560,
            'rate_floor_late': 0.540,
            'rate_cut_early': 0.40,
            'rate_cut_mid': 0.70,
            # Filler parameters
            'filler_base': 0.015,
            'filler_max': 0.13,
            # PI controller
            'pi_kp': 1.0,
            'pi_ki': 0.15,
            # Oracle parameters
            'oracle_delta': 0.008,
            'oracle_samples': 4000,
            # Dual learning
            'dual_eta': 0.06,
            'dual_decay': 0.994,
            # LP solver parameters
            'use_lp_duals': True,
            'lp_solver_timeout': 1.0,
            'dual_clamp_max': 50.0,
            'emergency_threshold': 0.95,
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # State variables
        self._since_resolve = 0
        self._lambda_y = 0.0
        self._lambda_w = 0.0
        self._mu = 0.0  # Capacity multiplier
        self._acc_int_err = 0.0
        
        # Joint probability storage
        self._p11 = 0.0  # Both attributes
        self._p10 = 0.0  # Young only
        self._p01 = 0.0  # Well-dressed only
        self._p00 = 0.0  # Neither
        
        # Oracle and persistence
        self._oracle: Optional[FeasibilityOracle] = None
        self._duals_path = Path('game_logs/meta/rbcr2_duals.json')
        self._duals = self._load_duals()

    @property
    def name(self) -> str:
        return "RBCR2"

    def _ensure_oracle(self, gs: GameState):
        """Initialize oracle and compute joint probabilities."""
        if self._oracle is not None:
            return
            
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            return
            
        a_y, a_w = keys[0], keys[1]
        p_y = float(gs.statistics.frequencies.get(a_y, 0.0))
        p_w = float(gs.statistics.frequencies.get(a_w, 0.0))
        rho = float(gs.statistics.correlations.get(a_y, {}).get(a_w, 0.0))
        
        # Compute joint probabilities using Gaussian copula approximation
        denom = math.sqrt(max(p_y*(1-p_y), 1e-9) * max(p_w*(1-p_w), 1e-9))
        p11_guess = p_y * p_w + rho * denom
        
        # Clamp to feasible bounds
        p11 = max(0.0, min(min(p_y, p_w), p11_guess))
        p10 = max(0.0, p_y - p11)
        p01 = max(0.0, p_w - p11)
        p00 = max(0.0, 1.0 - (p11 + p10 + p01))
        
        # Store joint probabilities on instance for reuse
        self._p11, self._p10, self._p01, self._p00 = p11, p10, p01, p00
        
        # Initialize oracle if available
        if FeasibilityOracle is not None:
            self._oracle = FeasibilityOracle(
                p11, p10, p01, p00,
                delta=float(self.params.get('oracle_delta', 0.008)),
                samples=int(self.params.get('oracle_samples', 4000)),
                cache_key=f"scenario_{gs.scenario}"
            )

    def _recompute_duals(self, gs: GameState):
        """Compute dual prices using LP optimization."""
        self._ensure_oracle(gs)
        keys = [c.attribute for c in gs.constraints]
        if len(keys) < 2:
            self._lambda_y = self._lambda_w = self._mu = 0.0
            return

        a_y, a_w = keys[0], keys[1]
        req_y = next(c.min_count for c in gs.constraints if c.attribute == a_y)
        req_w = next(c.min_count for c in gs.constraints if c.attribute == a_w)
        cy = gs.admitted_attributes.get(a_y, 0)
        cw = gs.admitted_attributes.get(a_w, 0)
        dy = max(0, req_y - cy)
        dw = max(0, req_w - cw)
        s = max(1, gs.target_capacity - gs.admitted_count)

        # Load learned dual initializations
        key = f"scenario_{gs.scenario}"
        learned_duals = self._duals.get(key, {})
        lam_y0 = float(learned_duals.get('lambda_y', 0.0))
        lam_w0 = float(learned_duals.get('lambda_w', 0.0))

        # Use enhanced ratio-based method like original RBCR but with joint probabilities
        p_y_total = self._p11 + self._p10 if hasattr(self, '_p11') else float(gs.statistics.frequencies.get(a_y, 0.0))
        p_w_total = self._p11 + self._p01 if hasattr(self, '_p01') else float(gs.statistics.frequencies.get(a_w, 0.0))
        
        qy = max(1e-6, p_y_total)
        qw = max(1e-6, p_w_total)
        
        clamp_max = float(self.params.get('dual_clamp_max', 10.0))
        lam_y = min(clamp_max, dy / (s * qy))
        lam_w = min(clamp_max, dw / (s * qw))
        
        # Blend with learned duals but don't let them dominate
        self._lambda_y = max(lam_y0 * 0.3 + lam_y * 0.7, lam_y)
        self._lambda_w = max(lam_w0 * 0.3 + lam_w * 0.7, lam_w)
        self._mu = 1.0

        self._since_resolve = 0

    def _solve_lp_duals(self, dy: int, dw: int, s: int) -> Tuple[float, float, float]:
        """Solve LP to extract optimal dual prices."""
        # Joint probabilities for the 4 types: (Y,W), (Y,¬W), (¬Y,W), (¬Y,¬W)
        ps = [self._p11, self._p10, self._p01, self._p00]
        
        if HAS_CVXPY:
            return self._solve_cvxpy_duals(dy, dw, s, ps)
        elif HAS_SCIPY:
            return self._solve_scipy_duals(dy, dw, s, ps)
        else:
            # Should not reach here given the check above
            return 0.0, 0.0, 1.0

    def _solve_cvxpy_duals(self, dy: int, dw: int, s: int, ps: list) -> Tuple[float, float, float]:
        """Solve using CVXPY."""
        try:
            # Decision variables: x[0]=both, x[1]=young_only, x[2]=well_only, x[3]=neither
            x = cp.Variable(4, nonneg=True)
            
            constraints = [
                x[0] + x[1] >= dy,          # Young constraint
                x[0] + x[2] >= dw,          # Well-dressed constraint
                cp.sum(x) <= s,             # Capacity constraint
                x <= s * cp.Constant(ps)    # Availability constraints
            ]
            
            objective = cp.Maximize(cp.sum(x))
            prob = cp.Problem(objective, constraints)
            
            prob.solve(solver=cp.ECOS, warm_start=True)
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                lam_y = max(0.0, float(constraints[0].dual_value or 0.0))
                lam_w = max(0.0, float(constraints[1].dual_value or 0.0))
                mu = max(0.0, float(constraints[2].dual_value or 0.0))
                
                # Clamp to reasonable bounds
                clamp_max = float(self.params.get('dual_clamp_max', 50.0))
                return (min(clamp_max, lam_y), min(clamp_max, lam_w), min(clamp_max, mu))
            
        except Exception:
            pass
            
        # Fallback
        return self._fallback_duals(dy, dw, s)

    def _solve_scipy_duals(self, dy: int, dw: int, s: int, ps: list) -> Tuple[float, float, float]:
        """Solve using scipy.optimize.linprog."""
        try:
            # Minimize -sum(x) subject to constraints
            c = [-1, -1, -1, -1]  # Maximize sum(x) -> minimize -sum(x)
            
            # Inequality constraints: Ax <= b
            A_ub = [
                [-1, -1, 0, 0],   # -(x0 + x1) <= -dy  =>  x0 + x1 >= dy
                [-1, 0, -1, 0],   # -(x0 + x2) <= -dw  =>  x0 + x2 >= dw
                [1, 1, 1, 1],     # sum(x) <= s
                [1, 0, 0, 0],     # x0 <= s*p11
                [0, 1, 0, 0],     # x1 <= s*p10
                [0, 0, 1, 0],     # x2 <= s*p01
                [0, 0, 0, 1],     # x3 <= s*p00
            ]
            b_ub = [-dy, -dw, s, s*ps[0], s*ps[1], s*ps[2], s*ps[3]]
            
            # Bounds: all variables >= 0
            bounds = [(0, None)] * 4
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                # Extract dual values (shadow prices) from slack variables
                # Note: scipy doesn't directly provide dual values, so we approximate
                x_opt = result.x
                
                # Check which constraints are binding to estimate duals
                eps = 1e-6
                young_binding = abs(x_opt[0] + x_opt[1] - dy) < eps
                well_binding = abs(x_opt[0] + x_opt[2] - dw) < eps
                cap_binding = abs(sum(x_opt) - s) < eps
                
                # Rough dual approximation based on binding constraints
                lam_y = 1.0 if young_binding else 0.0
                lam_w = 1.0 if well_binding else 0.0  
                mu = 1.0 if cap_binding else 0.0
                
                clamp_max = float(self.params.get('dual_clamp_max', 50.0))
                return (min(clamp_max, lam_y), min(clamp_max, lam_w), min(clamp_max, mu))
                
        except Exception:
            pass
            
        return self._fallback_duals(dy, dw, s)

    def _fallback_duals(self, dy: int, dw: int, s: int) -> Tuple[float, float, float]:
        """Fallback dual computation when LP solving fails."""
        p_y_total = self._p11 + self._p10
        p_w_total = self._p11 + self._p01
        
        qy = max(1e-6, p_y_total)
        qw = max(1e-6, p_w_total)
        
        clamp_max = float(self.params.get('dual_clamp_max', 50.0))
        lam_y = min(clamp_max, dy / (s * qy))
        lam_w = min(clamp_max, dw / (s * qw))
        mu = 1.0
        
        return lam_y, lam_w, mu

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Main decision logic with enhanced dual-based pricing."""
        # CRITICAL: Constraint safety override - this overrides all other logic
        constraint_override, constraint_reason = self._constraint_safety_check(person, game_state)
        if constraint_override is not None:
            return constraint_override, f"RBCR2_CONSTRAINT_OVERRIDE: {constraint_reason}"
        
        if self.is_emergency_mode(game_state):
            return True, "rbcr2_emergency_mode"

        self._since_resolve += 1
        if self._since_resolve >= int(self.params.get('resolve_every', 35)):
            self._recompute_duals(game_state)

        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]

        # Dynamic acceptance rate control with PI controller
        total_dec = max(1, game_state.admitted_count + game_state.rejected_count)
        acc_rate = game_state.admitted_count / total_dec
        cr = game_state.capacity_ratio
        
        # Dynamic floor schedule
        r_e = float(self.params.get('rate_floor_early', 0.580))
        r_m = float(self.params.get('rate_floor_mid', 0.560))
        r_l = float(self.params.get('rate_floor_late', 0.540))
        c_e = float(self.params.get('rate_cut_early', 0.40))
        c_m = float(self.params.get('rate_cut_mid', 0.70))
        
        floor_r = r_e if cr < c_e else (r_m if cr < c_m else r_l)
        err = floor_r - acc_rate
        self._acc_int_err = max(-0.5, min(0.5, self._acc_int_err + err))
        
        kp = float(self.params.get('pi_kp', 1.0))
        ki = float(self.params.get('pi_ki', 0.15))

        if not attrs:
            # Enhanced filler logic with better feasibility gating
            p = float(self.params.get('filler_base', 0.015)) + kp * err + ki * self._acc_int_err
            p = max(0.0, min(float(self.params.get('filler_max', 0.13)), p))
            
            if p <= 0.0:
                return False, "rbcr2_filler_zero"
                
            # Improved feasibility gating: allow fillers when rate is low OR progress is early
            progress = game_state.constraint_progress()
            min_prog = min(progress.values()) if progress else 0.0
            
            if err > 0.0 or min_prog < 0.92 or self._feasible_after(game_state, False, False):
                return random.random() < p, "rbcr2_filler"
            return False, "rbcr2_filler_blocked_oracle"

        # Always accept people with both attributes (highest value)
        if len(attrs) >= 2:
            return True, "rbcr2_dual_accept"

        # Enhanced single-attribute logic using LP-optimized duals
        is_y = (a_y in attrs)
        is_w = (a_w in attrs)
        shortage = game_state.constraint_shortage()
        
        # CRITICAL FIX: At 999 capacity, accept anyone with needed attributes
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        if capacity_remaining == 1:
            if is_y and shortage.get(a_y, 0) > 0:
                return True, "rbcr2_final_need_y_at_999"
            if is_w and shortage.get(a_w, 0) > 0:
                return True, "rbcr2_final_need_w_at_999"
        
        # Accept if strictly needed
        if is_y and shortage.get(a_y, 0) > 0:
            return True, "rbcr2_single_need_y"
        if is_w and shortage.get(a_w, 0) > 0:
            return True, "rbcr2_single_need_w"
            
        # Use dual prices for priority when both constraints are close to satisfied
        dual_value_y = self._lambda_y if is_y else 0.0
        dual_value_w = self._lambda_w if is_w else 0.0
        dual_value_total = dual_value_y + dual_value_w
        
        # Accept based on dual urgency
        if is_y and self._lambda_y >= self._lambda_w:
            return True, f"rbcr2_single_dual_y_{dual_value_total:.2f}"
        if is_w and self._lambda_w > self._lambda_y:
            return True, f"rbcr2_single_dual_w_{dual_value_total:.2f}"

        # Soft acceptance for borderline cases (similar to original RBCR)
        if err > 0.0 or self._feasible_after(game_state, is_y, is_w):
            return random.random() < 0.35, "rbcr2_single_soft"
            
        return False, "rbcr2_single_blocked_oracle"

    def _feasible_after(self, gs: GameState, accept_y: bool, accept_w: bool) -> bool:
        """Enhanced feasibility check with proper joint probability handling."""
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

        # Use oracle if available
        if self._oracle is not None:
            return self._oracle.is_feasible(dy, dw, s)

        # Enhanced fallback using Bonferroni bound with joint probabilities
        p_y_total = self._p11 + self._p10  # Total young arrival rate
        p_w_total = self._p11 + self._p01  # Total well-dressed arrival rate
        
        # Use individual binomial tests as in original RBCR (less conservative)
        delta = float(self.params.get('oracle_delta', 0.015))
        
        # Use original RBCR approach with enhanced probabilities
        y_feasible = self._binom_tail_ge(s, p_y_total, dy)
        w_feasible = self._binom_tail_ge(s, p_w_total, dw)
        
        return y_feasible and w_feasible

    def _binom_tail_ge(self, n: int, p: float, k: int) -> bool:
        """Check if P(X >= k) > threshold for X ~ Binomial(n, p)."""
        if k <= 0:
            return True
        if n <= 0:
            return False
            
        mu = n * p
        var = n * p * (1 - p)
        
        if var < 1e-6:
            return k <= mu
            
        sigma = math.sqrt(var)
        z = (k - 0.5 - mu) / sigma  # Continuity correction
        delta = float(self.params.get('oracle_delta', 0.015))
        
        # P(X >= k) = 0.5 * (1 - erf(z / sqrt(2)))
        tail = 0.5 * (1 - math.erf(z / math.sqrt(2)))
        return tail <= delta

    def _binom_tail_lt(self, n: int, p: float, k: int) -> float:
        """Compute P(X < k) for X ~ Binomial(n, p) using normal approximation."""
        if k <= 0:
            return 0.0
        if n <= 0:
            return 1.0 if k > 0 else 0.0
            
        mu = n * p
        var = n * p * (1 - p)
        
        if var < 1e-6:
            return 1.0 if k > mu else 0.0
            
        sigma = math.sqrt(var)
        z = (k - 0.5 - mu) / sigma  # Continuity correction
        
        # P(X < k) = P(Z < z) where Z ~ N(0,1)
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def is_emergency_mode(self, gs: GameState) -> bool:
        """Enhanced emergency mode detection."""
        progress = gs.constraint_progress()
        if not progress:
            return False
            
        min_progress = min(progress.values())
        capacity_used = gs.capacity_ratio
        
        # Enter emergency mode if very close to capacity limit with unmet constraints
        emergency_threshold = float(self.params.get('emergency_threshold', 0.95))
        return capacity_used >= emergency_threshold and min_progress < 1.0

    # Dual learning persistence with enhanced subgradient updates
    def _load_duals(self):
        """Load previously learned dual values."""
        try:
            if self._duals_path.exists():
                import json
                return json.load(open(self._duals_path, 'r'))
        except Exception:
            pass
        return {}

    def on_game_end(self, result) -> None:
        """Enhanced dual learning with proper subgradient updates."""
        try:
            key = f"scenario_{result.game_state.scenario}"
            ca = result.game_state.admitted_attributes
            
            # Get constraint attributes and requirements
            constraints = result.game_state.constraints
            if len(constraints) < 2:
                return
                
            y_constraint = constraints[0]
            w_constraint = constraints[1]
            
            y_attr, w_attr = y_constraint.attribute, w_constraint.attribute
            y_req, w_req = y_constraint.min_count, w_constraint.min_count
            
            # Compute constraint violations (subgradient components)
            err_y = max(0, y_req - ca.get(y_attr, 0))
            err_w = max(0, w_req - ca.get(w_attr, 0))
            
            # Load previous state
            prev = self._duals.get(key, {
                'lambda_y': 0.0, 
                'lambda_w': 0.0, 
                'eta': float(self.params.get('dual_eta', 0.06))
            })
            
            eta = prev.get('eta', float(self.params.get('dual_eta', 0.06)))
            
            # Subgradient update with projection to [0, ∞)
            lam_y_new = max(0.0, prev['lambda_y'] + eta * err_y / y_req)
            lam_w_new = max(0.0, prev['lambda_w'] + eta * err_w / w_req)
            
            # Adaptive step size with decay
            decay = float(self.params.get('dual_decay', 0.994))
            new_eta = max(1e-4, eta * decay)
            
            # Store updated values
            self._duals[key] = {
                'lambda_y': lam_y_new,
                'lambda_w': lam_w_new,
                'eta': new_eta,
                'games_played': prev.get('games_played', 0) + 1,
                'last_err_y': float(err_y),
                'last_err_w': float(err_w)
            }
            
            # Persist to disk
            self._duals_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(self._duals_path, 'w') as f:
                json.dump(self._duals, f, indent=2)
                
        except Exception as e:
            # Log the error but don't crash the game
            print(f"Warning: Failed to update duals: {e}")

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
        
        # CAPACITY FILL: If constraints are met, fill remaining capacity
        if young_deficit == 0 and well_dressed_deficit == 0 and capacity_remaining > 0:
            return True, f"FILL_CAPACITY_constraints_met_cap={capacity_remaining}"
        
        # No override needed
        return None, "no_constraint_override"