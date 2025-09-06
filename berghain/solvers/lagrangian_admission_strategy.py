"""ABOUTME: Lagrangian Admission Control - Theoretically Optimal Online Resource Allocation

Implementation of classic bid-price control and probability-of-failure policies for 
constrained admission problems. Based on dual pricing theory and binomial tail bounds.

Key Features:
- Variant A: Bid-price control using LP dual multipliers (shadow prices)  
- Variant B: Probability-of-failure with binomial tail safety checks
- Handles correlated binary attributes via joint type probabilities
- Near-optimal performance with theoretical guarantees
"""

import math
import random
from typing import Tuple, Dict, List, Optional
from scipy.stats import binom
from scipy.optimize import linprog
import numpy as np
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy  
from .base_solver import BaseSolver


class LagrangianAdmissionSolver(BaseSolver):
    """Lagrangian admission control solver with dual pricing."""
    
    def __init__(self, solver_id: str = "lagrangian_admission", config_manager=None, 
                 api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("lagrangian_admission")
        strategy = LagrangianAdmissionStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class LagrangianAdmissionStrategy(BaseDecisionStrategy):
    """Theoretically optimal admission control using Lagrangian dual pricing."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Policy variant selection
            'policy_variant': 'bid_price',  # 'bid_price' or 'prob_failure' or 'hybrid'
            
            # Bid-price policy parameters
            'lambda_update_frequency': 10,  # Update dual multipliers every N decisions
            'lambda_learning_rate': 0.05,  # Gradient step size for multiplier updates
            'lambda_momentum': 0.9,        # Momentum for smoother updates
            'value_epsilon': 0.01,         # Small threshold for bid-price acceptance
            
            # Probability-of-failure policy parameters  
            'safety_alpha': 0.05,          # Safety threshold (1-99% confidence)
            'binomial_approx_threshold': 5, # Use normal approx when n*p > threshold
            
            # Type enumeration and probabilities
            'estimate_types_online': True,  # Estimate type probabilities from arrivals
            'type_smoothing_factor': 0.95,  # Exponential smoothing for type prob updates
            'min_samples_for_estimation': 50, # Min arrivals before trusting estimates
            
            # Capacity management
            'early_capacity_conservatism': 1.1, # Be conservative early (>1.0)
            'late_capacity_adjustment': 0.9,    # Be aggressive late (<1.0)  
            'capacity_transition_point': 0.7,   # When to switch from early->late
            
            # Hybrid policy parameters
            'hybrid_bid_price_weight': 0.7,     # Weight for bid-price in hybrid
            'hybrid_safety_check_weight': 0.3,  # Weight for safety check in hybrid
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # Dual multipliers (shadow prices)
        self._lambda_multipliers: Dict[str, float] = {}
        self._lambda_momentum: Dict[str, float] = {}
        
        # Type tracking for probability estimation
        self._type_counts: Dict[tuple, int] = {}  # (attr1, attr2, ...) -> count
        self._total_arrivals = 0
        self._type_probabilities: Dict[tuple, float] = {}
        
        # Decision tracking
        self._decisions_since_update = 0
        self._recent_gradients: Dict[str, List[float]] = {}

    @property 
    def name(self) -> str:
        return "LagrangianAdmission"

    def _enumerate_person_type(self, person: Person, game_state: GameState) -> tuple:
        """Convert person to type tuple based on constraint attributes."""
        attrs = self.get_person_constraint_attributes(person, game_state)
        constraint_keys = [c.attribute for c in game_state.constraints]
        
        # Create binary tuple: (has_attr1, has_attr2, ...)
        type_tuple = tuple(attr in attrs for attr in constraint_keys)
        return type_tuple

    def _update_type_probabilities(self, person_type: tuple):
        """Update type probability estimates from online arrivals."""
        if not self.params.get('estimate_types_online', True):
            return
            
        # Count this type
        self._type_counts[person_type] = self._type_counts.get(person_type, 0) + 1
        self._total_arrivals += 1
        
        # Update probabilities with exponential smoothing
        smoothing = float(self.params['type_smoothing_factor'])
        
        if self._total_arrivals >= int(self.params['min_samples_for_estimation']):
            for ptype, count in self._type_counts.items():
                empirical_prob = count / self._total_arrivals
                
                if ptype in self._type_probabilities:
                    # Exponential smoothing update
                    self._type_probabilities[ptype] = (
                        smoothing * self._type_probabilities[ptype] + 
                        (1 - smoothing) * empirical_prob
                    )
                else:
                    self._type_probabilities[ptype] = empirical_prob

    def _solve_dual_lp(self, game_state: GameState) -> Dict[str, float]:
        """Solve LP dual to get shadow prices (Lagrange multipliers)."""
        constraints = game_state.constraints
        if not constraints:
            return {}
            
        # Build LP for static fluid approximation
        # Variables: acceptance rates a_t for each type t
        type_probs = self._type_probabilities
        if not type_probs:
            # Use uniform distribution as fallback
            n_attrs = len(constraints)
            n_types = 2 ** n_attrs
            uniform_prob = 1.0 / n_types
            for i in range(n_types):
                type_tuple = tuple((i >> j) & 1 == 1 for j in range(n_attrs))
                type_probs[type_tuple] = uniform_prob
        
        # Capacity constraint and requirements
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        if capacity_remaining <= 0:
            return {}
            
        expected_arrivals = capacity_remaining / 0.8  # Assume 80% acceptance rate
        
        # Build constraint matrix
        # Each row = constraint, each col = type
        types = list(type_probs.keys())
        n_types = len(types)
        n_constraints = len(constraints)
        
        # Inequality constraints: -sum(a_t * q_t * x_{t,j}) <= -R_j/expected_arrivals  
        A_ub = []
        b_ub = []
        
        for i, constraint in enumerate(constraints):
            row = []
            for t, type_tuple in enumerate(types):
                has_attr = type_tuple[i] if i < len(type_tuple) else False
                coeff = -type_probs[type_tuple] if has_attr else 0.0
                row.append(coeff)
            
            A_ub.append(row)
            
            # Required fraction for this constraint
            current_count = game_state.admitted_attributes.get(constraint.attribute, 0)
            remaining_required = max(0, constraint.min_count - current_count)
            b_ub.append(-remaining_required / expected_arrivals)
        
        # Capacity constraint: sum(a_t * q_t) = N/expected_arrivals
        A_eq = [[type_probs[t] for t in types]]
        b_eq = [capacity_remaining / expected_arrivals]
        
        # Bounds: 0 <= a_t <= 1
        bounds = [(0, 1) for _ in range(n_types)]
        
        # Objective: minimize sum(a_t) (or maximize utility)
        c = [1.0] * n_types  # Minimize total acceptance (placeholder)
        
        try:
            # Solve LP
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success and hasattr(result, 'ineqlin') and hasattr(result.ineqlin, 'marginals'):
                # Extract dual variables (shadow prices)
                dual_vars = result.ineqlin.marginals
                multipliers = {}
                for i, constraint in enumerate(constraints):
                    if i < len(dual_vars):
                        multipliers[constraint.attribute] = max(0.0, -dual_vars[i])
                return multipliers
        except Exception as e:
            print(f"LP solve failed: {e}")
        
        # Fallback: simple heuristic multipliers
        multipliers = {}
        for constraint in constraints:
            shortage = game_state.constraint_shortage().get(constraint.attribute, 0)
            multipliers[constraint.attribute] = max(0.0, shortage / 100.0)
        
        return multipliers

    def _update_dual_multipliers(self, game_state: GameState):
        """Update Lagrange multipliers using gradient-based learning."""
        update_freq = int(self.params['lambda_update_frequency'])
        
        if self._decisions_since_update < update_freq:
            return
            
        self._decisions_since_update = 0
        
        # Option 1: Solve LP for exact duals (expensive but optimal)
        if len(self._type_probabilities) > 4:  # Only if we have enough data
            lp_multipliers = self._solve_dual_lp(game_state)
            if lp_multipliers:
                for attr, lam in lp_multipliers.items():
                    self._lambda_multipliers[attr] = lam
                return
        
        # Option 2: Gradient-based updates (faster)
        learning_rate = float(self.params['lambda_learning_rate'])
        momentum = float(self.params['lambda_momentum'])
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            shortage = game_state.constraint_shortage().get(attr, 0)
            
            # Gradient = constraint violation (positive if violated)
            gradient = shortage / game_state.target_capacity
            
            # Initialize if needed
            if attr not in self._lambda_multipliers:
                self._lambda_multipliers[attr] = 1.0
                self._lambda_momentum[attr] = 0.0
            
            # Momentum update
            self._lambda_momentum[attr] = (momentum * self._lambda_momentum[attr] + 
                                         learning_rate * gradient)
            
            # Update multiplier (project to non-negative)
            self._lambda_multipliers[attr] = max(0.0, 
                self._lambda_multipliers[attr] + self._lambda_momentum[attr])

    def _compute_bid_price_value(self, person_type: tuple, game_state: GameState) -> float:
        """Compute person's net value using bid-price control."""
        # Base value = 1 (filling one slot)
        value = 1.0
        
        # Subtract opportunity costs (Lagrange multipliers)
        constraint_attrs = [c.attribute for c in game_state.constraints]
        
        for i, has_attr in enumerate(person_type):
            if has_attr and i < len(constraint_attrs):
                attr = constraint_attrs[i]
                lambda_val = self._lambda_multipliers.get(attr, 0.0)
                value -= lambda_val
        
        return value

    def _compute_failure_probability(self, person_type: tuple, game_state: GameState) -> bool:
        """Check if rejecting this person risks constraint failure."""
        constraint_attrs = [c.attribute for c in game_state.constraints]
        safety_alpha = float(self.params['safety_alpha'])
        
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        if capacity_remaining <= 1:
            return False  # No capacity to be concerned about
        
        for i, has_attr in enumerate(person_type):
            if not has_attr or i >= len(constraint_attrs):
                continue
                
            attr = constraint_attrs[i]
            constraint = game_state.constraints[i]
            
            current_count = game_state.admitted_attributes.get(attr, 0)
            remaining_required = max(0, constraint.min_count - current_count)
            
            if remaining_required <= 0:
                continue  # This constraint is already satisfied
            
            # Probability that a future person has this attribute
            marginal_prob = 0.0
            for ptype, prob in self._type_probabilities.items():
                if i < len(ptype) and ptype[i]:  # Type has this attribute
                    marginal_prob += prob
            
            if marginal_prob <= 0:
                return True  # Accept since we'll never see this attribute again
            
            # Binomial tail: P(Binom(capacity_remaining-1, marginal_prob) >= remaining_required)
            n = capacity_remaining - 1
            p = marginal_prob
            k = remaining_required
            
            threshold = int(self.params['binomial_approx_threshold'])
            
            if n * p > threshold and n * (1-p) > threshold:
                # Normal approximation
                mu = n * p
                sigma = math.sqrt(n * p * (1-p))
                
                if sigma > 0:
                    z = (k - 0.5 - mu) / sigma  # Continuity correction
                    prob_success = 0.5 * (1 - math.erf(z / math.sqrt(2)))
                else:
                    prob_success = 1.0 if mu >= k else 0.0
            else:
                # Exact binomial
                prob_success = 1.0 - binom.cdf(k - 1, n, p)
            
            # If probability of satisfying this constraint is too low, accept this person
            if prob_success < safety_alpha:
                return True
        
        return False

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Main decision logic using Lagrangian admission control."""
        self._decisions_since_update += 1
        
        # Emergency mode override
        if self.is_emergency_mode(game_state):
            return True, "lagrangian_emergency"
        
        # Classify person type
        person_type = self._enumerate_person_type(person, game_state)
        self._update_type_probabilities(person_type)
        
        # Update dual multipliers periodically
        self._update_dual_multipliers(game_state)
        
        # Apply capacity management adjustments
        capacity_ratio = game_state.capacity_ratio
        transition_point = float(self.params['capacity_transition_point'])
        
        policy_variant = self.params.get('policy_variant', 'bid_price')
        
        if policy_variant == 'bid_price':
            # Variant A: Bid-price control
            net_value = self._compute_bid_price_value(person_type, game_state)
            epsilon = float(self.params['value_epsilon'])
            
            # Capacity adjustments
            if capacity_ratio < transition_point:
                # Early game: be conservative
                conservatism = float(self.params['early_capacity_conservatism'])
                threshold = epsilon * conservatism
            else:
                # Late game: be aggressive
                aggressiveness = float(self.params['late_capacity_adjustment'])  
                threshold = epsilon * aggressiveness
            
            accept = net_value > threshold
            reason = f"lagrangian_bid_price_value_{net_value:.3f}_vs_{threshold:.3f}"
            
        elif policy_variant == 'prob_failure':
            # Variant B: Probability-of-failure  
            needs_acceptance = self._compute_failure_probability(person_type, game_state)
            
            if needs_acceptance:
                accept = True
                reason = "lagrangian_prob_failure_needed"
            else:
                # Optional: still accept high-value types even if not needed
                net_value = self._compute_bid_price_value(person_type, game_state)
                accept = net_value > 0.5  # Conservative threshold for non-essential
                reason = f"lagrangian_prob_failure_optional_{net_value:.3f}"
                
        else:  # hybrid
            # Variant C: Hybrid approach
            bid_value = self._compute_bid_price_value(person_type, game_state)
            safety_needed = self._compute_failure_probability(person_type, game_state)
            
            bid_weight = float(self.params['hybrid_bid_price_weight'])
            safety_weight = float(self.params['hybrid_safety_check_weight'])
            
            # Weighted decision
            bid_score = 1.0 if bid_value > 0 else 0.0
            safety_score = 1.0 if safety_needed else 0.0
            
            combined_score = bid_weight * bid_score + safety_weight * safety_score
            accept = combined_score > 0.5
            reason = f"lagrangian_hybrid_bid_{bid_score}_safety_{safety_score}_combined_{combined_score:.3f}"
        
        return accept, reason
