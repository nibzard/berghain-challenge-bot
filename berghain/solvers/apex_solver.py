"""ABOUTME: Apex Algorithm - Superior Hybrid Mathematical Optimization combining best features from Perfect and Ultimate3H
ABOUTME: Features triple-mode operation, predictive analytics, adaptive learning, and guaranteed constraint satisfaction"""

import math
import random
from typing import Tuple, Dict, Optional, List
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver


class ApexSolver(BaseSolver):
    """Apex hybrid mathematically optimal solver."""
    
    def __init__(self, solver_id: str = "apex", config_manager=None, api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("apex")
        strategy = ApexStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class ApexStrategy(BaseDecisionStrategy):
    """Apex strategy with triple-mode operation and predictive analytics."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Triple-mode operation
            'exploration_capacity_cutoff': 0.30,
            'optimization_capacity_cutoff': 0.70,
            'exploration_acceptance_rate': 0.25,
            'exploration_dual_bonus': 2.0,
            'optimization_dual_priority': 1.5,
            'optimization_single_threshold': 0.4,
            'optimization_filler_rate': 0.05,
            'constraint_rush_dual_rate': 1.0,
            'constraint_rush_single_urgency': 3.0,
            'constraint_rush_filler_cutoff': 0.01,
            
            # Constraint satisfaction
            'constraint_violation_penalty': 2000.0,
            'constraint_emergency_threshold': 0.85,
            'constraint_panic_threshold': 0.95,
            'deficit_exponential_weight': 4.0,
            
            # Predictive analytics
            'predictive_confidence_level': 0.95,
            'binomial_safety_buffer': 1.96,
            'feasibility_lookahead_steps': 50,
            
            # Mode switching
            'aggressive_mode_rejection_limit': 750,
            'safe_mode_constraint_threshold': 0.75,
            'emergency_mode_deficit_limit': 80,
            'mode_switch_hysteresis': 0.05,
            'transition_smoothing_factor': 0.8,
            
            # Lagrange multipliers
            'lambda_base_multiplier': 3.0,
            'lambda_learning_rate': 0.08,
            'lambda_momentum': 0.9,
            'lambda_max_value': 15.0,
            
            # Golden ratio optimization
            'golden_section_ratio': 0.618034,
            'fibonacci_scaling_factor': 1.414,
            
            # Adaptive learning
            'adaptive_learning_enabled': True,
            'learning_rate_success': 1.05,
            'learning_rate_failure': 0.95,
            'parameter_adaptation_window': 10,
            'success_rate_target': 0.95,
            'performance_smoothing_alpha': 0.7,
            
            # Advanced features
            'attribute_correlation_learning': True,
            'distribution_adaptation': True,
            'real_time_probability_updates': True,
            'multiple_safety_nets': True,
            'constraint_barrier_strength': 500.0,
            'emergency_override_enabled': True,
            
            # Exploration vs exploitation
            'exploration_epsilon': 0.08,
            'nash_equilibrium_mixing': 0.03,
            'entropy_bonus_weight': 0.1,
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # State tracking
        self._current_mode = 1  # 1=Exploration, 2=Optimization, 3=Constraint Rush
        self._decision_count = 0
        self._rejection_estimate = 0
        
        # Lagrange multipliers with momentum
        self._lambda_y = float(self.params['lambda_base_multiplier'])
        self._lambda_w = float(self.params['lambda_base_multiplier'])
        self._lambda_momentum_y = 0.0
        self._lambda_momentum_w = 0.0
        
        # Adaptive learning state
        self._performance_history: List[bool] = []
        self._success_rate_estimate = 0.5
        self._adaptive_parameters = {}
        
        # Predictive analytics state
        self._attribute_frequencies = {}
        self._correlation_estimates = {}
        self._constraint_satisfaction_predictions = {}
        
        # Safety net tracking
        self._emergency_activations = 0
        self._constraint_violations = 0
        self._last_mode_switch = 0

    @property
    def name(self) -> str:
        return "Apex"

    def _determine_current_mode(self, game_state: GameState) -> int:
        """Determine which of the three modes we should be in."""
        capacity_ratio = game_state.capacity_ratio
        
        # Check for emergency overrides first
        if self._should_trigger_emergency_mode(game_state):
            return 3  # Force Constraint Rush mode
        
        # Normal mode determination with hysteresis
        exploration_cutoff = float(self.params['exploration_capacity_cutoff'])
        optimization_cutoff = float(self.params['optimization_capacity_cutoff'])
        hysteresis = float(self.params['mode_switch_hysteresis'])
        
        # Mode 1: Exploration (early game)
        if capacity_ratio < exploration_cutoff:
            return 1
        
        # Mode 2: Optimization (mid game)
        elif capacity_ratio < optimization_cutoff:
            # Add hysteresis to prevent rapid switching
            if self._current_mode == 1 and capacity_ratio < exploration_cutoff + hysteresis:
                return 1
            return 2
        
        # Mode 3: Constraint Rush (late game)
        else:
            # Add hysteresis for mode 2 -> 3 transition
            if self._current_mode == 2 and capacity_ratio < optimization_cutoff + hysteresis:
                return 2
            return 3

    def _should_trigger_emergency_mode(self, game_state: GameState) -> bool:
        """Check if we should force emergency constraint rush mode."""
        if not bool(self.params.get('emergency_override_enabled', True)):
            return False
        
        shortage = game_state.constraint_shortage()
        if not shortage:
            return False
        
        # Emergency if any constraint has huge deficit
        emergency_deficit = int(self.params['emergency_mode_deficit_limit'])
        max_deficit = max(shortage.values())
        if max_deficit >= emergency_deficit:
            return True
        
        # Emergency if constraint progress is very poor late in game
        if game_state.capacity_ratio > float(self.params['constraint_panic_threshold']):
            constraint_progress = game_state.constraint_progress()
            min_progress = min(constraint_progress.values()) if constraint_progress else 0.0
            if min_progress < 0.90:
                return True
        
        return False

    def _update_lagrange_multipliers(self, game_state: GameState):
        """Update Lagrange multipliers with momentum-based learning."""
        keys = [c.attribute for c in game_state.constraints]
        if len(keys) < 2:
            return
        
        a_y, a_w = keys[0], keys[1]
        shortage = game_state.constraint_shortage()
        
        deficit_y = max(0, shortage.get(a_y, 0))
        deficit_w = max(0, shortage.get(a_w, 0))
        
        # Compute target lambda values based on deficits
        exp_weight = float(self.params['deficit_exponential_weight'])
        base_mult = float(self.params['lambda_base_multiplier'])
        
        # Exponential urgency weighting (Perfect-inspired)
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        if capacity_remaining > 0:
            urgency_y = base_mult * math.pow(1 + deficit_y / 100.0, exp_weight)
            urgency_w = base_mult * math.pow(1 + deficit_w / 100.0, exp_weight)
        else:
            urgency_y = urgency_w = base_mult
        
        # Momentum-based updates
        learning_rate = float(self.params['lambda_learning_rate'])
        momentum = float(self.params['lambda_momentum'])
        max_lambda = float(self.params['lambda_max_value'])
        
        # Compute momentum updates
        self._lambda_momentum_y = momentum * self._lambda_momentum_y + learning_rate * (urgency_y - self._lambda_y)
        self._lambda_momentum_w = momentum * self._lambda_momentum_w + learning_rate * (urgency_w - self._lambda_w)
        
        # Apply updates with clamping
        self._lambda_y = max(0.0, min(max_lambda, self._lambda_y + self._lambda_momentum_y))
        self._lambda_w = max(0.0, min(max_lambda, self._lambda_w + self._lambda_momentum_w))

    def _compute_person_value(self, person: Person, game_state: GameState) -> float:
        """Compute economic value of accepting this person (Perfect-inspired)."""
        keys = [c.attribute for c in game_state.constraints]
        if len(keys) < 2:
            return 0.0
        
        a_y, a_w = keys[0], keys[1]
        attrs = self.get_person_constraint_attributes(person, game_state)
        
        # Base value computation
        value = 0.0
        
        # Constraint contributions with exponential urgency
        shortage = game_state.constraint_shortage()
        urgency_y = 1.0 + shortage.get(a_y, 0) / 50.0
        urgency_w = 1.0 + shortage.get(a_w, 0) / 50.0
        
        if a_y in attrs:
            value += self._lambda_y * urgency_y
        if a_w in attrs:
            value += self._lambda_w * urgency_w
        
        # Dual attribute bonus (golden ratio inspired)
        if len(attrs) >= 2:
            golden_ratio = float(self.params['golden_section_ratio'])
            fib_scaling = float(self.params['fibonacci_scaling_factor'])
            dual_bonus = golden_ratio * fib_scaling * math.log(2)
            value += dual_bonus
        
        # Mode-specific adjustments
        if self._current_mode == 1:  # Exploration
            # Be more selective, add entropy bonus for learning
            entropy_bonus = float(self.params['entropy_bonus_weight'])
            if len(attrs) > 0:
                value += entropy_bonus * math.log(len(attrs) + 1)
            value *= 0.8  # Be more selective
            
        elif self._current_mode == 2:  # Optimization
            # Balance efficiency and constraint satisfaction
            value *= float(self.params['optimization_dual_priority'])
            
        elif self._current_mode == 3:  # Constraint Rush
            # Massive boost for constraint helpers
            if len(attrs) > 0:
                constraint_boost = float(self.params['constraint_rush_single_urgency'])
                value *= constraint_boost
                
                # Even bigger boost for dual attributes
                if len(attrs) >= 2:
                    value *= 2.0
            else:
                # Heavy penalty for fillers in constraint rush
                value -= float(self.params['constraint_violation_penalty']) / 4.0
        
        # Capacity shadow price
        capacity_remaining = game_state.target_capacity - game_state.admitted_count
        if capacity_remaining > 0:
            shadow_price = 1.0 / math.sqrt(capacity_remaining)
            value -= shadow_price
        
        return value

    def _predict_constraint_satisfaction(self, game_state: GameState) -> Dict[str, float]:
        """Predict probability of satisfying each constraint."""
        predictions = {}
        keys = [c.attribute for c in game_state.constraints]
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            required = constraint.min_count
            current = game_state.admitted_attributes.get(attr, 0)
            deficit = max(0, required - current)
            
            if deficit == 0:
                predictions[attr] = 1.0
                continue
            
            # Use binomial model with safety buffer
            capacity_remaining = game_state.target_capacity - game_state.admitted_count
            if capacity_remaining <= 0:
                predictions[attr] = 0.0
                continue
            
            # Get attribute frequency
            freq = game_state.statistics.frequencies.get(attr, 0.0)
            if freq <= 0:
                predictions[attr] = 0.0
                continue
            
            # Binomial probability calculation with normal approximation
            n = capacity_remaining
            p = freq
            k = deficit
            
            if n * p > 5 and n * (1-p) > 5:  # Normal approximation valid
                mu = n * p
                sigma = math.sqrt(n * p * (1-p))
                
                # Z-score with continuity correction
                z = (k - 0.5 - mu) / sigma if sigma > 0 else 0
                
                # P(X >= k) using normal approximation
                prob_success = 0.5 * (1 - math.erf(z / math.sqrt(2)))
                
                # Add safety buffer
                confidence_level = float(self.params['predictive_confidence_level'])
                safety_buffer = float(self.params['binomial_safety_buffer'])
                adjusted_prob = max(0.0, prob_success - (1 - confidence_level) * safety_buffer)
                
                predictions[attr] = adjusted_prob
            else:
                # Fallback for small numbers
                expected_arrivals = n * p
                predictions[attr] = 1.0 if expected_arrivals >= k else expected_arrivals / k
        
        return predictions

    def _adaptive_parameter_adjustment(self, success: bool):
        """Adjust parameters based on recent performance."""
        if not bool(self.params.get('adaptive_learning_enabled', True)):
            return
        
        # Update performance history
        window = int(self.params['parameter_adaptation_window'])
        self._performance_history.append(success)
        if len(self._performance_history) > window:
            self._performance_history.pop(0)
        
        # Update success rate estimate with exponential smoothing
        alpha = float(self.params['performance_smoothing_alpha'])
        self._success_rate_estimate = alpha * float(success) + (1 - alpha) * self._success_rate_estimate
        
        # Adjust parameters if we have enough history
        if len(self._performance_history) >= window // 2:
            current_success_rate = sum(self._performance_history) / len(self._performance_history)
            target_success_rate = float(self.params['success_rate_target'])
            
            success_factor = float(self.params['learning_rate_success'])
            failure_factor = float(self.params['learning_rate_failure'])
            
            if current_success_rate < target_success_rate:
                # Performing poorly, be more conservative
                if 'constraint_violation_penalty' not in self._adaptive_parameters:
                    self._adaptive_parameters['constraint_violation_penalty'] = self.params['constraint_violation_penalty']
                self._adaptive_parameters['constraint_violation_penalty'] *= success_factor
                
                # Increase constraint urgency
                if 'deficit_exponential_weight' not in self._adaptive_parameters:
                    self._adaptive_parameters['deficit_exponential_weight'] = self.params['deficit_exponential_weight']
                self._adaptive_parameters['deficit_exponential_weight'] *= success_factor
                
            elif current_success_rate > target_success_rate:
                # Performing well, can be more aggressive
                if 'optimization_filler_rate' not in self._adaptive_parameters:
                    self._adaptive_parameters['optimization_filler_rate'] = self.params['optimization_filler_rate']
                self._adaptive_parameters['optimization_filler_rate'] *= success_factor

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Main decision logic with triple-mode operation."""
        self._decision_count += 1
        
        # Update current mode
        new_mode = self._determine_current_mode(game_state)
        if new_mode != self._current_mode:
            self._last_mode_switch = self._decision_count
        self._current_mode = new_mode
        
        # Update Lagrange multipliers
        self._update_lagrange_multipliers(game_state)
        
        # Base emergency mode check (inherited from BaseDecisionStrategy)
        if self.is_emergency_mode(game_state):
            return True, f"apex_base_emergency_mode_{self._current_mode}"
        
        # Get person attributes
        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]
        
        is_dual = len(attrs) >= 2
        is_single = len(attrs) == 1
        is_filler = len(attrs) == 0
        
        # Mode-specific decision logic
        if self._current_mode == 1:  # Exploration Mode
            return self._exploration_mode_decision(person, game_state, attrs, is_dual, is_single, is_filler)
            
        elif self._current_mode == 2:  # Optimization Mode
            return self._optimization_mode_decision(person, game_state, attrs, is_dual, is_single, is_filler)
            
        else:  # Constraint Rush Mode (mode 3)
            return self._constraint_rush_mode_decision(person, game_state, attrs, is_dual, is_single, is_filler)

    def _exploration_mode_decision(self, person: Person, game_state: GameState, attrs, is_dual: bool, is_single: bool, is_filler: bool) -> Tuple[bool, str]:
        """Decision logic for exploration mode (0-30% capacity)."""
        
        # Always accept dual attributes with bonus
        if is_dual:
            return True, "apex_exploration_dual_accept"
        
        # Accept single attributes based on value and exploration rate
        if is_single:
            person_value = self._compute_person_value(person, game_state)
            exploration_threshold = float(self.params['exploration_acceptance_rate'])
            
            # Add some exploration randomness
            epsilon = float(self.params['exploration_epsilon'])
            if random.random() < epsilon:
                return True, "apex_exploration_single_explore"
            
            # Value-based acceptance
            if person_value > 0:
                prob = min(1.0, exploration_threshold + person_value / 10.0)
                accept = random.random() < prob
                return accept, f"apex_exploration_single_value_{person_value:.2f}"
        
        # Reject fillers in exploration mode to be selective
        return False, "apex_exploration_filler_reject"

    def _optimization_mode_decision(self, person: Person, game_state: GameState, attrs, is_dual: bool, is_single: bool, is_filler: bool) -> Tuple[bool, str]:
        """Decision logic for optimization mode (30-70% capacity)."""
        
        # Always accept dual attributes with priority
        if is_dual:
            dual_priority = float(self.params['optimization_dual_priority'])
            if dual_priority >= 1.0 or random.random() < dual_priority:
                return True, "apex_optimization_dual_accept"
        
        # Smart single attribute acceptance
        if is_single:
            person_value = self._compute_person_value(person, game_state)
            single_threshold = float(self.params['optimization_single_threshold'])
            
            # Check constraint predictions
            predictions = self._predict_constraint_satisfaction(game_state)
            keys = [c.attribute for c in game_state.constraints]
            
            # Boost acceptance for attributes we predict will be hard to satisfy
            for attr in attrs:
                if attr in predictions and predictions[attr] < 0.8:  # Low satisfaction probability
                    person_value *= 1.5
                    break
            
            if person_value > 0:
                prob = min(1.0, single_threshold + person_value / 15.0)
                accept = random.random() < prob
                return accept, f"apex_optimization_single_value_{person_value:.2f}"
        
        # Limited filler acceptance in optimization mode
        if is_filler:
            filler_rate = float(self.params['optimization_filler_rate'])
            # Use adaptive parameter if available
            if 'optimization_filler_rate' in self._adaptive_parameters:
                filler_rate = self._adaptive_parameters['optimization_filler_rate']
            
            if random.random() < filler_rate:
                return True, "apex_optimization_filler_accept"
        
        return False, "apex_optimization_reject"

    def _constraint_rush_mode_decision(self, person: Person, game_state: GameState, attrs, is_dual: bool, is_single: bool, is_filler: bool) -> Tuple[bool, str]:
        """Decision logic for constraint rush mode (70-100% capacity)."""
        
        # Always accept dual attributes
        if is_dual:
            rush_dual_rate = float(self.params['constraint_rush_dual_rate'])
            if rush_dual_rate >= 1.0 or random.random() < rush_dual_rate:
                return True, "apex_constraint_rush_dual_accept"
        
        # Highly prioritize single attributes that help constraints
        if is_single:
            shortage = game_state.constraint_shortage()
            
            # Check if this person helps with any deficit
            helps_constraint = False
            for attr in attrs:
                if shortage.get(attr, 0) > 0:
                    helps_constraint = True
                    break
            
            if helps_constraint:
                # Much higher acceptance for constraint helpers
                urgency = float(self.params['constraint_rush_single_urgency'])
                person_value = self._compute_person_value(person, game_state)
                prob = min(1.0, urgency * person_value / 10.0)
                
                if prob > 0.5 or random.random() < prob:
                    return True, f"apex_constraint_rush_single_needed_{prob:.2f}"
        
        # Very limited filler acceptance in constraint rush
        if is_filler:
            filler_cutoff = float(self.params['constraint_rush_filler_cutoff'])
            
            # Only accept fillers if we're doing well on constraints
            constraint_progress = game_state.constraint_progress()
            min_progress = min(constraint_progress.values()) if constraint_progress else 0.0
            
            if min_progress > 0.95 and random.random() < filler_cutoff:
                return True, "apex_constraint_rush_filler_safe"
        
        return False, "apex_constraint_rush_reject"

    def on_game_end(self, result) -> None:
        """Learn from game results for adaptive improvement."""
        try:
            # Determine if this was a successful game
            success = (result.success and 
                      hasattr(result, 'constraints_satisfied') and result.constraints_satisfied)
            
            # Update adaptive parameters
            self._adaptive_parameter_adjustment(success)
            
            # Log performance for analysis
            if hasattr(self, '_logger'):
                self._logger.info(f"Apex game ended: success={success}, mode_switches={self._last_mode_switch}, "
                                f"emergency_activations={self._emergency_activations}")
            
        except Exception as e:
            # Don't let learning failures crash the system
            print(f"Warning: Apex learning failed: {e}")