# ABOUTME: PEC v2 - Improved Predictive Equilibrium Control with aggressive early exploration
# ABOUTME: Uses phase-based operation, adaptive balancing, and stronger constraint satisfaction

import random
import math
from typing import Tuple, Dict, Set, List
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


def get_person_constraint_attrs(person: Person, game_state: GameState) -> Set[str]:
    """Get the constraint attributes that a person has."""
    constraint_attrs = {c.attribute for c in game_state.constraints}
    return {attr for attr in constraint_attrs if person.has_attribute(attr)}


class PECv2Strategy(BaseDecisionStrategy):
    """Improved Predictive Equilibrium Control with phase-based operation and adaptive balancing."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Phase-based operation
            'exploration_phase_cutoff': 0.30,
            'optimization_phase_cutoff': 0.75,
            
            # Exploration phase parameters
            'exploration_dual_rate': 1.0,
            'exploration_single_rate': 0.65,
            'exploration_filler_rate': 0.12,
            
            # Optimization phase parameters
            'optimization_dual_rate': 1.0,
            'optimization_equilibrium_tolerance': 0.025,
            'optimization_urgency_boost': 0.4,
            'optimization_balance_penalty': 0.3,
            'optimization_filler_rate': 0.06,
            
            # Constraint rush phase parameters
            'rush_dual_rate': 1.0,
            'rush_emergency_threshold': 0.88,
            'rush_deficit_panic_multiplier': 5.0,
            'rush_filler_cutoff': 0.02,
            
            # Learning parameters
            'learning_enabled': True,
            'learning_rate': 0.08,
            'learning_start_decisions': 30,
            
            # Predictive parameters
            'lookahead_window': 40,
            'confidence_level': 0.90,
            
            # Balance enforcement
            'balance_check_frequency': 25,
            'max_imbalance_tolerance': 0.05,
            'imbalance_correction_strength': 0.4,
            
            # Waste minimization
            'waste_penalty_base': 0.08,
            'waste_penalty_growth': 1.2,
            'max_waste_threshold': 8,
            
            # Emergency overrides
            'emergency_override_enabled': True,
            'constraint_violation_penalty': 1000.0,
            'panic_mode_deficit': 50,
            
            # Initial probabilities
            'p_young_initial': 0.325,
            'p_well_initial': 0.325,
            'p_both_initial': 0.148,
        }
        
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # Learned probabilities
        self.p_young = self.params['p_young_initial']
        self.p_well = self.params['p_well_initial']
        self.p_both = self.params['p_both_initial']
        
        # State tracking
        self.decision_count = 0
        self.waste_tracker = {'young': 0, 'well_dressed': 0}
        self.decision_history = []
        
        # Balance tracking
        self.last_balance_check = 0
        self.imbalance_history = []
        
    @property
    def name(self) -> str:
        return "PEC_v2"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        self.decision_count += 1
        attrs = get_person_constraint_attrs(person, game_state)
        capacity_ratio = game_state.capacity_ratio
        
        # Determine current phase
        if capacity_ratio <= self.params['exploration_phase_cutoff']:
            phase = "exploration"
        elif capacity_ratio <= self.params['optimization_phase_cutoff']:
            phase = "optimization"
        else:
            phase = "constraint_rush"
        
        # 1. DUAL ATTRIBUTES: Always accept in all phases
        if len(attrs) >= 2:
            decision = (True, f"PECv2_dual_{phase}")
            self._update_learning(decision, person, game_state)
            return decision
            
        # 2. SINGLE ATTRIBUTES: Phase-specific handling
        if len(attrs) == 1:
            attr = list(attrs)[0]
            if phase == "exploration":
                prob = self._calculate_exploration_single_prob(attr, game_state)
            elif phase == "optimization":
                prob = self._calculate_optimization_single_prob(attr, game_state)
            else:  # constraint_rush
                prob = self._calculate_rush_single_prob(attr, game_state)
            
            accept = random.random() < prob
            decision = (accept, f"PECv2_single_{attr}_{phase}_{prob:.2f}")
            self._update_learning(decision, person, game_state)
            return decision
            
        # 3. FILLER: Phase-specific minimal acceptance
        else:
            if phase == "exploration":
                filler_prob = self.params['exploration_filler_rate']
            elif phase == "optimization":
                filler_prob = self._calculate_optimization_filler_prob(game_state)
            else:  # constraint_rush
                filler_prob = self._calculate_rush_filler_prob(game_state)
            
            accept = random.random() < filler_prob
            decision = (accept, f"PECv2_filler_{phase}_{filler_prob:.2f}")
            self._update_learning(decision, person, game_state)
            return decision
    
    def _calculate_exploration_single_prob(self, attr: str, game_state: GameState) -> float:
        """Aggressive exploration phase acceptance."""
        base_rate = self.params['exploration_single_rate']
        
        # Check for emergency constraints even in exploration
        shortage = game_state.constraint_shortage()
        if shortage[attr] > self.params['panic_mode_deficit']:
            return min(1.0, base_rate * 1.5)
        
        # Waste penalty
        waste = self.waste_tracker.get(attr, 0)
        if waste > self.params['max_waste_threshold']:
            penalty = self.params['waste_penalty_base'] * (self.params['waste_penalty_growth'] ** (waste - self.params['max_waste_threshold']))
            base_rate = max(0.1, base_rate - penalty)
        
        return base_rate
    
    def _calculate_optimization_single_prob(self, attr: str, game_state: GameState) -> float:
        """Balanced optimization with predictive equilibrium."""
        # Calculate predictive equilibrium
        prediction = self._predict_future_state(game_state)
        equilibrium = self._calculate_equilibrium(game_state, prediction)
        
        # Base rate from equilibrium
        base_rate = equilibrium.get(f'{attr}_rate', 0.5)
        
        # Urgency adjustment
        shortage = game_state.constraint_shortage()
        progress = game_state.constraint_progress()
        
        deficit = shortage[attr]
        remaining = game_state.remaining_capacity
        urgency = deficit / max(1, remaining)
        
        if urgency > 0.6:
            base_rate += self.params['optimization_urgency_boost']
        elif urgency > 0.3:
            base_rate += self.params['optimization_urgency_boost'] * 0.5
        elif deficit <= 0:  # Constraint met
            base_rate *= 0.3  # Strong reduction
        
        # Balance adjustment
        other_attr = 'well_dressed' if attr == 'young' else 'young'
        imbalance = progress[other_attr] - progress[attr]
        
        if abs(imbalance) > self.params['max_imbalance_tolerance']:
            if imbalance > 0:  # We're behind on this attribute
                base_rate += self.params['imbalance_correction_strength']
            else:  # We're ahead on this attribute
                base_rate -= self.params['imbalance_correction_strength']
        
        # Waste penalty
        waste = self.waste_tracker.get(attr, 0)
        if waste > self.params['max_waste_threshold']:
            penalty = self.params['waste_penalty_base'] * (self.params['waste_penalty_growth'] ** (waste - self.params['max_waste_threshold']))
            base_rate = max(0.0, base_rate - penalty)
        
        return max(0.0, min(1.0, base_rate))
    
    def _calculate_rush_single_prob(self, attr: str, game_state: GameState) -> float:
        """Emergency constraint rush phase."""
        shortage = game_state.constraint_shortage()
        capacity_ratio = game_state.capacity_ratio
        deficit = shortage[attr]
        
        # Emergency mode
        if capacity_ratio > self.params['rush_emergency_threshold'] or deficit > self.params['panic_mode_deficit']:
            return 1.0 if deficit > 0 else 0.0
        
        # Normal rush mode - very selective
        if deficit <= 0:
            return 0.0  # Constraint already met
        
        # Calculate how critical this attribute is
        remaining = game_state.remaining_capacity
        urgency = deficit / max(1, remaining)
        
        if urgency > 0.8:
            return 1.0
        elif urgency > 0.5:
            return 0.9
        elif urgency > 0.2:
            return 0.7
        else:
            return 0.4
    
    def _calculate_optimization_filler_prob(self, game_state: GameState) -> float:
        """Calculate filler probability during optimization phase."""
        progress = game_state.constraint_progress()
        min_progress = min(progress.values())
        
        # Only allow filler if we're doing well on constraints
        if min_progress < 0.7:
            return 0.0
        
        # Check capacity buffer
        shortage = game_state.constraint_shortage()
        total_shortage = sum(shortage.values())
        buffer = game_state.remaining_capacity - total_shortage
        
        if buffer > 20:
            return self.params['optimization_filler_rate']
        else:
            return 0.0
    
    def _calculate_rush_filler_prob(self, game_state: GameState) -> float:
        """Minimal filler during constraint rush."""
        shortage = game_state.constraint_shortage()
        total_shortage = sum(shortage.values())
        buffer = game_state.remaining_capacity - total_shortage
        
        # Only if we have significant buffer and all constraints nearly met
        if buffer > 30 and total_shortage < 20:
            return self.params['rush_filler_cutoff']
        else:
            return 0.0
    
    def _predict_future_state(self, game_state: GameState) -> Dict:
        """Predict expected arrivals in lookahead window."""
        remaining_capacity = game_state.remaining_capacity
        window = min(self.params['lookahead_window'], remaining_capacity * 2)
        
        expected = {
            'both': window * self.p_both,
            'young_only': window * (self.p_young - self.p_both),
            'well_only': window * (self.p_well - self.p_both),
            'neither': window * (1 - self.p_young - self.p_well + self.p_both)
        }
        
        # Add confidence intervals
        confidence = self.params['confidence_level']
        z_score = 1.96 if confidence >= 0.95 else 1.645  # 95% or 90% confidence
        
        var_young = window * self.p_young * (1 - self.p_young)
        var_well = window * self.p_well * (1 - self.p_well)
        
        expected['young_std'] = math.sqrt(var_young)
        expected['well_std'] = math.sqrt(var_well)
        expected['young_ci'] = z_score * expected['young_std']
        expected['well_ci'] = z_score * expected['well_std']
        
        return expected
    
    def _calculate_equilibrium(self, game_state: GameState, prediction: Dict) -> Dict:
        """Calculate equilibrium acceptance rates with confidence intervals."""
        shortage = game_state.constraint_shortage()
        remaining_accepts = game_state.remaining_capacity
        
        # Conservative estimate using confidence intervals
        dual_contribution = prediction['both']
        young_singles = prediction['young_only'] - prediction['young_ci']  # Conservative
        well_singles = prediction['well_only'] - prediction['well_ci']
        
        # Net requirements
        young_net = max(0, shortage['young'] - dual_contribution)
        well_net = max(0, shortage['well_dressed'] - dual_contribution)
        
        # Required rates
        young_rate = min(1.0, young_net / max(1, young_singles))
        well_rate = min(1.0, well_net / max(1, well_singles))
        
        return {
            'young_rate': young_rate,
            'well_rate': well_rate,
            'balanced': abs(young_rate - well_rate) < self.params['optimization_equilibrium_tolerance']
        }
    
    def _update_learning(self, decision: Tuple[bool, str], person: Person, game_state: GameState):
        """Update learning parameters and state tracking."""
        accept, reasoning = decision
        
        # Record decision
        self.decision_history.append({
            'accepted': accept,
            'reasoning': reasoning,
            'person': person,
            'decision_count': self.decision_count
        })
        
        # Update waste tracking
        if accept:
            shortage = game_state.constraint_shortage()
            for attr in ['young', 'well_dressed']:
                if person.has_attribute(attr) and shortage.get(attr, 0) <= 0:
                    self.waste_tracker[attr] += 1
        
        # Update probability learning
        if (self.params['learning_enabled'] and 
            self.decision_count > self.params['learning_start_decisions'] and
            len(self.decision_history) >= 20):
            
            recent = self.decision_history[-20:]  # Use recent 20 decisions
            alpha = self.params['learning_rate']
            
            young_freq = sum(1 for d in recent if d['person'].has_attribute('young')) / len(recent)
            well_freq = sum(1 for d in recent if d['person'].has_attribute('well_dressed')) / len(recent)
            both_freq = sum(1 for d in recent if (d['person'].has_attribute('young') and 
                                                d['person'].has_attribute('well_dressed'))) / len(recent)
            
            # Exponential moving average
            self.p_young = (1 - alpha) * self.p_young + alpha * young_freq
            self.p_well = (1 - alpha) * self.p_well + alpha * well_freq
            self.p_both = (1 - alpha) * self.p_both + alpha * both_freq
            
            # Ensure consistency
            self.p_both = min(self.p_both, min(self.p_young, self.p_well))
        
        # Balance tracking
        if self.decision_count - self.last_balance_check >= self.params['balance_check_frequency']:
            progress = game_state.constraint_progress()
            imbalance = abs(progress['young'] - progress['well_dressed'])
            self.imbalance_history.append(imbalance)
            self.last_balance_check = self.decision_count
            
            # Keep only recent history
            if len(self.imbalance_history) > 10:
                self.imbalance_history = self.imbalance_history[-10:]