# ABOUTME: Predictive Equilibrium Control (PEC) strategy implementation
# ABOUTME: Uses lookahead prediction, dynamic equilibrium maintenance, and waste minimization

import random
import math
from typing import Tuple, Dict, Set, List
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


def get_person_constraint_attrs(person: Person, game_state: GameState) -> Set[str]:
    """Get the constraint attributes that a person has."""
    constraint_attrs = {c.attribute for c in game_state.constraints}
    return {attr for attr in constraint_attrs if person.has_attribute(attr)}


class PECStrategy(BaseDecisionStrategy):
    """Predictive Equilibrium Control strategy with lookahead prediction and dynamic equilibrium."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Core PEC parameters
            'lookahead_window': 50,
            'equilibrium_tolerance': 0.02,
            'waste_penalty_factor': 0.05,
            
            # Initial learned probabilities
            'p_young_initial': 0.323,
            'p_well_initial': 0.323,
            'p_both_initial': 0.144,
            
            # Learning parameters
            'learning_rate': 0.05,
            'learning_start': 50,
            
            # Decision thresholds
            'critical_boost': 0.15,
            'urgency_high': 0.7,
            'urgency_medium': 0.5,
            'urgency_high_boost': 0.4,
            'urgency_medium_boost': 0.2,
            
            # Balance adjustment
            'imbalance_threshold': 0.03,
            'imbalance_boost': 0.25,
            
            # Capacity phase adjustments
            'emergency_threshold': 0.92,
            'late_threshold': 0.85,
            'mid_threshold': 0.75,
            'late_penalty': 0.7,
            'mid_penalty': 0.85,
            
            # Filler parameters
            'filler_min_progress': 0.92,
            'filler_buffer_threshold': 15,
            'filler_max_rate': 0.08,
            'filler_target_rate': 0.515,
            
            # Constraint met penalty
            'surplus_penalty': 0.6,
            
            # Prediction confidence adjustment
            'high_uncertainty_threshold': 10,
            'uncertainty_boost': 0.1,
        }
        
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # Learned probabilities (updated online)
        self.p_young = self.params['p_young_initial']
        self.p_well = self.params['p_well_initial']
        self.p_both = self.params['p_both_initial']
        
        # Decision state
        self.waste_tracker = {'young': 0, 'well_dressed': 0}
        self.decision_history = []
        
    @property
    def name(self) -> str:
        return "PEC"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        attrs = get_person_constraint_attrs(person, game_state)
        
        # 1. DUAL ATTRIBUTES: Always accept (optimal)
        if len(attrs) >= 2:
            decision = (True, "PEC_dual")
            self._update_learning(decision, person, game_state)
            return decision
            
        # 2. Calculate predictive equilibrium
        prediction = self._predict_future_state(game_state)
        equilibrium = self._calculate_equilibrium(game_state, prediction)
        
        # 3. SINGLE ATTRIBUTES: Equilibrium-based decision
        if len(attrs) == 1:
            attr = list(attrs)[0]
            prob = self._calculate_single_prob(attr, game_state, equilibrium, prediction)
            accept = random.random() < prob
            decision = (accept, f"PEC_single_{attr}_{prob:.2f}")
            self._update_learning(decision, person, game_state)
            return decision
            
        # 4. FILLER: Strategic minimal acceptance
        else:
            filler_prob = self._calculate_filler_prob(game_state, equilibrium)
            accept = random.random() < filler_prob
            decision = (accept, f"PEC_filler_{filler_prob:.2f}")
            self._update_learning(decision, person, game_state)
            return decision
    
    def _predict_future_state(self, game_state: GameState) -> Dict:
        """Predict expected arrivals in lookahead window"""
        remaining_capacity = game_state.remaining_capacity
        window = min(self.params['lookahead_window'], remaining_capacity * 2)
        
        # Expected arrivals by type
        expected = {
            'both': window * self.p_both,
            'young_only': window * (self.p_young - self.p_both),
            'well_only': window * (self.p_well - self.p_both),
            'neither': window * (1 - self.p_young - self.p_well + self.p_both)
        }
        
        # Variance for confidence intervals
        var_young = window * self.p_young * (1 - self.p_young)
        var_well = window * self.p_well * (1 - self.p_well)
        
        expected['std_young'] = math.sqrt(var_young)
        expected['std_well'] = math.sqrt(var_well)
        
        return expected
    
    def _calculate_equilibrium(self, game_state: GameState, prediction: Dict) -> Dict:
        """Calculate equilibrium acceptance rates"""
        shortage = game_state.constraint_shortage()
        remaining_accepts = game_state.remaining_capacity
        
        # Account for expected dual contributions
        dual_contribution = min(prediction['both'], remaining_accepts)
        
        # Net requirements after duals
        young_net = max(0, shortage['young'] - dual_contribution)
        well_net = max(0, shortage['well_dressed'] - dual_contribution)
        
        # Required single acceptance rates for equilibrium
        young_rate = min(1.0, young_net / max(1, prediction['young_only']))
        well_rate = min(1.0, well_net / max(1, prediction['well_only']))
        
        # Identify critical constraint (higher required rate)
        critical = 'young' if young_rate > well_rate else 'well_dressed'
        
        return {
            'young_rate': young_rate,
            'well_rate': well_rate,
            'critical': critical,
            'balanced': abs(young_rate - well_rate) < self.params['equilibrium_tolerance']
        }
    
    def _calculate_single_prob(self, attr: str, game_state: GameState, equilibrium: Dict, prediction: Dict) -> float:
        """Calculate acceptance probability for single-attribute person"""
        shortage = game_state.constraint_shortage()
        progress = game_state.constraint_progress()
        capacity_ratio = game_state.capacity_ratio
        
        # Base rate from equilibrium calculation
        base_rate = equilibrium.get(f'{attr}_rate', 0.5)
        prob = base_rate
        
        # 1. Critical path boost
        if attr == equilibrium['critical']:
            prob += self.params['critical_boost']
            
        # 2. Shortage urgency adjustment
        deficit = shortage[attr]
        remaining = game_state.remaining_capacity
        urgency = deficit / max(1, remaining)
        
        if urgency > self.params['urgency_high']:  # Very urgent
            prob = min(1.0, prob + self.params['urgency_high_boost'])
        elif urgency > self.params['urgency_medium']:  # Urgent
            prob = min(1.0, prob + self.params['urgency_medium_boost'])
        elif deficit <= 0:  # Constraint met
            prob = max(0.0, prob - self.params['surplus_penalty'])  # Heavy penalty
            
        # 3. Balance adjustment
        other_attr = 'well_dressed' if attr == 'young' else 'young'
        imbalance = progress[other_attr] - progress[attr]
        
        if imbalance > self.params['imbalance_threshold']:  # We're behind
            prob = min(1.0, prob + self.params['imbalance_boost'])
        elif imbalance < -self.params['imbalance_threshold']:  # We're ahead
            prob = max(0.0, prob - self.params['imbalance_boost'])
            
        # 4. Capacity phase adjustment
        if capacity_ratio > self.params['emergency_threshold']:  # Emergency phase
            prob = 1.0 if deficit > 0 else 0.0
        elif capacity_ratio > self.params['late_threshold']:
            prob *= self.params['late_penalty']
        elif capacity_ratio > self.params['mid_threshold']:
            prob *= self.params['mid_penalty']
            
        # 5. Waste penalty
        waste = self.waste_tracker.get(attr, 0)
        if waste > 5:
            prob = max(0.0, prob - waste * self.params['waste_penalty_factor'])
            
        # 6. Confidence adjustment based on prediction variance
        std = prediction.get(f'std_{attr}', 1)
        if std > self.params['high_uncertainty_threshold']:  # High uncertainty
            prob = min(1.0, prob + self.params['uncertainty_boost'])  # Be slightly more accepting
            
        return max(0.0, min(1.0, prob))
    
    def _calculate_filler_prob(self, game_state: GameState, equilibrium: Dict) -> float:
        """Calculate filler acceptance probability"""
        progress = game_state.constraint_progress()
        min_progress = min(progress.values())
        
        # No filler until very late game
        if min_progress < self.params['filler_min_progress']:
            return 0.0
            
        # Check if we have buffer
        shortage = game_state.constraint_shortage()
        total_shortage = sum(shortage.values())
        buffer = game_state.remaining_capacity - total_shortage
        
        if buffer > self.params['filler_buffer_threshold']:
            # Can afford minimal filler to maintain acceptance rate
            total_people = game_state.admitted_count + game_state.rejected_count
            current_rate = game_state.admitted_count / max(1, total_people)
            if current_rate < self.params['filler_target_rate']:
                return min(self.params['filler_max_rate'], buffer / game_state.remaining_capacity)
                
        return 0.0
    
    def _update_learning(self, decision: Tuple[bool, str], person: Person, game_state: GameState):
        """Update learned parameters and waste tracking"""
        accept, reasoning = decision
        
        # Create decision-like record for history
        decision_record = {
            'accepted': accept,
            'reasoning': reasoning,
            'person': person
        }
        self.decision_history.append(decision_record)
        
        # Update waste tracker if accepting surplus
        if accept:
            shortage = game_state.constraint_shortage()
            for attr in ['young', 'well_dressed']:
                if person.has_attribute(attr) and shortage.get(attr, 0) <= 0:
                    self.waste_tracker[attr] += 1
                    
        # Update probability estimates (EMA) after enough decisions
        if len(self.decision_history) > self.params['learning_start']:
            recent = self.decision_history[-50:]
            alpha = self.params['learning_rate']
            
            young_freq = sum(1 for d in recent if d['person'].has_attribute('young')) / len(recent)
            well_freq = sum(1 for d in recent if d['person'].has_attribute('well_dressed')) / len(recent)
            both_freq = sum(1 for d in recent if (d['person'].has_attribute('young') and 
                                                d['person'].has_attribute('well_dressed'))) / len(recent)
            
            self.p_young = (1 - alpha) * self.p_young + alpha * young_freq
            self.p_well = (1 - alpha) * self.p_well + alpha * well_freq
            self.p_both = (1 - alpha) * self.p_both + alpha * both_freq