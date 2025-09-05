# ABOUTME: Adaptive strategy that adjusts based on real-time game state
# ABOUTME: Uses machine learning principles to optimize decisions dynamically

import random
import math
from typing import Tuple, Dict
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy


class AdaptiveStrategy(BaseDecisionStrategy):
    """Strategy that adapts thresholds based on real-time constraint progress."""
    
    def __init__(self, strategy_params: dict = None):
        # Default adaptive parameters
        default_params = {
            'base_ultra_rare_rate': 0.97,
            'base_rare_rate': 0.80,
            'base_common_rate': 0.50,
            'base_no_constraint_rate': 0.05,
            'adaptation_rate': 0.1,
            'progress_weight': 0.7,
            'urgency_weight': 0.3,
            'early_game_threshold': 0.30,
            'mid_game_threshold': 0.70,
            'ultra_rare_threshold': 0.10,
        }
        
        if strategy_params:
            default_params.update(strategy_params)
            
        super().__init__(default_params)
        
        # Adaptive state
        self.decision_history: list = []
        self.adaptive_multipliers: Dict[str, float] = {}
    
    @property
    def name(self) -> str:
        return "Adaptive"
    
    def _calculate_constraint_urgency(self, game_state: GameState) -> Dict[str, float]:
        """Calculate urgency for each constraint based on progress and time."""
        urgencies = {}
        progress = game_state.constraint_progress()
        shortage = game_state.constraint_shortage()
        remaining_ratio = game_state.remaining_capacity / game_state.target_capacity
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            current_progress = progress[attr]
            current_shortage = shortage[attr]
            
            # Base urgency on how far behind we are
            progress_urgency = max(0, 1.0 - current_progress) * 2
            
            # Time pressure - more urgent as capacity fills
            time_urgency = (1.0 - remaining_ratio) * 1.5
            
            # Rarity urgency - rarer attributes need more urgency
            rarity_urgency = game_state.statistics.get_rarity_score(attr) * 0.1
            
            # Shortage urgency - absolute number needed
            shortage_urgency = min(current_shortage / 100, 2.0)
            
            total_urgency = (
                progress_urgency * self.params['progress_weight'] +
                (time_urgency + rarity_urgency + shortage_urgency) * self.params['urgency_weight']
            )
            
            urgencies[attr] = total_urgency
            
        return urgencies
    
    def _adapt_acceptance_rates(self, game_state: GameState) -> Dict[str, float]:
        """Dynamically adjust acceptance rates based on game state."""
        urgencies = self._calculate_constraint_urgency(game_state)
        adapted_rates = {}
        
        # Get base rates
        base_rates = {
            'ultra_rare': self.params['base_ultra_rare_rate'],
            'rare': self.params['base_rare_rate'],
            'common': self.params['base_common_rate'],
            'no_constraint': self.params['base_no_constraint_rate']
        }
        
        # Adapt based on constraint urgency
        max_urgency = max(urgencies.values()) if urgencies else 1.0
        avg_urgency = sum(urgencies.values()) / len(urgencies) if urgencies else 1.0
        
        # If constraints are very urgent, be more accepting
        urgency_multiplier = 1.0 + (avg_urgency - 1.0) * self.params['adaptation_rate']
        urgency_multiplier = max(0.5, min(2.0, urgency_multiplier))  # Clamp to reasonable range
        
        # Rejection pressure adaptation
        rejection_pressure = game_state.rejection_ratio
        pressure_multiplier = 1.0 + (rejection_pressure * 2)  # More accepting under pressure
        
        # Combined multiplier
        total_multiplier = urgency_multiplier * pressure_multiplier
        
        # Apply adaptation
        for rate_type, base_rate in base_rates.items():
            adapted_rate = base_rate * total_multiplier
            adapted_rates[rate_type] = max(0.01, min(0.99, adapted_rate))  # Keep in bounds
            
        return adapted_rates
    
    def _categorize_person(self, person: Person, game_state: GameState) -> str:
        """Categorize person based on their attributes."""
        constraint_attrs = self.get_constraint_attributes(game_state)
        person_constraint_attrs = self.get_person_constraint_attributes(person, game_state)
        
        if not person_constraint_attrs:
            return 'no_constraint'
        
        # Check for ultra-rare attributes
        has_ultra_rare = any(
            game_state.statistics.get_frequency(attr) < self.params['ultra_rare_threshold']
            for attr in person_constraint_attrs
        )
        
        if has_ultra_rare:
            return 'ultra_rare'
        
        # Check rarity score
        total_rarity = sum(
            game_state.statistics.get_rarity_score(attr) 
            for attr in person_constraint_attrs
        )
        
        if total_rarity > 8.0:  # High combined rarity
            return 'rare'
        else:
            return 'common'
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Adaptive decision making based on real-time game state."""
        
        # Emergency mode
        if self.is_emergency_mode(game_state):
            return True, "emergency_adaptive_mode"
        
        # Get adapted acceptance rates
        adapted_rates = self._adapt_acceptance_rates(game_state)
        
        # Categorize person
        category = self._categorize_person(person, game_state)
        
        # Get current phase and urgencies
        phase = self.get_game_phase(game_state)
        urgencies = self._calculate_constraint_urgency(game_state)
        constraint_attrs = self.get_person_constraint_attributes(person, game_state)
        
        # Check for critical constraint help
        critical_constraints = self.get_critical_constraints(game_state, threshold=0.8)
        helps_critical = bool(constraint_attrs & critical_constraints)
        
        if helps_critical:
            # Always accept if helps critical constraints
            critical_attrs = list(constraint_attrs & critical_constraints)
            return True, f"adaptive_critical_help_{critical_attrs}"
        
        # Calculate person value with urgency weighting
        person_value = 0.0
        if constraint_attrs:
            for attr in constraint_attrs:
                person_value += urgencies.get(attr, 1.0)
        
        # Multi-attribute bonus
        if len(constraint_attrs) > 1:
            person_value *= 1.4
        
        # Get base acceptance probability
        base_prob = adapted_rates.get(category, 0.5)
        
        # Phase adjustments
        if phase == "early":
            if category == 'no_constraint':
                base_prob *= 0.3  # Very selective early
            elif len(constraint_attrs) >= 2:
                base_prob *= 1.2  # Bonus for multi-attribute
                
        elif phase == "mid":
            # Standard rates with value adjustment
            if person_value > 3.0:
                base_prob *= 1.3
                
        elif phase == "late":
            # Focus on needed attributes
            shortage = game_state.constraint_shortage()
            needed_attrs = [attr for attr in constraint_attrs if shortage[attr] > 0]
            if needed_attrs:
                base_prob *= 1.5
            else:
                base_prob *= 0.7  # Less interested in surplus
                
        else:  # panic
            if constraint_attrs:
                base_prob = 0.95
            else:
                base_prob = 0.6
        
        # Final probability
        final_prob = max(0.01, min(0.99, base_prob))
        
        # Make decision
        accept = random.random() < final_prob
        
        # Record decision for learning
        self.decision_history.append({
            'person_attrs': constraint_attrs,
            'category': category,
            'phase': phase,
            'urgency': person_value,
            'decision': accept,
            'final_prob': final_prob
        })
        
        # Reasoning
        reasoning = f"adaptive_{category}_{phase}_v{person_value:.1f}_p{final_prob:.2f}"
        
        return accept, reasoning