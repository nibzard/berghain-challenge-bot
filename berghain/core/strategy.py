# ABOUTME: Strategy interface and base classes for decision making
# ABOUTME: Clean abstraction that separates strategy from execution logic

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from .domain import GameState, Person, Decision


class DecisionStrategy(ABC):
    """Abstract base class for decision strategies."""
    
    def __init__(self, strategy_params: Dict[str, Any] = None):
        self.params = strategy_params or {}
    
    @abstractmethod
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """
        Make a decision about whether to accept a person.
        
        Returns:
            Tuple of (accept: bool, reasoning: str)
        """
        pass
    
    @property
    @abstractmethod 
    def name(self) -> str:
        """Strategy name for identification."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.params.copy()
    
    def update_params(self, new_params: Dict[str, Any]):
        """Update strategy parameters."""
        self.params.update(new_params)


class BaseDecisionStrategy(DecisionStrategy):
    """Base implementation with common utility methods."""
    
    def get_constraint_attributes(self, game_state: GameState) -> set[str]:
        """Get all constraint attributes."""
        return {c.attribute for c in game_state.constraints}
    
    def get_person_constraint_attributes(self, person: Person, game_state: GameState) -> set[str]:
        """Get constraint attributes that this person has."""
        constraint_attrs = self.get_constraint_attributes(game_state)
        return {attr for attr in constraint_attrs if person.has_attribute(attr)}
    
    def calculate_rarity_scores(self, game_state: GameState) -> Dict[str, float]:
        """Calculate rarity scores for all constraint attributes."""
        constraint_attrs = self.get_constraint_attributes(game_state)
        return {
            attr: game_state.statistics.get_rarity_score(attr)
            for attr in constraint_attrs
        }
    
    def get_game_phase(self, game_state: GameState) -> str:
        """Determine current game phase."""
        capacity_ratio = game_state.capacity_ratio
        rejection_ratio = game_state.rejection_ratio
        
        if capacity_ratio < 0.3:
            return "early"
        elif capacity_ratio < 0.7:
            return "mid" 
        elif rejection_ratio > 0.9:
            return "panic"
        else:
            return "late"
    
    def is_emergency_mode(self, game_state: GameState) -> bool:
        """Check if we should accept almost everyone."""
        return game_state.rejection_ratio > 0.95
    
    def get_critical_constraints(self, game_state: GameState, threshold: float = 0.8) -> set[str]:
        """Get constraints that are significantly behind target."""
        critical = set()
        progress = game_state.constraint_progress()
        
        for attr, prog in progress.items():
            if prog < threshold:
                shortage = game_state.constraint_shortage()[attr]
                if shortage > 50:  # Significant shortage
                    critical.add(attr)
        
        return critical