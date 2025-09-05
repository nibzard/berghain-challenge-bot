# ABOUTME: Core domain models for the Berghain Challenge
# ABOUTME: Clean separation of business logic from infrastructure concerns

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class GameStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Person:
    """Represents a person arriving at the club."""
    index: int
    attributes: Dict[str, bool]
    
    def has_attribute(self, attribute: str) -> bool:
        return self.attributes.get(attribute, False)
    
    def has_any_attributes(self, attributes: List[str]) -> bool:
        return any(self.has_attribute(attr) for attr in attributes)
    
    def has_all_attributes(self, attributes: List[str]) -> bool:
        return all(self.has_attribute(attr) for attr in attributes)
    
    def count_attributes(self, attributes: List[str]) -> int:
        return sum(1 for attr in attributes if self.has_attribute(attr))


@dataclass
class Constraint:
    """Represents a constraint that must be satisfied."""
    attribute: str
    min_count: int
    
    def is_satisfied(self, current_count: int) -> bool:
        return current_count >= self.min_count
    
    def shortage(self, current_count: int) -> int:
        return max(0, self.min_count - current_count)


@dataclass 
class Decision:
    """Represents a decision made about a person."""
    person: Person
    accepted: bool
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def person_index(self) -> int:
        return self.person.index


@dataclass
class AttributeStatistics:
    """Statistical information about attributes."""
    frequencies: Dict[str, float]
    correlations: Dict[str, Dict[str, float]]
    
    def get_frequency(self, attribute: str) -> float:
        return self.frequencies.get(attribute, 0.0)
    
    def get_correlation(self, attr1: str, attr2: str) -> float:
        return self.correlations.get(attr1, {}).get(attr2, 0.0)
    
    def get_rarity_score(self, attribute: str) -> float:
        """Higher score = more rare."""
        freq = self.get_frequency(attribute)
        return 1.0 / max(freq, 0.001)


@dataclass
class GameState:
    """Complete state of a game in progress."""
    game_id: str
    scenario: int
    constraints: List[Constraint]
    statistics: AttributeStatistics
    
    # Counters
    admitted_count: int = 0
    rejected_count: int = 0
    admitted_attributes: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    status: GameStatus = GameStatus.RUNNING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.admitted_attributes:
            self.admitted_attributes = {
                attr: 0 for attr in self.statistics.frequencies.keys()
            }
    
    @property
    def target_capacity(self) -> int:
        return 1000
    
    @property
    def max_rejections(self) -> int:
        return 20000
    
    @property
    def remaining_capacity(self) -> int:
        return self.target_capacity - self.admitted_count
    
    @property
    def rejection_ratio(self) -> float:
        return self.rejected_count / self.max_rejections
    
    @property
    def capacity_ratio(self) -> float:
        return self.admitted_count / self.target_capacity
    
    def constraint_progress(self) -> Dict[str, float]:
        """Progress toward each constraint (0.0 to 1.0+)."""
        return {
            constraint.attribute: self.admitted_attributes[constraint.attribute] / constraint.min_count
            for constraint in self.constraints
        }
    
    def constraint_shortage(self) -> Dict[str, int]:
        """How many more people needed for each constraint."""
        return {
            constraint.attribute: constraint.shortage(self.admitted_attributes[constraint.attribute])
            for constraint in self.constraints
        }
    
    def is_constraint_satisfied(self, attribute: str) -> bool:
        """Check if a specific constraint is satisfied."""
        constraint = next((c for c in self.constraints if c.attribute == attribute), None)
        if not constraint:
            return True
        return constraint.is_satisfied(self.admitted_attributes[attribute])
    
    def are_all_constraints_satisfied(self) -> bool:
        """Check if all constraints are satisfied."""
        return all(self.is_constraint_satisfied(c.attribute) for c in self.constraints)
    
    def update_decision(self, decision: Decision):
        """Update state based on a decision."""
        if decision.accepted:
            self.admitted_count += 1
            for attr, has_attr in decision.person.attributes.items():
                if has_attr:
                    self.admitted_attributes[attr] += 1
        else:
            self.rejected_count += 1
    
    def can_continue(self) -> bool:
        """Check if game can continue."""
        return (
            self.status == GameStatus.RUNNING and
            self.admitted_count < self.target_capacity and
            self.rejected_count < self.max_rejections
        )
    
    def complete_game(self, status: GameStatus, end_time: Optional[datetime] = None):
        """Mark game as completed."""
        self.status = status
        self.end_time = end_time or datetime.now()


@dataclass
class GameResult:
    """Final result of a completed game."""
    game_state: GameState
    decisions: List[Decision]
    solver_id: str
    strategy_params: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        return self.game_state.status == GameStatus.COMPLETED
    
    @property
    def total_decisions(self) -> int:
        return len(self.decisions)
    
    @property
    def acceptance_rate(self) -> float:
        if not self.decisions:
            return 0.0
        accepted = sum(1 for d in self.decisions if d.accepted)
        return accepted / len(self.decisions)
    
    @property
    def duration(self) -> float:
        """Game duration in seconds."""
        if not self.game_state.end_time:
            return 0.0
        return (self.game_state.end_time - self.game_state.start_time).total_seconds()
    
    def constraint_satisfaction_summary(self) -> Dict[str, Dict[str, Any]]:
        """Summary of constraint satisfaction."""
        summary = {}
        for constraint in self.game_state.constraints:
            current = self.game_state.admitted_attributes[constraint.attribute]
            summary[constraint.attribute] = {
                "current": current,
                "required": constraint.min_count,
                "satisfied": constraint.is_satisfied(current),
                "progress": current / constraint.min_count,
                "shortage": constraint.shortage(current)
            }
        return summary