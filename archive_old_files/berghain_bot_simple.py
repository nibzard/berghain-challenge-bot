# ABOUTME: This is the original simple algorithm for comparison purposes
# ABOUTME: Used to benchmark improvements from the optimized version

import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class Constraint:
    attribute: str
    min_count: int

@dataclass
class GameState:
    game_id: str
    constraints: List[Constraint]
    attribute_frequencies: Dict[str, float]
    attribute_correlations: Dict[str, Dict[str, float]]
    admitted_count: int = 0
    rejected_count: int = 0
    admitted_attributes: Dict[str, int] = None
    
    def __post_init__(self):
        if self.admitted_attributes is None:
            self.admitted_attributes = {attr: 0 for attr in self.attribute_frequencies.keys()}

class BerghainBotSimple:
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai"):
        self.base_url = base_url
        self.player_id = "3f60a32b-8232-4b52-a11d-31a82aaa0c61"
        self.target_capacity = 1000
        self.max_rejections = 20000
        
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def start_new_game(self, scenario: int) -> GameState:
        """Start a new game with the specified scenario."""
        url = f"{self.base_url}/new-game"
        params = {
            "scenario": scenario,
            "playerId": self.player_id
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        constraints = [Constraint(c["attribute"], c["minCount"]) for c in data["constraints"]]
        
        game_state = GameState(
            game_id=data["gameId"],
            constraints=constraints,
            attribute_frequencies=data["attributeStatistics"]["relativeFrequencies"],
            attribute_correlations=data["attributeStatistics"]["correlations"]
        )
        
        self.logger.info(f"Started new game: {game_state.game_id}")
        self.logger.info(f"Constraints: {[(c.attribute, c.min_count) for c in constraints]}")
        
        return game_state

    def make_decision(self, game_state: GameState, person_index: int, accept: Optional[bool] = None) -> Dict:
        """Make a decision about the current person and get next person."""
        url = f"{self.base_url}/decide-and-next"
        params = {
            "gameId": game_state.game_id,
            "personIndex": person_index
        }
        
        if accept is not None:
            params["accept"] = str(accept).lower()
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def calculate_constraint_urgency(self, game_state: GameState) -> Dict[str, float]:
        """Calculate how urgent each constraint is (higher = more urgent)."""
        urgency = {}
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            current_count = game_state.admitted_attributes[attr]
            needed = constraint.min_count - current_count
            remaining_capacity = self.target_capacity - game_state.admitted_count
            
            if needed <= 0:
                urgency[attr] = 0.0  # Already satisfied
            elif remaining_capacity <= 0:
                urgency[attr] = float('inf')  # Impossible to satisfy
            else:
                # Urgency based on: needed/remaining ratio and attribute rarity
                rarity_factor = 1.0 / max(game_state.attribute_frequencies[attr], 0.01)
                ratio_factor = needed / remaining_capacity
                urgency[attr] = rarity_factor * ratio_factor
                
        return urgency

    def should_accept_person(self, game_state: GameState, person_attributes: Dict[str, bool]) -> bool:
        """Original simple decision making."""
        # Always accept if we're running out of rejection budget
        rejection_ratio = game_state.rejected_count / self.max_rejections
        if rejection_ratio > 0.9:  # If we've used 90% of our rejections
            return True
        
        # Calculate constraint urgencies
        urgencies = self.calculate_constraint_urgency(game_state)
        
        # Calculate the "value" of this person
        person_value = 0.0
        constraint_contributions = 0
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            if person_attributes.get(attr, False):
                constraint_contributions += 1
                # Weight by urgency
                person_value += urgencies[attr]
        
        # Base acceptance threshold - more selective early, less selective later
        capacity_ratio = game_state.admitted_count / self.target_capacity
        base_threshold = 1.0 - (capacity_ratio * 0.5)  # Start at 1.0, decrease to 0.5
        
        # Adjust threshold based on rejection pressure
        rejection_pressure = min(rejection_ratio * 2, 1.0)  # 0 to 1
        adjusted_threshold = base_threshold * (1 - rejection_pressure)
        
        # Special case: if person helps with multiple constraints, be more likely to accept
        if constraint_contributions > 1:
            person_value *= 1.5
        
        # Accept if person's value exceeds threshold
        decision = person_value > adjusted_threshold
        
        return decision

    def update_game_state(self, game_state: GameState, response: Dict, accepted: bool, person_attributes: Dict[str, bool]):
        """Update game state with the latest response."""
        game_state.admitted_count = response.get("admittedCount", game_state.admitted_count)
        game_state.rejected_count = response.get("rejectedCount", game_state.rejected_count)
        
        if accepted:
            # Update admitted attributes count
            for attr, has_attr in person_attributes.items():
                if has_attr:
                    game_state.admitted_attributes[attr] += 1

    def play_game(self, scenario: int) -> Dict:
        """Play a complete game."""
        game_state = self.start_new_game(scenario)
        person_index = 0
        
        # Get first person (no decision needed)
        response = self.make_decision(game_state, person_index)
        
        while response["status"] == "running":
            person_data = response["nextPerson"]
            person_attributes = person_data["attributes"]
            person_index = person_data["personIndex"]
            
            # Make decision
            accept = self.should_accept_person(game_state, person_attributes)
            
            # Submit decision and get next person
            response = self.make_decision(game_state, person_index, accept)
            
            # Update our tracking
            self.update_game_state(game_state, response, accept, person_attributes)
            
        # Game ended
        final_status = response["status"]
        final_rejected = response.get("rejectedCount", game_state.rejected_count)
        
        result = {
            "status": final_status,
            "rejected_count": final_rejected,
            "admitted_count": game_state.admitted_count,
            "game_id": game_state.game_id
        }
        
        if final_status == "completed":
            print(f"🎉 SIMPLE BOT WON! Rejected {final_rejected} people")
        else:
            reason = response.get("reason", "Unknown")
            print(f"❌ SIMPLE BOT FAILED: {reason}. Rejected {final_rejected} people")
        
        return result