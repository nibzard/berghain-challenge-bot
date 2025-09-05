# ABOUTME: Empirically-optimized bot based on data analysis findings
# ABOUTME: Uses extreme selectivity to meet 60% constraint requirements from 32% arrivals

import requests
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

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

class BerghainBotOptimal:
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai"):
        self.base_url = base_url
        self.player_id = "3f60a32b-8232-4b52-a11d-31a82aaa0c61"
        self.target_capacity = 1000
        self.max_rejections = 20000
        
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def start_new_game(self, scenario: int) -> GameState:
        """Start a new game."""
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
        
        self.logger.info(f"🎯 Started game {game_state.game_id[:8]}")
        self.logger.info(f"📋 Constraints: {[(c.attribute, c.min_count) for c in constraints]}")
        
        return game_state

    def make_decision(self, game_state: GameState, person_index: int, accept: Optional[bool] = None) -> Dict:
        """Make decision and get next person."""
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

    def calculate_constraint_progress(self, game_state: GameState) -> Dict[str, float]:
        """Calculate how close we are to meeting each constraint."""
        progress = {}
        for constraint in game_state.constraints:
            current = game_state.admitted_attributes[constraint.attribute]
            progress[constraint.attribute] = current / constraint.min_count
        return progress

    def should_accept_person(self, game_state: GameState, person_attributes: Dict[str, bool]) -> bool:
        """Empirically-derived decision logic based on constraint math."""
        
        # Emergency: accept almost everyone if running out of rejections
        rejection_ratio = game_state.rejected_count / self.max_rejections
        if rejection_ratio > 0.9:
            return True
        
        # Get constraint attributes for this scenario
        constraint_attrs = {c.attribute for c in game_state.constraints}
        person_constraint_attrs = {attr for attr in constraint_attrs 
                                 if person_attributes.get(attr, False)}
        
        # Calculate current progress
        progress = self.calculate_constraint_progress(game_state)
        capacity_used = game_state.admitted_count / self.target_capacity
        
        # Strategy based on empirical findings:
        # Need ~92% acceptance of people with constraint attributes
        
        if len(person_constraint_attrs) == len(constraint_attrs):
            # Person has ALL constraint attributes - ALWAYS ACCEPT
            accept_prob = 1.0
        elif len(person_constraint_attrs) > 0:
            # Person has SOME constraint attributes
            
            # Check which constraints are lagging
            lagging_constraints = [attr for attr, prog in progress.items() if prog < 0.8]
            person_helps_lagging = any(attr in person_constraint_attrs 
                                     for attr in lagging_constraints)
            
            if person_helps_lagging:
                # Person helps with lagging constraint - very likely to accept
                accept_prob = 0.98
            else:
                # Person has constraint attrs but for satisfied constraints
                # Base acceptance depends on how full we are
                if capacity_used < 0.5:
                    accept_prob = 0.95  # Still very selective early
                elif capacity_used < 0.8:
                    accept_prob = 0.90  # Slightly less selective
                else:
                    accept_prob = 0.85  # Even less selective late game
        else:
            # Person has NO constraint attributes
            
            # Check if we have "room" for non-constraint people
            min_progress = min(progress.values())
            
            if capacity_used < 0.3 and min_progress < 0.5:
                # Early game, constraints not progressing well - be very strict
                accept_prob = 0.02
            elif capacity_used < 0.6:
                # Mid game - slightly more lenient
                accept_prob = 0.05
            elif capacity_used < 0.9:
                # Late game - more accepting to fill capacity
                accept_prob = 0.15
            else:
                # Very late game - accept more to avoid failure
                accept_prob = 0.30
        
        # Add rejection pressure adjustment
        pressure_multiplier = 1.0 + (rejection_ratio * 2)  # Become more accepting under pressure
        accept_prob = min(1.0, accept_prob * pressure_multiplier)
        
        # Make decision
        import random
        decision = random.random() < accept_prob
        
        # Enhanced logging
        attrs_str = ', '.join(person_constraint_attrs) if person_constraint_attrs else 'none'
        progress_str = ', '.join(f"{k}:{v:.2f}" for k,v in progress.items())
        
        self.logger.debug(f"Attrs: {attrs_str:15} | Progress: {progress_str} | "
                         f"Prob: {accept_prob:.3f} | {'✅' if decision else '❌'}")
        
        return decision

    def update_game_state(self, game_state: GameState, response: Dict, accepted: bool, person_attributes: Dict[str, bool]):
        """Update game state tracking."""
        game_state.admitted_count = response.get("admittedCount", game_state.admitted_count)
        game_state.rejected_count = response.get("rejectedCount", game_state.rejected_count)
        
        if accepted:
            for attr, has_attr in person_attributes.items():
                if has_attr:
                    game_state.admitted_attributes[attr] += 1

    def play_game(self, scenario: int) -> Dict:
        """Play complete game with optimal strategy."""
        game_state = self.start_new_game(scenario)
        person_index = 0
        
        # Get first person
        response = self.make_decision(game_state, person_index)
        
        while response["status"] == "running":
            person_data = response["nextPerson"]
            person_attributes = person_data["attributes"]
            person_index = person_data["personIndex"]
            
            # Make decision
            accept = self.should_accept_person(game_state, person_attributes)
            
            # Submit decision
            response = self.make_decision(game_state, person_index, accept)
            
            # Update state
            self.update_game_state(game_state, response, accept, person_attributes)
            
            # Progress logging
            if person_index % 200 == 0:
                progress = self.calculate_constraint_progress(game_state)
                progress_str = ', '.join(f"{k}: {v:.1%}" for k,v in progress.items())
                self.logger.info(f"👥 Person {person_index}: Admitted {game_state.admitted_count}, "
                               f"Rejected {game_state.rejected_count} | Progress: {progress_str}")
        
        # Game ended
        final_status = response["status"]
        final_rejected = response.get("rejectedCount", game_state.rejected_count)
        
        result = {
            "status": final_status,
            "rejected_count": final_rejected,
            "admitted_count": game_state.admitted_count,
            "game_id": game_state.game_id
        }
        
        # Final constraint check
        final_progress = self.calculate_constraint_progress(game_state)
        
        if final_status == "completed":
            self.logger.info(f"🎉 SUCCESS! Rejected {final_rejected} people")
        else:
            reason = response.get("reason", "Unknown")  
            self.logger.error(f"❌ FAILED: {reason}. Rejected {final_rejected} people")
        
        # Log final constraint satisfaction
        for constraint in game_state.constraints:
            current = game_state.admitted_attributes[constraint.attribute]
            satisfied = "✅" if current >= constraint.min_count else "❌"
            self.logger.info(f"📊 {constraint.attribute}: {current}/{constraint.min_count} {satisfied}")
        
        return result

def main():
    """Test the optimal bot."""
    bot = BerghainBotOptimal()
    
    for scenario in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"🎯 TESTING OPTIMAL BOT - SCENARIO {scenario}")
        print(f"{'='*60}")
        
        try:
            result = bot.play_game(scenario)
            print(f"\n📈 RESULT:")
            print(f"Status: {result['status']}")
            print(f"Rejections: {result['rejected_count']}")
            
            if result['status'] == 'completed':
                print(f"🏆 SUCCESS! Ready for Berghain with {result['rejected_count']} rejections!")
            
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()