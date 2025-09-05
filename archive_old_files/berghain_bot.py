# ABOUTME: This script plays the Berghain Challenge game using a dynamic threshold strategy
# ABOUTME: It balances meeting attribute constraints while minimizing total rejections

import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import random
import numpy as np
from scipy import stats

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

class BerghainBot:
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai"):
        self.base_url = base_url
        self.player_id = "3f60a32b-8232-4b52-a11d-31a82aaa0c61"
        self.target_capacity = 1000
        self.max_rejections = 20000
        
        # Setup logging
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

    def calculate_constraint_satisfaction_probability(self, game_state: GameState) -> Dict[str, float]:
        """Calculate probability of satisfying each constraint given current state."""
        probabilities = {}
        remaining_spots = self.target_capacity - game_state.admitted_count
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            current_count = game_state.admitted_attributes[attr]
            needed = constraint.min_count - current_count
            
            if needed <= 0:
                probabilities[attr] = 1.0  # Already satisfied
            elif remaining_spots <= 0:
                probabilities[attr] = 0.0  # Impossible to satisfy
            else:
                # Use binomial distribution to calculate probability
                p = game_state.attribute_frequencies[attr]
                # P(X >= needed) where X ~ Binomial(remaining_spots, p)
                prob = 1 - stats.binom.cdf(needed - 1, remaining_spots, p)
                probabilities[attr] = max(prob, 0.0001)  # Avoid zero for log calculations
                
        return probabilities
    
    def calculate_constraint_urgency(self, game_state: GameState) -> Dict[str, float]:
        """Calculate constraint urgency based on satisfaction probability."""
        probabilities = self.calculate_constraint_satisfaction_probability(game_state)
        urgencies = {}
        
        for attr, prob in probabilities.items():
            # Higher urgency for lower probability (more risk)
            # Use log scale to amplify differences
            urgencies[attr] = -math.log(prob)
            
        return urgencies
    
    def predict_expected_arrivals(self, game_state: GameState, remaining_spots: int) -> Dict[str, float]:
        """Predict expected number of people with each attribute in remaining spots."""
        expected = {}
        
        for attr, frequency in game_state.attribute_frequencies.items():
            # Adjust for correlations if we've accepted people with correlated attributes
            adjusted_frequency = frequency
            
            # Use correlations to adjust expectations
            for other_attr, correlation in game_state.attribute_correlations.get(attr, {}).items():
                if other_attr in game_state.admitted_attributes:
                    admitted_ratio = game_state.admitted_attributes[other_attr] / max(game_state.admitted_count, 1)
                    # Positive correlation increases expected frequency
                    adjustment = correlation * (admitted_ratio - game_state.attribute_frequencies[other_attr])
                    adjusted_frequency += adjustment * 0.1  # Scale down the effect
            
            adjusted_frequency = max(0.01, min(0.99, adjusted_frequency))  # Keep in bounds
            expected[attr] = remaining_spots * adjusted_frequency
            
        return expected
    
    def get_game_phase(self, game_state: GameState) -> str:
        """Determine current game phase based on capacity and constraint progress."""
        capacity_ratio = game_state.admitted_count / self.target_capacity
        rejection_ratio = game_state.rejected_count / self.max_rejections
        
        # Check constraint satisfaction progress
        constraint_progress = []
        for constraint in game_state.constraints:
            attr = constraint.attribute
            current = game_state.admitted_attributes[attr]
            progress = current / constraint.min_count
            constraint_progress.append(progress)
        
        avg_constraint_progress = sum(constraint_progress) / len(constraint_progress)
        
        if capacity_ratio < 0.3 and rejection_ratio < 0.1:
            return "ultra_selective"
        elif capacity_ratio < 0.7 and rejection_ratio < 0.5:
            return "balanced"
        elif capacity_ratio < 0.9 or avg_constraint_progress < 0.8:
            return "constraint_focused"
        else:
            return "completion_mode"
    
    def calculate_constraint_slack(self, game_state: GameState) -> Dict[str, float]:
        """Calculate slack for each constraint (negative means behind target)."""
        slack = {}
        capacity_ratio = game_state.admitted_count / self.target_capacity
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            current = game_state.admitted_attributes[attr]
            expected_by_now = constraint.min_count * capacity_ratio
            slack[attr] = current - expected_by_now
            
        return slack
    
    def check_feasibility(self, game_state: GameState) -> Dict[str, bool]:
        """Check if each constraint is still feasible to meet."""
        feasibility = {}
        remaining_spots = self.target_capacity - game_state.admitted_count
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            current = game_state.admitted_attributes[attr]
            needed = constraint.min_count - current
            
            if needed <= 0:
                feasibility[attr] = True
            elif remaining_spots <= 0:
                feasibility[attr] = False
            else:
                # Check if it's theoretically possible (even if unlikely)
                max_possible = current + remaining_spots
                feasibility[attr] = max_possible >= constraint.min_count
                
        return feasibility
    
    def monte_carlo_decision_analysis(self, game_state: GameState, person_attributes: Dict[str, bool], 
                                    num_simulations: int = 1000) -> Tuple[float, float]:
        """
        Run Monte Carlo simulation to estimate success probability for accept vs reject.
        Returns (accept_success_prob, reject_success_prob).
        """
        remaining_spots = self.target_capacity - game_state.admitted_count
        
        if remaining_spots <= 10 or num_simulations < 100:
            # Skip simulation for edge cases
            return 0.5, 0.5
            
        accept_successes = 0
        reject_successes = 0
        
        for _ in range(num_simulations):
            # Simulate accepting this person
            sim_state_accept = self._create_simulation_state(game_state, True, person_attributes)
            if self._simulate_game_completion(game_state, sim_state_accept, remaining_spots - 1):
                accept_successes += 1
                
            # Simulate rejecting this person
            sim_state_reject = self._create_simulation_state(game_state, False, person_attributes)
            if self._simulate_game_completion(game_state, sim_state_reject, remaining_spots):
                reject_successes += 1
                
        return accept_successes / num_simulations, reject_successes / num_simulations
    
    def _create_simulation_state(self, game_state: GameState, accept_person: bool, 
                               person_attributes: Dict[str, bool]) -> Dict[str, int]:
        """Create a simulation state dict for Monte Carlo."""
        sim_state = game_state.admitted_attributes.copy()
        
        if accept_person:
            for attr, has_attr in person_attributes.items():
                if has_attr:
                    sim_state[attr] += 1
                    
        return sim_state
    
    def _simulate_game_completion(self, game_state: GameState, sim_state: Dict[str, int], remaining_spots: int) -> bool:
        """Simulate the rest of the game to check if all constraints can be met."""
        if remaining_spots <= 0:
            # Check if all constraints are already satisfied
            for constraint in game_state.constraints:
                if sim_state[constraint.attribute] < constraint.min_count:
                    return False
            return True
            
        # Generate random people for remaining spots
        for _ in range(remaining_spots):
            # Generate person with random attributes based on frequencies
            for attr, freq in game_state.attribute_frequencies.items():
                if random.random() < freq:
                    sim_state[attr] += 1
                    
        # Check if all constraints are satisfied
        for constraint in game_state.constraints:
            if sim_state[constraint.attribute] < constraint.min_count:
                return False
                
        return True

    def should_accept_person(self, game_state: GameState, person_attributes: Dict[str, bool]) -> bool:
        """Advanced decision making using probabilistic model and multi-phase strategy."""
        phase = self.get_game_phase(game_state)
        rejection_ratio = game_state.rejected_count / self.max_rejections
        
        # Emergency mode: accept almost everyone
        if rejection_ratio > 0.95:
            return True
            
        # Check feasibility
        feasibility = self.check_feasibility(game_state)
        if not all(feasibility.values()):
            # Some constraints impossible - focus on possible ones
            person_helps_feasible = any(
                person_attributes.get(attr, False) and feasible 
                for attr, feasible in feasibility.items()
            )
            if person_helps_feasible:
                return True
        
        # Get constraint analysis
        urgencies = self.calculate_constraint_urgency(game_state)
        slack = self.calculate_constraint_slack(game_state)
        probabilities = self.calculate_constraint_satisfaction_probability(game_state)
        
        # Calculate person value
        person_value = 0.0
        constraint_contributions = 0
        critical_contributions = 0
        
        for constraint in game_state.constraints:
            attr = constraint.attribute
            if person_attributes.get(attr, False):
                constraint_contributions += 1
                
                # Weight by urgency and slack
                urgency_weight = urgencies[attr]
                slack_penalty = max(0, -slack[attr])  # Penalty for being behind
                prob_bonus = -math.log(probabilities[attr]) if probabilities[attr] > 0 else 10
                
                contribution = urgency_weight + slack_penalty + prob_bonus
                person_value += contribution
                
                # Check if this is a critical contribution
                if slack[attr] < -10 or probabilities[attr] < 0.1:
                    critical_contributions += 1
        
        # Multi-attribute bonus (quadratic scaling)
        if constraint_contributions > 1:
            person_value *= (1.0 + 0.5 * constraint_contributions)
        
        # Use Monte Carlo for critical decisions
        remaining_spots = self.target_capacity - game_state.admitted_count
        use_monte_carlo = (
            phase in ["balanced", "constraint_focused"] and
            any(probabilities[attr] < 0.2 for attr in probabilities) and
            rejection_ratio < 0.8 and
            remaining_spots > 50
        )
        
        if use_monte_carlo:
            accept_prob, reject_prob = self.monte_carlo_decision_analysis(game_state, person_attributes, 500)
            if abs(accept_prob - reject_prob) > 0.1:  # Significant difference
                decision = accept_prob > reject_prob
                self.logger.debug(f"Monte Carlo: Accept={accept_prob:.3f}, Reject={reject_prob:.3f}, "
                                f"Decision: {'ACCEPT' if decision else 'REJECT'}")
                return decision
        
        # Phase-specific thresholds
        if phase == "ultra_selective":
            # Only accept people with multiple attributes
            threshold = 8.0
            if constraint_contributions < 2:
                return False
        elif phase == "balanced":
            # Use probability-based threshold
            min_prob = min(probabilities.values())
            threshold = 3.0 + (-math.log(min_prob) if min_prob > 0 else 5.0)
        elif phase == "constraint_focused":
            # Focus on lagging constraints
            threshold = 2.0
            # Accept anyone who helps with critical constraints
            if critical_contributions > 0:
                return True
        else:  # completion_mode
            # Accept almost everyone
            threshold = 0.5
        
        # Adjust threshold based on rejection pressure
        pressure_adjustment = math.exp(-3 * (1 - rejection_ratio))  # Exponential decay
        adjusted_threshold = threshold * pressure_adjustment
        
        decision = person_value > adjusted_threshold
        
        self.logger.debug(f"Phase: {phase}, Person value: {person_value:.3f}, "
                         f"Threshold: {adjusted_threshold:.3f}, Contributions: {constraint_contributions}, "
                         f"Decision: {'ACCEPT' if decision else 'REJECT'}")
        
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
            
            # Log decision
            attr_list = [attr for attr, has_attr in person_attributes.items() if has_attr]
            self.logger.info(f"Person {person_index}: {attr_list} -> {'ACCEPT' if accept else 'REJECT'}")
            
            # Submit decision and get next person
            response = self.make_decision(game_state, person_index, accept)
            
            # Update our tracking
            self.update_game_state(game_state, response, accept, person_attributes)
            
            # Log progress periodically
            if person_index % 100 == 0:
                self.logger.info(f"Progress - Admitted: {game_state.admitted_count}, "
                               f"Rejected: {game_state.rejected_count}")
        
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
            self.logger.info(f"🎉 GAME WON! Rejected {final_rejected} people")
        else:
            reason = response.get("reason", "Unknown")
            self.logger.error(f"❌ GAME FAILED: {reason}. Rejected {final_rejected} people")
        
        # Log constraint satisfaction
        for constraint in game_state.constraints:
            current = game_state.admitted_attributes[constraint.attribute]
            satisfied = "✅" if current >= constraint.min_count else "❌"
            self.logger.info(f"Constraint {constraint.attribute}: {current}/{constraint.min_count} {satisfied}")
        
        return result

def main():
    """Main entry point for the script."""
    bot = BerghainBot()
    
    # Play all scenarios
    for scenario in [1, 2, 3]:
        print(f"\n{'='*50}")
        print(f"Playing Scenario {scenario}")
        print(f"{'='*50}")
        
        try:
            result = bot.play_game(scenario)
            print(f"\nScenario {scenario} Result:")
            print(f"Status: {result['status']}")
            print(f"Rejections: {result['rejected_count']}")
            print(f"Admissions: {result['admitted_count']}")
        except Exception as e:
            print(f"Error playing scenario {scenario}: {e}")

if __name__ == "__main__":
    main()