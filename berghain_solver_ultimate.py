# ABOUTME: Ultimate rarity-weighted Berghain solver based on mathematical constraint analysis
# ABOUTME: Uses extreme selectivity and real-time constraint tracking with TUI streaming

import requests
import json
import logging
import random
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import websocket
import socket

@dataclass
class Constraint:
    attribute: str
    min_count: int

@dataclass 
class GameState:
    game_id: str
    scenario: int
    constraints: List[Constraint]
    attribute_frequencies: Dict[str, float]
    attribute_correlations: Dict[str, Dict[str, float]]
    admitted_count: int = 0
    rejected_count: int = 0
    admitted_attributes: Dict[str, int] = None
    start_time: datetime = None
    
    def __post_init__(self):
        if self.admitted_attributes is None:
            self.admitted_attributes = {attr: 0 for attr in self.attribute_frequencies.keys()}
        if self.start_time is None:
            self.start_time = datetime.now()

@dataclass
class DecisionLog:
    person_index: int
    attributes: Dict[str, bool]
    decision: bool
    reasoning: str
    constraint_progress: Dict[str, float]
    timestamp: datetime

class UltimateSolver:
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai", 
                 strategy_params: Dict = None, solver_id: str = "ultimate"):
        self.base_url = base_url
        self.player_id = "3f60a32b-8232-4b52-a11d-31a82aaa0c61"
        self.solver_id = solver_id
        self.target_capacity = 1000
        self.max_rejections = 20000
        
        # Strategy parameters (tunable via genetic algorithm)
        self.params = strategy_params or {
            'ultra_rare_threshold': 0.1,      # Attributes below this frequency are ultra-rare
            'rare_accept_rate': 0.98,         # Accept rate for people with ultra-rare attributes
            'common_reject_rate': 0.05,       # Accept rate for people with no constraint attributes
            'phase1_multi_attr_only': True,   # Only accept people with 2+ attributes in phase 1
            'deficit_panic_threshold': 0.8,   # When to enter panic mode for lagging constraints
            'early_game_threshold': 0.3,      # Capacity ratio for early game phase
            'mid_game_threshold': 0.7,        # Capacity ratio for mid game phase
        }
        
        # Logging and streaming setup
        self.decision_log: List[DecisionLog] = []
        self.stream_callback: Optional[Callable] = None
        
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"UltimateSolver-{solver_id}")

    def set_stream_callback(self, callback: Callable):
        """Set callback for streaming updates to TUI."""
        self.stream_callback = callback

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
            scenario=scenario,
            constraints=constraints,
            attribute_frequencies=data["attributeStatistics"]["relativeFrequencies"],
            attribute_correlations=data["attributeStatistics"]["correlations"]
        )
        
        self.logger.info(f"🎯 [{self.solver_id}] Started game {game_state.game_id[:8]} - Scenario {scenario}")
        
        # Log constraint analysis
        rarity_scores = self.calculate_rarity_scores(game_state)
        for attr, score in sorted(rarity_scores.items(), key=lambda x: x[1], reverse=True):
            freq = game_state.attribute_frequencies[attr]
            constraint = next((c for c in constraints if c.attribute == attr), None)
            if constraint:
                self.logger.info(f"📊 {attr}: {freq:.3f} freq, {score:.1f} rarity, need {constraint.min_count}")
        
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

    def calculate_rarity_scores(self, game_state: GameState) -> Dict[str, float]:
        """Calculate rarity scores for each attribute (higher = more rare)."""
        scores = {}
        constraint_attrs = {c.attribute for c in game_state.constraints}
        
        for attr in constraint_attrs:
            freq = game_state.attribute_frequencies[attr]
            # Rarity score = 1/frequency, with minimum to avoid division by zero
            scores[attr] = 1.0 / max(freq, 0.001)
        
        return scores

    def calculate_constraint_progress(self, game_state: GameState) -> Dict[str, float]:
        """Calculate progress toward each constraint (0.0 to 1.0+)."""
        progress = {}
        for constraint in game_state.constraints:
            current = game_state.admitted_attributes[constraint.attribute]
            progress[constraint.attribute] = current / constraint.min_count
        return progress

    def calculate_constraint_deficit(self, game_state: GameState) -> Dict[str, int]:
        """Calculate how many more people we need for each constraint."""
        deficit = {}
        remaining_spots = self.target_capacity - game_state.admitted_count
        
        for constraint in game_state.constraints:
            current = game_state.admitted_attributes[constraint.attribute]
            needed = constraint.min_count - current
            # Can't admit more than remaining spots
            realistic_deficit = min(needed, remaining_spots)
            deficit[constraint.attribute] = max(0, realistic_deficit)
        
        return deficit

    def get_game_phase(self, game_state: GameState) -> str:
        """Determine current game phase."""
        capacity_ratio = game_state.admitted_count / self.target_capacity
        rejection_ratio = game_state.rejected_count / self.max_rejections
        
        if capacity_ratio < self.params['early_game_threshold']:
            return "early"
        elif capacity_ratio < self.params['mid_game_threshold']:
            return "mid"
        elif rejection_ratio > 0.85:
            return "panic"
        else:
            return "late"

    def should_accept_person(self, game_state: GameState, person_attributes: Dict[str, bool]) -> tuple[bool, str]:
        """Ultimate decision logic with reasoning."""
        
        # Emergency: accept almost everyone if running out of rejections
        rejection_ratio = game_state.rejected_count / self.max_rejections
        if rejection_ratio > 0.95:
            return True, "emergency_rejection_limit"
        
        # Get analysis data
        rarity_scores = self.calculate_rarity_scores(game_state)
        progress = self.calculate_constraint_progress(game_state)
        deficit = self.calculate_constraint_deficit(game_state)
        phase = self.get_game_phase(game_state)
        
        # Calculate person's value
        constraint_attrs = {c.attribute for c in game_state.constraints}
        person_constraint_attrs = {attr for attr in constraint_attrs 
                                 if person_attributes.get(attr, False)}
        
        if not person_constraint_attrs:
            # Person has no constraint attributes
            if phase == "early":
                return False, "early_phase_no_constraints"
            elif phase == "panic":
                return random.random() < 0.3, "panic_mode_filler"
            else:
                return random.random() < self.params['common_reject_rate'], "common_filler"
        
        # Person has some constraint attributes
        person_rarity_sum = sum(rarity_scores[attr] for attr in person_constraint_attrs)
        has_ultra_rare = any(game_state.attribute_frequencies[attr] < self.params['ultra_rare_threshold']
                           for attr in person_constraint_attrs)
        
        # Check for critical constraints (very behind)
        critical_constraints = {attr for attr, prog in progress.items() 
                              if prog < self.params['deficit_panic_threshold'] and deficit[attr] > 50}
        helps_critical = bool(person_constraint_attrs & critical_constraints)
        
        # Decision logic by phase
        if phase == "early":
            if self.params['phase1_multi_attr_only'] and len(person_constraint_attrs) < 2:
                return False, "early_phase_single_attr"
            if has_ultra_rare:
                return True, f"early_ultra_rare_{list(person_constraint_attrs)}"
            if len(person_constraint_attrs) >= 2:
                return random.random() < 0.9, f"early_multi_attr_{len(person_constraint_attrs)}"
            return False, "early_phase_common"
            
        elif phase == "mid":
            if helps_critical:
                return True, f"mid_critical_help_{list(person_constraint_attrs & critical_constraints)}"
            if has_ultra_rare:
                return random.random() < self.params['rare_accept_rate'], f"mid_ultra_rare_{list(person_constraint_attrs)}"
            if person_rarity_sum > 5.0:  # High combined rarity
                return random.random() < 0.8, f"mid_high_rarity_{person_rarity_sum:.1f}"
            return random.random() < 0.5, "mid_moderate_value"
            
        elif phase == "late":
            if helps_critical:
                return True, f"late_critical_help_{list(person_constraint_attrs & critical_constraints)}"
            if deficit[list(person_constraint_attrs)[0]] > 0:  # Still need this attribute
                return random.random() < 0.7, f"late_needed_attr_{list(person_constraint_attrs)}"
            return random.random() < 0.3, "late_surplus_attr"
            
        else:  # panic
            if person_constraint_attrs:
                return True, f"panic_any_constraint_{list(person_constraint_attrs)}"
            return random.random() < 0.5, "panic_filler"

    def update_game_state(self, game_state: GameState, response: Dict, accepted: bool, person_attributes: Dict[str, bool]):
        """Update game state tracking."""
        game_state.admitted_count = response.get("admittedCount", game_state.admitted_count)
        game_state.rejected_count = response.get("rejectedCount", game_state.rejected_count)
        
        if accepted:
            for attr, has_attr in person_attributes.items():
                if has_attr:
                    game_state.admitted_attributes[attr] += 1

    def save_game_log(self, game_state: GameState, final_result: Dict):
        """Save complete game log with decision history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultimate_solver_{self.solver_id}_s{game_state.scenario}_{timestamp}_{game_state.game_id[:8]}.json"
        filepath = Path("game_logs") / filename
        
        # Create comprehensive log
        game_log = {
            "solver_id": self.solver_id,
            "strategy_params": self.params,
            "game_id": game_state.game_id,
            "scenario": game_state.scenario,
            "start_time": game_state.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "constraints": [asdict(c) for c in game_state.constraints],
            "attribute_frequencies": game_state.attribute_frequencies,
            "attribute_correlations": game_state.attribute_correlations,
            "final_result": final_result,
            "decisions": [asdict(d) for d in self.decision_log[-1000:]],  # Last 1000 decisions
            "final_admitted_attributes": game_state.admitted_attributes
        }
        
        with open(filepath, 'w') as f:
            json.dump(game_log, f, indent=2, default=str)
        
        self.logger.info(f"💾 Saved game log: {filename}")
        return filepath

    def play_game(self, scenario: int) -> Dict:
        """Play complete game with ultimate strategy."""
        game_state = self.start_new_game(scenario)
        person_index = 0
        
        # Get first person
        response = self.make_decision(game_state, person_index)
        
        while response["status"] == "running":
            person_data = response["nextPerson"]
            person_attributes = person_data["attributes"]
            person_index = person_data["personIndex"]
            
            # Make decision with reasoning
            accept, reasoning = self.should_accept_person(game_state, person_attributes)
            
            # Log decision
            progress = self.calculate_constraint_progress(game_state)
            decision_log = DecisionLog(
                person_index=person_index,
                attributes=person_attributes,
                decision=accept,
                reasoning=reasoning,
                constraint_progress=progress.copy(),
                timestamp=datetime.now()
            )
            self.decision_log.append(decision_log)
            
            # Stream to TUI if callback set
            if self.stream_callback:
                self.stream_callback({
                    "type": "decision",
                    "solver_id": self.solver_id,
                    "game_id": game_state.game_id[:8],
                    "person_index": person_index,
                    "decision": accept,
                    "reasoning": reasoning,
                    "progress": progress,
                    "admitted": game_state.admitted_count,
                    "rejected": game_state.rejected_count
                })
            
            # Submit decision
            response = self.make_decision(game_state, person_index, accept)
            
            # Update state
            self.update_game_state(game_state, response, accept, person_attributes)
            
            # Periodic progress logging
            if person_index % 100 == 0:
                progress_str = ', '.join(f"{k}: {v:.1%}" for k,v in progress.items())
                phase = self.get_game_phase(game_state)
                self.logger.debug(f"👥 P{person_index} [{phase}]: A{game_state.admitted_count}, "
                                f"R{game_state.rejected_count} | {progress_str}")
        
        # Game ended
        final_status = response["status"]
        final_rejected = response.get("rejectedCount", game_state.rejected_count)
        
        result = {
            "status": final_status,
            "rejected_count": final_rejected,
            "admitted_count": game_state.admitted_count,
            "game_id": game_state.game_id,
            "solver_id": self.solver_id,
            "strategy_params": self.params,
            "total_decisions": len(self.decision_log)
        }
        
        # Log final results
        if final_status == "completed":
            self.logger.info(f"🎉 [{self.solver_id}] SUCCESS! Rejected {final_rejected} people")
        else:
            reason = response.get("reason", "Unknown")  
            self.logger.error(f"❌ [{self.solver_id}] FAILED: {reason}. Rejected {final_rejected}")
        
        # Log constraint satisfaction
        for constraint in game_state.constraints:
            current = game_state.admitted_attributes[constraint.attribute]
            satisfied = "✅" if current >= constraint.min_count else "❌"
            shortage = max(0, constraint.min_count - current)
            self.logger.info(f"📊 {constraint.attribute}: {current}/{constraint.min_count} {satisfied}"
                           + (f" (need {shortage} more)" if shortage > 0 else ""))
        
        # Save comprehensive log
        self.save_game_log(game_state, result)
        
        return result

def main():
    """Test the ultimate solver."""
    solver = UltimateSolver(solver_id="test-1")
    
    for scenario in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"🎯 TESTING ULTIMATE SOLVER - SCENARIO {scenario}")
        print(f"{'='*60}")
        
        try:
            result = solver.play_game(scenario)
            print(f"\n📈 RESULT:")
            print(f"Status: {result['status']}")
            print(f"Rejections: {result['rejected_count']}")
            print(f"Total decisions: {result['total_decisions']}")
            
            if result['status'] == 'completed':
                print(f"🏆 SUCCESS! Berghain trip with {result['rejected_count']} rejections!")
                break  # Stop on first success
            
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()