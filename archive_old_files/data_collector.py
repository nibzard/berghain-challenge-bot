# ABOUTME: Fast data collection bot that records complete game logs for analysis
# ABOUTME: Uses random decisions to speed through games and gather empirical data

import requests
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class PersonData:
    person_index: int
    attributes: Dict[str, bool]
    decision: bool
    timestamp: float

@dataclass
class GameLog:
    game_id: str
    scenario: int
    start_time: datetime
    end_time: Optional[datetime]
    constraints: List[Dict]
    attribute_frequencies: Dict[str, float]
    attribute_correlations: Dict[str, Dict[str, float]]
    people: List[PersonData]
    final_status: str
    final_rejected_count: int
    final_admitted_count: int
    total_time: float
    
class DataCollector:
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai"):
        self.base_url = base_url
        self.player_id = "3f60a32b-8232-4b52-a11d-31a82aaa0c61"
        self.data_dir = Path("game_logs")
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup minimal logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def make_random_decision(self, person_attributes: Dict[str, bool], game_state: Dict) -> bool:
        """Make random decisions with slight bias toward accepting people with any constraint attribute."""
        
        # Count how many constraint attributes this person has
        constraint_attrs = set()
        for constraint in game_state.get('constraints', []):
            constraint_attrs.add(constraint['attribute'])
        
        person_constraint_count = sum(1 for attr in constraint_attrs if person_attributes.get(attr, False))
        
        # Bias toward accepting people with constraint attributes
        if person_constraint_count > 0:
            accept_probability = 0.7  # 70% chance to accept
        else:
            accept_probability = 0.3  # 30% chance to accept
        
        return random.random() < accept_probability

    def play_fast_game(self, scenario: int) -> GameLog:
        """Play a single game quickly with random decisions."""
        start_time = datetime.now()
        
        # Start new game
        url = f"{self.base_url}/new-game"
        params = {
            "scenario": scenario,
            "playerId": self.player_id
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        game_log = GameLog(
            game_id=data["gameId"],
            scenario=scenario,
            start_time=start_time,
            end_time=None,
            constraints=data["constraints"],
            attribute_frequencies=data["attributeStatistics"]["relativeFrequencies"],
            attribute_correlations=data["attributeStatistics"]["correlations"],
            people=[],
            final_status="",
            final_rejected_count=0,
            final_admitted_count=0,
            total_time=0.0
        )
        
        # Game state for decision making
        game_state = {
            'constraints': data["constraints"],
            'admitted_count': 0,
            'rejected_count': 0
        }
        
        # Get first person
        url = f"{self.base_url}/decide-and-next"
        params = {
            "gameId": game_log.game_id,
            "personIndex": 0
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        response_data = response.json()
        
        while response_data["status"] == "running":
            person_data = response_data["nextPerson"]
            person_attributes = person_data["attributes"]
            person_index = person_data["personIndex"]
            
            # Make random decision
            accept = self.make_random_decision(person_attributes, game_state)
            
            # Record the person and decision
            game_log.people.append(PersonData(
                person_index=person_index,
                attributes=person_attributes,
                decision=accept,
                timestamp=time.time()
            ))
            
            # Submit decision
            params = {
                "gameId": game_log.game_id,
                "personIndex": person_index,
                "accept": str(accept).lower()
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            response_data = response.json()
            
            # Update game state
            game_state['admitted_count'] = response_data.get("admittedCount", game_state['admitted_count'])
            game_state['rejected_count'] = response_data.get("rejectedCount", game_state['rejected_count'])
            
            # Progress indicator
            if person_index % 500 == 0:
                print(f"  Person {person_index}: Admitted {game_state['admitted_count']}, "
                      f"Rejected {game_state['rejected_count']}")
        
        # Game ended
        end_time = datetime.now()
        game_log.end_time = end_time
        game_log.final_status = response_data["status"]
        game_log.final_rejected_count = response_data.get("rejectedCount", game_state['rejected_count'])
        game_log.final_admitted_count = game_state.get('admitted_count', 0)
        game_log.total_time = (end_time - start_time).total_seconds()
        
        return game_log

    def save_game_log(self, game_log: GameLog):
        """Save game log to JSON file."""
        timestamp = game_log.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"scenario_{game_log.scenario}_{timestamp}_{game_log.game_id[:8]}.json"
        filepath = self.data_dir / filename
        
        # Convert to dict for JSON serialization
        log_dict = asdict(game_log)
        log_dict['start_time'] = game_log.start_time.isoformat()
        if game_log.end_time:
            log_dict['end_time'] = game_log.end_time.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(log_dict, f, indent=2)
        
        print(f"💾 Saved game log: {filename}")
        return filepath

    def collect_data(self, scenario: int, num_games: int = 10):
        """Collect data from multiple games."""
        print(f"🎯 Starting data collection for Scenario {scenario}")
        print(f"📊 Target: {num_games} games")
        print("-" * 50)
        
        successful_games = 0
        failed_games = 0
        total_rejections = []
        
        for game_num in range(num_games):
            try:
                print(f"\n🎮 Game {game_num + 1}/{num_games}")
                
                game_log = self.play_fast_game(scenario)
                self.save_game_log(game_log)
                
                if game_log.final_status == "completed":
                    successful_games += 1
                    print(f"✅ SUCCESS - Rejected: {game_log.final_rejected_count}, "
                          f"Time: {game_log.total_time:.1f}s")
                else:
                    failed_games += 1
                    print(f"❌ FAILED - Rejected: {game_log.final_rejected_count}, "
                          f"Reason: {game_log.final_status}")
                
                total_rejections.append(game_log.final_rejected_count)
                
            except Exception as e:
                print(f"💥 Error in game {game_num + 1}: {e}")
                failed_games += 1
                continue
        
        # Summary statistics
        avg_rejections = sum(total_rejections) / len(total_rejections) if total_rejections else 0
        print(f"\n📈 DATA COLLECTION SUMMARY")
        print(f"{'='*40}")
        print(f"Successful games: {successful_games}")
        print(f"Failed games: {failed_games}")
        print(f"Success rate: {100*successful_games/(successful_games+failed_games):.1f}%")
        print(f"Average rejections: {avg_rejections:.1f}")
        print(f"Data files saved in: {self.data_dir}")

def main():
    collector = DataCollector()
    
    # Collect data for all scenarios
    for scenario in [1, 2, 3]:
        collector.collect_data(scenario, num_games=5)  # Start with 5 games per scenario
        
        # Brief pause between scenarios
        print(f"\n⏸️  Pausing 5 seconds before next scenario...")
        time.sleep(5)
    
    print(f"\n🎉 Data collection complete! Check the '{collector.data_dir}' directory for logs.")

if __name__ == "__main__":
    main()