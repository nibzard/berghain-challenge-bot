# ABOUTME: Clean API client for the Berghain Challenge API
# ABOUTME: Single responsibility - handle HTTP communication with the game API

import requests
import logging
from typing import Dict, Optional, Any
from .domain import GameState, Person, Constraint, AttributeStatistics, GameStatus


logger = logging.getLogger(__name__)


class BerghainAPIError(Exception):
    """Custom exception for API-related errors."""
    pass


class BerghainAPIClient:
    """Clean API client with single responsibility."""
    
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai", 
                 player_id: str = "3f60a32b-8232-4b52-a11d-31a82aaa0c61", 
                 timeout: int = 30):
        self.base_url = base_url
        self.player_id = player_id
        self.timeout = timeout
        self.session = requests.Session()
        
        # Configure session with timeout
        adapter = requests.adapters.HTTPAdapter()
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def start_new_game(self, scenario: int) -> GameState:
        """Start a new game and return initial game state."""
        url = f"{self.base_url}/new-game"
        params = {
            "scenario": scenario,
            "playerId": self.player_id
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Parse constraints
            constraints = [
                Constraint(c["attribute"], c["minCount"]) 
                for c in data["constraints"]
            ]
            
            # Parse statistics
            statistics = AttributeStatistics(
                frequencies=data["attributeStatistics"]["relativeFrequencies"],
                correlations=data["attributeStatistics"]["correlations"]
            )
            
            # Create game state
            game_state = GameState(
                game_id=data["gameId"],
                scenario=scenario,
                constraints=constraints,
                statistics=statistics
            )
            
            logger.info(f"Started new game {game_state.game_id[:8]} for scenario {scenario}")
            return game_state
            
        except requests.exceptions.RequestException as e:
            raise BerghainAPIError(f"Failed to start game: {e}")
        except (KeyError, ValueError) as e:
            raise BerghainAPIError(f"Invalid API response: {e}")
    
    def make_decision(self, game_state: GameState, person_index: int, 
                     accept: Optional[bool] = None) -> Dict[str, Any]:
        """Make a decision and get the next person or game result."""
        url = f"{self.base_url}/decide-and-next"
        params = {
            "gameId": game_state.game_id,
            "personIndex": person_index
        }
        
        if accept is not None:
            params["accept"] = str(accept).lower()
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise BerghainAPIError(f"Failed to make decision: {e}")
    
    def get_next_person(self, game_state: GameState, person_index: int) -> Optional[Person]:
        """Get the next person without making a decision."""
        response = self.make_decision(game_state, person_index)
        
        if response["status"] != "running":
            return None
            
        person_data = response["nextPerson"]
        return Person(
            index=person_data["personIndex"],
            attributes=person_data["attributes"]
        )
    
    def submit_decision(self, game_state: GameState, person: Person, accept: bool) -> Dict[str, Any]:
        """Submit a decision about a person."""
        response = self.make_decision(game_state, person.index, accept)
        
        # Update game state counters from API response
        if "admittedCount" in response:
            game_state.admitted_count = response["admittedCount"]
        if "rejectedCount" in response:
            game_state.rejected_count = response["rejectedCount"]
        
        # Update status if game ended
        status = response.get("status", "running")
        if status != "running":
            if status == "completed":
                game_state.complete_game(GameStatus.COMPLETED)
            else:
                game_state.complete_game(GameStatus.FAILED)
        
        return response
    
    def play_turn(self, game_state: GameState, person_index: int, accept: bool) -> Optional[Person]:
        """Complete turn: submit decision and get next person."""
        # Submit decision
        response = self.submit_decision(game_state, 
                                       Person(person_index, {}), accept)
        
        # Return next person if game continues
        if response["status"] == "running" and "nextPerson" in response:
            person_data = response["nextPerson"]
            return Person(
                index=person_data["personIndex"],
                attributes=person_data["attributes"]
            )
        
        return None
    
    def close(self):
        """Clean up resources."""
        self.session.close()