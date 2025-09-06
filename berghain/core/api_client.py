# ABOUTME: Clean API client for the Berghain Challenge API
# ABOUTME: Single responsibility - handle HTTP communication with the game API

import os
import requests
import logging
import time
import random
import threading
from typing import Dict, Optional, Any
from .domain import GameState, Person, Constraint, AttributeStatistics, GameStatus


logger = logging.getLogger(__name__)


class BerghainAPIError(Exception):
    """Custom exception for API-related errors."""
    pass


class BerghainAPIClient:
    """Clean API client with single responsibility."""
    
    # Global semaphore to limit concurrent API calls across all instances
    _api_concurrency_limit = int(os.getenv("BERGHAIN_MAX_API_CONCURRENCY", "10"))
    _api_semaphore = threading.Semaphore(_api_concurrency_limit)
    
    def __init__(self, base_url: str = "https://berghain.challenges.listenlabs.ai", 
                 player_id: str = "3f60a32b-8232-4b52-a11d-31a82aaa0c61", 
                 timeout: int = 60):
        # Allow overrides via environment
        self.base_url = os.getenv("BERGHAIN_BASE_URL", base_url)
        self.player_id = os.getenv("BERGHAIN_PLAYER_ID", player_id)
        self.timeout = timeout
        # Lightweight defaults; use per-request connections for simplicity
        self.session = requests.Session()
        self.default_headers = {
            "User-Agent": "python-requests",
            "Accept": "application/json",
            "Connection": "close",
        }
        
    def start_new_game(self, scenario: int) -> GameState:
        """Start a new game and return initial game state."""
        url = f"{self.base_url}/new-game"
        params = {
            "scenario": scenario,
            "playerId": self.player_id
        }
        
        # Simple, robust start with fresh connection per attempt
        data = None
        last_exc: Optional[Exception] = None
        for attempt in range(1, 7):
            try:
                # Backoff: 2s, 4s, 8s, 12s, 16s, 20s (+jitter)
                delay = 2.0 * attempt
                time.sleep(delay + random.uniform(0, 0.5))
                
                # Use semaphore to limit concurrent API calls
                logger.debug(f"Acquiring API semaphore for start_new_game (attempt {attempt})")
                with self._api_semaphore:
                    logger.debug(f"API semaphore acquired for start_new_game")
                    resp = requests.get(url, params=params, headers=self.default_headers, timeout=(10, self.timeout))
                    
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            time.sleep(min(15.0, float(ra)))
                        except Exception:
                            pass
                    last_exc = BerghainAPIError("HTTP 429 Too Many Requests")
                    continue
                if 500 <= resp.status_code < 600:
                    last_exc = BerghainAPIError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                last_exc = e
                continue
        if data is None:
            raise BerghainAPIError(f"Failed to start game after retries: {last_exc}")
        
        try:
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
        
        # Simple per-call request with modest backoff
        return self._simple_get_with_backoff(url, params)
    
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
        # Track previous count to detect rollbacks
        previous_admitted_count = game_state.admitted_count
        
        response = self.make_decision(game_state, person.index, accept)
        
        # Update game state counters from API response
        rollback_detected = False
        if "admittedCount" in response:
            new_admitted_count = response["admittedCount"]
            
            # Check for API rollback behavior (count went down despite accepting)
            if accept and new_admitted_count < previous_admitted_count:
                logger.warning(f"⚠️ API rollback detected! Accepted person {person.index} but count went from {previous_admitted_count} to {new_admitted_count}")
                rollback_detected = True
                response["rollback_detected"] = True
            
            game_state.admitted_count = new_admitted_count
            
        if "rejectedCount" in response:
            game_state.rejected_count = response["rejectedCount"]
        
        # Update status if game ended
        status = response.get("status", "running")
        if status != "running":
            if status == "completed":
                game_state.complete_game(GameStatus.COMPLETED)
            else:
                game_state.complete_game(GameStatus.FAILED)
        
        # Special handling for capacity edge cases - both 999 and 1000
        elif game_state.admitted_count >= 999:
            if game_state.admitted_count >= game_state.target_capacity:
                logger.info(f"Hit target capacity ({game_state.admitted_count}/{game_state.target_capacity}) but API still shows running")
            else:
                logger.info(f"At 999 admissions but API still shows running - potential completion state")
        
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
    
    def get_game_status(self, game_state: GameState) -> Dict[str, Any]:
        """Get current game status without making a decision."""
        # Use the make_decision method without an accept parameter to get status only
        return self.make_decision(game_state, 0)  # personIndex 0 is just for status check
    
    def is_game_effectively_complete(self, game_state: GameState) -> bool:
        """Check if game is effectively complete (handles 999 edge case)."""
        try:
            # If we're at 999 or 1000, make a status check to see if game is really done
            if game_state.admitted_count >= 999:
                response = self.get_game_status(game_state)
                # If API says running but provides no next person, game is effectively done
                if response.get("status") == "running" and "nextPerson" not in response:
                    return True
                # If API explicitly says completed or failed
                if response.get("status") in ["completed", "failed"]:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Could not check if game is effectively complete: {e}")
            # Default to false to avoid false positives
            return False
    
    def close(self):
        """Clean up resources."""
        self.session.close()

    # --- Internal helpers ---
    def _simple_get_with_backoff(self, url: str, params: Dict[str, Any], max_retries: int = 6) -> Dict[str, Any]:
        """Simple GET with small backoff and new connection per attempt."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    time.sleep(min(10.0, attempt * 1.5) + random.uniform(0, 0.5))
                
                # Use semaphore to limit concurrent API calls
                logger.debug(f"Acquiring API semaphore for decision API call (attempt {attempt})")
                with self._api_semaphore:
                    logger.debug(f"API semaphore acquired for decision API call")
                    resp = requests.get(url, params=params, headers=self.default_headers, timeout=(10, self.timeout))
                    
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            time.sleep(min(15.0, float(ra)))
                        except Exception:
                            pass
                    last_exc = BerghainAPIError("HTTP 429 Too Many Requests")
                    continue
                if 500 <= resp.status_code < 600:
                    last_exc = BerghainAPIError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                continue
        raise BerghainAPIError(f"HTTP request failed after retries: {last_exc}")
