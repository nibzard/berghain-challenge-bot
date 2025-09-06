# ABOUTME: Base solver with common game execution logic
# ABOUTME: Separates strategy from execution - clean architecture

import logging
import time
import random
from typing import Optional, Callable, List
from datetime import datetime

from ..core import GameState, Person, Decision, BerghainAPIClient, GameResult, GameStatus
from ..core.strategy import DecisionStrategy
from ..core.high_score_checker import HighScoreChecker


logger = logging.getLogger(__name__)


class BaseSolver:
    """Base game solver that executes strategies."""
    
    def __init__(self, strategy: DecisionStrategy, solver_id: str = "base", enable_high_score_check: bool = True, api_client: Optional[BerghainAPIClient] = None):
        self.strategy = strategy
        self.solver_id = solver_id
        self.api_client = api_client or BerghainAPIClient()
        self.decisions: List[Decision] = []
        self.stream_callback: Optional[Callable] = None
        self.enable_high_score_check = enable_high_score_check
        self.high_score_checker: Optional[HighScoreChecker] = None
        
    def set_stream_callback(self, callback: Callable):
        """Set callback for streaming updates."""
        self.stream_callback = callback
        
    def _stream_update(self, update_type: str, data: dict):
        """Send update to stream if callback is set."""
        if self.stream_callback:
            self.stream_callback({
                "type": update_type,
                "solver_id": self.solver_id,
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
    
    def play_game(self, scenario: int) -> GameResult:
        """Execute a complete game using the configured strategy."""
        logger.info(f"Starting game - Solver: {self.solver_id}, Strategy: {self.strategy.name}, Scenario: {scenario}")
        
        # Initialize high score checker
        if self.enable_high_score_check:
            self.high_score_checker = HighScoreChecker(scenario, enabled=True)
        else:
            self.high_score_checker = None
        
        # Start new game with robust retry/backoff
        max_attempts = 5
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                game_state = self.api_client.start_new_game(scenario)
                break
            except Exception as e:
                last_error = e
                if attempt == max_attempts:
                    logger.error(f"Failed to start new game after {max_attempts} attempts: {e}")
                    raise
                wait = 5 * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(f"Start new game failed (attempt {attempt}/{max_attempts}): {e}. "
                               f"Backing off {wait:.1f}s ...")
                time.sleep(wait)
        self.decisions = []
        
        # Stream game start
        self._stream_update("game_start", {
            "game_id": game_state.game_id,
            "scenario": scenario,
            "strategy": self.strategy.name,
            "constraints": [
                {"attribute": c.attribute, "min_count": c.min_count}
                for c in game_state.constraints
            ]
        })
        
        # Log strategy analysis
        rarity_scores = {
            attr: game_state.statistics.get_rarity_score(attr)
            for attr in [c.attribute for c in game_state.constraints]
        }
        for attr, score in sorted(rarity_scores.items(), key=lambda x: x[1], reverse=True):
            freq = game_state.statistics.get_frequency(attr)
            constraint = next((c for c in game_state.constraints if c.attribute == attr), None)
            logger.info(f"📊 {attr}: {freq:.3f} freq, {score:.1f} rarity, need {constraint.min_count if constraint else 'N/A'}")
        
        # Get first person
        person = self.api_client.get_next_person(game_state, 0)
        
        # Main game loop with safety counter
        max_iterations = 25000  # Safety limit to prevent infinite loops
        iteration_count = 0
        
        while person and game_state.can_continue() and iteration_count < max_iterations:
            iteration_count += 1
            # Make decision using strategy
            accept, reasoning = self.strategy.should_accept(person, game_state)
            
            # Create decision record
            decision = Decision(person, accept, reasoning)
            self.decisions.append(decision)
            
            # Stream decision
            self._stream_update("decision", {
                "person_index": person.index,
                "attributes": person.attributes,
                "decision": accept,
                "reasoning": reasoning,
                "admitted": game_state.admitted_count,
                "rejected": game_state.rejected_count,
                "progress": game_state.constraint_progress()
            })
            
            # Submit decision and get next person
            try:
                response = self.api_client.submit_decision(game_state, person, accept)
            except Exception as e:
                logger.error(f"❌ [{self.solver_id}] API error during decision submission: {e}")
                # If API fails, mark game as failed and exit
                game_state.complete_game(GameStatus.FAILED)
                break
            
            # Update game state
            game_state.update_decision(decision)

            # Stream API response snapshot (post-update row for append-only logs)
            try:
                self._stream_update("api_response", {
                    "person_index": decision.person.index,
                    "attributes": decision.person.attributes,
                    "decision": decision.accepted,
                    "reasoning": decision.reasoning,
                    "admitted": game_state.admitted_count,
                    "rejected": game_state.rejected_count,
                    "progress": game_state.constraint_progress(),
                    "status": game_state.status.value
                })
            except Exception:
                pass
            
            # Check high score threshold after each decision
            if self.high_score_checker and self.high_score_checker.should_terminate(game_state.rejected_count):
                reason = self.high_score_checker.get_termination_reason(game_state.rejected_count)
                logger.info(f"🏁 [{self.solver_id}] Early termination: {reason}")
                game_state.complete_game(GameStatus.ABORTED_HIGH_SCORE)
                person = None
            elif response["status"] == "running" and "nextPerson" in response:
                person_data = response["nextPerson"]
                person = Person(person_data["personIndex"], person_data["attributes"])
            else:
                person = None
                # Update final status from API
                if response["status"] != "running":
                    status_map = {"completed": GameStatus.COMPLETED, "failed": GameStatus.FAILED}
                    game_state.complete_game(status_map.get(response["status"], GameStatus.FAILED))
            
            # Additional check for capacity reached (defensive programming)
            if game_state.admitted_count >= game_state.target_capacity:
                if game_state.status == GameStatus.RUNNING:
                    logger.info(f"🏁 [{self.solver_id}] Capacity reached: {game_state.admitted_count}/{game_state.target_capacity}")
                    
                    # Make a final API call to synchronize the game completion status
                    try:
                        logger.info(f"🔄 [{self.solver_id}] Making final status check to synchronize game completion")
                        final_response = self.api_client.get_game_status(game_state)
                        
                        # Update status from final API response
                        final_status = final_response.get("status", "running")
                        if final_status != "running":
                            if final_status == "completed":
                                game_state.complete_game(GameStatus.COMPLETED)
                                logger.info(f"✅ [{self.solver_id}] Game completion confirmed by API")
                            else:
                                game_state.complete_game(GameStatus.FAILED)
                                logger.info(f"❌ [{self.solver_id}] Game marked as failed by API")
                        else:
                            # API still shows running even though we're at capacity - force completion locally
                            game_state.complete_game(GameStatus.COMPLETED)
                            logger.warning(f"⚠️ [{self.solver_id}] Forced local completion - API still shows running at capacity")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ [{self.solver_id}] Final status check failed: {e} - completing locally")
                        game_state.complete_game(GameStatus.COMPLETED)
                
                person = None
            
            # Periodic progress logging
            if person and person.index % 100 == 0:
                progress_str = ', '.join(f"{k}: {v:.1%}" for k,v in game_state.constraint_progress().items())
                phase = getattr(self.strategy, 'get_game_phase', lambda gs: 'unknown')(game_state)
                logger.debug(f"👥 P{person.index} [{phase}]: A{game_state.admitted_count}, "
                           f"R{game_state.rejected_count} | {progress_str}")
        
        # Check if loop ended due to iteration limit (safety measure)
        if iteration_count >= max_iterations:
            logger.warning(f"⚠️ [{self.solver_id}] Game ended due to iteration limit ({max_iterations}) - possible infinite loop")
            if game_state.status == GameStatus.RUNNING:
                game_state.complete_game(GameStatus.FAILED)
        
        # Game ended - create result
        result = GameResult(
            game_state=game_state,
            decisions=self.decisions,
            solver_id=self.solver_id,
            strategy_params=self.strategy.get_params()
        )
        
        # Log final results
        if result.success:
            logger.info(f"🎉 [{self.solver_id}] SUCCESS! Rejected {game_state.rejected_count} people")
        else:
            logger.error(f"❌ [{self.solver_id}] FAILED. Rejected {game_state.rejected_count}")
        
        # Log constraint satisfaction
        constraint_summary = result.constraint_satisfaction_summary()
        for attr, summary in constraint_summary.items():
            satisfied = "✅" if summary["satisfied"] else "❌"
            shortage = f" (need {summary['shortage']} more)" if summary['shortage'] > 0 else ""
            logger.info(f"📊 {attr}: {summary['current']}/{summary['required']} {satisfied}{shortage}")
        
        # Stream game end
        self._stream_update("game_end", {
            "game_id": game_state.game_id,
            "status": game_state.status.value,
            "success": result.success,
            "admitted_count": game_state.admitted_count,
            "rejected_count": game_state.rejected_count,
            "constraint_summary": constraint_summary,
            "total_decisions": len(self.decisions),
            "duration": result.duration
        })
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        self.api_client.close()
