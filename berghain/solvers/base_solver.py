# ABOUTME: Base solver with common game execution logic
# ABOUTME: Separates strategy from execution - clean architecture

import logging
from typing import Optional, Callable, List
from datetime import datetime

from ..core import GameState, Person, Decision, BerghainAPIClient, GameResult
from ..core.strategy import DecisionStrategy


logger = logging.getLogger(__name__)


class BaseSolver:
    """Base game solver that executes strategies."""
    
    def __init__(self, strategy: DecisionStrategy, solver_id: str = "base"):
        self.strategy = strategy
        self.solver_id = solver_id
        self.api_client = BerghainAPIClient()
        self.decisions: List[Decision] = []
        self.stream_callback: Optional[Callable] = None
        
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
        
        # Start new game
        game_state = self.api_client.start_new_game(scenario)
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
        
        # Main game loop
        while person and game_state.can_continue():
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
            response = self.api_client.submit_decision(game_state, person, accept)
            
            # Update game state
            game_state.update_decision(decision)
            
            # Get next person if game continues
            if response["status"] == "running" and "nextPerson" in response:
                person_data = response["nextPerson"]
                person = Person(person_data["personIndex"], person_data["attributes"])
            else:
                person = None
                # Update final status from API
                if response["status"] != "running":
                    status_map = {"completed": GameStatus.COMPLETED, "failed": GameStatus.FAILED}
                    game_state.complete_game(status_map.get(response["status"], GameStatus.FAILED))
            
            # Periodic progress logging
            if person and person.index % 100 == 0:
                progress_str = ', '.join(f"{k}: {v:.1%}" for k,v in game_state.constraint_progress().items())
                phase = getattr(self.strategy, 'get_game_phase', lambda gs: 'unknown')(game_state)
                logger.debug(f"👥 P{person.index} [{phase}]: A{game_state.admitted_count}, "
                           f"R{game_state.rejected_count} | {progress_str}")
        
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