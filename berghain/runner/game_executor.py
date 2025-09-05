# ABOUTME: Single game executor with clean separation of concerns
# ABOUTME: Handles configuration loading and result logging

import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from ..core import GameResult
from ..solvers import BaseSolver, RarityWeightedStrategy, AdaptiveStrategy


logger = logging.getLogger(__name__)


class GameExecutor:
    """Executes single games with configuration-driven setup."""
    
    def __init__(self, config_base_path: str = "berghain/config"):
        self.config_base_path = Path(config_base_path)
        self.logs_base_path = Path("game_logs")
        
        # Ensure logs directory exists
        self.logs_base_path.mkdir(exist_ok=True)
        
    def load_scenario_config(self, scenario_id: int) -> Dict[str, Any]:
        """Load scenario configuration from YAML."""
        scenario_file = self.config_base_path / "scenarios" / f"scenario_{scenario_id}.yaml"
        
        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario config not found: {scenario_file}")
        
        with open(scenario_file, 'r') as f:
            return yaml.safe_load(f)
    
    def load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Load strategy configuration from YAML."""
        strategy_file = self.config_base_path / "strategies" / f"{strategy_name}.yaml"
        
        if not strategy_file.exists():
            raise FileNotFoundError(f"Strategy config not found: {strategy_file}")
        
        with open(strategy_file, 'r') as f:
            return yaml.safe_load(f)
    
    def create_strategy(self, strategy_name: str, scenario_id: int) -> 'DecisionStrategy':
        """Create strategy instance with configuration."""
        strategy_config = self.load_strategy_config(strategy_name)
        
        # Get base parameters
        base_params = strategy_config.get('parameters', {})
        
        # Apply scenario-specific adjustments
        scenario_adjustments = strategy_config.get('scenario_adjustments', {})
        if str(scenario_id) in scenario_adjustments:
            base_params.update(scenario_adjustments[str(scenario_id)])
        
        # Create strategy instance
        if strategy_name == 'conservative' or strategy_name == 'aggressive':
            return RarityWeightedStrategy(base_params)
        elif strategy_name == 'adaptive':
            return AdaptiveStrategy(base_params)
        else:
            # Default to rarity weighted
            logger.warning(f"Unknown strategy '{strategy_name}', using RarityWeighted")
            return RarityWeightedStrategy(base_params)
    
    def execute_game(self, 
                    scenario_id: int,
                    strategy_name: str = 'conservative',
                    solver_id: str = None,
                    stream_callback: Optional[Callable] = None) -> GameResult:
        """Execute a single game."""
        
        if solver_id is None:
            solver_id = f"{strategy_name}_{scenario_id}_{datetime.now().strftime('%H%M%S')}"
        
        logger.info(f"Executing game - Scenario: {scenario_id}, Strategy: {strategy_name}, Solver: {solver_id}")
        
        # Create live status file for monitoring
        live_status_file = self.logs_base_path / f"live_{solver_id}.json"
        
        def live_stream_callback(update):
            """Write live updates to status file for monitoring."""
            try:
                with open(live_status_file, 'w') as f:
                    json.dump(update, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write live status: {e}")
            
            # Also call original callback if provided
            if stream_callback:
                stream_callback(update)
        
        # Load configurations
        scenario_config = self.load_scenario_config(scenario_id)
        
        # Create strategy and solver
        strategy = self.create_strategy(strategy_name, scenario_id)
        solver = BaseSolver(strategy, solver_id)
        
        # Set up streaming with live status file
        solver.stream_callback = live_stream_callback
        
        try:
            # Execute the game
            result = solver.play_game(scenario_id)
            
            # Save result log
            self._save_game_log(result, scenario_config)
            
            logger.info(f"Game completed - {solver_id}: {'SUCCESS' if result.success else 'FAILED'}")
            
            return result
            
        finally:
            # Clean up live status file
            if live_status_file.exists():
                live_status_file.unlink()
            solver.cleanup()
    
    def _save_game_log(self, result: GameResult, scenario_config: Dict[str, Any]):
        """Save comprehensive game log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{result.solver_id}_{timestamp}_{result.game_state.game_id[:8]}.json"
        filepath = self.logs_base_path / filename
        
        # Create comprehensive log
        log_data = {
            # Game metadata
            "solver_id": result.solver_id,
            "game_id": result.game_state.game_id,
            "scenario_id": result.game_state.scenario,
            "scenario_name": scenario_config.get('name', f"Scenario {result.game_state.scenario}"),
            
            # Timestamps
            "start_time": result.game_state.start_time.isoformat(),
            "end_time": result.game_state.end_time.isoformat() if result.game_state.end_time else None,
            "duration_seconds": result.duration,
            
            # Strategy info
            "strategy_name": result.game_state.__class__.__name__,  # Will need to be passed down
            "strategy_params": result.strategy_params,
            
            # Game constraints and statistics
            "constraints": [
                {
                    "attribute": c.attribute,
                    "min_count": c.min_count
                }
                for c in result.game_state.constraints
            ],
            "attribute_frequencies": result.game_state.statistics.frequencies,
            "attribute_correlations": result.game_state.statistics.correlations,
            
            # Final results
            "status": result.game_state.status.value,
            "success": result.success,
            "admitted_count": result.game_state.admitted_count,
            "rejected_count": result.game_state.rejected_count,
            "total_decisions": result.total_decisions,
            "acceptance_rate": result.acceptance_rate,
            
            # Constraint satisfaction
            "final_admitted_attributes": result.game_state.admitted_attributes,
            "constraint_satisfaction": result.constraint_satisfaction_summary(),
            
            # Decision summary
            "decisions": [
                {
                    "person_index": d.person.index,
                    "attributes": d.person.attributes,
                    "decision": d.accepted,
                    "reasoning": d.reasoning,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in result.decisions[-1000:]  # Last 1000 decisions to avoid huge files
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"💾 Saved game log: {filename}")
        return filepath