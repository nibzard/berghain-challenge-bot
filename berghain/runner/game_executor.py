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
    
    def create_strategy(self, strategy_name: str, scenario_id: int, override_params: Optional[Dict[str, Any]] = None) -> 'DecisionStrategy':
        """Create strategy instance with configuration and optional overrides."""
        strategy_config = self.load_strategy_config(strategy_name)
        
        # Get base parameters
        base_params = dict(strategy_config.get('parameters', {}))
        
        # Apply scenario-specific adjustments (support int or string keys)
        scenario_adjustments = strategy_config.get('scenario_adjustments', {})
        try:
            # Normalize keys to strings for robust lookup
            normalized_adjustments = {str(k): v for k, v in scenario_adjustments.items()}
        except AttributeError:
            normalized_adjustments = {}
        if str(scenario_id) in normalized_adjustments:
            base_params.update(normalized_adjustments[str(scenario_id)])
        
        # Apply external overrides (may be full config or just params)
        if override_params:
            if isinstance(override_params, dict) and 'parameters' in override_params and isinstance(override_params['parameters'], dict):
                base_params.update(override_params['parameters'])
            elif isinstance(override_params, dict):
                base_params.update(override_params)
        
        # Create strategy instance
        name = strategy_name.lower()
        if name in ('conservative', 'aggressive'):
            return RarityWeightedStrategy(base_params)
        elif name == 'adaptive':
            return AdaptiveStrategy(base_params)
        else:
            # Try additional strategies if available
            try:
                from ..solvers import BalancedStrategy, GreedyConstraintStrategy, DiversityFirstStrategy, QuotaTrackerStrategy, DualDeficitController, RBCRStrategy, RBCR2Strategy, DVOStrategy, RamanujanStrategy, UltimateStrategy, Ultimate2Strategy, Ultimate3Strategy, Ultimate3HStrategy, PerfectStrategy, MecStrategy
                if name == 'balanced':
                    return BalancedStrategy(base_params)
                if name == 'greedy':
                    return GreedyConstraintStrategy(base_params)
                if name == 'diversity':
                    return DiversityFirstStrategy(base_params)
                if name in ('quota', 'quota_tracker'):
                    return QuotaTrackerStrategy(base_params)
                if name in ('dual', 'dual_deficit', 'dualdeficit'):
                    return DualDeficitController(base_params)
                if name in ('rbcr', 'bidprice', 'bid_price'):
                    return RBCRStrategy(base_params)
                if name in ('rbcr2', 'rbcr_2', 'enhanced_rbcr', 'lp_rbcr'):
                    return RBCR2Strategy(base_params)
                if name in ('dvo', 'dynamic'):
                    return DVOStrategy(base_params)
                if name in ('ramanujan', 'rimo', 'mathematical'):
                    return RamanujanStrategy(base_params)
                if name in ('ultimate', 'umo', 'optimal'):
                    return UltimateStrategy(base_params)
                if name in ('ultimate2', 'ultimate_2', 'u2'):
                    return Ultimate2Strategy(base_params)
                if name in ('ultimate3', 'ultimate_3', 'u3'):
                    return Ultimate3Strategy(base_params)
                if name in ('ultimate3h', 'ultimate_3h', 'u3h', 'hybrid'):
                    return Ultimate3HStrategy(base_params)
                if name in ('perfect', 'pbo', 'balance'):
                    return PerfectStrategy(base_params)
                if name in ('mec', 'exact', 'mathematician'):
                    return MecStrategy(base_params)
            except Exception:
                pass
            
            # Default to rarity weighted
            logger.warning(f"Unknown strategy '{strategy_name}', using RarityWeighted")
            return RarityWeightedStrategy(base_params)
    
    def execute_game(self, 
                    scenario_id: int,
                    strategy_name: str = 'conservative',
                    solver_id: str = None,
                    stream_callback: Optional[Callable] = None,
                    enable_high_score_check: bool = True,
                    strategy_params: Optional[Dict[str, Any]] = None,
                    mode: str = 'local') -> GameResult:
        """Execute a single game."""
        
        if solver_id is None:
            solver_id = f"{strategy_name}_{scenario_id}_{datetime.now().strftime('%H%M%S')}"
        
        logger.info(f"Executing game - Scenario: {scenario_id}, Strategy: {strategy_name}, Solver: {solver_id}")
        
        # Create live status file for monitoring and append-only event log
        live_status_file = self.logs_base_path / f"live_{solver_id}.json"
        events_file = self.logs_base_path / f"events_{solver_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        def live_stream_callback(update):
            """Write live updates to status file for monitoring."""
            try:
                with open(live_status_file, 'w') as f:
                    json.dump(update, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write live status: {e}")
            
            # Append one row per API response to events file (NDJSON)
            try:
                if isinstance(update, dict) and update.get('type') == 'api_response':
                    with open(events_file, 'a') as f:
                        f.write(json.dumps(update) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append event row: {e}")
            
            # Also call original callback if provided
            if stream_callback:
                stream_callback(update)
        
        # Load configurations
        scenario_config = self.load_scenario_config(scenario_id)
        
        # Create strategy and solver
        strategy = self.create_strategy(strategy_name, scenario_id, override_params=strategy_params)
        # Select client backend
        api_client = None
        try:
            if mode and mode.lower() == 'local':
                from ..core.local_simulator import LocalSimulatorClient
                api_client = LocalSimulatorClient()
            else:
                from ..core import BerghainAPIClient as _RealClient
                api_client = _RealClient()
        except Exception as e:
            logger.warning(f"Falling back to real API client due to: {e}")
            from ..core import BerghainAPIClient as _RealClient
            api_client = _RealClient()

        # Create specialized solver for strategies that need it
        name = strategy_name.lower()
        if name in ('mec', 'exact', 'mathematician'):
            from ..solvers import MecSolver
            solver = MecSolver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('ultimate', 'umo', 'optimal'):
            from ..solvers import UltimateSolver
            solver = UltimateSolver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('ultimate2', 'ultimate_2', 'u2'):
            from ..solvers import Ultimate2Solver
            solver = Ultimate2Solver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('ultimate3', 'ultimate_3', 'u3'):
            from ..solvers import Ultimate3Solver
            solver = Ultimate3Solver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('ultimate3h', 'ultimate_3h', 'u3h', 'hybrid'):
            from ..solvers import Ultimate3HSolver
            solver = Ultimate3HSolver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('perfect', 'pbo', 'balance'):
            from ..solvers import PerfectSolver
            solver = PerfectSolver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('dvo', 'dynamic'):
            from ..solvers import DVOSolver
            solver = DVOSolver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        elif name in ('ramanujan', 'rimo', 'mathematical'):
            from ..solvers import RamanujanSolver
            solver = RamanujanSolver(solver_id, api_client=api_client, enable_high_score_check=enable_high_score_check)
        else:
            # Use BaseSolver for strategies that don't have specialized solvers
            solver = BaseSolver(strategy, solver_id, enable_high_score_check, api_client=api_client)
        
        # Set up streaming with live status file
        solver.stream_callback = live_stream_callback
        
        try:
            # Execute the game
            result = solver.play_game(scenario_id)
            
            # Strategy post-run hook (learning/persistence)
            try:
                if hasattr(strategy, 'on_game_end') and callable(getattr(strategy, 'on_game_end')):
                    strategy.on_game_end(result)
            except Exception:
                pass

            # Save result log
            self._save_game_log(result, scenario_config, strategy_name=strategy.name)
            
            logger.info(f"Game completed - {solver_id}: {'SUCCESS' if result.success else 'FAILED'}")
            
            return result
            
        finally:
            # Clean up live status file
            if live_status_file.exists():
                live_status_file.unlink()
            # Keep events_file for post-run analysis
            solver.cleanup()
    
    def _save_game_log(self, result: GameResult, scenario_config: Dict[str, Any], strategy_name: str):
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
            "strategy_name": strategy_name,
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
