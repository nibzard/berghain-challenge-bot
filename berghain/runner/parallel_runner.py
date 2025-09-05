# ABOUTME: Parallel runner for executing multiple games simultaneously
# ABOUTME: Clean implementation using ThreadPoolExecutor with proper resource management

import concurrent.futures
import logging
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

from ..core import GameResult
from .game_executor import GameExecutor


logger = logging.getLogger(__name__)


@dataclass
class GameTask:
    """Configuration for a single game task."""
    scenario_id: int
    strategy_name: str
    solver_id: str
    strategy_params: Optional[Dict[str, Any]] = None


@dataclass  
class BatchResult:
    """Results from a batch of parallel games."""
    tasks: List[GameTask]
    results: List[GameResult]
    successful_count: int
    total_duration: float
    best_result: Optional[GameResult] = None
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.successful_count / len(self.results)


class ParallelRunner:
    """Runs multiple games in parallel with clean resource management."""
    
    def __init__(self, max_workers: int = 4, config_base_path: str = "berghain/config"):
        self.max_workers = max_workers
        self.executor = GameExecutor(config_base_path)
        self.logs_base_path = Path("game_logs")
        
    def generate_strategy_variations(self, 
                                   base_strategy: str,
                                   scenario_id: int, 
                                   num_variations: int = 10) -> List[Dict[str, Any]]:
        """Generate parameter variations for strategy exploration."""
        
        # Load base strategy config
        strategy_config = self.executor.load_strategy_config(base_strategy)
        base_params = strategy_config.get('parameters', {})
        
        # Apply scenario adjustments to base
        scenario_adjustments = strategy_config.get('scenario_adjustments', {})
        if str(scenario_id) in scenario_adjustments:
            base_params.update(scenario_adjustments[str(scenario_id)])
        
        variations = [base_params.copy()]  # Include base params
        
        # Parameter ranges for exploration
        param_ranges = {
            'ultra_rare_threshold': (0.05, 0.15),
            'rare_accept_rate': (0.90, 0.99),
            'common_reject_rate': (0.01, 0.08),
            'deficit_panic_threshold': (0.6, 0.9),
            'early_game_threshold': (0.2, 0.4),
            'mid_game_threshold': (0.6, 0.8),
        }
        
        # Generate variations
        for i in range(num_variations - 1):
            variation = base_params.copy()
            
            # Randomly modify 2-3 parameters
            num_changes = random.randint(2, 3)
            params_to_change = random.sample(
                [p for p in param_ranges.keys() if p in base_params], 
                min(num_changes, len([p for p in param_ranges.keys() if p in base_params]))
            )
            
            for param in params_to_change:
                if param in param_ranges:
                    low, high = param_ranges[param]
                    variation[param] = random.uniform(low, high)
            
            variations.append(variation)
        
        return variations
    
    def create_tasks(self, 
                    scenarios: List[int],
                    strategies: List[str],
                    games_per_combination: int = 1) -> List[GameTask]:
        """Create game tasks for parallel execution."""
        tasks = []
        
        for scenario_id in scenarios:
            for strategy_name in strategies:
                # Generate parameter variations
                variations = self.generate_strategy_variations(
                    strategy_name, scenario_id, games_per_combination
                )
                
                for i, params in enumerate(variations):
                    solver_id = f"{strategy_name}_s{scenario_id}_v{i:02d}"
                    
                    task = GameTask(
                        scenario_id=scenario_id,
                        strategy_name=strategy_name,
                        solver_id=solver_id,
                        strategy_params=params
                    )
                    tasks.append(task)
        
        return tasks
    
    def execute_task(self, task: GameTask, stream_callback: Optional[Callable] = None) -> GameResult:
        """Execute a single game task."""
        try:
            # For now, always use the standard executor path
            # Custom strategy parameters can be handled in GameExecutor
            return self.executor.execute_game(
                task.scenario_id,
                task.strategy_name,
                task.solver_id,
                stream_callback
            )
                
        except Exception as e:
            logger.error(f"Task {task.solver_id} failed: {e}")
            # Return a failed result
            from ..core import GameState, GameStatus
            
            failed_state = GameState(
                game_id="failed",
                scenario=task.scenario_id,
                constraints=[],
                statistics=None
            )
            failed_state.complete_game(GameStatus.FAILED)
            
            return GameResult(
                game_state=failed_state,
                decisions=[],
                solver_id=task.solver_id,
                strategy_params=task.strategy_params or {}
            )
    
    def run_batch(self, 
                 tasks: List[GameTask],
                 stream_callback: Optional[Callable] = None) -> BatchResult:
        """Execute a batch of tasks in parallel."""
        
        start_time = datetime.now()
        logger.info(f"🚀 Starting parallel batch: {len(tasks)} games, {self.max_workers} workers")
        
        results = []
        successful_count = 0
        best_result = None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.execute_task, task, stream_callback): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        successful_count += 1
                        
                        # Track best result (lowest rejections among successful)
                        if (best_result is None or 
                            result.game_state.rejected_count < best_result.game_state.rejected_count):
                            best_result = result
                    
                    # Log result
                    status_emoji = "🎉" if result.success else "❌"
                    logger.info(f"{status_emoji} {result.solver_id}: {result.game_state.rejected_count} rejections "
                              f"({result.duration:.1f}s) {'SUCCESS' if result.success else 'FAILED'}")
                    
                except Exception as e:
                    logger.error(f"💥 Exception in {task.solver_id}: {e}")
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        batch_result = BatchResult(
            tasks=tasks,
            results=results,
            successful_count=successful_count,
            total_duration=total_duration,
            best_result=best_result
        )
        
        logger.info(f"📈 Batch completed: {successful_count}/{len(tasks)} successful "
                   f"({batch_result.success_rate:.1%}) in {total_duration:.1f}s")
        
        return batch_result
    
    def save_batch_summary(self, batch_result: BatchResult) -> Path:
        """Save comprehensive batch summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parallel_batch_{timestamp}.json"
        filepath = self.logs_base_path / filename
        
        # Group results by scenario and strategy
        by_scenario = {}
        by_strategy = {}
        
        for result in batch_result.results:
            scenario = result.game_state.scenario
            strategy = result.solver_id.split('_')[0]  # Extract strategy from solver_id
            
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            by_scenario[scenario].append({
                'solver_id': result.solver_id,
                'success': result.success,
                'rejected_count': result.game_state.rejected_count,
                'duration': result.duration
            })
            
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append({
                'solver_id': result.solver_id,
                'scenario': scenario,
                'success': result.success,
                'rejected_count': result.game_state.rejected_count
            })
        
        # Calculate statistics
        summary = {
            'timestamp': timestamp,
            'total_games': len(batch_result.results),
            'successful_games': batch_result.successful_count,
            'success_rate': batch_result.success_rate,
            'total_duration': batch_result.total_duration,
            'best_result': {
                'solver_id': batch_result.best_result.solver_id,
                'scenario': batch_result.best_result.game_state.scenario,
                'rejected_count': batch_result.best_result.game_state.rejected_count,
                'strategy_params': batch_result.best_result.strategy_params
            } if batch_result.best_result else None,
            'by_scenario': by_scenario,
            'by_strategy': by_strategy
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Saved batch summary: {filename}")
        return filepath