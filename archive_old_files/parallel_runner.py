# ABOUTME: Parallel game runner for brute-force strategy optimization
# ABOUTME: Runs multiple solver instances with different parameters simultaneously

import asyncio
import concurrent.futures
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue
import copy

from berghain_solver_ultimate import UltimateSolver

@dataclass
class SolverConfig:
    solver_id: str
    strategy_params: Dict[str, Any]
    scenario: int
    priority: int = 0  # Higher = run first

@dataclass
class GameResult:
    solver_id: str
    scenario: int
    success: bool
    rejected_count: int
    strategy_params: Dict[str, Any]
    completion_time: float
    game_id: str
    constraint_satisfaction: Dict[str, float]

class ParallelRunner:
    def __init__(self, max_workers: int = 6):
        self.max_workers = max_workers
        self.results: List[GameResult] = []
        self.active_games: Dict[str, threading.Thread] = {}
        self.result_queue = queue.Queue()
        self.best_params: Dict[int, Dict[str, Any]] = {}  # scenario -> best params
        self.logs_dir = Path("game_logs")
        
    def generate_param_variations(self, base_params: Dict[str, Any], num_variations: int = 20) -> List[Dict[str, Any]]:
        """Generate parameter variations using random sampling."""
        variations = [base_params.copy()]  # Include base params
        
        # Define parameter ranges for exploration
        param_ranges = {
            'ultra_rare_threshold': (0.05, 0.2),
            'rare_accept_rate': (0.90, 0.99),
            'common_reject_rate': (0.01, 0.1),
            'phase1_multi_attr_only': [True, False],
            'deficit_panic_threshold': (0.6, 0.9),
            'early_game_threshold': (0.2, 0.4),
            'mid_game_threshold': (0.6, 0.8),
        }
        
        for i in range(num_variations - 1):
            variation = base_params.copy()
            
            # Randomly modify 2-4 parameters
            num_changes = random.randint(2, 4)
            params_to_change = random.sample(list(param_ranges.keys()), num_changes)
            
            for param in params_to_change:
                if isinstance(param_ranges[param], tuple):
                    # Continuous parameter
                    low, high = param_ranges[param]
                    variation[param] = random.uniform(low, high)
                else:
                    # Discrete parameter
                    variation[param] = random.choice(param_ranges[param])
            
            variations.append(variation)
            
        return variations
    
    def create_solver_configs(self, scenarios: List[int], num_variants_per_scenario: int = 10) -> List[SolverConfig]:
        """Create solver configurations for testing."""
        configs = []
        
        # Base parameters from ultimate solver
        base_params = {
            'ultra_rare_threshold': 0.1,
            'rare_accept_rate': 0.98,
            'common_reject_rate': 0.05,
            'phase1_multi_attr_only': True,
            'deficit_panic_threshold': 0.8,
            'early_game_threshold': 0.3,
            'mid_game_threshold': 0.7,
        }
        
        for scenario in scenarios:
            # Generate parameter variations for this scenario
            param_variations = self.generate_param_variations(base_params, num_variants_per_scenario)
            
            for i, params in enumerate(param_variations):
                config = SolverConfig(
                    solver_id=f"s{scenario}_v{i:02d}",
                    strategy_params=params,
                    scenario=scenario,
                    priority=1 if i == 0 else 0  # Base params get priority
                )
                configs.append(config)
        
        # Sort by priority (higher first)
        configs.sort(key=lambda x: x.priority, reverse=True)
        return configs
    
    def run_single_game(self, config: SolverConfig) -> GameResult:
        """Run a single game with given configuration."""
        start_time = time.time()
        
        try:
            solver = UltimateSolver(strategy_params=config.strategy_params, 
                                   solver_id=config.solver_id)
            
            result = solver.play_game(config.scenario)
            
            # Calculate constraint satisfaction rate
            constraint_satisfaction = {}
            if hasattr(solver, 'decision_log') and solver.decision_log:
                # Get final constraint progress from last decision
                final_progress = solver.decision_log[-1].constraint_progress
                constraint_satisfaction = final_progress
            
            game_result = GameResult(
                solver_id=config.solver_id,
                scenario=config.scenario,
                success=(result['status'] == 'completed'),
                rejected_count=result['rejected_count'],
                strategy_params=config.strategy_params.copy(),
                completion_time=time.time() - start_time,
                game_id=result['game_id'][:8],
                constraint_satisfaction=constraint_satisfaction
            )
            
            return game_result
            
        except Exception as e:
            print(f"❌ Error in solver {config.solver_id}: {e}")
            return GameResult(
                solver_id=config.solver_id,
                scenario=config.scenario,
                success=False,
                rejected_count=20000,
                strategy_params=config.strategy_params.copy(),
                completion_time=time.time() - start_time,
                game_id="error",
                constraint_satisfaction={}
            )
    
    def run_parallel_batch(self, configs: List[SolverConfig]) -> List[GameResult]:
        """Run a batch of games in parallel."""
        print(f"🚀 Starting parallel batch: {len(configs)} games, {self.max_workers} workers")
        
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self.run_single_game, config): config 
                for config in configs
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                    self.results.append(result)
                    
                    # Log result
                    status_emoji = "🎉" if result.success else "❌"
                    print(f"{status_emoji} {result.solver_id}: {result.rejected_count} rejections "
                          f"({result.completion_time:.1f}s) {'SUCCESS' if result.success else 'FAILED'}")
                    
                    # Update best params if this was successful
                    if result.success:
                        if result.scenario not in self.best_params:
                            self.best_params[result.scenario] = result.strategy_params.copy()
                            print(f"🏆 New best params for scenario {result.scenario}!")
                        elif result.rejected_count < min(r.rejected_count for r in batch_results 
                                                       if r.success and r.scenario == result.scenario):
                            self.best_params[result.scenario] = result.strategy_params.copy()
                            print(f"⭐ Improved best params for scenario {result.scenario}!")
                    
                except Exception as e:
                    print(f"💥 Exception in {config.solver_id}: {e}")
        
        return batch_results
    
    def save_results_summary(self):
        """Save comprehensive results summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parallel_results_{timestamp}.json"
        filepath = self.logs_dir / filename
        
        # Group results by scenario
        by_scenario = {}
        for result in self.results:
            scenario = result.scenario
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            by_scenario[scenario].append(asdict(result))
        
        # Calculate statistics
        stats = {}
        for scenario, scenario_results in by_scenario.items():
            successful = [r for r in scenario_results if r['success']]
            stats[scenario] = {
                'total_games': len(scenario_results),
                'successful_games': len(successful),
                'success_rate': len(successful) / len(scenario_results) if scenario_results else 0,
                'best_rejection_count': min([r['rejected_count'] for r in successful]) if successful else None,
                'avg_rejection_count': sum([r['rejected_count'] for r in successful]) / len(successful) if successful else None,
                'best_params': self.best_params.get(scenario)
            }
        
        summary = {
            'timestamp': timestamp,
            'total_games': len(self.results),
            'total_successful': len([r for r in self.results if r.success]),
            'overall_success_rate': len([r for r in self.results if r.success]) / len(self.results) if self.results else 0,
            'by_scenario': by_scenario,
            'statistics': stats,
            'best_parameters': self.best_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved results summary: {filename}")
        return filepath
    
    def adaptive_run(self, scenarios: List[int], max_total_games: int = 50):
        """Adaptive parameter exploration with learning."""
        print(f"🎯 Starting adaptive parameter exploration")
        print(f"📊 Scenarios: {scenarios}, Max games: {max_total_games}")
        
        total_games_run = 0
        iteration = 0
        
        while total_games_run < max_total_games:
            iteration += 1
            remaining_games = max_total_games - total_games_run
            batch_size = min(remaining_games, self.max_workers * 3)  # 3 batches worth
            
            print(f"\n🔄 Iteration {iteration}: {batch_size} games ({total_games_run}/{max_total_games})")
            
            if iteration == 1:
                # First iteration: broad exploration
                configs = self.create_solver_configs(scenarios, batch_size // len(scenarios))
            else:
                # Subsequent iterations: exploit successful parameters
                configs = []
                for scenario in scenarios:
                    if scenario in self.best_params:
                        # Use best params as base for new variations
                        variations = self.generate_param_variations(
                            self.best_params[scenario], 
                            batch_size // len(scenarios)
                        )
                    else:
                        # Still exploring if no success yet
                        base_params = {
                            'ultra_rare_threshold': 0.1,
                            'rare_accept_rate': 0.98,
                            'common_reject_rate': 0.05,
                            'phase1_multi_attr_only': True,
                            'deficit_panic_threshold': 0.8,
                            'early_game_threshold': 0.3,
                            'mid_game_threshold': 0.7,
                        }
                        variations = self.generate_param_variations(base_params, batch_size // len(scenarios))
                    
                    for i, params in enumerate(variations):
                        config = SolverConfig(
                            solver_id=f"s{scenario}_i{iteration}_v{i:02d}",
                            strategy_params=params,
                            scenario=scenario
                        )
                        configs.append(config)
            
            # Run batch
            batch_results = self.run_parallel_batch(configs[:batch_size])
            total_games_run += len(batch_results)
            
            # Print iteration summary
            successful_this_batch = [r for r in batch_results if r.success]
            print(f"📈 Iteration {iteration} results: {len(successful_this_batch)}/{len(batch_results)} successful")
            
            if successful_this_batch:
                for result in successful_this_batch:
                    print(f"   🎉 {result.solver_id}: {result.rejected_count} rejections")
            
            # Early termination if we have success on all scenarios
            if len(self.best_params) == len(scenarios):
                print("🏆 SUCCESS on all scenarios! Stopping early.")
                break
        
        # Final summary
        self.save_results_summary()
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        print(f"\n{'='*80}")
        print(f"🏁 FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        
        total_successful = len([r for r in self.results if r.success])
        total_games = len(self.results)
        
        print(f"📊 Overall: {total_successful}/{total_games} games successful ({100*total_successful/total_games:.1f}%)")
        
        # By scenario
        scenarios = sorted(set(r.scenario for r in self.results))
        for scenario in scenarios:
            scenario_results = [r for r in self.results if r.scenario == scenario]
            successful = [r for r in scenario_results if r.success]
            
            print(f"\n🎯 Scenario {scenario}:")
            print(f"   Success rate: {len(successful)}/{len(scenario_results)} ({100*len(successful)/len(scenario_results):.1f}%)")
            
            if successful:
                best_result = min(successful, key=lambda x: x.rejected_count)
                print(f"   Best result: {best_result.rejected_count} rejections ({best_result.solver_id})")
                print(f"   Avg rejections: {sum(r.rejected_count for r in successful)/len(successful):.0f}")
                
                # Show best parameters
                if scenario in self.best_params:
                    print(f"   Best parameters:")
                    for param, value in self.best_params[scenario].items():
                        print(f"     {param}: {value}")
            else:
                print(f"   ❌ No successful games yet")
        
        if total_successful > 0:
            print(f"\n🎉 SUCCESS! Ready for Berghain with optimized strategies!")
        else:
            print(f"\n🔧 No successful games yet. Consider:")
            print(f"   - Increasing parameter exploration ranges")
            print(f"   - Running more iterations")
            print(f"   - Analyzing failure patterns in logs")

def main():
    """Run parallel optimization."""
    runner = ParallelRunner(max_workers=4)  # Adjust based on your system
    
    # Run adaptive parameter exploration
    scenarios = [1, 2, 3]  # Test all scenarios
    runner.adaptive_run(scenarios, max_total_games=30)

if __name__ == "__main__":
    main()