# ABOUTME: Dynamic strategy runner with real-time optimization
# ABOUTME: Adaptive resource allocation and continuous strategy evolution

import logging
import asyncio
import time
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .strategy_monitor import StrategyPerformanceMonitor, StrategyMetrics
from .strategy_evolution import StrategyEvolution, StrategyGenome
from ..runner import ParallelRunner, GameTask
from ..config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class DynamicRunConfig:
    """Configuration for dynamic strategy running."""
    scenario_id: int
    max_concurrent_games: int = 6
    evolution_interval: int = 180  # Evolve every 3 minutes
    min_games_per_strategy: int = 3
    max_generations: int = 10
    target_success_rate: float = 0.8


class DynamicStrategyRunner:
    """Runs strategies with continuous evolution and optimization."""
    
    def __init__(self, config: DynamicRunConfig):
        self.config = config
        self.monitor = StrategyPerformanceMonitor()
        self.evolution = StrategyEvolution()
        self.config_manager = ConfigManager()
        self.runner = ParallelRunner(max_workers=config.max_concurrent_games)
        
        self.active_games: Dict[str, GameTask] = {}
        self.completed_games: List = []
        self.current_population: List[StrategyGenome] = []
        
        # Performance tracking
        self.strategy_results: Dict[str, List[float]] = {}
        self.generation_stats: List[Dict] = []
        
        # Setup monitoring callbacks
        self.monitor.add_termination_callback(self._on_strategy_terminated)
    
    async def run_dynamic_optimization(self):
        """Run continuous strategy optimization."""
        logger.info(f"🚀 Starting dynamic strategy optimization for scenario {self.config.scenario_id}")
        
        # Initialize with base strategies
        await self._initialize_population()
        
        generation = 0
        while generation < self.config.max_generations:
            logger.info(f"🧬 Generation {generation + 1}/{self.config.max_generations}")
            
            # Run current population
            await self._run_generation()
            
            # Analyze results
            performance_data = self._analyze_performance()
            
            # Check if we've achieved target success rate
            avg_success_rate = sum(performance_data.values()) / len(performance_data) if performance_data else 0
            logger.info(f"📊 Generation {generation + 1} avg success rate: {avg_success_rate:.1%}")
            
            if avg_success_rate >= self.config.target_success_rate:
                logger.info(f"🎯 Target success rate achieved! Stopping optimization.")
                break
            
            # Evolve population for next generation
            if generation < self.config.max_generations - 1:
                self.current_population = self.evolution.evolve_population(
                    performance_data, 
                    self.config.max_concurrent_games
                )
            
            generation += 1
        
        # Final results
        await self._generate_final_report()
    
    async def _initialize_population(self):
        """Initialize the population with base strategies."""
        base_strategies = []
        
        # Load existing strategies
        available_strategies = self.config_manager.list_available_strategies()
        for strategy_name in available_strategies[:4]:  # Limit initial population
            strategy_config = self.config_manager.get_strategy_config(strategy_name)
            if strategy_config:
                base_strategies.append({
                    'name': strategy_name,
                    'type': strategy_name,
                    'parameters': strategy_config.get('parameters', {})
                })
        
        self.current_population = self.evolution.create_base_population(base_strategies)
        logger.info(f"🧬 Initialized population with {len(self.current_population)} strategies")
    
    async def _run_generation(self):
        """Run all strategies in current generation."""
        tasks = []
        
        # Create tasks for current population
        for i, genome in enumerate(self.current_population):
            for game_num in range(self.config.min_games_per_strategy):
                solver_id = f"{genome.name}_{game_num:03d}"
                
                task = GameTask(
                    scenario_id=self.config.scenario_id,
                    strategy_name=genome.base_strategy,
                    solver_id=solver_id,
                    strategy_params=genome.parameters,
                    enable_high_score_check=True
                )
                
                tasks.append(task)
                self.active_games[solver_id] = task
                
                # Register with monitor
                self.monitor.register_strategy(solver_id, genome.name)
        
        # Run tasks with monitoring
        def monitoring_callback(update):
            self._process_game_update(update)
        
        logger.info(f"🎮 Running {len(tasks)} games across {len(self.current_population)} strategies")
        
        # Execute generation
        results = self.runner.run_batch(tasks, stream_callback=monitoring_callback)
        
        # Process completed games
        for result in results.results:
            self.completed_games.append(result)
            
            # Extract strategy name from solver_id
            strategy_name = result.solver_id.rsplit('_', 1)[0]
            if strategy_name not in self.strategy_results:
                self.strategy_results[strategy_name] = []
            
            # Calculate fitness (higher = better)
            if result.success:
                # Success: lower rejections = higher fitness
                fitness = 1.0 / (1.0 + result.game_state.rejected_count / 1000)
            else:
                # Failure: small fitness based on progress
                progress_sum = sum(result.game_state.constraint_progress().values())
                fitness = progress_sum * 0.1  # Small reward for partial progress
            
            self.strategy_results[strategy_name].append(fitness)
        
        logger.info(f"📊 Generation completed: {results.successful_count}/{len(results.results)} successful")
    
    def _process_game_update(self, update):
        """Process real-time game updates for monitoring."""
        if update.get("type") == "decision":
            solver_id = update.get("solver_id")
            data = update.get("data", {})
            
            admitted = data.get("admitted", 0)
            rejected = data.get("rejected", 0)
            progress = data.get("progress", {})
            
            self.monitor.update_strategy(solver_id, admitted, rejected, progress)
    
    def _on_strategy_terminated(self, solver_id: str, reason: str):
        """Handle strategy termination callback."""
        logger.warning(f"🛑 Strategy {solver_id} terminated: {reason}")
        
        # TODO: Could spawn new mutant to replace terminated strategy
        # This would require dynamic task injection into the runner
    
    def _analyze_performance(self) -> Dict[str, float]:
        """Analyze performance of current generation."""
        performance_data = {}
        
        for strategy_name, fitness_scores in self.strategy_results.items():
            if fitness_scores:
                # Use average fitness as performance measure
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                performance_data[strategy_name] = avg_fitness
        
        return performance_data
    
    async def _generate_final_report(self):
        """Generate final optimization report."""
        logger.info("📋 Generating final optimization report...")
        
        # Find best performing strategies
        performance_data = self._analyze_performance()
        best_strategies = sorted(performance_data.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 TOP PERFORMING STRATEGIES:")
        for i, (strategy_name, fitness) in enumerate(best_strategies[:5]):
            logger.info(f"   #{i+1}: {strategy_name} (fitness: {fitness:.3f})")
        
        # Success rate analysis
        total_games = len(self.completed_games)
        successful_games = sum(1 for r in self.completed_games if r.success)
        final_success_rate = successful_games / total_games if total_games > 0 else 0
        
        logger.info(f"📊 Final Results:")
        logger.info(f"   Total games: {total_games}")
        logger.info(f"   Successful: {successful_games}")
        logger.info(f"   Success rate: {final_success_rate:.1%}")
        logger.info(f"   Generations: {len(self.generation_stats)}")
        
        # Save best strategies for future use
        self._save_evolved_strategies(best_strategies[:3])
    
    def _save_evolved_strategies(self, best_strategies: List[tuple]):
        """Save evolved strategies to config files."""
        import yaml
        from pathlib import Path
        
        evolved_dir = Path("berghain/config/strategies/evolved")
        evolved_dir.mkdir(exist_ok=True)
        
        for strategy_name, fitness in best_strategies:
            # Find the genome
            genome = None
            for g in self.current_population:
                if g.name == strategy_name:
                    genome = g
                    break
            
            if genome:
                config = self.evolution.get_strategy_config(genome)
                config['fitness_score'] = fitness
                config['evolved_from'] = genome.parent_ids
                
                filename = evolved_dir / f"{strategy_name}.yaml"
                with open(filename, 'w') as f:
                    yaml.dump(config, f, indent=2)
                
                logger.info(f"💾 Saved evolved strategy: {filename}")


# Convenience function for quick optimization runs
async def optimize_strategies(scenario_id: int = 1, max_workers: int = 6, 
                            generations: int = 5) -> DynamicStrategyRunner:
    """Run strategy optimization with default settings."""
    config = DynamicRunConfig(
        scenario_id=scenario_id,
        max_concurrent_games=max_workers,
        max_generations=generations
    )
    
    runner = DynamicStrategyRunner(config)
    await runner.run_dynamic_optimization()
    return runner
