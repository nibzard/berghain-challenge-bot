# ABOUTME: Parameter fine-tuning system for optimizing strategy parameters based on game scenarios
# ABOUTME: Uses Bayesian optimization and reinforcement learning to adapt strategy configurations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from berghain.core import GameState, Person
from berghain.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class ParameterConfig:
    """Configuration for a tunable parameter"""
    name: str
    min_value: float
    max_value: float
    current_value: float
    param_type: str  # 'float', 'int', 'bool'
    importance: float = 1.0  # Weight for optimization
    scenario_specific: bool = True

@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    strategy_name: str
    scenario_id: int
    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    performance_improvement: float
    avg_rejections_before: float
    avg_rejections_after: float
    confidence: float

class ParameterOptimizer:
    """Bayesian optimization for strategy parameters"""
    
    def __init__(self, strategy_name: str, scenario_id: int):
        self.strategy_name = strategy_name
        self.scenario_id = scenario_id
        self.config_manager = ConfigManager()
        self.optimization_history: List[Dict] = []
        self.gp_regressor: Optional[GaussianProcessRegressor] = None
        
        # Load base strategy configuration
        self.base_config = self.config_manager.get_strategy_config(strategy_name)
        if not self.base_config:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Define tunable parameters for each strategy type
        self.tunable_params = self._get_tunable_parameters()
        
    def _get_tunable_parameters(self) -> Dict[str, ParameterConfig]:
        """Define tunable parameters based on strategy type"""
        base_params = {}
        
        # Common parameters across all strategies
        if 'ultra_rare_threshold' in self.base_config.get('parameters', {}):
            base_params['ultra_rare_threshold'] = ParameterConfig(
                name='ultra_rare_threshold',
                min_value=0.001,
                max_value=0.05,
                current_value=self.base_config['parameters'].get('ultra_rare_threshold', 0.01),
                param_type='float',
                importance=1.5
            )
            
        if 'phase1_multi_attr_only' in self.base_config.get('parameters', {}):
            base_params['phase1_multi_attr_only'] = ParameterConfig(
                name='phase1_multi_attr_only',
                min_value=0,
                max_value=1,
                current_value=int(self.base_config['parameters'].get('phase1_multi_attr_only', True)),
                param_type='bool',
                importance=1.2
            )
            
        if 'deficit_panic_threshold' in self.base_config.get('parameters', {}):
            base_params['deficit_panic_threshold'] = ParameterConfig(
                name='deficit_panic_threshold',
                min_value=0.1,
                max_value=0.9,
                current_value=self.base_config['parameters'].get('deficit_panic_threshold', 0.7),
                param_type='float',
                importance=1.8
            )
            
        if 'multi_attr_bonus' in self.base_config.get('parameters', {}):
            base_params['multi_attr_bonus'] = ParameterConfig(
                name='multi_attr_bonus',
                min_value=1.0,
                max_value=5.0,
                current_value=self.base_config['parameters'].get('multi_attr_bonus', 2.0),
                param_type='float',
                importance=1.3
            )
        
        # Strategy-specific parameters
        if self.strategy_name == 'rbcr2':
            if 'rbcr2_switch_threshold' in self.base_config.get('parameters', {}):
                base_params['rbcr2_switch_threshold'] = ParameterConfig(
                    name='rbcr2_switch_threshold',
                    min_value=0.3,
                    max_value=0.8,
                    current_value=self.base_config['parameters'].get('rbcr2_switch_threshold', 0.6),
                    param_type='float',
                    importance=1.4
                )
                
        elif 'lstm' in self.strategy_name.lower():
            if 'temperature' in self.base_config.get('parameters', {}):
                base_params['temperature'] = ParameterConfig(
                    name='temperature',
                    min_value=0.1,
                    max_value=2.0,
                    current_value=self.base_config['parameters'].get('temperature', 1.0),
                    param_type='float',
                    importance=1.6
                )
                
        elif self.strategy_name in ['ultimate3', 'ultimate3h']:
            if 'risk_aversion' in self.base_config.get('parameters', {}):
                base_params['risk_aversion'] = ParameterConfig(
                    name='risk_aversion',
                    min_value=0.1,
                    max_value=2.0,
                    current_value=self.base_config['parameters'].get('risk_aversion', 1.0),
                    param_type='float',
                    importance=1.5
                )
        
        return base_params
    
    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range for optimization"""
        normalized = []
        for param_name, param_config in self.tunable_params.items():
            if param_name in params:
                value = params[param_name]
                normalized_value = (value - param_config.min_value) / (param_config.max_value - param_config.min_value)
                normalized.append(normalized_value)
        return np.array(normalized)
    
    def _denormalize_params(self, normalized_params: np.ndarray) -> Dict[str, Any]:
        """Convert normalized parameters back to original ranges"""
        denormalized = {}
        param_names = list(self.tunable_params.keys())
        
        for i, (param_name, param_config) in enumerate(self.tunable_params.items()):
            if i < len(normalized_params):
                normalized_value = np.clip(normalized_params[i], 0, 1)
                original_value = param_config.min_value + normalized_value * (param_config.max_value - param_config.min_value)
                
                if param_config.param_type == 'int':
                    denormalized[param_name] = int(round(original_value))
                elif param_config.param_type == 'bool':
                    denormalized[param_name] = bool(original_value > 0.5)
                else:
                    denormalized[param_name] = float(original_value)
                    
        return denormalized
    
    async def evaluate_parameters(self, params: Dict[str, Any], num_games: int = 5) -> Tuple[float, Dict[str, Any]]:
        """Evaluate parameter configuration by running games"""
        from berghain.runner import ParallelRunner
        from berghain.runner.parallel_runner import GameTask
        
        # Create temporary strategy config
        temp_config = self.base_config.copy()
        temp_config['parameters'].update(params)
        
        # Set up parallel runner
        runner = ParallelRunner(max_workers=min(num_games, 5))
        
        # Create game tasks
        tasks = []
        for i in range(num_games):
            tasks.append(GameTask(
                scenario_id=self.scenario_id,
                strategy_name=self.strategy_name,
                solver_id=f"{self.strategy_name}_opt_{i:03d}",
                strategy_params=temp_config,
                enable_high_score_check=False,
                mode='local'
            ))
        
        # Run games
        batch_result = runner.run_batch(tasks)
        
        # Calculate performance metrics
        successful_results = [r for r in batch_result.results if r.success]
        
        if not successful_results:
            # Heavy penalty for failed configurations
            return 10000.0, {'avg_rejections': 10000, 'success_rate': 0.0, 'constraint_violations': num_games}
        
        avg_rejections = sum(r.game_state.rejected_count for r in successful_results) / len(successful_results)
        success_rate = len(successful_results) / num_games
        constraint_violations = num_games - len(successful_results)
        
        # Multi-objective fitness function
        fitness = avg_rejections + (constraint_violations * 1000)  # Heavy penalty for constraint violations
        
        metrics = {
            'avg_rejections': avg_rejections,
            'success_rate': success_rate,
            'constraint_violations': constraint_violations,
            'fitness': fitness
        }
        
        return fitness, metrics
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """Upper Confidence Bound acquisition function for Bayesian optimization"""
        if self.gp_regressor is None:
            return 0.0
            
        x_reshaped = x.reshape(1, -1)
        mu, sigma = self.gp_regressor.predict(x_reshaped, return_std=True)
        
        # UCB with exploration parameter
        kappa = 2.0
        return -(mu[0] - kappa * sigma[0])  # Minimize, so negate
    
    async def optimize_parameters(self, max_iterations: int = 20, initial_samples: int = 5) -> OptimizationResult:
        """Run Bayesian optimization to find optimal parameters"""
        logger.info(f"Starting parameter optimization for {self.strategy_name} on scenario {self.scenario_id}")
        
        # Store original parameters
        original_params = {name: config.current_value for name, config in self.tunable_params.items()}
        
        # Evaluate original configuration
        original_fitness, original_metrics = await self.evaluate_parameters(original_params)
        
        # Initialize with random samples
        X_samples = []
        y_samples = []
        
        logger.info(f"Collecting {initial_samples} initial samples...")
        for i in range(initial_samples):
            # Random sample in normalized space
            random_params = np.random.random(len(self.tunable_params))
            denormalized_params = self._denormalize_params(random_params)
            
            fitness, metrics = await self.evaluate_parameters(denormalized_params)
            
            X_samples.append(random_params)
            y_samples.append(fitness)
            
            self.optimization_history.append({
                'iteration': i,
                'params': denormalized_params,
                'fitness': fitness,
                'metrics': metrics
            })
            
            logger.info(f"Sample {i+1}: fitness={fitness:.1f}, rejections={metrics.get('avg_rejections', 0):.1f}")
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        # Initialize Gaussian Process
        kernel = Matern(length_scale=0.5, nu=2.5)
        self.gp_regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        
        best_fitness = float('inf')
        best_params = original_params.copy()
        best_metrics = original_metrics
        
        # Bayesian optimization loop
        for iteration in range(initial_samples, max_iterations):
            # Fit GP to current data
            self.gp_regressor.fit(X_samples, y_samples)
            
            # Find next point to evaluate using acquisition function
            bounds = [(0, 1) for _ in range(len(self.tunable_params))]
            
            # Multiple random starts for global optimization
            best_acquisition = float('inf')
            next_point = None
            
            for _ in range(10):
                x0 = np.random.random(len(self.tunable_params))
                result = minimize(
                    self._acquisition_function,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_acquisition:
                    best_acquisition = result.fun
                    next_point = result.x
            
            if next_point is None:
                logger.warning("Failed to find next point, using random sample")
                next_point = np.random.random(len(self.tunable_params))
            
            # Evaluate next point
            denormalized_params = self._denormalize_params(next_point)
            fitness, metrics = await self.evaluate_parameters(denormalized_params)
            
            # Update samples
            X_samples = np.vstack([X_samples, next_point.reshape(1, -1)])
            y_samples = np.append(y_samples, fitness)
            
            # Track best result
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = denormalized_params.copy()
                best_metrics = metrics.copy()
                logger.info(f"New best! Iteration {iteration}: fitness={fitness:.1f}, rejections={metrics.get('avg_rejections', 0):.1f}")
            
            self.optimization_history.append({
                'iteration': iteration,
                'params': denormalized_params,
                'fitness': fitness,
                'metrics': metrics
            })
            
            logger.info(f"Iteration {iteration}: fitness={fitness:.1f}, rejections={metrics.get('avg_rejections', 0):.1f}")
        
        # Calculate improvement
        performance_improvement = (original_metrics['avg_rejections'] - best_metrics['avg_rejections']) / original_metrics['avg_rejections']
        
        result = OptimizationResult(
            strategy_name=self.strategy_name,
            scenario_id=self.scenario_id,
            original_params=original_params,
            optimized_params=best_params,
            performance_improvement=performance_improvement,
            avg_rejections_before=original_metrics['avg_rejections'],
            avg_rejections_after=best_metrics['avg_rejections'],
            confidence=1.0 - (best_fitness / max(y_samples))
        )
        
        logger.info(f"Optimization complete! Improvement: {performance_improvement*100:.1f}%")
        return result
    
    def save_optimization_result(self, result: OptimizationResult, output_dir: str = "berghain/config/strategies/optimized"):
        """Save optimized parameters to a new strategy config"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create optimized strategy config
        optimized_config = self.base_config.copy()
        optimized_config['parameters'].update(result.optimized_params)
        optimized_config['name'] = f"{optimized_config['name']} (Optimized S{self.scenario_id})"
        optimized_config['description'] = f"{optimized_config.get('description', '')} - Optimized for scenario {self.scenario_id}"
        
        # Add optimization metadata
        optimized_config['optimization'] = {
            'base_strategy': self.strategy_name,
            'scenario_id': self.scenario_id,
            'performance_improvement': result.performance_improvement,
            'avg_rejections_before': result.avg_rejections_before,
            'avg_rejections_after': result.avg_rejections_after,
            'confidence': result.confidence,
            'optimized_params': result.optimized_params
        }
        
        # Save config
        config_filename = f"{self.strategy_name}_optimized_s{self.scenario_id}.yaml"
        config_path = output_path / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, sort_keys=False)
        
        # Save optimization history
        history_filename = f"{self.strategy_name}_optimization_history_s{self.scenario_id}.json"
        history_path = output_path / history_filename
        
        with open(history_path, 'w') as f:
            json.dump({
                'result': asdict(result),
                'history': self.optimization_history
            }, f, indent=2)
        
        logger.info(f"Saved optimized configuration to {config_path}")
        logger.info(f"Saved optimization history to {history_path}")


class MultiScenarioOptimizer:
    """Optimize parameters across multiple scenarios"""
    
    def __init__(self, strategy_name: str, scenarios: List[int] = None):
        self.strategy_name = strategy_name
        self.scenarios = scenarios or [1]  # Default to scenario 1
        self.results: Dict[int, OptimizationResult] = {}
    
    async def optimize_all_scenarios(self, max_iterations: int = 15) -> Dict[int, OptimizationResult]:
        """Optimize parameters for all specified scenarios"""
        logger.info(f"Multi-scenario optimization for {self.strategy_name} across scenarios: {self.scenarios}")
        
        for scenario_id in self.scenarios:
            logger.info(f"\n=== Optimizing for Scenario {scenario_id} ===")
            
            optimizer = ParameterOptimizer(self.strategy_name, scenario_id)
            result = await optimizer.optimize_parameters(max_iterations=max_iterations)
            
            self.results[scenario_id] = result
            optimizer.save_optimization_result(result)
            
            logger.info(f"Scenario {scenario_id} complete: {result.performance_improvement*100:.1f}% improvement")
        
        return self.results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of multi-scenario optimization"""
        if not self.results:
            return {'error': 'No optimization results available'}
        
        total_improvement = sum(r.performance_improvement for r in self.results.values())
        avg_improvement = total_improvement / len(self.results)
        
        best_scenario = max(self.results.keys(), key=lambda s: self.results[s].performance_improvement)
        worst_scenario = min(self.results.keys(), key=lambda s: self.results[s].performance_improvement)
        
        return {
            'strategy_name': self.strategy_name,
            'scenarios_optimized': list(self.results.keys()),
            'average_improvement': avg_improvement,
            'best_scenario': {
                'scenario_id': best_scenario,
                'improvement': self.results[best_scenario].performance_improvement,
                'rejections_improvement': self.results[best_scenario].avg_rejections_before - self.results[best_scenario].avg_rejections_after
            },
            'worst_scenario': {
                'scenario_id': worst_scenario,
                'improvement': self.results[worst_scenario].performance_improvement,
                'rejections_improvement': self.results[worst_scenario].avg_rejections_before - self.results[worst_scenario].avg_rejections_after
            },
            'per_scenario_results': {
                scenario_id: {
                    'improvement_pct': result.performance_improvement * 100,
                    'rejections_before': result.avg_rejections_before,
                    'rejections_after': result.avg_rejections_after,
                    'confidence': result.confidence
                }
                for scenario_id, result in self.results.items()
            }
        }