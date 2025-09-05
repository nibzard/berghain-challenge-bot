#!/usr/bin/env python3
"""
ABOUTME: Evaluation script for trained RL models
ABOUTME: Compares RL performance against existing strategies
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
import json
from datetime import datetime

from berghain.runner.game_executor import GameExecutor
from berghain.solvers.rl_lstm_solver import create_rl_lstm_solver, create_rl_lstm_hybrid_solver
from berghain.solvers.ogds_solver import OGDSSolver
from berghain.analysis.comparison import ComparisonAnalyzer


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def evaluate_strategy(
    strategy_name: str,
    solver_factory,
    scenario: int,
    num_games: int,
    workers: int = 1
) -> Dict:
    """
    Evaluate a strategy across multiple games.
    
    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {strategy_name} with {num_games} games")
    
    # Create executor
    executor = GameExecutor()
    
    # Run games
    results = executor.run_batch_games(
        solvers=[solver_factory()],
        scenario=scenario,
        count=num_games,
        workers=workers
    )
    
    if not results:
        logger.error(f"No results for {strategy_name}")
        return {}
    
    # Calculate statistics
    success_rate = sum(1 for r in results if r.success) / len(results)
    rejection_counts = [r.game_state.rejected_count for r in results]
    admission_counts = [r.game_state.admitted_count for r in results]
    durations = [r.duration for r in results if r.duration]
    
    # Constraint satisfaction analysis
    constraint_satisfaction = {}
    for result in results:
        if result.success:
            summary = result.constraint_satisfaction_summary()
            for attr, data in summary.items():
                if attr not in constraint_satisfaction:
                    constraint_satisfaction[attr] = []
                constraint_satisfaction[attr].append(data['current'])
    
    stats = {
        'strategy': strategy_name,
        'num_games': len(results),
        'success_rate': success_rate,
        'success_count': sum(1 for r in results if r.success),
        
        # Rejection statistics
        'rejection_mean': np.mean(rejection_counts),
        'rejection_std': np.std(rejection_counts),
        'rejection_median': np.median(rejection_counts),
        'rejection_min': np.min(rejection_counts),
        'rejection_max': np.max(rejection_counts),
        
        # Admission statistics
        'admission_mean': np.mean(admission_counts),
        'admission_std': np.std(admission_counts),
        
        # Duration statistics
        'duration_mean': np.mean(durations) if durations else 0,
        'duration_std': np.std(durations) if durations else 0,
        
        # Constraint satisfaction
        'constraint_stats': {
            attr: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            } for attr, values in constraint_satisfaction.items()
        }
    }
    
    logger.info(f"{strategy_name}: Success rate = {success_rate:.2%}, "
               f"Avg rejections = {stats['rejection_mean']:.1f} ± {stats['rejection_std']:.1f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL models")
    
    # Evaluation parameters
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained RL model')
    parser.add_argument('--scenario', type=int, default=1,
                       help='Game scenario to evaluate on')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games to run for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    
    # Comparison strategies
    parser.add_argument('--compare-with', nargs='+', default=['ogds'],
                       choices=['ogds', 'ultimate', 'greedy', 'adaptive'],
                       help='Baseline strategies to compare against')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save-detailed', action='store_true',
                       help='Save detailed game results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    # RL model options
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for RL inference')
    parser.add_argument('--eval-hybrid', action='store_true',
                       help='Also evaluate hybrid RL strategy')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting RL model evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Games per strategy: {args.num_games}")
    
    # Define strategies to evaluate
    strategies = {}
    
    # RL strategies
    strategies['rl_lstm'] = lambda: create_rl_lstm_solver(
        model_path=str(args.model_path),
        device=args.device
    )
    
    if args.eval_hybrid:
        strategies['rl_lstm_hybrid'] = lambda: create_rl_lstm_hybrid_solver(
            model_path=str(args.model_path),
            device=args.device
        )
    
    # Baseline strategies
    if 'ogds' in args.compare_with:
        strategies['ogds'] = lambda: OGDSSolver()
    
    # Add other baseline strategies as needed
    # TODO: Add factories for other strategies when needed
    
    # Run evaluations
    results = {}
    for strategy_name, solver_factory in strategies.items():
        try:
            result = evaluate_strategy(
                strategy_name=strategy_name,
                solver_factory=solver_factory,
                scenario=args.scenario,
                num_games=args.num_games,
                workers=args.workers
            )
            results[strategy_name] = result
        except Exception as e:
            logger.error(f"Error evaluating {strategy_name}: {e}")
            results[strategy_name] = {'error': str(e)}
    
    # Create comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary table
    summary_data = []
    for strategy, stats in results.items():
        if 'error' not in stats:
            summary_data.append({
                'Strategy': strategy,
                'Success Rate': f"{stats['success_rate']:.2%}",
                'Success Count': f"{stats['success_count']}/{stats['num_games']}",
                'Avg Rejections': f"{stats['rejection_mean']:.1f} ± {stats['rejection_std']:.1f}",
                'Median Rejections': f"{stats['rejection_median']:.0f}",
                'Best Rejection': f"{stats['rejection_min']:.0f}",
                'Avg Duration (s)': f"{stats['duration_mean']:.1f}" if stats['duration_mean'] > 0 else 'N/A'
            })
    
    # Save results
    summary_file = output_dir / f"rl_evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create human-readable report
    report_file = output_dir / f"rl_evaluation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(f"RL Model Evaluation Report\n")
        f.write(f"========================\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Scenario: {args.scenario}\n")
        f.write(f"Games per Strategy: {args.num_games}\n")
        f.write(f"Workers: {args.workers}\n\n")
        
        f.write("Results Summary:\n")
        f.write("================\n")
        
        # Sort strategies by success rate
        sorted_results = sorted(
            [(name, stats) for name, stats in results.items() if 'error' not in stats],
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        for strategy, stats in sorted_results:
            f.write(f"\n{strategy.upper()}:\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2%} ({stats['success_count']}/{stats['num_games']})\n")
            f.write(f"  Rejection Count: {stats['rejection_mean']:.1f} ± {stats['rejection_std']:.1f} (median: {stats['rejection_median']:.0f})\n")
            f.write(f"  Best Performance: {stats['rejection_min']:.0f} rejections\n")
            f.write(f"  Worst Performance: {stats['rejection_max']:.0f} rejections\n")
            if stats['duration_mean'] > 0:
                f.write(f"  Average Duration: {stats['duration_mean']:.1f} ± {stats['duration_std']:.1f} seconds\n")
            
            # Constraint satisfaction details
            if stats['constraint_stats']:
                f.write(f"  Constraint Satisfaction:\n")
                for attr, cstats in stats['constraint_stats'].items():
                    f.write(f"    {attr}: {cstats['mean']:.1f} ± {cstats['std']:.1f} (range: {cstats['min']:.0f}-{cstats['max']:.0f})\n")
        
        # Comparison analysis
        if len(sorted_results) > 1:
            f.write(f"\nComparison Analysis:\n")
            f.write(f"====================\n")
            
            rl_results = [(name, stats) for name, stats in sorted_results if name.startswith('rl_')]
            baseline_results = [(name, stats) for name, stats in sorted_results if not name.startswith('rl_')]
            
            if rl_results and baseline_results:
                best_rl = max(rl_results, key=lambda x: x[1]['success_rate'])
                best_baseline = max(baseline_results, key=lambda x: x[1]['success_rate'])
                
                f.write(f"Best RL Strategy: {best_rl[0]} ({best_rl[1]['success_rate']:.2%} success)\n")
                f.write(f"Best Baseline Strategy: {best_baseline[0]} ({best_baseline[1]['success_rate']:.2%} success)\n")
                
                improvement = best_rl[1]['success_rate'] - best_baseline[1]['success_rate']
                f.write(f"RL Improvement: {improvement:+.1%}\n")
                
                rejection_improvement = best_baseline[1]['rejection_mean'] - best_rl[1]['rejection_mean']
                f.write(f"Rejection Count Improvement: {rejection_improvement:+.1f} (lower is better)\n")
    
    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {summary_file}")
    print(f"Report saved to: {report_file}")
    
    print(f"\nQuick Summary:")
    for strategy, stats in sorted(results.items(), key=lambda x: x[1].get('success_rate', 0), reverse=True):
        if 'error' not in stats:
            print(f"  {strategy}: {stats['success_rate']:.2%} success, {stats['rejection_mean']:.1f} avg rejections")
        else:
            print(f"  {strategy}: ERROR - {stats['error']}")


if __name__ == "__main__":
    main()