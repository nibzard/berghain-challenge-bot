# ABOUTME: Test script to evaluate trained strategy controller performance against baselines
# ABOUTME: Compares hybrid transformer with trained controller vs individual strategies

import asyncio
import logging
from pathlib import Path
from typing import Dict, List

from berghain.runner import ParallelRunner
from berghain.runner.parallel_runner import GameTask
from berghain.config import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

async def test_strategy_performance(strategy_name: str, scenario_id: int, num_games: int = 10) -> Dict:
    """Test performance of a single strategy"""
    config_manager = ConfigManager()
    
    # Load strategy configuration
    strategy_config = config_manager.get_strategy_config(strategy_name)
    if not strategy_config:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    # Set up parallel runner
    runner = ParallelRunner(max_workers=min(num_games, 5))
    
    # Create game tasks
    tasks = []
    for i in range(num_games):
        tasks.append(GameTask(
            scenario_id=scenario_id,
            strategy_name=strategy_name,
            solver_id=f"{strategy_name}_test_{i:03d}",
            strategy_params=strategy_config,
            enable_high_score_check=False,
            mode='local'
        ))
    
    # Run games
    logger.info(f"Testing {strategy_name} with {num_games} games on scenario {scenario_id}")
    batch_result = runner.run_batch(tasks)
    
    # Calculate metrics
    successful_results = [r for r in batch_result.results if r.success]
    
    if not successful_results:
        return {
            'strategy': strategy_name,
            'scenario': scenario_id,
            'games_played': num_games,
            'success_rate': 0.0,
            'avg_rejections': None,
            'min_rejections': None,
            'max_rejections': None,
            'std_rejections': None,
            'status': 'FAILED - No successful games'
        }
    
    rejections = [r.game_state.rejected_count for r in successful_results]
    avg_rejections = sum(rejections) / len(rejections)
    min_rejections = min(rejections)
    max_rejections = max(rejections)
    
    # Calculate standard deviation
    variance = sum((r - avg_rejections) ** 2 for r in rejections) / len(rejections)
    std_rejections = variance ** 0.5
    
    success_rate = len(successful_results) / num_games
    
    result = {
        'strategy': strategy_name,
        'scenario': scenario_id,
        'games_played': num_games,
        'successful_games': len(successful_results),
        'success_rate': success_rate,
        'avg_rejections': avg_rejections,
        'min_rejections': min_rejections,
        'max_rejections': max_rejections,
        'std_rejections': std_rejections,
        'status': 'SUCCESS' if success_rate > 0.8 else 'PARTIAL' if success_rate > 0.0 else 'FAILED'
    }
    
    return result

async def compare_strategies(strategies: List[str], scenario_id: int = 1, num_games: int = 10):
    """Compare multiple strategies head-to-head"""
    print(f"🎯 Strategy Performance Comparison - Scenario {scenario_id}")
    print(f"   Games per strategy: {num_games}")
    print(f"   Strategies: {', '.join(strategies)}")
    print()
    
    results = []
    
    for strategy in strategies:
        try:
            result = await test_strategy_performance(strategy, scenario_id, num_games)
            results.append(result)
            
            # Print individual result
            if result['success_rate'] > 0:
                print(f"✅ {strategy}")
                print(f"   Success: {result['success_rate']*100:.1f}% ({result['successful_games']}/{result['games_played']})")
                print(f"   Rejections: {result['avg_rejections']:.1f} ± {result['std_rejections']:.1f}")
                print(f"   Range: {result['min_rejections']}-{result['max_rejections']}")
            else:
                print(f"❌ {strategy}")
                print(f"   Status: {result['status']}")
            print()
            
        except Exception as e:
            print(f"❌ {strategy}: Error - {e}")
            print()
            results.append({
                'strategy': strategy,
                'scenario': scenario_id,
                'status': f'ERROR: {e}',
                'success_rate': 0.0,
                'avg_rejections': None
            })
    
    # Summary comparison
    successful_results = [r for r in results if r['success_rate'] > 0]
    
    if successful_results:
        print("📊 Performance Ranking (by average rejections):")
        
        # Sort by average rejections (lower is better)
        successful_results.sort(key=lambda x: x['avg_rejections'])
        
        for i, result in enumerate(successful_results, 1):
            print(f"   {i}. {result['strategy']}: {result['avg_rejections']:.1f} rejections")
            
        print()
        
        # Calculate improvements vs baseline (assume first strategy is baseline)
        if len(successful_results) > 1:
            baseline = successful_results[0]  # Best performing strategy
            baseline_score = baseline['avg_rejections']
            
            print(f"💡 Performance vs Best ({baseline['strategy']}):")
            for result in successful_results[1:]:
                diff = result['avg_rejections'] - baseline_score
                pct_diff = (diff / baseline_score) * 100
                print(f"   {result['strategy']}: +{diff:.1f} rejections (+{pct_diff:.1f}%)")
            print()
    
    return results

async def comprehensive_evaluation():
    """Run comprehensive evaluation of key strategies"""
    
    # Test trained hybrid transformer
    trained_model_path = "models/strategy_controller/trained_strategy_controller.pt"
    
    if not Path(trained_model_path).exists():
        print("⚠️  Trained model not found at models/strategy_controller/trained_strategy_controller.pt")
        print("   Run: python -m berghain.training.train_strategy_controller")
        print()
    
    # Strategies to compare
    strategies_to_test = [
        'hybrid_transformer',  # Our trained controller
        'rbcr2',               # Current baseline
        'ultra_elite_lstm',    # Best LSTM
        'constraint_focused_lstm',  # Constraint-focused LSTM
        'ultimate3h',          # Ultimate3H
        'dual_deficit'         # Dual deficit
    ]
    
    print("🚀 Comprehensive Strategy Evaluation")
    print("=" * 60)
    
    # Test on scenario 1 first
    results = await compare_strategies(strategies_to_test, scenario_id=1, num_games=15)
    
    # Show summary
    print("📋 Final Summary:")
    print("=" * 40)
    
    successful_results = [r for r in results if r['success_rate'] > 0.8]  # Only highly successful strategies
    
    if successful_results:
        # Sort by performance
        successful_results.sort(key=lambda x: x['avg_rejections'])
        
        print(f"🏆 Top Performing Strategies:")
        for i, result in enumerate(successful_results[:3], 1):
            print(f"   {i}. {result['strategy']}: {result['avg_rejections']:.1f} ± {result['std_rejections']:.1f} rejections")
        
        # Check if hybrid transformer is in top 3
        hybrid_result = next((r for r in results if r['strategy'] == 'hybrid_transformer'), None)
        if hybrid_result and hybrid_result in successful_results[:3]:
            rank = successful_results.index(hybrid_result) + 1
            print(f"\n🎉 Hybrid Transformer ranked #{rank}!")
        elif hybrid_result and hybrid_result['success_rate'] > 0:
            print(f"\n📈 Hybrid Transformer: {hybrid_result['avg_rejections']:.1f} rejections (needs improvement)")
        else:
            print(f"\n⚠️  Hybrid Transformer failed to complete games successfully")
    
    return results

async def quick_test():
    """Quick test of just hybrid transformer vs rbcr2"""
    strategies = ['hybrid_transformer', 'rbcr2']
    
    print("⚡ Quick Performance Test")
    print("=" * 30)
    
    return await compare_strategies(strategies, scenario_id=1, num_games=5)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        asyncio.run(quick_test())
    else:
        asyncio.run(comprehensive_evaluation())