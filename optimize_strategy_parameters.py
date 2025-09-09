# ABOUTME: Main script to run parameter optimization for specific strategies across scenarios  
# ABOUTME: Uses Bayesian optimization to find optimal parameter configurations

import asyncio
import argparse
import logging
from pathlib import Path

from berghain.training.parameter_optimizer import ParameterOptimizer, MultiScenarioOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

async def optimize_single_strategy(strategy_name: str, scenario_id: int, max_iterations: int = 20):
    """Optimize parameters for a single strategy on a single scenario"""
    print(f"🔧 Optimizing {strategy_name} for scenario {scenario_id}")
    print(f"   Max iterations: {max_iterations}")
    
    optimizer = ParameterOptimizer(strategy_name, scenario_id)
    
    # Check if we have tunable parameters
    if not optimizer.tunable_params:
        print(f"❌ No tunable parameters found for strategy '{strategy_name}'")
        return None
    
    print(f"   Tunable parameters: {list(optimizer.tunable_params.keys())}")
    
    # Run optimization
    result = await optimizer.optimize_parameters(max_iterations=max_iterations)
    
    # Save results
    optimizer.save_optimization_result(result)
    
    # Print summary
    print(f"\n✅ Optimization Complete!")
    print(f"   Strategy: {result.strategy_name}")
    print(f"   Scenario: {result.scenario_id}")
    print(f"   Improvement: {result.performance_improvement*100:.1f}%")
    print(f"   Rejections: {result.avg_rejections_before:.1f} → {result.avg_rejections_after:.1f}")
    print(f"   Confidence: {result.confidence:.2f}")
    
    print(f"\n📋 Parameter Changes:")
    for param, new_value in result.optimized_params.items():
        old_value = result.original_params.get(param, 'N/A')
        print(f"   {param}: {old_value} → {new_value}")
    
    return result

async def optimize_multi_scenario(strategy_name: str, scenarios: list, max_iterations: int = 15):
    """Optimize parameters for a strategy across multiple scenarios"""
    print(f"🔧 Multi-scenario optimization for {strategy_name}")
    print(f"   Scenarios: {scenarios}")
    print(f"   Max iterations per scenario: {max_iterations}")
    
    optimizer = MultiScenarioOptimizer(strategy_name, scenarios)
    results = await optimizer.optimize_all_scenarios(max_iterations=max_iterations)
    
    # Generate summary report
    summary = optimizer.generate_summary_report()
    
    print(f"\n📊 Multi-Scenario Summary:")
    print(f"   Strategy: {summary['strategy_name']}")
    print(f"   Scenarios optimized: {summary['scenarios_optimized']}")
    print(f"   Average improvement: {summary['average_improvement']*100:.1f}%")
    
    print(f"\n🏆 Best scenario: {summary['best_scenario']['scenario_id']}")
    print(f"   Improvement: {summary['best_scenario']['improvement']*100:.1f}%")
    print(f"   Rejection reduction: {summary['best_scenario']['rejections_improvement']:.1f}")
    
    if summary['worst_scenario']['improvement'] < 0:
        print(f"\n⚠️  Worst scenario: {summary['worst_scenario']['scenario_id']}")
        print(f"   Change: {summary['worst_scenario']['improvement']*100:.1f}%")
    
    print(f"\n📈 Per-Scenario Results:")
    for scenario_id, data in summary['per_scenario_results'].items():
        print(f"   Scenario {scenario_id}: {data['improvement_pct']:.1f}% ({data['rejections_before']:.1f} → {data['rejections_after']:.1f})")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument('strategy', help='Strategy name to optimize')
    parser.add_argument('--scenario', type=int, default=1, help='Scenario ID (default: 1)')
    parser.add_argument('--multi-scenario', nargs='+', type=int, help='Optimize across multiple scenarios')
    parser.add_argument('--iterations', type=int, default=20, help='Max optimization iterations (default: 20)')
    parser.add_argument('--all-key-strategies', action='store_true', help='Optimize all key strategies for scenario 1')
    
    args = parser.parse_args()
    
    if args.all_key_strategies:
        # Optimize all key strategies for scenario 1
        key_strategies = ['rbcr2', 'ultra_elite_lstm', 'constraint_focused_lstm', 'ultimate3h']
        
        print("🚀 Optimizing all key strategies for scenario 1...")
        for strategy in key_strategies:
            try:
                print(f"\n{'='*60}")
                await optimize_single_strategy(strategy, 1, args.iterations)
            except Exception as e:
                print(f"❌ Failed to optimize {strategy}: {e}")
        
        print(f"\n🎉 All optimizations complete!")
        print(f"Check berghain/config/strategies/optimized/ for results")
        
    elif args.multi_scenario:
        # Multi-scenario optimization
        await optimize_multi_scenario(args.strategy, args.multi_scenario, args.iterations)
        
    else:
        # Single strategy, single scenario
        await optimize_single_strategy(args.strategy, args.scenario, args.iterations)

if __name__ == "__main__":
    asyncio.run(main())