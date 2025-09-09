# ABOUTME: Main entry point for Berghain Challenge bot
# ABOUTME: Provides CLI interface for running games, monitoring, and analysis

import asyncio
import argparse
import logging
from pathlib import Path

from berghain.runner import ParallelRunner
from berghain.monitoring import TUIDashboard, GameLogWatcher
from berghain.analysis import GameAnalyzer, StatisticalAnalyzer
from berghain.config import ConfigManager
from berghain.core import BerghainAPIClient
from berghain.optimization import DynamicStrategyRunner, DynamicRunConfig


def run_games(args):
    """Run games with specified strategy/strategies and scenario."""
    config_manager = ConfigManager()
    api_client = BerghainAPIClient()
    
    # Load scenario configuration
    scenario_config = config_manager.get_scenario_config(args.scenario)
    if not scenario_config:
        print(f"❌ Scenario {args.scenario} not found")
        return
    
    # Parse strategies (comma-separated, single, or "all")
    if args.strategy.lower() == "all":
        strategies = config_manager.list_available_strategies()
        if not strategies:
            print(f"❌ No strategies found in config directory")
            return
        print(f"🎯 Using all available strategies: {', '.join(strategies)}")
    elif isinstance(args.strategy, list):
        strategies = args.strategy
    else:
        strategies = [s.strip() for s in args.strategy.split(',')]
    
    # Validate all strategies exist and apply CLI overrides
    strategy_configs = {}
    for strategy in strategies:
        config = config_manager.get_strategy_config(strategy)
        if not config:
            print(f"❌ Strategy '{strategy}' not found")
            return
        
        # Apply CLI temperature override if provided
        if args.temp is not None:
            if 'parameters' not in config:
                config['parameters'] = {}
            config['parameters']['temperature'] = args.temp
            print(f"🌡️  Temperature override: {args.temp} for {strategy} strategy")
        
        strategy_configs[strategy] = config
    
    total_games = args.count * len(strategies)
    if len(strategies) > 1:
        print(f"🎯 Running {total_games} games - Scenario {args.scenario} with {len(strategies)} strategies: {', '.join(strategies)}")
    else:
        print(f"🎯 Running {args.count} games - Scenario {args.scenario} with {strategies[0]} strategy")
    
    # Show high score checking status
    enable_high_score_check = not args.no_high_score_check
    if enable_high_score_check:
        config_manager = ConfigManager()
        threshold = config_manager.get_high_score_threshold(args.scenario)
        if threshold:
            buffer_pct = config_manager.get_buffer_percentage()
            effective_threshold = int(threshold * buffer_pct)
            print(f"🏆 High score checking enabled: will stop at {effective_threshold} rejections ({buffer_pct*100:.0f}% of record {threshold})")
        else:
            print(f"🏆 High score checking enabled but no threshold found for scenario {args.scenario}")
    else:
        print(f"⏭️  High score checking disabled")
    
    # Set up parallel runner
    runner = ParallelRunner(
        max_workers=args.workers
    )
    
    # Create game tasks for each strategy
    from berghain.runner.parallel_runner import GameTask
    tasks = []
    enable_high_score_check = not args.no_high_score_check
    for strategy in strategies:
        for i in range(args.count):
            tasks.append(GameTask(
                scenario_id=args.scenario,
                strategy_name=strategy,
                solver_id=f"{strategy}_{i:03d}",
                strategy_params=strategy_configs[strategy],
                enable_high_score_check=enable_high_score_check,
                mode=args.mode
            ))
    
    # Run games
    batch_result = runner.run_batch(tasks)
    
    # Print summary
    print(f"\n📊 Batch Complete:")
    print(f"   Total games: {len(batch_result.results)}")
    print(f"   Successful: {batch_result.successful_count} ({batch_result.success_rate*100:.1f}%)")
    print(f"   Duration: {batch_result.total_duration:.1f}s")
    
    if batch_result.best_result:
        print(f"   Best result: {batch_result.best_result.game_state.rejected_count} rejections")
    
    # Show per-strategy breakdown if multiple strategies
    if len(strategies) > 1:
        print(f"\n📈 Per-Strategy Results:")
        from collections import defaultdict
        strategy_results = defaultdict(list)
        
        for result in batch_result.results:
            strategy_name = result.solver_id.rsplit('_', 1)[0]  # Extract strategy from solver_id
            strategy_results[strategy_name].append(result)
        
        for strategy in strategies:
            results = strategy_results[strategy]
            if results:
                successful = sum(1 for r in results if r.success)
                success_rate = successful / len(results) if results else 0
                avg_rejections = sum(r.game_state.rejected_count for r in results if r.success) / max(1, successful)
                print(f"   {strategy}: {success_rate:.1%} success ({successful}/{len(results)}), avg {avg_rejections:.0f} rejections")


async def monitor_games(args):
    """Start the TUI dashboard for monitoring games."""
    print("🖥️  Starting TUI Dashboard...")
    
    if args.file_watch:
        # File-based monitoring
        log_watcher = GameLogWatcher(
            logs_directory="game_logs", 
            process_existing=getattr(args, 'process_existing', False),
            max_existing_files=getattr(args, 'max_existing', 20)
        )
        log_watcher.start_watching()
    else:
        # TUI dashboard
        dashboard = TUIDashboard(
            process_existing=getattr(args, 'process_existing', False),
            max_existing_files=getattr(args, 'max_existing', 20)
        )
        await dashboard.run()


def analyze_games(args):
    """Analyze recent games or compare strategies."""
    analyzer = GameAnalyzer(logs_directory="game_logs")
    stat_analyzer = StatisticalAnalyzer(logs_directory="game_logs")
    
    if args.compare:
        # Strategy comparison
        strategy_a, strategy_b = args.compare.split(',')
        print(f"📊 Comparing {strategy_a} vs {strategy_b}")
        
        result = stat_analyzer.compare_strategies(
            strategy_a=strategy_a.strip(),
            strategy_b=strategy_b.strip(),
            scenario=args.scenario
        )
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        print(f"\n🆚 Strategy Comparison:")
        print(f"   {result['strategy_a']}: {result['success_rate_a']:.1%} success rate ({result['games_a']} games)")
        print(f"   {result['strategy_b']}: {result['success_rate_b']:.1%} success rate ({result['games_b']} games)")
        
        if "rejection_comparison" in result:
            rc = result["rejection_comparison"]
            print(f"\n🎯 Rejection Counts (successful games only):")
            print(f"   {result['strategy_a']}: {rc['mean_rejections_a']:.1f} ± {rc['std_rejections_a']:.1f}")
            print(f"   {result['strategy_b']}: {rc['mean_rejections_b']:.1f} ± {rc['std_rejections_b']:.1f}")
            
            if rc["significant_difference"]:
                print(f"   🏆 Winner: {rc['better_strategy']} (p={rc['p_value']:.4f})")
            else:
                print(f"   🤷 No significant difference (p={rc['p_value']:.4f})")
    
    elif args.parameter:
        # Parameter impact analysis
        print(f"📈 Analyzing parameter '{args.parameter}' impact")
        
        result = stat_analyzer.analyze_parameter_impact(
            strategy=args.strategy,
            parameter_name=args.parameter,
            scenario=args.scenario
        )
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        print(f"\n📊 Parameter Analysis for '{args.parameter}':")
        print(f"   Range: {result['parameter_range']['min']:.3f} - {result['parameter_range']['max']:.3f}")
        print(f"   Mean: {result['parameter_range']['mean']:.3f} ± {result['parameter_range']['std']:.3f}")
        
        if "correlation_analysis" in result:
            ca = result["correlation_analysis"]
            print(f"\n🔗 Correlations:")
            print(f"   Success rate: r={ca['success_correlation']:.3f} (p={ca['success_p_value']:.4f})")
            if ca["rejection_correlation"] is not None:
                print(f"   Rejection count: r={ca['rejection_correlation']:.3f} (p={ca['rejection_p_value']:.4f})")
    
    else:
        # General analysis
        print(f"📊 Analyzing last {args.limit} games")
        
        result = analyzer.analyze_recent_games(limit=args.limit)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        summary = result["summary"]
        print(f"\n📈 Recent Games Summary:")
        print(f"   Total games: {summary['total_games']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Avg rejections (all): {summary['avg_rejections_all']:.1f}")
        print(f"   Avg rejections (successful): {summary['avg_rejections_successful']:.1f}")
        
        # Best scenarios
        print(f"\n🎯 By Scenario:")
        for scenario, data in result["by_scenario"].items():
            print(f"   Scenario {scenario}: {data['success_rate']:.1%} success ({data['successful_games']}/{data['total_games']})")
        
        # Best solvers
        print(f"\n🤖 By Solver:")
        for solver, data in result["by_solver"].items():
            print(f"   {solver}: {data['success_rate']:.1%} success ({data['successful_games']}/{data['total_games']})")
        
        # Recommendations
        if result["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in result["recommendations"]:
                print(f"   • {rec}")


def generate_report(args):
    """Generate comprehensive performance report."""
    stat_analyzer = StatisticalAnalyzer(logs_directory="game_logs")
    
    print(f"📋 Generating performance report ({args.days} days)")
    
    result = stat_analyzer.generate_performance_report(days_back=args.days)
    
    if "error" in result:
        print(f"❌ {result['error']}")
        return
    
    print(f"\n📊 Performance Report - {result['period']}")
    print(f"   Total games: {result['total_games']}")
    print(f"   Successful games: {result['successful_games']}")
    print(f"   Overall success rate: {result['overall_success_rate']:.1%}")
    
    if "best_performance" in result:
        best = result["best_performance"]
        print(f"\n🏆 Best Performance:")
        print(f"   Solver: {best['solver_id']}")
        print(f"   Scenario: {best['scenario']}")
        print(f"   Rejections: {best['rejected_count']}")
        print(f"   Strategy: {best['strategy_params']}")
    
    if "trend_analysis" in result:
        trend = result["trend_analysis"]
        print(f"\n📈 Trend Analysis:")
        print(f"   Early success rate: {trend['early_success_rate']:.1%}")
        print(f"   Recent success rate: {trend['late_success_rate']:.1%}")
        improvement = trend['improvement']
        if improvement > 0:
            print(f"   📈 Improving by {improvement:.1%}")
        elif improvement < 0:
            print(f"   📉 Declining by {abs(improvement):.1%}")
        else:
            print(f"   ➡️  Stable performance")


async def optimize_strategies(args):
    """Run dynamic strategy optimization with evolution."""
    print(f"🧬 Starting strategy optimization for scenario {args.scenario}")
    print(f"   Workers: {args.workers}")
    print(f"   Generations: {args.generations}")
    print(f"   Games per strategy: {args.games_per_strategy}")
    
    config = DynamicRunConfig(
        scenario_id=args.scenario,
        max_concurrent_games=args.workers,
        max_generations=args.generations,
        min_games_per_strategy=args.games_per_strategy,
        target_success_rate=args.target_success_rate / 100.0
    )
    
    runner = DynamicStrategyRunner(config)
    await runner.run_dynamic_optimization()
    
    print(f"\n🎉 Strategy optimization completed!")
    print(f"   Check berghain/config/strategies/evolved/ for new strategies")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Berghain Challenge Bot")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run games')
    run_parser.add_argument('--scenario', type=int, default=1, help='Scenario ID (default: 1)')
    run_parser.add_argument('--strategy', default='conservative', help='Strategy name(s) - comma-separated for multiple, or "all" for all available (default: conservative)')
    run_parser.add_argument('--count', type=int, default=10, help='Number of games per strategy (default: 10)')
    run_parser.add_argument('--workers', type=int, default=10, help='Parallel workers (default: 10, API calls limited to 10 concurrent)')
    run_parser.add_argument('--no-high-score-check', action='store_true', help='Disable high score checking for early termination')
    run_parser.add_argument('--mode', choices=['local','api'], default='local', help='Backend mode: local simulator or live API (default: local)')
    run_parser.add_argument('--temp', type=float, help='Temperature for transformer strategy (overrides config file)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring dashboard')
    monitor_parser.add_argument('--file-watch', action='store_true', help='Use file watching instead of TUI')
    monitor_parser.add_argument('--process-existing', action='store_true', help='Process existing log files on startup (slower but shows historical games)')
    monitor_parser.add_argument('--max-existing', type=int, default=20, help='Maximum existing files to process (default: 20)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze game results')
    analyze_parser.add_argument('--limit', type=int, default=20, help='Number of recent games (default: 20)')
    analyze_parser.add_argument('--scenario', type=int, help='Filter by scenario')
    analyze_parser.add_argument('--compare', help='Compare two strategies (e.g., "conservative,aggressive")')
    analyze_parser.add_argument('--parameter', help='Analyze parameter impact')
    analyze_parser.add_argument('--strategy', help='Strategy for parameter analysis')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--days', type=int, default=7, help='Days to include (default: 7)')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run dynamic strategy optimization')
    optimize_parser.add_argument('--scenario', type=int, default=1, help='Scenario ID (default: 1)')
    optimize_parser.add_argument('--workers', type=int, default=10, help='Concurrent games (default: 10, API calls limited to 10 concurrent)')
    optimize_parser.add_argument('--generations', type=int, default=5, help='Number of generations (default: 5)')
    optimize_parser.add_argument('--games-per-strategy', type=int, default=3, help='Games per strategy (default: 3)')
    optimize_parser.add_argument('--target-success-rate', type=float, default=80, help='Target success rate percentage (default: 80)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'run':
        run_games(args)
    elif args.command == 'monitor':
        asyncio.run(monitor_games(args))
    elif args.command == 'analyze':
        analyze_games(args)
    elif args.command == 'report':
        generate_report(args)
    elif args.command == 'optimize':
        asyncio.run(optimize_strategies(args))


if __name__ == "__main__":
    main()
