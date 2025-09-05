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


def run_games(args):
    """Run games with specified strategy and scenario."""
    config_manager = ConfigManager()
    api_client = BerghainAPIClient()
    
    # Load scenario configuration
    scenario_config = config_manager.get_scenario_config(args.scenario)
    if not scenario_config:
        print(f"❌ Scenario {args.scenario} not found")
        return
    
    # Get strategy configuration
    strategy_config = config_manager.get_strategy_config(args.strategy)
    if not strategy_config:
        print(f"❌ Strategy '{args.strategy}' not found")
        return
    
    print(f"🎯 Running {args.count} games - Scenario {args.scenario} with {args.strategy} strategy")
    
    # Set up parallel runner
    runner = ParallelRunner(
        max_workers=args.workers
    )
    
    # Create game tasks
    from berghain.runner.parallel_runner import GameTask
    tasks = []
    for i in range(args.count):
        tasks.append(GameTask(
            scenario_id=args.scenario,
            strategy_name=args.strategy,
            solver_id=f"{args.strategy}_{i:03d}",
            strategy_params=strategy_config
        ))
    
    # Run games
    batch_result = runner.run_batch(tasks)
    
    # Print summary
    print(f"\n📊 Batch Complete:")
    print(f"   Total games: {len(batch_result.results)}")
    print(f"   Successful: {batch_result.successful_count} ({batch_result.success_rate*100:.1f}%)")
    print(f"   Duration: {batch_result.duration:.1f}s")
    
    if batch_result.best_result:
        print(f"   Best result: {batch_result.best_result.game_state.rejected_count} rejections")


async def monitor_games(args):
    """Start the TUI dashboard for monitoring games."""
    print("🖥️  Starting TUI Dashboard...")
    
    if args.file_watch:
        # File-based monitoring
        log_watcher = GameLogWatcher(logs_directory="game_logs")
        await log_watcher.start_watching()
    else:
        # TUI dashboard
        dashboard = TUIDashboard()
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
    run_parser.add_argument('--strategy', default='conservative', help='Strategy name (default: conservative)')
    run_parser.add_argument('--count', type=int, default=10, help='Number of games to run (default: 10)')
    run_parser.add_argument('--workers', type=int, default=3, help='Parallel workers (default: 3)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring dashboard')
    monitor_parser.add_argument('--file-watch', action='store_true', help='Use file watching instead of TUI')
    
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


if __name__ == "__main__":
    main()