# ABOUTME: This script compares the optimized vs simple algorithms across scenarios
# ABOUTME: Shows the improvement in rejection minimization

from berghain_bot import BerghainBot
from berghain_bot_simple import BerghainBotSimple
import sys
import time

def run_comparison(scenario: int, num_runs: int = 1):
    """Compare optimized vs simple algorithm on a scenario."""
    
    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario} COMPARISON")
    print(f"{'='*60}")
    
    optimized_results = []
    simple_results = []
    
    for run in range(num_runs):
        if num_runs > 1:
            print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Test optimized algorithm
        print(f"\n🧠 Testing OPTIMIZED Algorithm...")
        optimized_bot = BerghainBot()
        optimized_bot.logger.disabled = True  # Disable verbose logging for comparison
        
        start_time = time.time()
        try:
            result_opt = optimized_bot.play_game(scenario)
            optimized_results.append({
                'success': result_opt['status'] == 'completed',
                'rejections': result_opt['rejected_count'],
                'time': time.time() - start_time
            })
        except Exception as e:
            print(f"❌ Optimized algorithm failed: {e}")
            optimized_results.append({
                'success': False,
                'rejections': 20000,
                'time': time.time() - start_time
            })
        
        # Test simple algorithm  
        print(f"\n🤖 Testing SIMPLE Algorithm...")
        simple_bot = BerghainBotSimple()
        simple_bot.logger.disabled = True  # Disable verbose logging for comparison
        
        start_time = time.time()
        try:
            result_simple = simple_bot.play_game(scenario)
            simple_results.append({
                'success': result_simple['status'] == 'completed',
                'rejections': result_simple['rejected_count'],
                'time': time.time() - start_time
            })
        except Exception as e:
            print(f"❌ Simple algorithm failed: {e}")
            simple_results.append({
                'success': False,
                'rejections': 20000,
                'time': time.time() - start_time
            })
    
    # Calculate statistics
    opt_successes = sum(1 for r in optimized_results if r['success'])
    simple_successes = sum(1 for r in simple_results if r['success'])
    
    if opt_successes > 0:
        opt_avg_rejections = sum(r['rejections'] for r in optimized_results if r['success']) / opt_successes
        opt_avg_time = sum(r['time'] for r in optimized_results) / len(optimized_results)
    else:
        opt_avg_rejections = 20000
        opt_avg_time = sum(r['time'] for r in optimized_results) / len(optimized_results)
    
    if simple_successes > 0:
        simple_avg_rejections = sum(r['rejections'] for r in simple_results if r['success']) / simple_successes
        simple_avg_time = sum(r['time'] for r in simple_results) / len(simple_results)
    else:
        simple_avg_rejections = 20000
        simple_avg_time = sum(r['time'] for r in simple_results) / len(simple_results)
    
    # Display results
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"OPTIMIZED Algorithm:")
    print(f"  Success Rate: {opt_successes}/{num_runs} ({100*opt_successes/num_runs:.1f}%)")
    print(f"  Avg Rejections (successful runs): {opt_avg_rejections:.1f}")
    print(f"  Avg Time per game: {opt_avg_time:.1f}s")
    
    print(f"\nSIMPLE Algorithm:")
    print(f"  Success Rate: {simple_successes}/{num_runs} ({100*simple_successes/num_runs:.1f}%)")
    print(f"  Avg Rejections (successful runs): {simple_avg_rejections:.1f}")
    print(f"  Avg Time per game: {simple_avg_time:.1f}s")
    
    # Calculate improvement
    if simple_successes > 0 and opt_successes > 0:
        rejection_improvement = ((simple_avg_rejections - opt_avg_rejections) / simple_avg_rejections) * 100
        print(f"\n🎯 IMPROVEMENT:")
        print(f"  Rejection Reduction: {rejection_improvement:+.1f}%")
        if rejection_improvement > 0:
            print(f"  🏆 OPTIMIZED algorithm performs {rejection_improvement:.1f}% better!")
        else:
            print(f"  📉 Simple algorithm performs {abs(rejection_improvement):.1f}% better")
    
    return {
        'optimized': optimized_results,
        'simple': simple_results,
        'improvement': rejection_improvement if 'rejection_improvement' in locals() else 0
    }

def main():
    if len(sys.argv) > 1:
        scenario = int(sys.argv[1])
        scenarios = [scenario]
    else:
        scenarios = [1, 2, 3]
    
    overall_improvements = []
    
    for scenario in scenarios:
        result = run_comparison(scenario, num_runs=1)  # Single run due to time constraints
        if result['improvement'] != 0:
            overall_improvements.append(result['improvement'])
    
    if overall_improvements:
        avg_improvement = sum(overall_improvements) / len(overall_improvements)
        print(f"\n🎊 OVERALL IMPROVEMENT: {avg_improvement:.1f}% reduction in rejections")
        
        if avg_improvement > 20:
            print("🚀 EXCELLENT optimization! Ready for Berghain!")
        elif avg_improvement > 10:
            print("✅ Good optimization! Significant improvement achieved.")
        elif avg_improvement > 0:
            print("📈 Positive optimization. Room for further improvements.")
        else:
            print("🔧 Optimization needs tuning. Consider adjusting parameters.")

if __name__ == "__main__":
    main()