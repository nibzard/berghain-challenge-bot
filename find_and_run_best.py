#!/usr/bin/env python3
"""
ABOUTME: Automatically find and run the best available strategy for record attempts
ABOUTME: Detects working strategies and runs them systematically for record breaking
"""

import subprocess
import json
import time
from pathlib import Path
import glob
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_strategy(strategy: str) -> dict:
    """Test if a strategy works and get its baseline performance."""
    logger.info(f"🧪 Testing strategy: {strategy}")
    
    try:
        # Try to run one game with this strategy
        cmd = ["python", "main.py", "run", "--strategy", strategy, "--count", "1", "--scenario", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
        
        if result.returncode == 0:
            # Strategy worked, check the result
            game_logs = glob.glob(f"game_logs/game_{strategy}_*.json") + glob.glob(f"game_logs/*{strategy}*.json")
            
            if game_logs:
                latest_log = max(game_logs, key=lambda x: Path(x).stat().st_mtime)
                
                with open(latest_log, 'r') as f:
                    game_data = json.load(f)
                
                rejections = game_data.get('rejected_count', float('inf'))
                success = game_data.get('success', False)
                
                logger.info(f"✅ {strategy}: {rejections} rejections, success: {success}")
                
                return {
                    'strategy': strategy,
                    'works': True,
                    'rejections': rejections,
                    'success': success,
                    'log_file': latest_log
                }
            else:
                logger.warning(f"⚠️  {strategy}: No game logs found")
                return {'strategy': strategy, 'works': False, 'reason': 'no_logs'}
        else:
            logger.warning(f"❌ {strategy} failed: {result.stderr}")
            return {'strategy': strategy, 'works': False, 'reason': 'execution_failed'}
            
    except subprocess.TimeoutExpired:
        logger.warning(f"⏰ {strategy} timed out")
        return {'strategy': strategy, 'works': False, 'reason': 'timeout'}
    except Exception as e:
        logger.warning(f"💥 {strategy} error: {e}")
        return {'strategy': strategy, 'works': False, 'reason': str(e)}

def find_available_strategies() -> list:
    """Find all available strategies by checking the config directory."""
    
    strategies_to_try = [
        # Based on our analysis, these are the best performers
        "ultra_elite_lstm",    # Best: 750 rejections
        "rbcr2",              # Reliable: 771 rejections  
        "constraint_focused_lstm",  # Good: 775 rejections
        "rbcr",               # Solid: 781 rejections
        "transformer",        # Neural: 790+ rejections
        "dual",               # Alternative: 810+ rejections
        "perfect",            # Baseline: 815+ rejections
        "ultimate3",          # Fallback: 812+ rejections
        "apex",               # Advanced: 823+ rejections
        # Add any other strategies that might exist
        "ultimate3h",
        "ultimate2", 
        "elite_lstm",
        "rl_lstm"
    ]
    
    logger.info(f"🔍 Testing {len(strategies_to_try)} potential strategies...")
    
    working_strategies = []
    
    for strategy in strategies_to_try:
        result = test_strategy(strategy)
        if result['works']:
            working_strategies.append(result)
    
    # Sort by performance (successful games with fewer rejections first)
    working_strategies.sort(key=lambda x: (not x['success'], x['rejections']))
    
    return working_strategies

def run_record_attempt(strategy_info: dict, num_games: int = 15) -> list:
    """Run multiple games with the best strategy to attempt record breaking."""
    
    strategy = strategy_info['strategy']
    logger.info(f"🎯 Running {num_games} record attempts with {strategy}")
    logger.info(f"📊 Baseline: {strategy_info['rejections']} rejections")
    
    results = []
    best_so_far = strategy_info['rejections']
    
    for game_num in range(1, num_games + 1):
        logger.info(f"⚡ Game {game_num}/{num_games} - Best so far: {best_so_far}")
        
        try:
            cmd = ["python", "main.py", "run", "--strategy", strategy, "--count", "1", "--scenario", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
            
            if result.returncode == 0:
                # Find the latest game log
                game_logs = glob.glob(f"game_logs/game_{strategy}_*.json") + glob.glob(f"game_logs/*{strategy}*.json")
                
                if game_logs:
                    latest_log = max(game_logs, key=lambda x: Path(x).stat().st_mtime)
                    
                    with open(latest_log, 'r') as f:
                        game_data = json.load(f)
                    
                    rejections = game_data.get('rejected_count', float('inf'))
                    success = game_data.get('success', False)
                    
                    results.append({
                        'game': game_num,
                        'rejections': rejections,
                        'success': success,
                        'log_file': latest_log
                    })
                    
                    # Status indicators
                    success_icon = "✅" if success else "❌"
                    
                    if success and rejections < 716:
                        record_icon = "🏆 NEW RECORD!"
                        logger.info(f"Game {game_num}: {rejections} rejections {success_icon} {record_icon}")
                        logger.info(f"🎉🎉🎉 WORLD RECORD BROKEN! {rejections} < 716! 🎉🎉🎉")
                        break  # Mission accomplished!
                    elif success and rejections < best_so_far:
                        record_icon = "📈 NEW BEST!"
                        best_so_far = rejections
                        logger.info(f"Game {game_num}: {rejections} rejections {success_icon} {record_icon}")
                    elif success and rejections < 750:
                        record_icon = "🎯 EXCELLENT!"
                        logger.info(f"Game {game_num}: {rejections} rejections {success_icon} {record_icon}")
                    elif success:
                        record_icon = "👍 GOOD"
                        logger.info(f"Game {game_num}: {rejections} rejections {success_icon} {record_icon}")
                    else:
                        logger.info(f"Game {game_num}: {rejections} rejections {success_icon}")
                        
        except Exception as e:
            logger.error(f"❌ Game {game_num} failed: {e}")
        
        # Brief pause between games
        if game_num < num_games:
            time.sleep(1)
    
    return results

def main():
    """Main execution: find best strategy and attempt record."""
    
    print("\n🚀 AUTOMATIC BERGHAIN RECORD-BREAKING SYSTEM")
    print("=" * 60)
    print("🎯 Goal: Beat 716 rejection world record")
    print("🔍 Strategy: Auto-detect best available approach")
    print("")
    
    # Step 1: Find working strategies
    logger.info("Phase 1: Discovering available strategies...")
    working_strategies = find_available_strategies()
    
    if not working_strategies:
        print("❌ No working strategies found!")
        print("💡 Troubleshooting suggestions:")
        print("   • Check if 'python main.py' works")
        print("   • Ensure internet connection for API")
        print("   • Verify dependencies: pip install -r requirements.txt")
        return
    
    print(f"\n✅ Found {len(working_strategies)} working strategies:")
    for i, strat in enumerate(working_strategies[:5]):  # Show top 5
        success_text = "✅" if strat['success'] else "❌"
        print(f"   {i+1}. {strat['strategy']}: {strat['rejections']} rejections {success_text}")
    
    # Step 2: Use the best strategy for record attempts
    best_strategy = working_strategies[0]
    print(f"\n🎯 Selected: {best_strategy['strategy']} ({best_strategy['rejections']} rejections)")
    
    if best_strategy['rejections'] > 800:
        print(f"⚠️  Warning: Best available strategy has {best_strategy['rejections']} rejections")
        print("   This is far from the 716 record. Consider improving strategies first.")
        
        proceed = input("Continue anyway? (y/N): ").lower().strip()
        if proceed != 'y':
            print("Aborted.")
            return
    
    # Step 3: Attempt record breaking
    logger.info("Phase 2: Attempting record-breaking runs...")
    
    try:
        results = run_record_attempt(best_strategy, 20)  # 20 attempts
        
        # Final analysis
        print(f"\n🎯 FINAL RESULTS")
        print("=" * 40)
        
        if results:
            successful = [r for r in results if r['success']]
            if successful:
                best = min(successful, key=lambda x: x['rejections'])
                avg = sum(r['rejections'] for r in successful) / len(successful)
                
                print(f"📊 Successful runs: {len(successful)}/{len(results)}")
                print(f"🏆 Best performance: {best['rejections']} rejections")
                print(f"📈 Average performance: {avg:.1f} rejections") 
                
                # Record status
                record_games = [r for r in successful if r['rejections'] < 716]
                if record_games:
                    print(f"\n🎉🎉🎉 NEW WORLD RECORDS SET: {len(record_games)} 🎉🎉🎉")
                    for record in record_games:
                        print(f"   🏆 Game {record['game']}: {record['rejections']} rejections")
                        print(f"      📁 Log: {record['log_file']}")
                else:
                    gap = best['rejections'] - 715
                    print(f"\n🎯 Gap to record: {gap} rejections")
                    if gap < 50:
                        print("💡 Very close! Try running more games or fine-tuning.")
                    elif gap < 100:
                        print("💡 Getting close. Consider strategy improvements.")
                    else:
                        print("💡 Need significant improvements to reach record.")
            else:
                print("❌ No successful games completed")
        else:
            print("❌ No games completed")
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n💥 Error during record attempts: {e}")

if __name__ == "__main__":
    main()