#!/usr/bin/env python3
"""
ABOUTME: Simple script to run the best-performing strategy for record breaking
ABOUTME: Uses existing main.py infrastructure with the ultra_elite_lstm strategy (750 rejections)
"""

import subprocess
import json
import time
from pathlib import Path
import glob
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_best_strategy(num_games: int = 10) -> None:
    """Run multiple games with the best strategy to maximize record-breaking chances."""
    
    logger.info(f"🎯 Running {num_games} games with ultra_elite_lstm strategy (750 rejection record)")
    
    results = []
    
    for game_num in range(1, num_games + 1):
        logger.info(f"⚡ Starting game {game_num}/{num_games}")
        
        try:
            # Run the best strategy using main.py
            cmd = ["python", "main.py", "run", "--strategy", "ultra_elite_lstm", "--count", "1", "--scenario", "1"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Game {game_num} completed successfully")
                
                # Find the most recent game log
                game_logs = glob.glob("game_logs/game_ultra_elite_lstm_*.json")
                if game_logs:
                    latest_log = max(game_logs, key=lambda x: Path(x).stat().st_mtime)
                    
                    try:
                        with open(latest_log, 'r') as f:
                            game_data = json.load(f)
                        
                        rejections = game_data.get('rejected_count', 'unknown')
                        success = game_data.get('success', False)
                        
                        results.append({
                            'game': game_num,
                            'rejections': rejections,
                            'success': success,
                            'log_file': latest_log
                        })
                        
                        status = "✅ SUCCESS" if success else "❌ FAILED"
                        record_status = "🏆 RECORD!" if rejections < 716 and success else "📈" if rejections < 750 else ""
                        
                        logger.info(f"Game {game_num}: {rejections} rejections - {status} {record_status}")
                        
                        # If we beat the record, celebrate!
                        if rejections < 716 and success:
                            logger.info(f"🎉🎉🎉 NEW WORLD RECORD! {rejections} rejections! 🎉🎉🎉")
                            break  # Stop here, we won!
                            
                    except Exception as e:
                        logger.warning(f"Could not parse game log: {e}")
                        
            else:
                logger.error(f"❌ Game {game_num} failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Game {game_num} timed out after 5 minutes")
        except Exception as e:
            logger.error(f"💥 Game {game_num} crashed: {e}")
        
        # Brief pause between games
        if game_num < num_games:
            time.sleep(2)
    
    # Final summary
    logger.info("\n🎯 FINAL RESULTS SUMMARY")
    logger.info("=" * 50)
    
    if results:
        successful_games = [r for r in results if r['success']]
        if successful_games:
            best_game = min(successful_games, key=lambda x: x['rejections'])
            avg_rejections = sum(r['rejections'] for r in successful_games) / len(successful_games)
            
            logger.info(f"📊 Successful games: {len(successful_games)}/{len(results)}")
            logger.info(f"🏆 Best performance: {best_game['rejections']} rejections")
            logger.info(f"📈 Average rejections: {avg_rejections:.1f}")
            
            # Check if we beat the record
            record_beaters = [r for r in successful_games if r['rejections'] < 716]
            if record_beaters:
                logger.info(f"🎉 NEW RECORDS SET: {len(record_beaters)} games!")
                for record in record_beaters:
                    logger.info(f"   🏆 Game {record['game']}: {record['rejections']} rejections - {record['log_file']}")
            else:
                closest = best_game['rejections']
                gap = closest - 715  # 715 to beat the 716 record
                logger.info(f"🎯 Closest to record: {closest} rejections (need {gap} fewer)")
                logger.info(f"💡 Recommendation: Run more games or fine-tune strategy")
        else:
            logger.info("❌ No successful games completed")
    else:
        logger.info("❌ No games completed successfully")

def run_alternative_strategies() -> None:
    """Try other top strategies if ultra_elite_lstm doesn't work."""
    
    strategies = [
        ("rbcr2", "Most reliable strategy - 771 rejections"),
        ("constraint_focused_lstm", "Constraint specialist - 775 rejections"), 
        ("rbcr", "Solid performer - 781 rejections"),
        ("transformer", "Neural approach - 790+ rejections")
    ]
    
    logger.info("🔄 Trying alternative strategies...")
    
    for strategy, description in strategies:
        logger.info(f"⚡ Testing {strategy}: {description}")
        
        try:
            cmd = ["python", "main.py", "run", "--strategy", strategy, "--count", "1", "--scenario", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ {strategy} ran successfully")
                
                # Check the result
                game_logs = glob.glob(f"game_logs/game_{strategy}_*.json")
                if game_logs:
                    latest_log = max(game_logs, key=lambda x: Path(x).stat().st_mtime)
                    
                    with open(latest_log, 'r') as f:
                        game_data = json.load(f)
                    
                    rejections = game_data.get('rejected_count', 'unknown')
                    success = game_data.get('success', False)
                    
                    status = "✅" if success else "❌"
                    logger.info(f"   Result: {rejections} rejections {status}")
                    
                    if success and rejections < 750:
                        logger.info(f"🎯 {strategy} shows promise! Consider running more games.")
                        return strategy  # Found a good alternative
                        
            else:
                logger.warning(f"⚠️  {strategy} failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"❌ Error testing {strategy}: {e}")
    
    return None

def main():
    """Main execution."""
    print("\n🚀 BERGHAIN RECORD-BREAKING ATTEMPT")
    print("=" * 50)
    print("🎯 Target: Beat 716 rejection record")
    print("🏆 Current best: 750 rejections (ultra_elite_lstm)")
    print("📊 Strategy: Run best algorithm multiple times")
    print("")
    
    # First try the best strategy
    try:
        run_best_strategy(20)  # Run 20 games for good chances
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n💥 Error with main strategy: {e}")
        print("🔄 Trying alternative strategies...")
        
        # Try alternatives if main strategy fails
        working_strategy = run_alternative_strategies()
        
        if working_strategy:
            print(f"\n🎯 Found working strategy: {working_strategy}")
            print("Consider running more games with this strategy!")
        else:
            print("\n❌ No strategies worked. Check your setup:")
            print("   • Ensure 'python main.py run --strategy ultra_elite_lstm --count 1' works")
            print("   • Check internet connection for API access") 
            print("   • Verify all dependencies are installed")

if __name__ == "__main__":
    main()