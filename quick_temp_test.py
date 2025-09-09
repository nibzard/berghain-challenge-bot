#!/usr/bin/env python3
"""
ABOUTME: Quick temperature optimization test to find best ultimate3 settings
ABOUTME: Tests different temperature values to optimize for minimum rejections
"""

import subprocess
import json
import glob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_temperature(temp: float, games: int = 3) -> dict:
    """Test a specific temperature setting."""
    logger.info(f"🌡️  Testing temperature {temp} with {games} games")
    
    results = []
    
    for i in range(games):
        try:
            cmd = ["python", "main.py", "run", "--strategy", "ultimate3", "--count", "1", "--temp", str(temp)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Find latest game log
                game_logs = glob.glob("game_logs/game_ultimate3_*.json")
                if game_logs:
                    latest_log = max(game_logs, key=lambda x: Path(x).stat().st_mtime)
                    
                    with open(latest_log, 'r') as f:
                        game_data = json.load(f)
                    
                    rejections = game_data.get('rejected_count', float('inf'))
                    success = game_data.get('success', False)
                    
                    results.append({
                        'rejections': rejections,
                        'success': success
                    })
                    
                    status = "✅" if success else "❌"
                    logger.info(f"  Game {i+1}: {rejections} rejections {status}")
        
        except Exception as e:
            logger.warning(f"  Game {i+1} failed: {e}")
    
    # Calculate statistics
    if results:
        successful = [r for r in results if r['success']]
        if successful:
            avg_rej = sum(r['rejections'] for r in successful) / len(successful)
            min_rej = min(r['rejections'] for r in successful)
            success_rate = len(successful) / len(results)
            
            return {
                'temperature': temp,
                'avg_rejections': avg_rej,
                'min_rejections': min_rej,
                'success_rate': success_rate,
                'total_games': len(results)
            }
    
    return {
        'temperature': temp,
        'avg_rejections': float('inf'),
        'min_rejections': float('inf'),
        'success_rate': 0.0,
        'total_games': len(results)
    }

def main():
    """Test different temperature values."""
    print("\n🌡️  TEMPERATURE OPTIMIZATION TEST")
    print("=" * 40)
    print("🎯 Goal: Find optimal temperature for ultimate3")
    print("📊 Testing range: 0.1 to 1.0")
    print("")
    
    # Temperature values to test (based on theory: lower = more selective, higher = more accepting)
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    all_results = []
    
    for temp in temperatures:
        result = test_temperature(temp, games=3)  # 3 games per temperature
        all_results.append(result)
        
        if result['success_rate'] > 0:
            print(f"🌡️  Temp {temp}: {result['min_rejections']:.0f} best, {result['avg_rejections']:.0f} avg, {result['success_rate']:.1%} success")
        else:
            print(f"🌡️  Temp {temp}: No successful games")
    
    # Find best temperature
    successful_results = [r for r in all_results if r['success_rate'] > 0]
    if successful_results:
        best_by_min = min(successful_results, key=lambda x: x['min_rejections'])
        best_by_avg = min(successful_results, key=lambda x: x['avg_rejections'])
        
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        print(f"   Best minimum: Temp {best_by_min['temperature']} → {best_by_min['min_rejections']:.0f} rejections")
        print(f"   Best average: Temp {best_by_avg['temperature']} → {best_by_avg['avg_rejections']:.0f} rejections")
        
        # Recommendation
        if best_by_min['min_rejections'] < 800:
            print(f"💡 RECOMMENDATION: Use temp {best_by_min['temperature']} for record attempts!")
            print(f"   Command: python main.py run --strategy ultimate3 --count 20 --temp {best_by_min['temperature']}")
        else:
            print(f"💡 Temperature optimization didn't find sub-800 performance")
            print(f"   Best option: temp {best_by_min['temperature']} with {best_by_min['min_rejections']:.0f} rejections")
    
    else:
        print("❌ No successful temperature configurations found")

if __name__ == "__main__":
    main()