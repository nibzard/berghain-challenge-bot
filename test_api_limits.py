#!/usr/bin/env python3
"""Test script to determine API rate limits and optimal worker count."""

import subprocess
import time
import json
from typing import Dict, List

def run_test(workers: int, count: int = 3, strategy: str = "conservative") -> Dict:
    """Run a test with specified workers and return results."""
    print(f"🧪 Testing {workers} workers with {count} games...")
    
    start_time = time.time()
    
    try:
        # Run the command and capture output
        cmd = [
            "python", "main.py", "run",
            "--scenario", "1",
            "--strategy", strategy,
            "--count", str(count),
            "--workers", str(workers),
            "--no-high-score-check"  # Disable for faster testing
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Parse the output for success/failure info
        stdout = result.stdout
        stderr = result.stderr
        
        # Look for batch results
        successful = 0
        total = 0
        
        for line in stdout.split('\n'):
            if "Successful:" in line:
                # Extract numbers from "Successful: 3 (100.0%)"
                parts = line.split("Successful:")[-1].strip()
                successful = int(parts.split()[0])
                total_part = parts.split('(')[0].strip()
                
            elif "Total games:" in line:
                total = int(line.split("Total games:")[-1].strip())
        
        # Count error messages
        api_errors = stdout.count("Max retries exceeded") + stderr.count("Max retries exceeded")
        connection_errors = stdout.count("Failed to start game") + stderr.count("Failed to start game")
        
        return {
            "workers": workers,
            "count": count,
            "successful": successful,
            "total": total,
            "duration": duration,
            "api_errors": api_errors,
            "connection_errors": connection_errors,
            "success_rate": successful / total if total > 0 else 0,
            "return_code": result.returncode,
            "stdout_sample": stdout[:500],  # First 500 chars for debugging
        }
        
    except subprocess.TimeoutExpired:
        return {
            "workers": workers,
            "count": count,
            "successful": 0,
            "total": count,
            "duration": 120,
            "api_errors": 0,
            "connection_errors": count,
            "success_rate": 0,
            "return_code": -1,
            "error": "TIMEOUT"
        }
    except Exception as e:
        return {
            "workers": workers,
            "count": count,
            "successful": 0,
            "total": count,
            "duration": time.time() - start_time,
            "api_errors": 0,
            "connection_errors": count,
            "success_rate": 0,
            "return_code": -2,
            "error": str(e)
        }

def main():
    """Run systematic API rate limit tests."""
    print("🚀 Starting API Rate Limit Tests")
    print("=" * 50)
    
    results = []
    
    # Test configurations: (workers, games_per_worker)
    test_configs = [
        (1, 3),   # Baseline
        (2, 2),   # 4 total games
        (3, 2),   # 6 total games  
        (4, 2),   # 8 total games
        (5, 2),   # 10 total games
        (6, 2),   # 12 total games
        (8, 1),   # 8 total games
        (10, 1),  # 10 total games
    ]
    
    for workers, count in test_configs:
        result = run_test(workers, count)
        results.append(result)
        
        # Print immediate results
        if result["success_rate"] == 1.0:
            print(f"✅ {workers} workers: {result['successful']}/{result['total']} SUCCESS ({result['duration']:.1f}s)")
        elif result["success_rate"] > 0.5:
            print(f"⚠️  {workers} workers: {result['successful']}/{result['total']} PARTIAL ({result['duration']:.1f}s, {result['api_errors']} API errors)")
        else:
            print(f"❌ {workers} workers: {result['successful']}/{result['total']} FAILED ({result['duration']:.1f}s)")
        
        # Give API a moment between tests
        time.sleep(2)
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY RESULTS")
    print("=" * 50)
    
    print(f"{'Workers':<8} {'Games':<6} {'Success':<8} {'Rate':<6} {'Duration':<8} {'Errors':<7}")
    print("-" * 50)
    
    best_config = None
    best_throughput = 0
    
    for r in results:
        throughput = r['successful'] / r['duration'] if r['duration'] > 0 else 0
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = r
        
        print(f"{r['workers']:<8} {r['total']:<6} {r['successful']:<8} {r['success_rate']:.1%} "
              f"{r['duration']:<8.1f} {r['api_errors']:<7}")
    
    if best_config:
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   Workers: {best_config['workers']}")
        print(f"   Throughput: {best_throughput:.2f} successful games/second")
        print(f"   Success Rate: {best_config['success_rate']:.1%}")
    
    # Save detailed results
    with open("api_rate_limit_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: api_rate_limit_test_results.json")

if __name__ == "__main__":
    main()
