#!/usr/bin/env python3
"""
ABOUTME: Fix elite hunter stats to reflect the actual elite games found
ABOUTME: Count elite games from CSV and update the stats file correctly
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

def fix_elite_stats():
    """Fix the elite hunter stats based on actual elite games found."""
    
    # Count elite games from CSV
    elite_games_csv = "elite_games_summary.csv"
    elite_per_strategy = defaultdict(int)
    total_elite = 0
    
    if Path(elite_games_csv).exists():
        with open(elite_games_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                strategy = row['strategy']
                elite_per_strategy[strategy] += 1
                total_elite += 1
    
    print(f"Found {total_elite} elite games in CSV:")
    for strategy, count in elite_per_strategy.items():
        print(f"  {strategy}: {count}")
    
    # Load and update stats
    stats_file = "elite_hunter_stats.json"
    if Path(stats_file).exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Update elite counts
        stats['elite_games'] = total_elite
        stats['elite_per_strategy'].update(dict(elite_per_strategy))
        
        # Recalculate rates
        stats['elite_rate'] = total_elite / max(stats['total_games'], 1)
        stats['elite_per_hour'] = total_elite / max(stats['runtime_hours'], 0.01)
        
        # Save updated stats
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Updated stats file:")
        print(f"   Total elite games: {total_elite}")
        print(f"   Elite rate: {stats['elite_rate']:.1%}")
        print(f"   Elite per hour: {stats['elite_per_hour']:.2f}")
    else:
        print("❌ Stats file not found")

if __name__ == "__main__":
    fix_elite_stats()