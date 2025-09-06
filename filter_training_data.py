#!/usr/bin/env python3
"""
ABOUTME: Filter game logs to create high-quality training dataset
ABOUTME: Only keeps games with excellent performance (< 850 rejections) from best strategies
"""

import json
import glob
import shutil
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any


def analyze_game_quality(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a game's quality metrics."""
    if not game_data.get('success', False):
        return {'quality': 'failed', 'score': 0}
    
    rejections = game_data.get('rejected_count', 10000)
    admitted = game_data.get('admitted_count', 0)
    
    # Quality thresholds
    if rejections <= 716:  # Record performance
        quality = 'elite'
        score = 100
    elif rejections <= 800:  # Excellent
        quality = 'excellent'
        score = 90
    elif rejections <= 850:  # Good
        quality = 'good'
        score = 75
    elif rejections <= 900:  # Average
        quality = 'average'
        score = 50
    else:  # Poor
        quality = 'poor'
        score = 25
    
    return {
        'quality': quality,
        'score': score,
        'rejections': rejections,
        'admitted': admitted
    }


def filter_training_data(
    input_dir: str = "game_logs",
    output_dir: str = "game_logs_filtered",
    min_rejections_threshold: int = 850,
    preferred_strategies: List[str] = None,
    max_games_per_strategy: int = 50
) -> None:
    """
    Filter game logs to create a high-quality training dataset.
    
    Args:
        input_dir: Directory containing all game logs
        output_dir: Directory to save filtered games
        min_rejections_threshold: Maximum rejections to include
        preferred_strategies: List of preferred strategy names
        max_games_per_strategy: Maximum games per strategy to prevent imbalance
    """
    if preferred_strategies is None:
        preferred_strategies = [
            'optimal', 'apex', 'ultimate3', 'ultimate3h', 'ultimate2', 
            'perfect', 'dual', 'ultimate'  # Best performing strategies
        ]
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze all games
    games_by_strategy = {}
    quality_stats = {'elite': 0, 'excellent': 0, 'good': 0, 'average': 0, 'poor': 0, 'failed': 0}
    
    for file_path in glob.glob(f"{input_dir}/game_*.json"):
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            # Extract strategy from filename
            filename = os.path.basename(file_path)
            strategy = filename.split('_')[1]  # game_STRATEGY_xxx_timestamp.json
            
            # Analyze quality
            quality_info = analyze_game_quality(game_data)
            quality_stats[quality_info['quality']] += 1
            
            # Store game info
            if strategy not in games_by_strategy:
                games_by_strategy[strategy] = []
            
            games_by_strategy[strategy].append({
                'file_path': file_path,
                'filename': filename,
                'quality_info': quality_info,
                'game_data': game_data
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("=== Game Quality Analysis ===")
    total_games = sum(quality_stats.values())
    for quality, count in quality_stats.items():
        pct = 100 * count / total_games if total_games > 0 else 0
        print(f"{quality:10}: {count:4d} games ({pct:5.1f}%)")
    
    print(f"\nTotal games analyzed: {total_games}")
    
    # Filter and copy high-quality games
    filtered_count = 0
    strategy_counts = {}
    
    print("\n=== Filtering Strategy ===")
    print(f"Max rejections: {min_rejections_threshold}")
    print(f"Preferred strategies: {', '.join(preferred_strategies)}")
    print(f"Max games per strategy: {max_games_per_strategy}")
    
    for strategy in preferred_strategies:
        if strategy not in games_by_strategy:
            print(f"Warning: Strategy '{strategy}' not found in dataset")
            continue
        
        # Sort by quality (best first)
        strategy_games = games_by_strategy[strategy]
        strategy_games.sort(key=lambda x: x['quality_info']['score'], reverse=True)
        
        # Filter by quality threshold
        high_quality_games = [
            game for game in strategy_games 
            if (game['quality_info'].get('rejections', 10000) <= min_rejections_threshold and 
                game['game_data'].get('success', False))
        ]
        
        # Limit number of games per strategy
        selected_games = high_quality_games[:max_games_per_strategy]
        strategy_counts[strategy] = len(selected_games)
        
        # Copy selected games
        for game in selected_games:
            src_path = game['file_path']
            dst_path = os.path.join(output_dir, game['filename'])
            shutil.copy2(src_path, dst_path)
            filtered_count += 1
        
        if selected_games:
            avg_rejections = np.mean([g['quality_info']['rejections'] for g in selected_games])
            print(f"  {strategy:15}: {len(selected_games):3d} games (avg {avg_rejections:.0f} rejections)")
        else:
            print(f"  {strategy:15}:   0 games (no high-quality games found)")
    
    print(f"\n=== Results ===")
    print(f"Total filtered games: {filtered_count}")
    print(f"Reduction: {total_games} -> {filtered_count} ({100*filtered_count/total_games:.1f}% kept)")
    print(f"Output directory: {output_dir}")
    
    # Save filtering metadata
    metadata = {
        'filtering_config': {
            'min_rejections_threshold': min_rejections_threshold,
            'preferred_strategies': preferred_strategies,
            'max_games_per_strategy': max_games_per_strategy
        },
        'quality_stats': quality_stats,
        'strategy_counts': strategy_counts,
        'total_original_games': total_games,
        'total_filtered_games': filtered_count
    }
    
    with open(os.path.join(output_dir, 'filtering_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_dir}/filtering_metadata.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter game logs for high-quality training data")
    parser.add_argument('--input-dir', default='game_logs', help='Input directory with game logs')
    parser.add_argument('--output-dir', default='game_logs_filtered', help='Output directory for filtered games')
    parser.add_argument('--max-rejections', type=int, default=850, help='Maximum rejections threshold')
    parser.add_argument('--max-per-strategy', type=int, default=50, help='Maximum games per strategy')
    
    args = parser.parse_args()
    
    filter_training_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_rejections_threshold=args.max_rejections,
        max_games_per_strategy=args.max_per_strategy
    )