#!/usr/bin/env python3
"""
ABOUTME: Filter and prepare ultra-elite games (<800 rejections) for improved LSTM training
ABOUTME: Creates a curated dataset of only the highest quality elite games
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def filter_ultra_elite_games(
    elite_games_dir: str = "elite_games",
    output_dir: str = "ultra_elite_games", 
    max_rejections: int = 800
) -> List[str]:
    """
    Filter elite games to only include ultra-elite performances.
    
    Args:
        elite_games_dir: Source directory with elite games
        output_dir: Output directory for filtered games  
        max_rejections: Maximum rejections allowed for ultra-elite status
        
    Returns:
        List of ultra-elite game files copied
    """
    elite_dir = Path(elite_games_dir)
    output_path = Path(output_dir)
    
    if not elite_dir.exists():
        raise ValueError(f"Elite games directory not found: {elite_games_dir}")
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    ultra_elite_files = []
    game_stats = []
    
    for json_file in elite_dir.glob("elite_*.json"):
        try:
            with open(json_file, 'r') as f:
                game_data = json.load(f)
            
            rejected_count = game_data.get('rejected_count', float('inf'))
            
            if rejected_count <= max_rejections:
                # Copy to ultra-elite directory
                output_file = output_path / json_file.name
                shutil.copy2(json_file, output_file)
                ultra_elite_files.append(str(output_file))
                
                game_stats.append({
                    'file': json_file.name,
                    'strategy': game_data.get('strategy', 'unknown'),
                    'rejections': rejected_count,
                    'admitted': game_data.get('admitted_count', 0),
                    'success': game_data.get('success', False)
                })
                
                logger.info(f"✅ Ultra-elite: {json_file.name} - {rejected_count} rejections")
            
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
    
    # Save stats
    stats_file = output_path / "ultra_elite_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'total_ultra_elite_games': len(ultra_elite_files),
            'max_rejections_threshold': max_rejections,
            'games': game_stats,
            'strategy_breakdown': _get_strategy_breakdown(game_stats),
            'best_performance': min(stat['rejections'] for stat in game_stats) if game_stats else None,
            'avg_rejections': sum(stat['rejections'] for stat in game_stats) / len(game_stats) if game_stats else 0
        }, f, indent=2)
    
    logger.info(f"🎯 Filtered {len(ultra_elite_files)} ultra-elite games (≤{max_rejections} rejections)")
    logger.info(f"📊 Average rejections: {sum(stat['rejections'] for stat in game_stats) / len(game_stats):.1f}")
    logger.info(f"🏆 Best performance: {min(stat['rejections'] for stat in game_stats)} rejections")
    logger.info(f"📁 Ultra-elite games saved to: {output_dir}/")
    
    return ultra_elite_files

def _get_strategy_breakdown(game_stats: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get breakdown of games by strategy."""
    breakdown = {}
    for stat in game_stats:
        strategy = stat['strategy']
        breakdown[strategy] = breakdown.get(strategy, 0) + 1
    return breakdown

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter ultra-elite games")
    parser.add_argument('--elite-dir', default='elite_games', help='Elite games directory')
    parser.add_argument('--output-dir', default='ultra_elite_games', help='Output directory')
    parser.add_argument('--max-rejections', type=int, default=800, help='Max rejections for ultra-elite')
    
    args = parser.parse_args()
    
    ultra_elite_files = filter_ultra_elite_games(
        args.elite_dir, 
        args.output_dir, 
        args.max_rejections
    )
    
    print(f"\n✨ Successfully filtered {len(ultra_elite_files)} ultra-elite games!")