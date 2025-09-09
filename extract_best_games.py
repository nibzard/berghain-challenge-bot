#!/usr/bin/env python3
"""
ABOUTME: Simple script to extract the best successful games for transformer training
ABOUTME: Focus on games with <800 rejections that actually succeeded
"""

import json
import glob
from pathlib import Path
import shutil
import re

def extract_best_games():
    """Extract the best successful games."""
    
    print("🔍 Searching for best successful games...")
    
    best_games = []
    for file in glob.glob('game_logs/game_*.json'):
        try:
            with open(file) as f:
                data = json.load(f)
                
                success = data.get('success', False)
                admitted = data.get('admitted_count', 0)
                rejections = data.get('rejected_count', float('inf'))
                
                if success and admitted >= 900 and rejections < 850:
                    # Extract strategy from filename
                    fname = Path(file).name
                    match = re.match(r'game_([^_]+)_', fname)
                    strategy = match.group(1) if match else 'unknown'
                    
                    best_games.append({
                        'file': file,
                        'strategy': strategy,
                        'rejections': rejections,
                        'admitted': admitted
                    })
        except:
            pass
    
    # Sort by rejection count
    best_games.sort(key=lambda x: x['rejections'])
    
    print(f"Found {len(best_games)} successful elite games")
    
    # Create training directory
    training_dir = Path("ultra_elite_training")
    training_dir.mkdir(exist_ok=True)
    
    # Copy best games
    for i, game in enumerate(best_games):
        src = Path(game['file'])
        dst = training_dir / f"elite_{i:03d}_{game['rejections']}rej_{src.name}"
        shutil.copy2(src, dst)
        
        if i < 20:  # Print top 20
            print(f"  {game['rejections']:3d} rejections - {game['strategy']:15s} - {game['admitted']:4d} admitted")
    
    print(f"\n📁 Copied {len(best_games)} games to {training_dir}/")
    
    # Create metadata
    metadata = {
        'total_games': len(best_games),
        'best_rejection': min(g['rejections'] for g in best_games),
        'avg_rejection': sum(g['rejections'] for g in best_games) / len(best_games),
        'games': best_games
    }
    
    with open(training_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(training_dir)

if __name__ == "__main__":
    extract_best_games()