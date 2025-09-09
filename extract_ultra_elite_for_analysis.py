#!/usr/bin/env python3
"""
ABOUTME: Extract ultra-elite games to a temporary directory for pattern analysis
ABOUTME: Uses the tier analysis results to get the best games efficiently
"""

import json
import shutil
from pathlib import Path

def extract_ultra_elite_games():
    """Extract ultra-elite games based on tier analysis."""
    
    # Find the most recent tier analysis
    tier_dirs = list(Path("training_tiers").glob("ultra_elite_*"))
    if not tier_dirs:
        print("No ultra-elite tier directories found")
        return
    
    latest_tier_dir = sorted(tier_dirs, key=lambda x: x.name)[-1]
    print(f"Using tier directory: {latest_tier_dir}")
    
    # Read the metadata
    metadata_file = latest_tier_dir / "tier_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    file_paths = metadata['file_paths']
    print(f"Found {len(file_paths)} ultra-elite games")
    
    # Create temporary directory for analysis
    analysis_dir = Path("temp_ultra_elite_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Copy the best games for analysis
    best_games = metadata.get('best_games', file_paths[:100])
    
    copied = 0
    for game_path in best_games[:50]:  # Top 50 for analysis
        src = Path(game_path)
        if src.exists():
            dst = analysis_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
    
    print(f"Copied {copied} ultra-elite games to {analysis_dir}")
    return str(analysis_dir)

if __name__ == "__main__":
    extract_ultra_elite_games()