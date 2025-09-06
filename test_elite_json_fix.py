#!/usr/bin/env python3
"""
ABOUTME: Test the JSON saving fix by running a single elite game
ABOUTME: Verify that the complete decision data is saved properly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elite_game_hunter import EliteGameHunter, EliteGameConfig
import json

# Test with a single RBCR game
config = EliteGameConfig(
    max_rejections_threshold=850,
    batch_size=1,
    max_workers=1,
    strategies=['rbcr'],
    elite_games_dir="test_elite_games"
)

hunter = EliteGameHunter(config)

# Run a single game to test
print("🧪 Testing elite JSON saving...")
elite_count = hunter.run_batch()

if elite_count > 0:
    # Check if JSON is complete
    import glob
    json_files = glob.glob("test_elite_games/elite_*.json")
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)
        print(f"📄 Checking {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            print(f"✅ JSON is valid!")
            print(f"   Decisions: {len(data.get('decisions', []))}")
            print(f"   Constraints: {len(data.get('constraints', []))}")
            print(f"   Game ID: {data.get('game_id', 'missing')}")
            
            # Check last decision
            if data.get('decisions'):
                last_decision = data['decisions'][-1]
                print(f"   Last decision complete: {'timestamp' in last_decision}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON is still invalid: {e}")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print("❌ No elite JSON files created")
else:
    print("ℹ️  No elite games found in this test run")