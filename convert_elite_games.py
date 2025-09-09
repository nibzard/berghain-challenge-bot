#!/usr/bin/env python3
"""Convert elite game JSON files to transformer-compatible JSONL format."""

import json
from pathlib import Path
from datetime import datetime
import argparse


def convert_elite_game_to_jsonl(elite_game_path: Path, output_dir: Path):
    """Convert a single elite game JSON file to JSONL format."""
    
    with open(elite_game_path) as f:
        game_data = json.load(f)
    
    # Extract game info
    game_id = game_data.get('game_id', 'unknown')
    timestamp = game_data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    scenario = game_data.get('scenario', 1)
    strategy = game_data.get('strategy', 'unknown')
    
    # Create output filename
    output_filename = f"events_scenario_{scenario}_{strategy}_{timestamp}.jsonl"
    output_path = output_dir / output_filename
    
    events = []
    
    # Create game_started event
    game_started_event = {
        'event_type': 'game_started',
        'game_id': game_id,
        'scenario': scenario,
        'timestamp': timestamp,
        'constraints': [
            {
                'attribute': constraint['attribute'],
                'required': constraint['min_count']
            }
            for constraint in game_data.get('constraints', [])
        ]
    }
    events.append(game_started_event)
    
    # Convert each decision to person_evaluated event
    for decision in game_data.get('decisions', []):
        person = decision['person']
        
        # Convert attributes format from {attr: bool} to list of attributes
        attributes = [attr for attr, value in person['attributes'].items() if value]
        
        person_evaluated_event = {
            'event_type': 'person_evaluated',
            'person': {
                'index': person['index'],
                'attributes': attributes  # List format expected by transformer
            },
            'decision': {
                'admitted': decision['accepted']
            },
            'reasoning': decision.get('reasoning', ''),
            'timestamp': decision.get('timestamp', timestamp)
        }
        events.append(person_evaluated_event)
    
    # Create game_ended event
    game_ended_event = {
        'event_type': 'game_ended',
        'game_id': game_id,
        'success': game_data.get('success', False),
        'rejected_count': game_data.get('rejected_count', 0),
        'admitted_count': game_data.get('admitted_count', 0),
        'timestamp': timestamp
    }
    events.append(game_ended_event)
    
    # Write JSONL file
    with open(output_path, 'w') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert elite games to transformer format')
    parser.add_argument('--elite-dir', type=Path, 
                       default=Path('elite_games'),
                       help='Directory containing elite game JSON files')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('game_logs'),
                       help='Output directory for JSONL files')
    parser.add_argument('--pattern', type=str,
                       default='elite_*.json',
                       help='Pattern to match elite game files')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all elite game files
    elite_files = list(args.elite_dir.glob(args.pattern))
    
    if not elite_files:
        print(f"No elite game files found in {args.elite_dir} with pattern {args.pattern}")
        return
    
    print(f"Found {len(elite_files)} elite game files")
    
    converted = 0
    for elite_file in elite_files:
        try:
            output_path = convert_elite_game_to_jsonl(elite_file, args.output_dir)
            print(f"Converted {elite_file.name} -> {output_path.name}")
            converted += 1
        except Exception as e:
            print(f"Error converting {elite_file.name}: {e}")
    
    print(f"\nSuccessfully converted {converted}/{len(elite_files)} elite games")


if __name__ == '__main__':
    main()