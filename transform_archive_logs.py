#!/usr/bin/env python3
"""
Transform old archive logs to new game_logs format for RL training.

This script converts the old JSON format from archive_old_files to the new
structured format used in game_logs, creating both the summary JSON file 
and the detailed JSONL events file.
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_game_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from old filename format."""
    # Match patterns like: game_scenario_1_20250905_173446_7374e8b6.json
    scenario_match = re.search(r'scenario_(\d+)', filename)
    date_match = re.search(r'(\d{8})_(\d{6})', filename)
    id_match = re.search(r'([a-f0-9]{8})\.json', filename)
    
    metadata = {}
    
    if scenario_match:
        metadata['scenario_id'] = int(scenario_match.group(1))
        metadata['scenario_name'] = get_scenario_name(int(scenario_match.group(1)))
    
    if date_match:
        date_str = date_match.group(1)
        time_str = date_match.group(2)
        # Parse YYYYMMDD_HHMMSS
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        metadata['start_time'] = dt.isoformat()
    
    if id_match:
        metadata['short_id'] = id_match.group(1)
    
    return metadata


def get_scenario_name(scenario_id: int) -> str:
    """Map scenario ID to name."""
    scenario_names = {
        1: "Friday Night",
        2: "Creative Night",
        3: "Special Event"
    }
    return scenario_names.get(scenario_id, f"Scenario {scenario_id}")


def infer_strategy_from_data(old_data: Dict[str, Any], filename: str) -> str:
    """Infer strategy name from old data or filename."""
    if "ultimate" in filename.lower():
        return "Ultimate"
    elif "optimal" in filename.lower():
        return "Optimal"
    elif "simple" in filename.lower():
        return "Simple"
    else:
        return "Legacy"


def calculate_duration(old_data: Dict[str, Any]) -> float:
    """Calculate duration from start/end times."""
    try:
        start = datetime.fromisoformat(old_data["start_time"])
        end = datetime.fromisoformat(old_data["end_time"])
        return (end - start).total_seconds()
    except:
        return 0.0


def determine_game_success(old_data: Dict[str, Any]) -> bool:
    """Determine if game was successful based on constraints."""
    if "constraints" not in old_data:
        return False
    
    # Count admitted people by attribute
    attribute_counts = {}
    for person in old_data.get("people", []):
        if person.get("decision", False):  # Only count admitted people
            for attr, has_attr in person.get("attributes", {}).items():
                if has_attr:
                    attribute_counts[attr] = attribute_counts.get(attr, 0) + 1
    
    # Check if all constraints are met
    for constraint in old_data["constraints"]:
        attr = constraint["attribute"]
        required = constraint["minCount"]
        actual = attribute_counts.get(attr, 0)
        if actual < required:
            return False
    
    return True


def transform_old_game_to_new_format(old_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Transform old game format to new format."""
    metadata = extract_game_metadata_from_filename(filename)
    
    # Generate solver ID based on filename and strategy
    strategy_name = infer_strategy_from_data(old_data, filename)
    solver_id = f"legacy_{metadata.get('short_id', 'unknown')}"
    
    # Count decisions
    total_people = len(old_data.get("people", []))
    admitted = sum(1 for p in old_data.get("people", []) if p.get("decision", False))
    rejected = total_people - admitted
    
    # Build new format
    new_data = {
        "solver_id": solver_id,
        "game_id": old_data.get("game_id", str(uuid.uuid4())),
        "scenario_id": metadata.get("scenario_id", 1),
        "scenario_name": metadata.get("scenario_name", "Unknown Scenario"),
        "start_time": old_data.get("start_time", metadata.get("start_time", "")),
        "end_time": old_data.get("end_time", ""),
        "duration_seconds": calculate_duration(old_data),
        "strategy_name": strategy_name,
        "strategy_params": {
            "legacy_conversion": True,
            "source_file": filename
        },
        "constraints": old_data.get("constraints", []),
        "attribute_frequencies": old_data.get("attribute_frequencies", {}),
        "attribute_correlations": old_data.get("attribute_correlations", {}),
        "final_stats": {
            "total_people_seen": total_people,
            "total_admitted": admitted,
            "total_rejected": rejected,
            "game_successful": determine_game_success(old_data),
            "constraints_met": {
                constraint["attribute"]: sum(
                    1 for p in old_data.get("people", []) 
                    if p.get("decision", False) and 
                    p.get("attributes", {}).get(constraint["attribute"], False)
                ) >= constraint["minCount"]
                for constraint in old_data.get("constraints", [])
            }
        },
        "decisions": []
    }
    
    # Transform individual decisions
    for person in old_data.get("people", []):
        decision = {
            "person_index": person.get("person_index", 0),
            "attributes": person.get("attributes", {}),
            "decision": person.get("decision", False),
            "reasoning": "legacy_decision",
            "timestamp": person.get("timestamp", 0),
            "admitted_count": None,  # Not available in old format
            "rejected_count": None,  # Not available in old format
            "constraint_progress": {}  # Not available in old format
        }
        new_data["decisions"].append(decision)
    
    return new_data


def create_events_jsonl(game_data: Dict[str, Any]) -> List[str]:
    """Create JSONL events from game data."""
    events = []
    
    # Create events for each decision
    admitted = 0
    rejected = 0
    
    for decision in game_data["decisions"]:
        if decision["decision"]:
            admitted += 1
        else:
            rejected += 1
        
        # Calculate progress
        progress = {}
        for constraint in game_data.get("constraints", []):
            attr = constraint["attribute"]
            count = sum(1 for d in game_data["decisions"][:decision["person_index"]+1] 
                       if d["decision"] and d["attributes"].get(attr, False))
            progress[attr] = count / constraint["minCount"]
        
        event = {
            "type": "api_response",
            "solver_id": game_data["solver_id"],
            "timestamp": datetime.fromtimestamp(decision["timestamp"]).isoformat() if decision["timestamp"] else "",
            "data": {
                "person_index": decision["person_index"],
                "attributes": decision["attributes"],
                "decision": decision["decision"],
                "reasoning": decision["reasoning"],
                "admitted": admitted,
                "rejected": rejected,
                "progress": progress,
                "status": "running"
            }
        }
        events.append(json.dumps(event))
    
    return events


def process_archive_files():
    """Process all archive files and create consolidated output."""
    archive_dir = Path("archive_old_files")
    game_logs_dir = Path("game_logs")
    
    # Find all JSON files in archive
    json_files = list(archive_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in archive_old_files directory")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    all_games = []
    all_events = []
    
    for json_file in json_files:
        print(f"Processing {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                old_data = json.load(f)
            
            # Transform to new format
            new_game = transform_old_game_to_new_format(old_data, json_file.name)
            all_games.append(new_game)
            
            # Create events
            events = create_events_jsonl(new_game)
            all_events.extend(events)
            
            print(f"  ✓ Converted {len(new_game['decisions'])} decisions")
            
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
            continue
    
    # Process JSONL files if any
    jsonl_files = list(archive_dir.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.name} (copying events)")
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Parse and potentially transform old JSONL format
                        event_data = json.loads(line)
                        # Add legacy marker
                        if "solver_id" in event_data:
                            event_data["solver_id"] = f"legacy_{event_data['solver_id']}"
                        all_events.append(json.dumps(event_data))
        except Exception as e:
            print(f"  ✗ Error processing {jsonl_file.name}: {e}")
    
    # Save consolidated files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save consolidated games
    games_file = game_logs_dir / f"consolidated_legacy_games_{timestamp}.json"
    with open(games_file, 'w') as f:
        json.dump({
            "metadata": {
                "source": "archive_old_files",
                "conversion_timestamp": datetime.now().isoformat(),
                "total_games": len(all_games),
                "total_events": len(all_events)
            },
            "games": all_games
        }, f, indent=2)
    
    # Save consolidated events
    events_file = game_logs_dir / f"consolidated_legacy_events_{timestamp}.jsonl"
    with open(events_file, 'w') as f:
        for event in all_events:
            f.write(event + '\n')
    
    print(f"\nConversion complete!")
    print(f"Games saved to: {games_file}")
    print(f"Events saved to: {events_file}")
    print(f"Total games: {len(all_games)}")
    print(f"Total events: {len(all_events)}")


if __name__ == "__main__":
    process_archive_files()