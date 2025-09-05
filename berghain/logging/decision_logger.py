# ABOUTME: Decision-level logging for detailed analysis
# ABOUTME: Handles streaming decision logs for real-time analysis

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from ..core import Decision


class DecisionLogger:
    """Logs individual decisions for detailed analysis."""
    
    def __init__(self, logs_directory: str = "game_logs"):
        self.logs_directory = Path(logs_directory)
        self.logs_directory.mkdir(exist_ok=True)
        self.decision_stream_file = self.logs_directory / "decision_stream.jsonl"
    
    def log_decision(self, decision: Decision, game_context: Dict[str, Any] = None):
        """Log a single decision to the decision stream."""
        decision_data = {
            "timestamp": decision.timestamp.isoformat(),
            "person_index": decision.person.index,
            "person_attributes": decision.person.attributes,
            "decision": decision.accepted,
            "reasoning": decision.reasoning,
            "game_context": game_context or {}
        }
        
        # Append to JSONL stream
        with open(self.decision_stream_file, 'a') as f:
            f.write(json.dumps(decision_data) + '\n')
    
    def log_decision_batch(self, decisions: List[Decision], game_id: str):
        """Log a batch of decisions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decisions_{game_id[:8]}_{timestamp}.jsonl"
        filepath = self.logs_directory / filename
        
        with open(filepath, 'w') as f:
            for decision in decisions:
                decision_data = {
                    "game_id": game_id,
                    "timestamp": decision.timestamp.isoformat(),
                    "person_index": decision.person.index,
                    "person_attributes": decision.person.attributes,
                    "decision": decision.accepted,
                    "reasoning": decision.reasoning
                }
                f.write(json.dumps(decision_data) + '\n')
        
        return filepath
    
    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decisions from the stream."""
        if not self.decision_stream_file.exists():
            return []
        
        decisions = []
        with open(self.decision_stream_file, 'r') as f:
            lines = f.readlines()
            
        # Get last N lines
        for line in lines[-limit:]:
            try:
                decision_data = json.loads(line.strip())
                decisions.append(decision_data)
            except json.JSONDecodeError:
                continue
        
        return decisions
    
    def clear_old_decisions(self, days_to_keep: int = 7):
        """Clear old decisions from the stream file."""
        if not self.decision_stream_file.exists():
            return
        
        cutoff_time = datetime.now() - datetime.timedelta(days=days_to_keep)
        temp_file = self.decision_stream_file.with_suffix('.tmp')
        
        kept_count = 0
        with open(self.decision_stream_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                try:
                    decision_data = json.loads(line.strip())
                    decision_time = datetime.fromisoformat(decision_data['timestamp'])
                    
                    if decision_time >= cutoff_time:
                        outfile.write(line)
                        kept_count += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        # Replace original with cleaned file
        temp_file.replace(self.decision_stream_file)
        
        return kept_count