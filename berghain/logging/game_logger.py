# ABOUTME: Structured game logging with consistent format
# ABOUTME: Single responsibility for game data persistence

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from ..core import GameResult


logger = logging.getLogger(__name__)


class GameLogger:
    """Handles structured logging of game results."""
    
    def __init__(self, logs_directory: str = "game_logs"):
        self.logs_directory = Path(logs_directory)
        self.logs_directory.mkdir(exist_ok=True)
    
    def log_game_result(self, result: GameResult, additional_data: Dict[str, Any] = None) -> Path:
        """Log a complete game result to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{result.solver_id}_{timestamp}_{result.game_state.game_id[:8]}.json"
        filepath = self.logs_directory / filename
        
        # Create structured log data
        log_data = {
            # Metadata
            "log_version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "solver_id": result.solver_id,
            "game_id": result.game_state.game_id,
            
            # Game info
            "scenario_id": result.game_state.scenario,
            "start_time": result.game_state.start_time.isoformat(),
            "end_time": result.game_state.end_time.isoformat() if result.game_state.end_time else None,
            "duration_seconds": result.duration,
            
            # Strategy
            "strategy_params": result.strategy_params,
            
            # Constraints and statistics
            "constraints": [
                {
                    "attribute": c.attribute,
                    "min_count": c.min_count
                }
                for c in result.game_state.constraints
            ],
            "attribute_frequencies": result.game_state.statistics.frequencies,
            "attribute_correlations": result.game_state.statistics.correlations,
            
            # Results
            "status": result.game_state.status.value,
            "success": result.success,
            "admitted_count": result.game_state.admitted_count,
            "rejected_count": result.game_state.rejected_count,
            "total_decisions": result.total_decisions,
            "acceptance_rate": result.acceptance_rate,
            
            # Final state
            "final_admitted_attributes": result.game_state.admitted_attributes,
            "constraint_satisfaction": result.constraint_satisfaction_summary(),
            
            # Decision log (sample to avoid huge files)
            "decisions_sample": [
                {
                    "person_index": d.person.index,
                    "attributes": d.person.attributes,
                    "decision": d.accepted,
                    "reasoning": d.reasoning,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in self._sample_decisions(result.decisions)
            ]
        }
        
        # Add any additional data
        if additional_data:
            log_data.update(additional_data)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"💾 Game log saved: {filename}")
        return filepath
    
    def _sample_decisions(self, decisions: List, max_decisions: int = 1000) -> List:
        """Sample decisions to avoid huge log files."""
        if len(decisions) <= max_decisions:
            return decisions
        
        # Keep first 200, last 200, and sample from middle
        first_chunk = decisions[:200]
        last_chunk = decisions[-200:]
        middle_chunk = decisions[200:-200]
        
        # Sample from middle
        if middle_chunk:
            step = max(1, len(middle_chunk) // (max_decisions - 400))
            sampled_middle = middle_chunk[::step][:max_decisions - 400]
        else:
            sampled_middle = []
        
        return first_chunk + sampled_middle + last_chunk
    
    def log_batch_summary(self, batch_results: List[GameResult], batch_metadata: Dict[str, Any] = None) -> Path:
        """Log summary of batch results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_summary_{timestamp}.json"
        filepath = self.logs_directory / filename
        
        # Calculate batch statistics
        successful_games = [r for r in batch_results if r.success]
        by_scenario = {}
        by_solver = {}
        
        for result in batch_results:
            # Group by scenario
            scenario = result.game_state.scenario
            if scenario not in by_scenario:
                by_scenario[scenario] = {"total": 0, "successful": 0, "results": []}
            by_scenario[scenario]["total"] += 1
            if result.success:
                by_scenario[scenario]["successful"] += 1
            by_scenario[scenario]["results"].append({
                "solver_id": result.solver_id,
                "success": result.success,
                "rejected_count": result.game_state.rejected_count,
                "duration": result.duration
            })
            
            # Group by solver type
            solver_type = result.solver_id.split('_')[0] if '_' in result.solver_id else result.solver_id
            if solver_type not in by_solver:
                by_solver[solver_type] = {"total": 0, "successful": 0, "results": []}
            by_solver[solver_type]["total"] += 1
            if result.success:
                by_solver[solver_type]["successful"] += 1
            by_solver[solver_type]["results"].append({
                "solver_id": result.solver_id,
                "scenario": scenario,
                "success": result.success,
                "rejected_count": result.game_state.rejected_count
            })
        
        # Create summary
        summary_data = {
            "log_version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "batch_metadata": batch_metadata or {},
            
            # Overall statistics
            "total_games": len(batch_results),
            "successful_games": len(successful_games),
            "success_rate": len(successful_games) / len(batch_results) if batch_results else 0,
            
            # Best result
            "best_result": None,
            
            # Breakdowns
            "by_scenario": by_scenario,
            "by_solver_type": by_solver
        }
        
        # Find best result (successful with lowest rejections)
        if successful_games:
            best_result = min(successful_games, key=lambda r: r.game_state.rejected_count)
            summary_data["best_result"] = {
                "solver_id": best_result.solver_id,
                "scenario": best_result.game_state.scenario,
                "rejected_count": best_result.game_state.rejected_count,
                "duration": best_result.duration,
                "strategy_params": best_result.strategy_params
            }
        
        # Write summary
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"💾 Batch summary saved: {filename}")
        return filepath
    
    def get_recent_games(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent game summaries."""
        json_files = list(self.logs_directory.glob("game_*.json"))
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        recent_games = []
        for json_file in json_files[:limit]:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.loads(f.read())
                
                # Extract summary - handle both new and legacy formats
                summary = {
                    "filepath": str(json_file),
                    "game_id": game_data.get("game_id", "unknown"),
                    "solver_id": game_data.get("solver_id", "unknown"),
                    "scenario": game_data.get("scenario_id", game_data.get("scenario", 0)),
                    "success": game_data.get("success", game_data.get("final_status") == "success"),
                    "rejected_count": game_data.get("rejected_count", game_data.get("final_rejected_count", 0)),
                    "duration": game_data.get("duration_seconds", game_data.get("total_time", 0)),
                    "timestamp": game_data.get("timestamp", game_data.get("start_time", ""))
                }
                
                recent_games.append(summary)
                
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        return recent_games