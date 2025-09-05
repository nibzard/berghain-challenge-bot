# ABOUTME: Game analysis tools for post-game insights
# ABOUTME: Analyzes patterns and provides recommendations

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import statistics

from ..logging import GameLogger


class GameAnalyzer:
    """Analyzes game results for insights and patterns."""
    
    def __init__(self, logs_directory: str = "game_logs"):
        self.logs_directory = Path(logs_directory)
        self.game_logger = GameLogger(logs_directory)
    
    def analyze_recent_games(self, limit: int = 20) -> Dict[str, Any]:
        """Analyze recent games for patterns."""
        recent_games = self.game_logger.get_recent_games(limit)
        
        if not recent_games:
            return {"error": "No games found"}
        
        # Basic statistics
        total_games = len(recent_games)
        successful_games = [g for g in recent_games if g["success"]]
        success_rate = len(successful_games) / total_games
        
        # Rejection statistics
        all_rejections = [g["rejected_count"] for g in recent_games]
        successful_rejections = [g["rejected_count"] for g in successful_games]
        
        analysis = {
            "summary": {
                "total_games": total_games,
                "successful_games": len(successful_games),
                "success_rate": success_rate,
                "avg_rejections_all": statistics.mean(all_rejections) if all_rejections else 0,
                "avg_rejections_successful": statistics.mean(successful_rejections) if successful_rejections else 0,
                "min_rejections": min(all_rejections) if all_rejections else 0,
                "max_rejections": max(all_rejections) if all_rejections else 0
            }
        }
        
        # Group by scenario
        by_scenario = defaultdict(list)
        for game in recent_games:
            by_scenario[game["scenario"]].append(game)
        
        scenario_analysis = {}
        for scenario, games in by_scenario.items():
            successful = [g for g in games if g["success"]]
            scenario_analysis[scenario] = {
                "total_games": len(games),
                "successful_games": len(successful),
                "success_rate": len(successful) / len(games),
                "avg_rejections": statistics.mean([g["rejected_count"] for g in successful]) if successful else None,
                "best_result": min(successful, key=lambda x: x["rejected_count"]) if successful else None
            }
        
        analysis["by_scenario"] = scenario_analysis
        
        # Group by solver type
        by_solver = defaultdict(list)
        for game in recent_games:
            solver_type = game["solver_id"].split('_')[0] if '_' in game["solver_id"] else game["solver_id"]
            by_solver[solver_type].append(game)
        
        solver_analysis = {}
        for solver_type, games in by_solver.items():
            successful = [g for g in games if g["success"]]
            solver_analysis[solver_type] = {
                "total_games": len(games),
                "successful_games": len(successful),
                "success_rate": len(successful) / len(games),
                "avg_rejections": statistics.mean([g["rejected_count"] for g in successful]) if successful else None
            }
        
        analysis["by_solver"] = solver_analysis
        
        # Recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        summary = analysis["summary"]
        success_rate = summary["success_rate"]
        
        if success_rate < 0.1:
            recommendations.append("Success rate is very low. Consider using more conservative strategies.")
        elif success_rate < 0.3:
            recommendations.append("Success rate is low. Try adjusting strategy parameters or using adaptive approach.")
        elif success_rate > 0.7:
            recommendations.append("Good success rate! Try optimizing for lower rejection counts.")
        
        # Scenario-specific recommendations
        scenario_analysis = analysis.get("by_scenario", {})
        for scenario, data in scenario_analysis.items():
            if data["success_rate"] == 0:
                recommendations.append(f"Scenario {scenario}: No successful games yet. This scenario may need special strategy tuning.")
            elif data["success_rate"] < 0.2:
                recommendations.append(f"Scenario {scenario}: Low success rate. Consider more aggressive parameter tuning.")
        
        # Solver recommendations
        solver_analysis = analysis.get("by_solver", {})
        if len(solver_analysis) > 1:
            best_solver = max(solver_analysis.items(), key=lambda x: x[1]["success_rate"])
            recommendations.append(f"Best performing solver type: {best_solver[0]} ({best_solver[1]['success_rate']:.1%} success rate)")
        
        return recommendations
    
    def analyze_constraint_patterns(self, scenario: int, limit: int = 10) -> Dict[str, Any]:
        """Analyze constraint satisfaction patterns for a specific scenario."""
        # Load detailed game logs for constraint analysis
        json_files = list(self.logs_directory.glob(f"game_*_s{scenario}_*.json"))
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        constraint_data = []
        for json_file in json_files[:limit]:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.loads(f.read())
                
                constraint_satisfaction = game_data.get("constraint_satisfaction", {})
                if constraint_satisfaction:
                    constraint_data.append({
                        "game_id": game_data.get("game_id", "unknown"),
                        "success": game_data.get("success", False),
                        "constraints": constraint_satisfaction
                    })
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        if not constraint_data:
            return {"error": f"No constraint data found for scenario {scenario}"}
        
        # Analyze constraint satisfaction patterns
        constraint_names = set()
        for game in constraint_data:
            constraint_names.update(game["constraints"].keys())
        
        analysis = {
            "scenario": scenario,
            "games_analyzed": len(constraint_data),
            "successful_games": len([g for g in constraint_data if g["success"]]),
            "constraint_analysis": {}
        }
        
        for constraint_name in constraint_names:
            satisfactions = []
            shortages = []
            
            for game in constraint_data:
                if constraint_name in game["constraints"]:
                    constraint_info = game["constraints"][constraint_name]
                    satisfactions.append(constraint_info.get("satisfied", False))
                    shortages.append(constraint_info.get("shortage", 0))
            
            if satisfactions:
                analysis["constraint_analysis"][constraint_name] = {
                    "satisfaction_rate": sum(satisfactions) / len(satisfactions),
                    "avg_shortage": statistics.mean(shortages) if shortages else 0,
                    "max_shortage": max(shortages) if shortages else 0,
                    "games_satisfied": sum(satisfactions),
                    "total_games": len(satisfactions)
                }
        
        return analysis
    
    def find_best_strategies(self, scenario: int = None, limit: int = 50) -> Dict[str, Any]:
        """Find the best performing strategies."""
        recent_games = self.game_logger.get_recent_games(limit)
        
        if scenario is not None:
            recent_games = [g for g in recent_games if g["scenario"] == scenario]
        
        successful_games = [g for g in recent_games if g["success"]]
        
        if not successful_games:
            return {"error": "No successful games found"}
        
        # Sort by rejection count (lower is better)
        successful_games.sort(key=lambda x: x["rejected_count"])
        
        # Get top strategies
        top_strategies = successful_games[:min(5, len(successful_games))]
        
        # Load detailed strategy parameters for top games
        strategy_details = []
        for game_summary in top_strategies:
            try:
                # Find the actual game file
                pattern = f"*{game_summary['game_id']}*.json"
                game_files = list(self.logs_directory.glob(pattern))
                
                if game_files:
                    with open(game_files[0], 'r') as f:
                        game_data = json.loads(f.read())
                    
                    strategy_details.append({
                        "game_id": game_summary["game_id"],
                        "solver_id": game_summary["solver_id"],
                        "scenario": game_summary["scenario"],
                        "rejected_count": game_summary["rejected_count"],
                        "strategy_params": game_data.get("strategy_params", {}),
                        "duration": game_summary["duration"]
                    })
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        return {
            "scenario_filter": scenario,
            "total_successful_games": len(successful_games),
            "top_strategies": strategy_details,
            "best_rejection_count": successful_games[0]["rejected_count"] if successful_games else None
        }