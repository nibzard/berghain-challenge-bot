# ABOUTME: Statistical analysis utilities for game data
# ABOUTME: Provides statistical methods for pattern detection

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis tools for game data."""
    
    def __init__(self, logs_directory: str = "game_logs"):
        self.logs_directory = Path(logs_directory)
    
    def compare_strategies(self, strategy_a: str, strategy_b: str, 
                         scenario: int = None) -> Dict[str, Any]:
        """Compare two strategies statistically."""
        
        # Load games for each strategy
        games_a = self._load_games_for_strategy(strategy_a, scenario)
        games_b = self._load_games_for_strategy(strategy_b, scenario)
        
        if not games_a or not games_b:
            return {"error": "Insufficient data for comparison"}
        
        # Extract metrics
        rejections_a = [g["rejected_count"] for g in games_a if g["success"]]
        rejections_b = [g["rejected_count"] for g in games_b if g["success"]]
        
        success_rate_a = len([g for g in games_a if g["success"]]) / len(games_a)
        success_rate_b = len([g for g in games_b if g["success"]]) / len(games_b)
        
        analysis = {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "scenario_filter": scenario,
            "games_a": len(games_a),
            "games_b": len(games_b),
            "success_rate_a": success_rate_a,
            "success_rate_b": success_rate_b,
            "successful_games_a": len(rejections_a),
            "successful_games_b": len(rejections_b)
        }
        
        # Statistical comparison of rejection counts (for successful games)
        if rejections_a and rejections_b:
            # T-test for rejection counts
            t_stat, p_value = stats.ttest_ind(rejections_a, rejections_b)
            
            analysis["rejection_comparison"] = {
                "mean_rejections_a": np.mean(rejections_a),
                "mean_rejections_b": np.mean(rejections_b),
                "std_rejections_a": np.std(rejections_a),
                "std_rejections_b": np.std(rejections_b),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_difference": p_value < 0.05,
                "better_strategy": strategy_a if np.mean(rejections_a) < np.mean(rejections_b) else strategy_b
            }
        
        # Chi-square test for success rates
        contingency_table = [
            [len([g for g in games_a if g["success"]]), len([g for g in games_a if not g["success"]])],
            [len([g for g in games_b if g["success"]]), len([g for g in games_b if not g["success"]])]
        ]
        
        chi2_stat, chi2_p = stats.chi2_contingency(contingency_table)[:2]
        
        analysis["success_rate_comparison"] = {
            "chi2_statistic": chi2_stat,
            "p_value": chi2_p,
            "significant_difference": chi2_p < 0.05,
            "better_success_rate": strategy_a if success_rate_a > success_rate_b else strategy_b
        }
        
        return analysis
    
    def analyze_parameter_impact(self, strategy: str, parameter_name: str, 
                               scenario: int = None) -> Dict[str, Any]:
        """Analyze the impact of a specific parameter on performance."""
        
        games = self._load_games_for_strategy(strategy, scenario)
        if not games:
            return {"error": f"No games found for strategy {strategy}"}
        
        # Extract parameter values and outcomes
        parameter_values = []
        outcomes = []
        
        for game in games:
            strategy_params = game.get("strategy_params", {})
            if parameter_name in strategy_params:
                parameter_values.append(strategy_params[parameter_name])
                outcomes.append({
                    "success": game["success"],
                    "rejected_count": game["rejected_count"],
                    "duration": game.get("duration", 0)
                })
        
        if not parameter_values:
            return {"error": f"Parameter {parameter_name} not found in strategy data"}
        
        # Analyze correlation with success rate
        success_values = [1 if o["success"] else 0 for o in outcomes]
        rejection_values = [o["rejected_count"] for o in outcomes]
        
        analysis = {
            "strategy": strategy,
            "parameter": parameter_name,
            "scenario_filter": scenario,
            "total_games": len(parameter_values),
            "parameter_range": {
                "min": min(parameter_values),
                "max": max(parameter_values),
                "mean": np.mean(parameter_values),
                "std": np.std(parameter_values)
            }
        }
        
        # Correlation analysis
        if len(set(parameter_values)) > 1:  # Need variation in parameter values
            # Correlation with success rate
            success_corr, success_p = stats.pearsonr(parameter_values, success_values)
            
            # Correlation with rejection count (for successful games only)
            successful_indices = [i for i, o in enumerate(outcomes) if o["success"]]
            if len(successful_indices) > 1:
                successful_param_values = [parameter_values[i] for i in successful_indices]
                successful_rejections = [rejection_values[i] for i in successful_indices]
                
                rejection_corr, rejection_p = stats.pearsonr(successful_param_values, successful_rejections)
            else:
                rejection_corr, rejection_p = None, None
            
            analysis["correlation_analysis"] = {
                "success_correlation": success_corr,
                "success_p_value": success_p,
                "success_significant": success_p < 0.05 if success_p is not None else False,
                "rejection_correlation": rejection_corr,
                "rejection_p_value": rejection_p,
                "rejection_significant": rejection_p < 0.05 if rejection_p is not None else False
            }
        
        # Binned analysis (if enough data)
        if len(parameter_values) >= 10:
            analysis["binned_analysis"] = self._analyze_parameter_bins(
                parameter_values, outcomes, num_bins=3
            )
        
        return analysis
    
    def _analyze_parameter_bins(self, parameter_values: List[float], 
                               outcomes: List[Dict], num_bins: int = 3) -> Dict[str, Any]:
        """Analyze parameter impact by binning values."""
        
        # Create bins
        min_val, max_val = min(parameter_values), max(parameter_values)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        
        bins = defaultdict(list)
        for param_val, outcome in zip(parameter_values, outcomes):
            bin_idx = min(num_bins - 1, int((param_val - min_val) / (max_val - min_val) * num_bins))
            bins[bin_idx].append(outcome)
        
        # Analyze each bin
        bin_analysis = {}
        for bin_idx, bin_outcomes in bins.items():
            if bin_outcomes:
                successful = [o for o in bin_outcomes if o["success"]]
                
                bin_analysis[f"bin_{bin_idx}"] = {
                    "range": f"{bin_edges[bin_idx]:.3f} - {bin_edges[bin_idx + 1]:.3f}",
                    "total_games": len(bin_outcomes),
                    "successful_games": len(successful),
                    "success_rate": len(successful) / len(bin_outcomes),
                    "avg_rejections": np.mean([o["rejected_count"] for o in successful]) if successful else None
                }
        
        return bin_analysis
    
    def _load_games_for_strategy(self, strategy: str, scenario: int = None) -> List[Dict[str, Any]]:
        """Load game data for a specific strategy."""
        pattern = f"game_*{strategy}*.json"
        json_files = list(self.logs_directory.glob(pattern))
        
        games = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.loads(f.read())
                
                # Filter by scenario if specified
                if scenario is not None and game_data.get("scenario_id", 0) != scenario:
                    continue
                
                games.append({
                    "game_id": game_data.get("game_id", "unknown"),
                    "solver_id": game_data.get("solver_id", "unknown"),
                    "scenario": game_data.get("scenario_id", 0),
                    "success": game_data.get("success", False),
                    "rejected_count": game_data.get("rejected_count", 0),
                    "duration": game_data.get("duration_seconds", 0),
                    "strategy_params": game_data.get("strategy_params", {})
                })
                
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        return games
    
    def generate_performance_report(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Get all recent games
        json_files = list(self.logs_directory.glob("game_*.json"))
        
        # Filter by date
        import time
        cutoff_time = time.time() - (days_back * 24 * 60 * 60)
        recent_files = [f for f in json_files if f.stat().st_mtime >= cutoff_time]
        
        if not recent_files:
            return {"error": f"No games found in the last {days_back} days"}
        
        # Load all games
        all_games = []
        for json_file in recent_files:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.loads(f.read())
                all_games.append(game_data)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        # Overall statistics
        total_games = len(all_games)
        successful_games = [g for g in all_games if g.get("success", False)]
        
        report = {
            "period": f"Last {days_back} days",
            "total_games": total_games,
            "successful_games": len(successful_games),
            "overall_success_rate": len(successful_games) / total_games if total_games > 0 else 0
        }
        
        # Best performances
        if successful_games:
            best_game = min(successful_games, key=lambda x: x.get("rejected_count", float('inf')))
            report["best_performance"] = {
                "solver_id": best_game.get("solver_id", "unknown"),
                "scenario": best_game.get("scenario_id", 0),
                "rejected_count": best_game.get("rejected_count", 0),
                "strategy_params": best_game.get("strategy_params", {})
            }
        
        # Trend analysis (if enough games)
        if total_games >= 10:
            # Sort by timestamp
            all_games.sort(key=lambda x: x.get("timestamp", ""))
            
            # Calculate success rate trend
            window_size = max(5, total_games // 4)
            early_games = all_games[:window_size]
            late_games = all_games[-window_size:]
            
            early_success_rate = len([g for g in early_games if g.get("success", False)]) / len(early_games)
            late_success_rate = len([g for g in late_games if g.get("success", False)]) / len(late_games)
            
            report["trend_analysis"] = {
                "early_success_rate": early_success_rate,
                "late_success_rate": late_success_rate,
                "improvement": late_success_rate - early_success_rate,
                "trending_up": late_success_rate > early_success_rate
            }
        
        return report