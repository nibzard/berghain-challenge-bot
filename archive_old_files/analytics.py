# ABOUTME: Advanced analytics engine for Berghain game performance analysis
# ABOUTME: Provides pattern detection, success prediction, and strategy extraction

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
import statistics
from collections import defaultdict, Counter
import math

@dataclass
class GameAnalytics:
    game_id: str
    scenario: int
    bot_type: str
    success: bool
    efficiency: float
    constraint_fulfillment: Dict[str, float]  # attribute -> fulfillment rate
    decision_patterns: Dict[str, Any]
    temporal_patterns: Dict[str, Any]
    attribute_correlations: Dict[str, float]
    performance_score: float

@dataclass
class StrategyProfile:
    name: str
    success_rate: float
    average_efficiency: float
    typical_patterns: Dict[str, Any]
    optimal_conditions: Dict[str, Any]
    failure_modes: List[str]
    games_analyzed: int

@dataclass
class PerformanceTrend:
    time_period: str
    success_rate_trend: List[float]
    efficiency_trend: List[float]
    pattern_evolution: Dict[str, List[float]]

class GameAnalyzer:
    """Analyzes individual game performance and extracts patterns"""
    
    def __init__(self):
        pass
        
    def analyze_game(self, game_data: Dict[str, Any]) -> GameAnalytics:
        """Perform comprehensive analysis of a single game"""
        
        # Basic metrics
        success = self._determine_success(game_data)
        efficiency = self._calculate_efficiency(game_data)
        constraint_fulfillment = self._analyze_constraints(game_data)
        
        # Pattern analysis
        decision_patterns = self._extract_decision_patterns(game_data)
        temporal_patterns = self._analyze_temporal_patterns(game_data)
        attribute_correlations = self._calculate_attribute_correlations(game_data)
        
        # Overall performance score
        performance_score = self._calculate_performance_score(
            success, efficiency, constraint_fulfillment, decision_patterns
        )
        
        return GameAnalytics(
            game_id=game_data['game_id'][:8],
            scenario=game_data['scenario'],
            bot_type=self._extract_bot_type(game_data),
            success=success,
            efficiency=efficiency,
            constraint_fulfillment=constraint_fulfillment,
            decision_patterns=decision_patterns,
            temporal_patterns=temporal_patterns,
            attribute_correlations=attribute_correlations,
            performance_score=performance_score
        )
        
    def _determine_success(self, game_data: Dict[str, Any]) -> bool:
        """Determine if game was successful"""
        if game_data.get('final_status') != 'completed':
            return False
            
        # Check capacity limit
        if game_data.get('final_admitted_count', 0) > 1000:
            return False
            
        # Check rejection limit  
        if game_data.get('final_rejected_count', 0) > 20000:
            return False
            
        # Check constraints
        constraints = game_data.get('constraints', [])
        people = game_data.get('people', [])
        
        for constraint in constraints:
            attr = constraint['attribute']
            required = constraint.get('minCount', constraint.get('requiredCount', 0))
            
            admitted_with_attr = sum(1 for person in people
                                   if person.get('decision', False) and 
                                   person.get('attributes', {}).get(attr, False))
                                   
            if admitted_with_attr < required:
                return False
                
        return True
        
    def _calculate_efficiency(self, game_data: Dict[str, Any]) -> float:
        """Calculate admission efficiency"""
        admitted = game_data.get('final_admitted_count', 0)
        rejected = game_data.get('final_rejected_count', 0)
        total = admitted + rejected
        
        return (admitted / total) if total > 0 else 0.0
        
    def _analyze_constraints(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze constraint fulfillment rates"""
        constraints = game_data.get('constraints', [])
        people = game_data.get('people', [])
        
        fulfillment = {}
        for constraint in constraints:
            attr = constraint['attribute']
            required = constraint.get('minCount', constraint.get('requiredCount', 0))
            
            admitted_with_attr = sum(1 for person in people
                                   if person.get('decision', False) and 
                                   person.get('attributes', {}).get(attr, False))
                                   
            fulfillment[attr] = (admitted_with_attr / required) if required > 0 else 0.0
            
        return fulfillment
        
    def _extract_decision_patterns(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract decision-making patterns"""
        people = game_data.get('people', [])
        if not people:
            return {}
            
        # Basic decision statistics
        total_decisions = len(people)
        accepted = sum(1 for p in people if p.get('decision', False))
        acceptance_rate = accepted / total_decisions if total_decisions > 0 else 0
        
        # Attribute-based decision analysis
        attribute_decisions = defaultdict(lambda: {'accepted': 0, 'total': 0})
        
        for person in people:
            decision = person.get('decision', False)
            attributes = person.get('attributes', {})
            
            for attr, value in attributes.items():
                if value:  # Person has this attribute
                    attribute_decisions[attr]['total'] += 1
                    if decision:
                        attribute_decisions[attr]['accepted'] += 1
                        
        # Calculate acceptance rates by attribute
        attribute_acceptance_rates = {}
        for attr, stats in attribute_decisions.items():
            rate = stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
            attribute_acceptance_rates[attr] = rate
            
        # Decision consistency analysis
        decision_sequence = [p.get('decision', False) for p in people]
        consistency_score = self._calculate_decision_consistency(decision_sequence)
        
        # Selectivity analysis (how picky the bot is)
        selectivity_score = 1.0 - acceptance_rate
        
        return {
            'acceptance_rate': acceptance_rate,
            'total_decisions': total_decisions,
            'attribute_acceptance_rates': attribute_acceptance_rates,
            'consistency_score': consistency_score,
            'selectivity_score': selectivity_score
        }
        
    def _calculate_decision_consistency(self, decisions: List[bool]) -> float:
        """Calculate how consistent decisions are (less random = higher score)"""
        if len(decisions) < 2:
            return 1.0
            
        # Calculate runs (sequences of same decision)
        runs = 1
        for i in range(1, len(decisions)):
            if decisions[i] != decisions[i-1]:
                runs += 1
                
        # Normalize by expected runs for random sequence
        expected_runs = (2 * sum(decisions) * (len(decisions) - sum(decisions))) / len(decisions) + 1
        
        return 1.0 - min(runs / expected_runs, 2.0) / 2.0 if expected_runs > 0 else 1.0
        
    def _analyze_temporal_patterns(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal decision patterns"""
        people = game_data.get('people', [])
        if not people:
            return {}
            
        # Decision rate over time
        timestamps = [p.get('timestamp', 0) for p in people if 'timestamp' in p]
        if len(timestamps) < 2:
            return {'decisions_per_second': 0}
            
        time_span = max(timestamps) - min(timestamps)
        decisions_per_second = len(people) / time_span if time_span > 0 else 0
        
        # Early vs late game behavior
        mid_point = len(people) // 2
        early_acceptance_rate = sum(1 for p in people[:mid_point] if p.get('decision', False)) / mid_point if mid_point > 0 else 0
        late_acceptance_rate = sum(1 for p in people[mid_point:] if p.get('decision', False)) / (len(people) - mid_point) if len(people) - mid_point > 0 else 0
        
        return {
            'decisions_per_second': decisions_per_second,
            'game_duration': game_data.get('total_time', 0),
            'early_acceptance_rate': early_acceptance_rate,
            'late_acceptance_rate': late_acceptance_rate,
            'acceptance_rate_change': late_acceptance_rate - early_acceptance_rate
        }
        
    def _calculate_attribute_correlations(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how well bot leveraged attribute correlations"""
        correlations = game_data.get('attribute_correlations', {})
        people = game_data.get('people', [])
        
        if not correlations or not people:
            return {}
            
        # Measure how well bot exploited known correlations
        exploitation_scores = {}
        
        for attr1, attr1_correls in correlations.items():
            for attr2, correlation in attr1_correls.items():
                if attr1 == attr2 or abs(correlation) < 0.1:
                    continue
                    
                # Find people with both attributes
                both_attrs = [p for p in people if 
                             p.get('attributes', {}).get(attr1, False) and 
                             p.get('attributes', {}).get(attr2, False)]
                             
                if both_attrs:
                    acceptance_rate = sum(1 for p in both_attrs if p.get('decision', False)) / len(both_attrs)
                    # Higher acceptance for positive correlations, lower for negative
                    expected_exploitation = 0.5 + (correlation * 0.3)  # Normalize to 0.2-0.8 range
                    exploitation_score = 1.0 - abs(acceptance_rate - expected_exploitation)
                    exploitation_scores[f"{attr1}_{attr2}"] = exploitation_score
                    
        return exploitation_scores
        
    def _calculate_performance_score(self, success: bool, efficiency: float, 
                                   constraints: Dict[str, float], patterns: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        if not success:
            return 0.0
            
        # Base score from efficiency
        score = efficiency * 0.3
        
        # Constraint fulfillment bonus
        constraint_score = statistics.mean(min(1.0, rate) for rate in constraints.values()) if constraints else 0
        score += constraint_score * 0.4
        
        # Pattern quality bonus
        consistency_bonus = patterns.get('consistency_score', 0) * 0.1
        selectivity_bonus = min(patterns.get('selectivity_score', 0), 0.8) * 0.1  # Reward selectivity up to 80%
        pattern_bonus = consistency_bonus + selectivity_bonus
        score += pattern_bonus
        
        # Speed bonus (if temporal data available)
        temporal = patterns.get('temporal_patterns', {})
        if 'decisions_per_second' in temporal and temporal['decisions_per_second'] > 0:
            speed_bonus = min(temporal['decisions_per_second'] / 10, 0.1)  # Up to 0.1 bonus
            score += speed_bonus
            
        return min(score, 1.0)  # Cap at 1.0
        
    def _extract_bot_type(self, game_data: Dict[str, Any]) -> str:
        """Attempt to extract bot type from game data or filename"""
        # This would need to be enhanced based on how bot types are stored
        return "unknown"

class SuccessPredictor:
    """Predicts game success based on early indicators"""
    
    def __init__(self):
        self.models = {}  # Will store predictive models per scenario
        
    def train_from_games(self, game_analytics: List[GameAnalytics]):
        """Train success prediction models from historical game data"""
        # Group by scenario
        by_scenario = defaultdict(list)
        for analytics in game_analytics:
            by_scenario[analytics.scenario].append(analytics)
            
        # Train simple heuristic models for each scenario
        for scenario, analytics_list in by_scenario.items():
            self.models[scenario] = self._train_heuristic_model(analytics_list)
            
    def _train_heuristic_model(self, analytics_list: List[GameAnalytics]) -> Dict[str, Any]:
        """Train a simple heuristic model"""
        successful = [a for a in analytics_list if a.success]
        failed = [a for a in analytics_list if not a.success]
        
        if not successful or not failed:
            return {'type': 'insufficient_data'}
            
        # Find distinguishing patterns
        success_patterns = {
            'avg_efficiency': statistics.mean(a.efficiency for a in successful),
            'avg_selectivity': statistics.mean(a.decision_patterns.get('selectivity_score', 0) for a in successful),
            'avg_consistency': statistics.mean(a.decision_patterns.get('consistency_score', 0) for a in successful)
        }
        
        failure_patterns = {
            'avg_efficiency': statistics.mean(a.efficiency for a in failed),
            'avg_selectivity': statistics.mean(a.decision_patterns.get('selectivity_score', 0) for a in failed),
            'avg_consistency': statistics.mean(a.decision_patterns.get('consistency_score', 0) for a in failed)
        }
        
        return {
            'type': 'heuristic',
            'success_patterns': success_patterns,
            'failure_patterns': failure_patterns,
            'sample_size': len(analytics_list),
            'success_rate': len(successful) / len(analytics_list)
        }
        
    def predict_success_probability(self, partial_game_data: Dict[str, Any], 
                                  scenario: int) -> float:
        """Predict success probability from partial game data"""
        if scenario not in self.models:
            return 0.5  # Default uncertainty
            
        model = self.models[scenario]
        if model['type'] == 'insufficient_data':
            return 0.5
            
        # Extract current patterns from partial data
        analyzer = GameAnalyzer()
        try:
            current_patterns = analyzer._extract_decision_patterns(partial_game_data)
            current_efficiency = analyzer._calculate_efficiency(partial_game_data)
            
            # Compare to success patterns
            success_patterns = model['success_patterns']
            
            efficiency_similarity = 1.0 - abs(current_efficiency - success_patterns['avg_efficiency'])
            selectivity_similarity = 1.0 - abs(current_patterns.get('selectivity_score', 0) - success_patterns['avg_selectivity'])
            consistency_similarity = 1.0 - abs(current_patterns.get('consistency_score', 0) - success_patterns['avg_consistency'])
            
            # Weighted average
            similarity_score = (efficiency_similarity * 0.4 + 
                              selectivity_similarity * 0.3 + 
                              consistency_similarity * 0.3)
                              
            # Convert similarity to probability (with some uncertainty)
            base_success_rate = model['success_rate']
            probability = base_success_rate * similarity_score + (1 - base_success_rate) * (1 - similarity_score)
            
            return max(0.1, min(0.9, probability))  # Clamp between 10-90%
            
        except:
            return 0.5  # Return uncertainty on error

class PerformanceAnalytics:
    """High-level analytics for overall performance trends"""
    
    def __init__(self):
        self.game_analytics: List[GameAnalytics] = []
        
    def add_game_analysis(self, analytics: GameAnalytics):
        """Add a game analysis to the dataset"""
        self.game_analytics.append(analytics)
        
    def get_bot_performance_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different bots"""
        by_bot = defaultdict(list)
        for analytics in self.game_analytics:
            by_bot[analytics.bot_type].append(analytics)
            
        comparison = {}
        for bot_type, analytics_list in by_bot.items():
            successful = [a for a in analytics_list if a.success]
            
            comparison[bot_type] = {
                'total_games': len(analytics_list),
                'successful_games': len(successful),
                'success_rate': len(successful) / len(analytics_list) if analytics_list else 0,
                'avg_efficiency': statistics.mean(a.efficiency for a in analytics_list),
                'avg_performance_score': statistics.mean(a.performance_score for a in analytics_list),
                'best_performance': max(a.performance_score for a in analytics_list) if analytics_list else 0
            }
            
        return comparison
        
    def get_scenario_analysis(self) -> Dict[int, Dict[str, Any]]:
        """Analyze performance by scenario"""
        by_scenario = defaultdict(list)
        for analytics in self.game_analytics:
            by_scenario[analytics.scenario].append(analytics)
            
        analysis = {}
        for scenario, analytics_list in by_scenario.items():
            successful = [a for a in analytics_list if a.success]
            
            analysis[scenario] = {
                'total_games': len(analytics_list),
                'success_rate': len(successful) / len(analytics_list) if analytics_list else 0,
                'avg_efficiency': statistics.mean(a.efficiency for a in analytics_list),
                'difficulty_score': 1.0 - (len(successful) / len(analytics_list)) if analytics_list else 0,
                'top_performers': sorted(analytics_list, key=lambda a: a.performance_score, reverse=True)[:3]
            }
            
        return analysis
        
    def detect_performance_trends(self, time_window_hours: int = 24) -> PerformanceTrend:
        """Detect performance trends over time"""
        # Sort by game completion time (would need timestamp in analytics)
        # For now, use order as proxy for time
        recent_games = self.game_analytics[-100:]  # Last 100 games
        
        # Calculate rolling success rates
        window_size = 10
        success_rates = []
        efficiency_rates = []
        
        for i in range(window_size, len(recent_games)):
            window = recent_games[i-window_size:i]
            success_rate = sum(1 for g in window if g.success) / len(window)
            avg_efficiency = statistics.mean(g.efficiency for g in window)
            
            success_rates.append(success_rate)
            efficiency_rates.append(avg_efficiency)
            
        return PerformanceTrend(
            time_period=f"Last {len(recent_games)} games",
            success_rate_trend=success_rates,
            efficiency_trend=efficiency_rates,
            pattern_evolution={}  # Could be enhanced
        )
        
    def identify_success_factors(self) -> Dict[str, Any]:
        """Identify key factors that lead to success"""
        successful = [a for a in self.game_analytics if a.success]
        failed = [a for a in self.game_analytics if not a.success]
        
        if not successful or not failed:
            return {'error': 'Insufficient data for comparison'}
            
        # Compare patterns between successful and failed games
        factors = {}
        
        # Efficiency factor
        success_efficiency = statistics.mean(a.efficiency for a in successful)
        failed_efficiency = statistics.mean(a.efficiency for a in failed)
        factors['efficiency_difference'] = success_efficiency - failed_efficiency
        
        # Selectivity factor
        success_selectivity = statistics.mean(
            a.decision_patterns.get('selectivity_score', 0) for a in successful
        )
        failed_selectivity = statistics.mean(
            a.decision_patterns.get('selectivity_score', 0) for a in failed
        )
        factors['selectivity_difference'] = success_selectivity - failed_selectivity
        
        # Consistency factor
        success_consistency = statistics.mean(
            a.decision_patterns.get('consistency_score', 0) for a in successful
        )
        failed_consistency = statistics.mean(
            a.decision_patterns.get('consistency_score', 0) for a in failed
        )
        factors['consistency_difference'] = success_consistency - failed_consistency
        
        return factors

class AnalyticsEngine:
    """Main analytics engine coordinating all analysis components"""
    
    def __init__(self, logs_dir: str = "game_logs"):
        self.logs_dir = Path(logs_dir)
        self.analyzer = GameAnalyzer()
        self.predictor = SuccessPredictor()
        self.performance_analytics = PerformanceAnalytics()
        self.game_analytics_cache: Dict[str, GameAnalytics] = {}
        
    def analyze_all_games(self):
        """Analyze all games in the logs directory"""
        print("🔍 Analyzing all games...")
        
        for json_file in self.logs_dir.glob("*.json"):
            game_id = json_file.stem
            
            if game_id in self.game_analytics_cache:
                continue  # Skip already analyzed
                
            try:
                with open(json_file, 'r') as f:
                    game_data = json.load(f)
                    
                analytics = self.analyzer.analyze_game(game_data)
                self.game_analytics_cache[game_id] = analytics
                self.performance_analytics.add_game_analysis(analytics)
                
            except Exception as e:
                print(f"Error analyzing {json_file}: {e}")
                continue
                
        # Train predictor with analyzed games
        self.predictor.train_from_games(list(self.game_analytics_cache.values()))
        
        print(f"✅ Analyzed {len(self.game_analytics_cache)} games")
        
    def get_top_performers(self, limit: int = 5) -> List[GameAnalytics]:
        """Get top performing games"""
        return sorted(
            self.game_analytics_cache.values(),
            key=lambda a: a.performance_score,
            reverse=True
        )[:limit]
        
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        if not self.game_analytics_cache:
            self.analyze_all_games()
            
        report = {
            'summary': {
                'total_games': len(self.game_analytics_cache),
                'total_successful': sum(1 for a in self.game_analytics_cache.values() if a.success),
                'overall_success_rate': sum(1 for a in self.game_analytics_cache.values() if a.success) / len(self.game_analytics_cache) if self.game_analytics_cache else 0,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'bot_comparison': self.performance_analytics.get_bot_performance_comparison(),
            'scenario_analysis': self.performance_analytics.get_scenario_analysis(),
            'success_factors': self.performance_analytics.identify_success_factors(),
            'top_performers': [asdict(a) for a in self.get_top_performers()],
            'performance_trends': asdict(self.performance_analytics.detect_performance_trends())
        }
        
        return report
        
    def save_analytics_report(self, output_file: str):
        """Save comprehensive analytics report to file"""
        report = self.get_comprehensive_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Analytics report saved: {output_file}")

def main():
    """CLI interface for analytics engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Berghain Game Analytics Engine')
    parser.add_argument('--logs-dir', default='game_logs', help='Directory containing game logs')
    parser.add_argument('--output', default='analytics_report.json', help='Output file for report')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top performers to show')
    
    args = parser.parse_args()
    
    engine = AnalyticsEngine(args.logs_dir)
    
    print("🚀 Starting Berghain Game Analytics")
    engine.analyze_all_games()
    
    # Show quick summary
    top_performers = engine.get_top_performers(args.top_n)
    
    print(f"\n🏆 Top {len(top_performers)} Performers:")
    print("-" * 60)
    for i, analytics in enumerate(top_performers, 1):
        print(f"{i:2d}. {analytics.game_id} (S{analytics.scenario}) - "
              f"Score: {analytics.performance_score:.3f}, "
              f"Efficiency: {analytics.efficiency:.1%}")
              
    # Generate full report
    engine.save_analytics_report(args.output)
    
    print(f"\n✅ Analysis complete! Full report saved to {args.output}")

if __name__ == "__main__":
    main()