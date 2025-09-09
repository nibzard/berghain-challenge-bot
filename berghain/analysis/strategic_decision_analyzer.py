# ABOUTME: Analyzes existing game logs to extract strategic decision points and transitions
# ABOUTME: Identifies critical moments where strategy switches would have been optimal

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategicDecision:
    """Represents a critical strategic decision point in a game"""
    game_id: str
    strategy_name: str
    person_index: int
    game_phase: str
    decision_type: str  # 'strategy_switch', 'parameter_change', 'risk_escalation'
    context: Dict[str, Any]
    outcome_impact: float  # How much this decision affected final outcome
    constraint_risk: float  # Risk to constraint satisfaction at this point
    optimal_action: str  # What would have been optimal
    reasoning: str


@dataclass
class GameAnalysis:
    """Complete analysis of a single game"""
    game_id: str
    strategy_name: str
    final_rejections: int
    success: bool
    constraint_satisfaction: Dict[str, bool]
    strategic_decisions: List[StrategicDecision]
    performance_score: float
    failure_reason: Optional[str]


class StrategicDecisionAnalyzer:
    """Analyzes game logs to extract strategic decision patterns"""
    
    def __init__(self, game_logs_dir: str = "game_logs"):
        self.game_logs_dir = Path(game_logs_dir)
        self.analyses: List[GameAnalysis] = []
        
    def analyze_all_games(self) -> List[GameAnalysis]:
        """Analyze all available game logs"""
        logger.info("Starting strategic analysis of all game logs...")
        
        # Find all game log files
        game_files = list(self.game_logs_dir.glob("game_*.json"))
        event_files = list(self.game_logs_dir.glob("events_*.jsonl"))
        
        logger.info(f"Found {len(game_files)} game logs and {len(event_files)} event logs")
        
        # Analyze game logs
        for game_file in game_files[:100]:  # Start with first 100 for testing
            try:
                analysis = self._analyze_single_game(game_file)
                if analysis:
                    self.analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {game_file}: {e}")
        
        # Analyze event logs for real-time decision patterns
        for event_file in event_files[:50]:  # Start with 50 event logs
            try:
                self._analyze_event_log(event_file)
            except Exception as e:
                logger.warning(f"Failed to analyze {event_file}: {e}")
        
        logger.info(f"Completed analysis of {len(self.analyses)} games")
        return self.analyses
    
    def _analyze_single_game(self, game_file: Path) -> Optional[GameAnalysis]:
        """Analyze a single game log file"""
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        # Extract basic info
        game_id = game_data.get('game_id', game_file.stem)
        strategy_name = game_data.get('strategy_name', 'unknown')
        final_rejections = game_data.get('final_results', {}).get('rejected_count', 0)
        success = game_data.get('final_results', {}).get('success', False)
        
        # Extract constraint satisfaction
        constraint_satisfaction = {}
        final_results = game_data.get('final_results', {})
        if 'constraint_summary' in final_results:
            for attr, summary in final_results['constraint_summary'].items():
                constraint_satisfaction[attr] = summary.get('satisfied', False)
        
        # Analyze strategic decisions from decision history
        strategic_decisions = self._extract_strategic_decisions(game_data, strategy_name)
        
        # Calculate performance score (lower rejections = higher score, with success bonus)
        if success:
            performance_score = max(0, 2000 - final_rejections)  # Success bonus
        else:
            performance_score = max(0, 1000 - final_rejections)  # No success penalty
        
        # Determine failure reason if applicable
        failure_reason = None
        if not success:
            if not all(constraint_satisfaction.values()):
                failure_reason = "constraint_failure"
            elif final_rejections > 1200:
                failure_reason = "rejection_limit"
            else:
                failure_reason = "unknown"
        
        return GameAnalysis(
            game_id=game_id,
            strategy_name=strategy_name,
            final_rejections=final_rejections,
            success=success,
            constraint_satisfaction=constraint_satisfaction,
            strategic_decisions=strategic_decisions,
            performance_score=performance_score,
            failure_reason=failure_reason
        )
    
    def _extract_strategic_decisions(self, game_data: Dict, strategy_name: str) -> List[StrategicDecision]:
        """Extract strategic decision points from game data"""
        decisions = []
        
        # Get decision history if available
        decision_history = game_data.get('decisions', [])
        if not decision_history:
            return decisions
        
        # Analyze decision patterns for strategic moments
        prev_accept_rate = 0.5
        window_size = 50
        
        for i, decision in enumerate(decision_history):
            if i < window_size:
                continue
                
            # Calculate recent acceptance rate
            recent_decisions = decision_history[i-window_size:i]
            recent_accept_rate = sum(1 for d in recent_decisions if d.get('accepted', False)) / len(recent_decisions)
            
            # Check for significant strategy shifts
            rate_change = abs(recent_accept_rate - prev_accept_rate)
            if rate_change > 0.3:  # Significant change in acceptance pattern
                
                # Determine game phase
                person_index = decision.get('person_index', i)
                if person_index < 300:
                    game_phase = 'early'
                elif person_index < 700:
                    game_phase = 'mid'
                else:
                    game_phase = 'late'
                
                # Extract context
                context = {
                    'person_index': person_index,
                    'recent_accept_rate': recent_accept_rate,
                    'prev_accept_rate': prev_accept_rate,
                    'rate_change': rate_change,
                    'reasoning': decision.get('reasoning', '')
                }
                
                # Create strategic decision
                strategic_decision = StrategicDecision(
                    game_id=game_data.get('game_id', 'unknown'),
                    strategy_name=strategy_name,
                    person_index=person_index,
                    game_phase=game_phase,
                    decision_type='acceptance_pattern_shift',
                    context=context,
                    outcome_impact=rate_change,  # Magnitude of change
                    constraint_risk=self._estimate_constraint_risk(game_data, i),
                    optimal_action='analyze_needed',
                    reasoning=f"Acceptance rate changed from {prev_accept_rate:.2f} to {recent_accept_rate:.2f}"
                )
                
                decisions.append(strategic_decision)
            
            prev_accept_rate = recent_accept_rate
        
        return decisions
    
    def _analyze_event_log(self, event_file: Path):
        """Analyze real-time event logs for decision patterns"""
        strategic_decisions = []
        
        with open(event_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        # Extract strategy name from filename
        strategy_name = event_file.stem.split('_')[1] if '_' in event_file.stem else 'unknown'
        
        # Look for decision pattern changes in real-time events
        api_responses = [e for e in events if e.get('type') == 'api_response']
        
        if len(api_responses) < 50:
            return
        
        # Analyze constraint progress patterns
        for i, event in enumerate(api_responses[::10]):  # Sample every 10th event
            progress = event.get('data', {}).get('progress', {})
            if not progress:
                continue
                
            # Check for constraint risk situations
            young_progress = progress.get('young', 0)
            well_dressed_progress = progress.get('well_dressed', 0)
            
            min_progress = min(young_progress, well_dressed_progress)
            person_index = event.get('data', {}).get('person_index', i * 10)
            
            # Identify high-risk situations
            expected_progress = person_index / 1000  # Expected progress if accepting everyone
            if min_progress < expected_progress * 0.6:  # Significantly behind
                constraint_risk = 1.0 - (min_progress / max(expected_progress, 0.1))
                
                # This represents a strategic decision point where intervention needed
                strategic_decision = StrategicDecision(
                    game_id=event_file.stem,
                    strategy_name=strategy_name,
                    person_index=person_index,
                    game_phase='mid' if person_index < 700 else 'late',
                    decision_type='constraint_risk_escalation',
                    context={
                        'young_progress': young_progress,
                        'well_dressed_progress': well_dressed_progress,
                        'expected_progress': expected_progress,
                        'admitted': event.get('data', {}).get('admitted', 0),
                        'rejected': event.get('data', {}).get('rejected', 0)
                    },
                    outcome_impact=constraint_risk,
                    constraint_risk=constraint_risk,
                    optimal_action='increase_acceptance_for_constraints',
                    reasoning=f"Constraint progress ({min_progress:.2f}) behind expected ({expected_progress:.2f})"
                )
                
                strategic_decisions.append(strategic_decision)
        
        # Add to analyses if we found strategic decisions
        if strategic_decisions:
            # Create synthetic analysis for this event log
            analysis = GameAnalysis(
                game_id=event_file.stem,
                strategy_name=strategy_name,
                final_rejections=0,  # Unknown from event log
                success=False,  # Unknown from event log
                constraint_satisfaction={},
                strategic_decisions=strategic_decisions,
                performance_score=0,
                failure_reason=None
            )
            self.analyses.append(analysis)
    
    def _estimate_constraint_risk(self, game_data: Dict, decision_index: int) -> float:
        """Estimate constraint satisfaction risk at a given decision point"""
        # This is a simplified estimate - in real implementation would be more sophisticated
        total_decisions = len(game_data.get('decisions', []))
        if total_decisions == 0:
            return 0.5
        
        progress_ratio = decision_index / total_decisions
        return max(0, 1.0 - progress_ratio)  # Risk decreases over time
    
    def get_strategy_patterns(self) -> Dict[str, List[StrategicDecision]]:
        """Group strategic decisions by strategy type"""
        patterns = defaultdict(list)
        
        for analysis in self.analyses:
            for decision in analysis.strategic_decisions:
                patterns[analysis.strategy_name].append(decision)
        
        return dict(patterns)
    
    def get_successful_patterns(self, min_performance_score: float = 1200) -> List[StrategicDecision]:
        """Extract patterns from high-performing games"""
        successful_decisions = []
        
        for analysis in self.analyses:
            if analysis.performance_score >= min_performance_score:
                successful_decisions.extend(analysis.strategic_decisions)
        
        return successful_decisions
    
    def get_failure_patterns(self) -> Dict[str, List[StrategicDecision]]:
        """Extract patterns from failed games, grouped by failure reason"""
        failure_patterns = defaultdict(list)
        
        for analysis in self.analyses:
            if not analysis.success and analysis.failure_reason:
                failure_patterns[analysis.failure_reason].extend(analysis.strategic_decisions)
        
        return dict(failure_patterns)
    
    def export_analysis_results(self, output_file: str = "strategic_analysis_results.json"):
        """Export analysis results to JSON for further processing"""
        results = {
            'summary': {
                'total_games_analyzed': len(self.analyses),
                'successful_games': sum(1 for a in self.analyses if a.success),
                'average_performance_score': np.mean([a.performance_score for a in self.analyses]),
                'strategies_analyzed': list(set(a.strategy_name for a in self.analyses))
            },
            'strategic_patterns': {},
            'successful_patterns': [],
            'failure_patterns': {}
        }
        
        # Add strategic patterns by strategy
        strategy_patterns = self.get_strategy_patterns()
        for strategy, decisions in strategy_patterns.items():
            results['strategic_patterns'][strategy] = [
                {
                    'game_id': d.game_id,
                    'person_index': d.person_index,
                    'game_phase': d.game_phase,
                    'decision_type': d.decision_type,
                    'context': d.context,
                    'outcome_impact': d.outcome_impact,
                    'constraint_risk': d.constraint_risk,
                    'optimal_action': d.optimal_action,
                    'reasoning': d.reasoning
                }
                for d in decisions
            ]
        
        # Add successful patterns
        successful_patterns = self.get_successful_patterns()
        results['successful_patterns'] = [
            {
                'strategy_name': d.strategy_name,
                'person_index': d.person_index,
                'game_phase': d.game_phase,
                'decision_type': d.decision_type,
                'context': d.context,
                'outcome_impact': d.outcome_impact,
                'constraint_risk': d.constraint_risk,
                'optimal_action': d.optimal_action,
                'reasoning': d.reasoning
            }
            for d in successful_patterns
        ]
        
        # Add failure patterns
        failure_patterns = self.get_failure_patterns()
        for failure_reason, decisions in failure_patterns.items():
            results['failure_patterns'][failure_reason] = [
                {
                    'strategy_name': d.strategy_name,
                    'person_index': d.person_index,
                    'game_phase': d.game_phase,
                    'decision_type': d.decision_type,
                    'context': d.context,
                    'outcome_impact': d.outcome_impact,
                    'constraint_risk': d.constraint_risk,
                    'optimal_action': d.optimal_action,
                    'reasoning': d.reasoning
                }
                for d in decisions
            ]
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results exported to {output_file}")
        return results


def main():
    """Run strategic decision analysis on all available game logs"""
    analyzer = StrategicDecisionAnalyzer()
    
    # Analyze all games
    analyses = analyzer.analyze_all_games()
    
    # Export results
    results = analyzer.export_analysis_results()
    
    # Print summary
    print(f"\n=== Strategic Decision Analysis Summary ===")
    print(f"Total games analyzed: {results['summary']['total_games_analyzed']}")
    print(f"Successful games: {results['summary']['successful_games']}")
    print(f"Average performance score: {results['summary']['average_performance_score']:.1f}")
    print(f"Strategies analyzed: {', '.join(results['summary']['strategies_analyzed'])}")
    
    # Print pattern counts
    print(f"\n=== Strategic Patterns by Strategy ===")
    for strategy, decisions in results['strategic_patterns'].items():
        print(f"{strategy}: {len(decisions)} strategic decision points")
    
    print(f"\n=== Success/Failure Patterns ===")
    print(f"Successful patterns: {len(results['successful_patterns'])}")
    for failure_reason, decisions in results['failure_patterns'].items():
        print(f"{failure_reason} patterns: {len(decisions)}")


if __name__ == "__main__":
    main()