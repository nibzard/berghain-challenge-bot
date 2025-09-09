# ABOUTME: Analyzes constraint satisfaction patterns across successful games
# ABOUTME: Identifies optimal decision sequences for meeting young/well_dressed requirements

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConstraintPattern:
    """Represents a constraint satisfaction pattern"""
    strategy_name: str
    game_phase: str
    constraint_type: str  # 'young', 'well_dressed', 'both'
    pattern_type: str  # 'early_focus', 'balanced', 'late_catch_up', 'emergency'
    success_rate: float
    avg_rejections: float
    decision_sequence: List[Dict[str, Any]]
    context: Dict[str, Any]
    effectiveness_score: float


@dataclass 
class ConstraintTrajectory:
    """Tracks how constraints progress over time in a game"""
    game_id: str
    strategy_name: str
    final_success: bool
    final_rejections: int
    young_trajectory: List[Tuple[int, float]]  # (person_index, progress)
    well_dressed_trajectory: List[Tuple[int, float]]
    critical_points: List[Dict[str, Any]]  # Points where constraint risk was high


class ConstraintPatternFinder:
    """Finds patterns in how successful strategies satisfy constraints"""
    
    def __init__(self, game_logs_dir: str = "game_logs"):
        self.game_logs_dir = Path(game_logs_dir) 
        self.trajectories: List[ConstraintTrajectory] = []
        self.patterns: List[ConstraintPattern] = []
        
    def analyze_constraint_patterns(self) -> List[ConstraintPattern]:
        """Analyze constraint satisfaction patterns across all games"""
        logger.info("Starting constraint pattern analysis...")
        
        # Load game logs and extract trajectories
        self._extract_constraint_trajectories()
        
        # Find successful patterns
        self._identify_constraint_patterns()
        
        logger.info(f"Found {len(self.patterns)} constraint patterns from {len(self.trajectories)} games")
        return self.patterns
    
    def _extract_constraint_trajectories(self):
        """Extract constraint progress trajectories from game logs"""
        game_files = list(self.game_logs_dir.glob("game_*.json"))
        event_files = list(self.game_logs_dir.glob("events_*.jsonl"))
        
        logger.info(f"Processing {len(game_files)} game logs...")
        
        # Process game logs (more complete data)
        for game_file in game_files[:200]:  # Process first 200 for analysis
            try:
                trajectory = self._extract_trajectory_from_game_log(game_file)
                if trajectory:
                    self.trajectories.append(trajectory)
            except Exception as e:
                logger.warning(f"Failed to extract trajectory from {game_file}: {e}")
        
        # Process event logs (real-time data)
        logger.info(f"Processing {len(event_files)} event logs...")
        for event_file in event_files[:100]:  # Process first 100 event logs
            try:
                trajectory = self._extract_trajectory_from_event_log(event_file)
                if trajectory:
                    self.trajectories.append(trajectory)
            except Exception as e:
                logger.warning(f"Failed to extract trajectory from {event_file}: {e}")
    
    def _extract_trajectory_from_game_log(self, game_file: Path) -> Optional[ConstraintTrajectory]:
        """Extract constraint trajectory from a complete game log"""
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        # Extract basic info
        game_id = game_data.get('game_id', game_file.stem)
        strategy_name = game_data.get('strategy_name', 'unknown')
        final_results = game_data.get('final_results', {})
        final_success = final_results.get('success', False)
        final_rejections = final_results.get('rejected_count', 0)
        
        # Extract decision history
        decisions = game_data.get('decisions', [])
        if len(decisions) < 50:  # Skip games with insufficient data
            return None
        
        # Build constraint trajectories
        young_trajectory = []
        well_dressed_trajectory = []
        critical_points = []
        
        young_count = 0
        well_dressed_count = 0
        total_admitted = 0
        
        for i, decision in enumerate(decisions):
            person_index = decision.get('person_index', i)
            accepted = decision.get('accepted', False)
            person_attrs = decision.get('person_attributes', {})
            
            if accepted:
                total_admitted += 1
                if person_attrs.get('young', False):
                    young_count += 1
                if person_attrs.get('well_dressed', False):
                    well_dressed_count += 1
            
            # Calculate progress (normalize by requirements: 600 each)
            young_progress = young_count / 600.0
            well_dressed_progress = well_dressed_count / 600.0
            
            # Record trajectory points every 50 decisions
            if person_index % 50 == 0:
                young_trajectory.append((person_index, young_progress))
                well_dressed_trajectory.append((person_index, well_dressed_progress))
            
            # Identify critical points (constraint risk situations)
            if person_index > 300:  # After early game
                expected_progress = total_admitted / 1000.0  # Expected if balanced admission
                min_progress = min(young_progress, well_dressed_progress)
                
                if min_progress < expected_progress * 0.6:  # Significantly behind
                    critical_point = {
                        'person_index': person_index,
                        'young_progress': young_progress,
                        'well_dressed_progress': well_dressed_progress,
                        'expected_progress': expected_progress,
                        'constraint_risk': 1.0 - (min_progress / max(expected_progress, 0.1)),
                        'total_admitted': total_admitted,
                        'reasoning': decision.get('reasoning', '')
                    }
                    critical_points.append(critical_point)
        
        return ConstraintTrajectory(
            game_id=game_id,
            strategy_name=strategy_name,
            final_success=final_success,
            final_rejections=final_rejections,
            young_trajectory=young_trajectory,
            well_dressed_trajectory=well_dressed_trajectory,
            critical_points=critical_points
        )
    
    def _extract_trajectory_from_event_log(self, event_file: Path) -> Optional[ConstraintTrajectory]:
        """Extract constraint trajectory from real-time event log"""
        with open(event_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        # Filter for API response events (contain progress data)
        api_events = [e for e in events if e.get('type') == 'api_response']
        if len(api_events) < 20:
            return None
        
        # Extract strategy name from filename
        strategy_name = event_file.stem.split('_')[1] if '_' in event_file.stem else 'unknown'
        
        young_trajectory = []
        well_dressed_trajectory = []
        critical_points = []
        
        for event in api_events:
            data = event.get('data', {})
            person_index = data.get('person_index', 0)
            progress = data.get('progress', {})
            
            if not progress:
                continue
                
            young_progress = progress.get('young', 0.0)
            well_dressed_progress = progress.get('well_dressed', 0.0)
            
            # Record trajectory
            young_trajectory.append((person_index, young_progress))
            well_dressed_trajectory.append((person_index, well_dressed_progress))
            
            # Check for critical points
            if person_index > 200:
                admitted = data.get('admitted', 0)
                expected_progress = admitted / 1000.0
                min_progress = min(young_progress, well_dressed_progress)
                
                if min_progress < expected_progress * 0.6:
                    critical_point = {
                        'person_index': person_index,
                        'young_progress': young_progress,
                        'well_dressed_progress': well_dressed_progress,
                        'expected_progress': expected_progress,
                        'constraint_risk': 1.0 - (min_progress / max(expected_progress, 0.1)),
                        'total_admitted': admitted,
                        'reasoning': data.get('reasoning', '')
                    }
                    critical_points.append(critical_point)
        
        # Determine success from final progress
        final_young = young_trajectory[-1][1] if young_trajectory else 0
        final_well_dressed = well_dressed_trajectory[-1][1] if well_dressed_trajectory else 0
        final_success = final_young >= 1.0 and final_well_dressed >= 1.0
        
        return ConstraintTrajectory(
            game_id=event_file.stem,
            strategy_name=strategy_name,
            final_success=final_success,
            final_rejections=0,  # Not available from event log
            young_trajectory=young_trajectory,
            well_dressed_trajectory=well_dressed_trajectory,
            critical_points=critical_points
        )
    
    def _identify_constraint_patterns(self):
        """Identify successful constraint satisfaction patterns"""
        # Group trajectories by strategy and success
        successful_trajectories = [t for t in self.trajectories if t.final_success]
        failed_trajectories = [t for t in self.trajectories if not t.final_success]
        
        logger.info(f"Analyzing {len(successful_trajectories)} successful and {len(failed_trajectories)} failed trajectories")
        
        # Analyze patterns by strategy
        strategy_groups = defaultdict(list)
        for trajectory in successful_trajectories:
            strategy_groups[trajectory.strategy_name].append(trajectory)
        
        for strategy_name, trajectories in strategy_groups.items():
            if len(trajectories) < 5:  # Need at least 5 examples
                continue
                
            logger.info(f"Analyzing {len(trajectories)} successful {strategy_name} games...")
            
            # Find different constraint satisfaction patterns
            self._find_early_focus_patterns(strategy_name, trajectories)
            self._find_balanced_patterns(strategy_name, trajectories)
            self._find_recovery_patterns(strategy_name, trajectories)
    
    def _find_early_focus_patterns(self, strategy_name: str, trajectories: List[ConstraintTrajectory]):
        """Find patterns where strategies focus on constraints early"""
        early_focus_trajectories = []
        
        for trajectory in trajectories:
            # Check if constraints were satisfied early (by 50% capacity)
            mid_game_idx = len(trajectory.young_trajectory) // 2
            if mid_game_idx > 0:
                mid_young = trajectory.young_trajectory[mid_game_idx][1]
                mid_well_dressed = trajectory.well_dressed_trajectory[mid_game_idx][1]
                
                if mid_young > 0.7 and mid_well_dressed > 0.7:  # 70% progress by midgame
                    early_focus_trajectories.append(trajectory)
        
        if len(early_focus_trajectories) >= 3:
            avg_rejections = np.mean([t.final_rejections for t in early_focus_trajectories if t.final_rejections > 0])
            success_rate = len(early_focus_trajectories) / len(trajectories)
            
            pattern = ConstraintPattern(
                strategy_name=strategy_name,
                game_phase='early_to_mid',
                constraint_type='both',
                pattern_type='early_focus',
                success_rate=success_rate,
                avg_rejections=avg_rejections,
                decision_sequence=self._extract_decision_sequence(early_focus_trajectories),
                context={
                    'sample_size': len(early_focus_trajectories),
                    'avg_mid_young_progress': np.mean([t.young_trajectory[len(t.young_trajectory)//2][1] for t in early_focus_trajectories]),
                    'avg_mid_well_dressed_progress': np.mean([t.well_dressed_trajectory[len(t.well_dressed_trajectory)//2][1] for t in early_focus_trajectories])
                },
                effectiveness_score=success_rate * (2000 - avg_rejections) / 2000
            )
            
            self.patterns.append(pattern)
    
    def _find_balanced_patterns(self, strategy_name: str, trajectories: List[ConstraintTrajectory]):
        """Find patterns with balanced constraint progress"""
        balanced_trajectories = []
        
        for trajectory in trajectories:
            # Check for balanced progress (similar rates for both constraints)
            if len(trajectory.young_trajectory) < 5:
                continue
                
            balance_scores = []
            for young_pt, well_dressed_pt in zip(trajectory.young_trajectory, trajectory.well_dressed_trajectory):
                young_prog = young_pt[1]
                well_dressed_prog = well_dressed_pt[1]
                
                if young_prog > 0 and well_dressed_prog > 0:
                    balance = 1.0 - abs(young_prog - well_dressed_prog) / max(young_prog, well_dressed_prog)
                    balance_scores.append(balance)
            
            if balance_scores and np.mean(balance_scores) > 0.8:  # Good balance
                balanced_trajectories.append(trajectory)
        
        if len(balanced_trajectories) >= 3:
            avg_rejections = np.mean([t.final_rejections for t in balanced_trajectories if t.final_rejections > 0])
            success_rate = len(balanced_trajectories) / len(trajectories)
            
            pattern = ConstraintPattern(
                strategy_name=strategy_name,
                game_phase='full_game',
                constraint_type='both',
                pattern_type='balanced',
                success_rate=success_rate,
                avg_rejections=avg_rejections,
                decision_sequence=self._extract_decision_sequence(balanced_trajectories),
                context={
                    'sample_size': len(balanced_trajectories),
                    'avg_balance_score': np.mean([
                        np.mean([1.0 - abs(y[1] - w[1]) / max(y[1], w[1], 0.01) 
                                for y, w in zip(t.young_trajectory, t.well_dressed_trajectory) 
                                if y[1] > 0 and w[1] > 0])
                        for t in balanced_trajectories
                    ])
                },
                effectiveness_score=success_rate * (2000 - avg_rejections) / 2000
            )
            
            self.patterns.append(pattern)
    
    def _find_recovery_patterns(self, strategy_name: str, trajectories: List[ConstraintTrajectory]):
        """Find patterns where strategies recover from constraint deficits"""
        recovery_trajectories = []
        
        for trajectory in trajectories:
            # Look for trajectories that had critical points but still succeeded
            if len(trajectory.critical_points) > 0:
                # Check if they recovered from high constraint risk
                max_risk = max(cp['constraint_risk'] for cp in trajectory.critical_points)
                if max_risk > 0.5:  # Had significant constraint risk
                    recovery_trajectories.append(trajectory)
        
        if len(recovery_trajectories) >= 3:
            avg_rejections = np.mean([t.final_rejections for t in recovery_trajectories if t.final_rejections > 0])
            success_rate = len(recovery_trajectories) / len([t for t in trajectories if len(t.critical_points) > 0])
            
            pattern = ConstraintPattern(
                strategy_name=strategy_name,
                game_phase='recovery',
                constraint_type='both',
                pattern_type='recovery',
                success_rate=success_rate,
                avg_rejections=avg_rejections,
                decision_sequence=self._extract_decision_sequence(recovery_trajectories),
                context={
                    'sample_size': len(recovery_trajectories),
                    'avg_max_risk': np.mean([max(cp['constraint_risk'] for cp in t.critical_points) for t in recovery_trajectories]),
                    'avg_critical_points': np.mean([len(t.critical_points) for t in recovery_trajectories])
                },
                effectiveness_score=success_rate * (2000 - avg_rejections) / 2000
            )
            
            self.patterns.append(pattern)
    
    def _extract_decision_sequence(self, trajectories: List[ConstraintTrajectory]) -> List[Dict[str, Any]]:
        """Extract common decision sequence patterns from trajectories"""
        # This is a simplified version - in practice would do more sophisticated sequence analysis
        sequence_points = []
        
        # Sample key decision points from trajectories
        for trajectory in trajectories[:5]:  # Use first 5 as examples
            for i, (young_pt, well_dressed_pt) in enumerate(zip(trajectory.young_trajectory, trajectory.well_dressed_trajectory)):
                if i % 3 == 0:  # Sample every 3rd point
                    sequence_points.append({
                        'person_index': young_pt[0],
                        'young_progress': young_pt[1],
                        'well_dressed_progress': well_dressed_pt[1],
                        'game_phase': 'early' if young_pt[0] < 300 else 'mid' if young_pt[0] < 700 else 'late'
                    })
        
        return sequence_points
    
    def get_best_patterns(self, min_effectiveness: float = 0.5) -> List[ConstraintPattern]:
        """Get the most effective constraint patterns"""
        return [p for p in self.patterns if p.effectiveness_score >= min_effectiveness]
    
    def get_patterns_by_strategy(self) -> Dict[str, List[ConstraintPattern]]:
        """Group patterns by strategy"""
        strategy_patterns = defaultdict(list)
        for pattern in self.patterns:
            strategy_patterns[pattern.strategy_name].append(pattern)
        return dict(strategy_patterns)
    
    def export_constraint_analysis(self, output_file: str = "constraint_pattern_analysis.json"):
        """Export constraint pattern analysis results"""
        results = {
            'summary': {
                'total_trajectories': len(self.trajectories),
                'successful_trajectories': len([t for t in self.trajectories if t.final_success]),
                'patterns_found': len(self.patterns),
                'strategies_analyzed': list(set(t.strategy_name for t in self.trajectories))
            },
            'patterns': [],
            'trajectory_analysis': {}
        }
        
        # Export patterns
        for pattern in self.patterns:
            results['patterns'].append({
                'strategy_name': pattern.strategy_name,
                'game_phase': pattern.game_phase,
                'constraint_type': pattern.constraint_type,
                'pattern_type': pattern.pattern_type,
                'success_rate': pattern.success_rate,
                'avg_rejections': pattern.avg_rejections,
                'effectiveness_score': pattern.effectiveness_score,
                'context': pattern.context,
                'decision_sequence': pattern.decision_sequence
            })
        
        # Export trajectory analysis by strategy
        strategy_groups = defaultdict(list)
        for trajectory in self.trajectories:
            strategy_groups[trajectory.strategy_name].append(trajectory)
        
        for strategy, trajectories in strategy_groups.items():
            successful = [t for t in trajectories if t.final_success]
            failed = [t for t in trajectories if not t.final_success]
            
            results['trajectory_analysis'][strategy] = {
                'total_games': len(trajectories),
                'successful_games': len(successful),
                'success_rate': len(successful) / len(trajectories),
                'avg_rejections_successful': np.mean([t.final_rejections for t in successful if t.final_rejections > 0]) if successful else 0,
                'avg_critical_points': np.mean([len(t.critical_points) for t in trajectories]),
                'constraint_satisfaction_patterns': {
                    'early_focus': len([t for t in successful if self._is_early_focus(t)]),
                    'balanced': len([t for t in successful if self._is_balanced(t)]),
                    'recovery': len([t for t in successful if len(t.critical_points) > 0])
                }
            }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Constraint pattern analysis exported to {output_file}")
        return results
    
    def _is_early_focus(self, trajectory: ConstraintTrajectory) -> bool:
        """Check if trajectory shows early constraint focus"""
        if len(trajectory.young_trajectory) < 2:
            return False
        mid_idx = len(trajectory.young_trajectory) // 2
        mid_young = trajectory.young_trajectory[mid_idx][1]
        mid_well_dressed = trajectory.well_dressed_trajectory[mid_idx][1]
        return mid_young > 0.7 and mid_well_dressed > 0.7
    
    def _is_balanced(self, trajectory: ConstraintTrajectory) -> bool:
        """Check if trajectory shows balanced progress"""
        if len(trajectory.young_trajectory) < 3:
            return False
        
        balance_scores = []
        for young_pt, well_dressed_pt in zip(trajectory.young_trajectory, trajectory.well_dressed_trajectory):
            if young_pt[1] > 0 and well_dressed_pt[1] > 0:
                balance = 1.0 - abs(young_pt[1] - well_dressed_pt[1]) / max(young_pt[1], well_dressed_pt[1])
                balance_scores.append(balance)
        
        return len(balance_scores) > 0 and np.mean(balance_scores) > 0.8


def main():
    """Run constraint pattern analysis"""
    finder = ConstraintPatternFinder()
    
    # Analyze constraint patterns
    patterns = finder.analyze_constraint_patterns()
    
    # Export results
    results = finder.export_constraint_analysis()
    
    # Print summary
    print(f"\n=== Constraint Pattern Analysis Summary ===")
    print(f"Total trajectories analyzed: {results['summary']['total_trajectories']}")
    print(f"Successful trajectories: {results['summary']['successful_trajectories']}")
    print(f"Patterns found: {results['summary']['patterns_found']}")
    print(f"Strategies analyzed: {', '.join(results['summary']['strategies_analyzed'])}")
    
    # Print best patterns
    best_patterns = finder.get_best_patterns(min_effectiveness=0.3)
    print(f"\n=== Most Effective Patterns (effectiveness >= 0.3) ===")
    for pattern in sorted(best_patterns, key=lambda p: p.effectiveness_score, reverse=True):
        print(f"{pattern.strategy_name} - {pattern.pattern_type}: {pattern.effectiveness_score:.3f} effectiveness, {pattern.success_rate:.1%} success rate, {pattern.avg_rejections:.0f} avg rejections")
    
    # Print strategy analysis
    print(f"\n=== Strategy Performance Analysis ===")
    for strategy, analysis in results['trajectory_analysis'].items():
        print(f"{strategy}: {analysis['success_rate']:.1%} success ({analysis['successful_games']}/{analysis['total_games']}), {analysis['avg_rejections_successful']:.0f} avg rejections")


if __name__ == "__main__":
    main()