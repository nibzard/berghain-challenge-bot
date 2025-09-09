# ABOUTME: Extracts high-quality training data for strategy controller transformer
# ABOUTME: Focuses on successful games and strategic decision points for reinforcement learning

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyTransition:
    """Represents a point where strategy should transition"""
    game_id: str
    strategy_name: str
    person_index: int
    game_state: Dict[str, Any]
    optimal_next_strategy: str
    confidence: float
    reasoning: str


@dataclass
class TrainingExample:
    """Single training example for strategy controller"""
    game_id: str
    original_strategy: str
    game_phase: str
    state_sequence: List[Dict[str, Any]]
    strategy_decision: str
    parameter_adjustments: Dict[str, Any]
    outcome_quality: float  # 0-1, how good was this decision
    final_success: bool
    final_rejections: int


class TrainingDataExtractor:
    """Extracts training data from game logs for strategy controller"""
    
    def __init__(self, game_logs_dir: str = "game_logs"):
        self.game_logs_dir = Path(game_logs_dir)
        self.successful_games: List[Dict] = []
        self.failed_games: List[Dict] = []
        self.training_examples: List[TrainingExample] = []
        
    def extract_all_training_data(self) -> List[TrainingExample]:
        """Extract comprehensive training data from all game logs"""
        logger.info("Starting training data extraction...")
        
        # Load and categorize all games
        self._load_all_games()
        
        # Extract training examples from successful games
        self._extract_successful_patterns()
        
        # Extract failure avoidance patterns
        self._extract_failure_patterns()
        
        # Extract strategy transition opportunities
        self._extract_transition_opportunities()
        
        logger.info(f"Extracted {len(self.training_examples)} training examples")
        return self.training_examples
    
    def _load_all_games(self):
        """Load and categorize all available games"""
        game_files = list(self.game_logs_dir.glob("game_*.json"))
        
        logger.info(f"Loading {len(game_files)} game files...")
        
        for game_file in game_files:
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                
                # Extract key info
                success = game_data.get('success', False)
                rejected_count = game_data.get('rejected_count', 0)
                strategy_name = game_data.get('strategy_name', 'unknown')
                
                # Add derived fields
                game_data['_file_path'] = str(game_file)
                game_data['_performance_score'] = self._calculate_performance_score(success, rejected_count)
                
                if success:
                    self.successful_games.append(game_data)
                else:
                    self.failed_games.append(game_data)
                    
            except Exception as e:
                logger.warning(f"Failed to load {game_file}: {e}")
        
        logger.info(f"Loaded {len(self.successful_games)} successful and {len(self.failed_games)} failed games")
        
        # Sort by performance
        self.successful_games.sort(key=lambda x: x['_performance_score'], reverse=True)
    
    def _calculate_performance_score(self, success: bool, rejected_count: int) -> float:
        """Calculate performance score for a game (higher = better)"""
        if success:
            return max(0, 2000 - rejected_count)  # Success bonus
        else:
            return max(0, 1000 - rejected_count)  # Penalty for failure
    
    def _extract_successful_patterns(self):
        """Extract patterns from successful games as positive examples"""
        logger.info(f"Extracting patterns from {len(self.successful_games)} successful games...")
        
        # Focus on best performing games
        top_games = self.successful_games[:50]  # Top 50 successful games
        
        for game_data in top_games:
            strategy_name = game_data.get('strategy_name', 'unknown')
            rejected_count = game_data.get('rejected_count', 0)
            
            # Extract key decision points from this successful game
            decision_points = self._extract_decision_points(game_data)
            
            for decision_point in decision_points:
                # Create training example showing this was a good decision
                example = TrainingExample(
                    game_id=game_data.get('game_id', 'unknown'),
                    original_strategy=strategy_name,
                    game_phase=decision_point['game_phase'],
                    state_sequence=decision_point['state_sequence'],
                    strategy_decision=strategy_name,  # Successful strategy stayed with itself
                    parameter_adjustments=decision_point['parameters'],
                    outcome_quality=min(1.0, game_data['_performance_score'] / 1500),  # Normalize to 0-1
                    final_success=True,
                    final_rejections=rejected_count
                )
                
                self.training_examples.append(example)
    
    def _extract_failure_patterns(self):
        """Extract patterns from failed games as negative examples"""
        logger.info(f"Extracting failure patterns from {len(self.failed_games)} failed games...")
        
        # Focus on games that almost succeeded (for learning)
        near_miss_games = [g for g in self.failed_games if g.get('rejected_count', 0) < 1200]
        near_miss_games.sort(key=lambda x: x['_performance_score'], reverse=True)
        
        for game_data in near_miss_games[:30]:  # Top 30 near-misses
            strategy_name = game_data.get('strategy_name', 'unknown')
            
            # Identify where strategy should have switched
            failure_reason = self._identify_failure_reason(game_data)
            
            if failure_reason:
                decision_points = self._extract_decision_points(game_data)
                
                for decision_point in decision_points[-5:]:  # Focus on late-game decisions
                    # Suggest what strategy should have been used instead
                    suggested_strategy = self._suggest_alternative_strategy(failure_reason, decision_point)
                    
                    example = TrainingExample(
                        game_id=game_data.get('game_id', 'unknown'),
                        original_strategy=strategy_name,
                        game_phase=decision_point['game_phase'],
                        state_sequence=decision_point['state_sequence'],
                        strategy_decision=suggested_strategy,
                        parameter_adjustments=decision_point['parameters'],
                        outcome_quality=0.2,  # Low quality since game failed
                        final_success=False,
                        final_rejections=game_data.get('rejected_count', 0)
                    )
                    
                    self.training_examples.append(example)
    
    def _extract_transition_opportunities(self):
        """Extract points where strategy transitions would be beneficial"""
        logger.info("Identifying strategy transition opportunities...")
        
        # Compare successful vs failed games of same strategy
        strategy_groups = defaultdict(lambda: {'successful': [], 'failed': []})
        
        for game in self.successful_games:
            strategy_groups[game['strategy_name']]['successful'].append(game)
        
        for game in self.failed_games:
            strategy_groups[game['strategy_name']]['failed'].append(game)
        
        # Find strategies that could benefit from transitions
        for strategy_name, games in strategy_groups.items():
            successful = games['successful']
            failed = games['failed']
            
            if len(successful) >= 3 and len(failed) >= 3:
                # Look for patterns where successful games differ from failed ones
                self._analyze_strategy_differences(strategy_name, successful, failed)
    
    def _extract_decision_points(self, game_data: Dict) -> List[Dict[str, Any]]:
        """Extract key decision points from a game"""
        decision_points = []
        
        rejected_count = game_data.get('rejected_count', 0)
        admitted_count = game_data.get('admitted_count', 0)
        constraint_satisfaction = game_data.get('constraint_satisfaction', {})
        
        # Sample decision points at different game phases
        phases = [
            ('early', 0.2),    # 20% through game
            ('mid', 0.5),      # 50% through game  
            ('late', 0.8),     # 80% through game
            ('final', 0.95)    # 95% through game
        ]
        
        total_decisions = admitted_count + rejected_count
        
        for phase_name, phase_ratio in phases:
            decision_index = int(total_decisions * phase_ratio)
            
            # Build state representation for this point
            state = {
                'person_index': decision_index,
                'admitted_count': int(admitted_count * phase_ratio),
                'rejected_count': int(rejected_count * phase_ratio),
                'young_progress': min(1.0, constraint_satisfaction.get('young', {}).get('progress', 0) * phase_ratio),
                'well_dressed_progress': min(1.0, constraint_satisfaction.get('well_dressed', {}).get('progress', 0) * phase_ratio),
                'capacity_ratio': phase_ratio,
                'rejection_ratio': (rejected_count * phase_ratio) / 20000,
                'constraint_risk': self._calculate_constraint_risk(constraint_satisfaction, phase_ratio)
            }
            
            decision_point = {
                'game_phase': phase_name,
                'decision_index': decision_index,
                'state_sequence': [state],  # Simplified - could be expanded to sequence
                'parameters': game_data.get('strategy_params', {})
            }
            
            decision_points.append(decision_point)
        
        return decision_points
    
    def _calculate_constraint_risk(self, constraint_satisfaction: Dict, phase_ratio: float) -> float:
        """Calculate constraint satisfaction risk at given game phase"""
        young_progress = constraint_satisfaction.get('young', {}).get('progress', 0) * phase_ratio
        well_dressed_progress = constraint_satisfaction.get('well_dressed', {}).get('progress', 0) * phase_ratio
        
        min_progress = min(young_progress, well_dressed_progress)
        expected_progress = phase_ratio  # Expected if admitting everyone
        
        if expected_progress > 0:
            return max(0, 1.0 - (min_progress / expected_progress))
        return 0.5
    
    def _identify_failure_reason(self, game_data: Dict) -> Optional[str]:
        """Identify why a game failed"""
        constraint_satisfaction = game_data.get('constraint_satisfaction', {})
        rejected_count = game_data.get('rejected_count', 0)
        
        young_satisfied = constraint_satisfaction.get('young', {}).get('satisfied', False)
        well_dressed_satisfied = constraint_satisfaction.get('well_dressed', {}).get('satisfied', False)
        
        if not young_satisfied and not well_dressed_satisfied:
            return 'both_constraints_failed'
        elif not young_satisfied:
            return 'young_constraint_failed'
        elif not well_dressed_satisfied:
            return 'well_dressed_constraint_failed'
        elif rejected_count > 1500:
            return 'too_many_rejections'
        else:
            return 'unknown_failure'
    
    def _suggest_alternative_strategy(self, failure_reason: str, decision_point: Dict) -> str:
        """Suggest what strategy should have been used based on failure reason"""
        game_phase = decision_point['game_phase']
        
        # Strategy recommendations based on failure patterns
        if failure_reason in ['young_constraint_failed', 'well_dressed_constraint_failed', 'both_constraints_failed']:
            if game_phase in ['early', 'mid']:
                return 'dual_deficit'  # Focus on constraint satisfaction
            else:
                return 'perfect'  # Emergency constraint filling
        elif failure_reason == 'too_many_rejections':
            return 'rbcr2'  # More efficient rejection strategy
        else:
            return 'ultimate3'  # Balanced fallback
    
    def _analyze_strategy_differences(self, strategy_name: str, successful: List[Dict], failed: List[Dict]):
        """Analyze differences between successful and failed games of same strategy"""
        # This is a simplified analysis - could be much more sophisticated
        
        # Calculate average performance metrics
        successful_rejections = [g.get('rejected_count', 0) for g in successful]
        failed_rejections = [g.get('rejected_count', 0) for g in failed]
        
        avg_successful_rejections = np.mean(successful_rejections)
        avg_failed_rejections = np.mean(failed_rejections)
        
        # If failed games had much higher rejections, suggest more efficient strategies
        if avg_failed_rejections > avg_successful_rejections * 1.5:
            # Create training examples suggesting strategy switches for high-rejection scenarios
            for game in failed[:5]:  # Top 5 failed games
                decision_points = self._extract_decision_points(game)
                
                for dp in decision_points[-2:]:  # Late game decisions
                    example = TrainingExample(
                        game_id=game.get('game_id', 'unknown'),
                        original_strategy=strategy_name,
                        game_phase=dp['game_phase'],
                        state_sequence=dp['state_sequence'],
                        strategy_decision='rbcr2',  # Suggest more efficient strategy
                        parameter_adjustments=dp['parameters'],
                        outcome_quality=0.3,
                        final_success=False,
                        final_rejections=game.get('rejected_count', 0)
                    )
                    
                    self.training_examples.append(example)
    
    def export_training_data(self, output_file: str = "strategy_controller_training_data.json") -> Dict[str, Any]:
        """Export training data for strategy controller"""
        training_data = {
            'metadata': {
                'total_examples': len(self.training_examples),
                'successful_games_analyzed': len(self.successful_games),
                'failed_games_analyzed': len(self.failed_games),
                'strategies_covered': list(set(ex.original_strategy for ex in self.training_examples)),
                'extraction_timestamp': pd.Timestamp.now().isoformat()
            },
            'training_examples': []
        }
        
        # Export training examples
        for example in self.training_examples:
            training_data['training_examples'].append({
                'game_id': example.game_id,
                'original_strategy': example.original_strategy,
                'game_phase': example.game_phase,
                'state_sequence': example.state_sequence,
                'strategy_decision': example.strategy_decision,
                'parameter_adjustments': example.parameter_adjustments,
                'outcome_quality': example.outcome_quality,
                'final_success': example.final_success,
                'final_rejections': example.final_rejections
            })
        
        # Add statistics
        strategy_stats = defaultdict(lambda: {'count': 0, 'avg_quality': 0, 'success_rate': 0})
        
        for example in self.training_examples:
            strategy = example.strategy_decision
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['avg_quality'] += example.outcome_quality
            if example.final_success:
                strategy_stats[strategy]['success_rate'] += 1
        
        # Normalize statistics
        for strategy, stats in strategy_stats.items():
            count = stats['count']
            stats['avg_quality'] /= count
            stats['success_rate'] /= count
        
        training_data['strategy_statistics'] = dict(strategy_stats)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Training data exported to {output_file}")
        return training_data
    
    def get_best_examples(self, min_quality: float = 0.7) -> List[TrainingExample]:
        """Get high-quality training examples"""
        return [ex for ex in self.training_examples if ex.outcome_quality >= min_quality]
    
    def get_examples_by_strategy(self) -> Dict[str, List[TrainingExample]]:
        """Group training examples by recommended strategy"""
        strategy_examples = defaultdict(list)
        for example in self.training_examples:
            strategy_examples[example.strategy_decision].append(example)
        return dict(strategy_examples)


def main():
    """Extract training data from game logs"""
    extractor = TrainingDataExtractor()
    
    # Extract all training data
    training_examples = extractor.extract_all_training_data()
    
    # Export training data
    training_data = extractor.export_training_data()
    
    # Print summary
    print(f"\n=== Training Data Extraction Summary ===")
    print(f"Total training examples: {len(training_examples)}")
    print(f"Successful games analyzed: {len(extractor.successful_games)}")
    print(f"Failed games analyzed: {len(extractor.failed_games)}")
    
    # Print strategy statistics
    print(f"\n=== Strategy Recommendations ===")
    for strategy, stats in training_data['strategy_statistics'].items():
        print(f"{strategy}: {stats['count']} examples, {stats['avg_quality']:.3f} avg quality, {stats['success_rate']:.1%} success rate")
    
    # Print best examples count
    best_examples = extractor.get_best_examples(min_quality=0.7)
    print(f"\nHigh-quality examples (>0.7 quality): {len(best_examples)}")
    
    # Print phase distribution
    phase_counts = defaultdict(int)
    for example in training_examples:
        phase_counts[example.game_phase] += 1
    
    print(f"\n=== Game Phase Distribution ===")
    for phase, count in phase_counts.items():
        print(f"{phase}: {count} examples")


if __name__ == "__main__":
    main()