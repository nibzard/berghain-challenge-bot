#!/usr/bin/env python3
"""
ABOUTME: Advanced decision pattern analyzer to identify optimal sequences from elite games  
ABOUTME: Extracts insights for synthetic data generation and transformer architecture improvements
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class DecisionSequence:
    """Represents a sequence of decisions with context."""
    game_id: str
    strategy: str
    start_index: int
    end_index: int
    decisions: List[Dict[str, Any]]
    context: Dict[str, Any]
    outcome_quality: float
    final_rejections: int

@dataclass  
class PatternInsight:
    """Insight discovered from pattern analysis."""
    pattern_type: str
    description: str
    frequency: int
    avg_performance: float
    examples: List[str]
    recommendation: str

class DecisionPatternAnalyzer:
    """Advanced analyzer for extracting optimal decision patterns from elite games."""
    
    def __init__(self, elite_games_dir: str = "ultra_elite_games"):
        self.elite_games_dir = Path(elite_games_dir)
        self.patterns = defaultdict(list)
        self.sequence_db = []
        self.insights = []
        
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis of elite games.
        
        Returns:
            Analysis report with actionable insights
        """
        logger.info("🔍 Starting decision pattern analysis...")
        
        # Load all elite games
        game_files = list(self.elite_games_dir.glob("*.json"))
        if not game_files:
            logger.warning(f"No games found in {self.elite_games_dir}")
            return {}
        
        logger.info(f"📁 Analyzing patterns from {len(game_files)} elite games")
        
        # Extract decision sequences
        for game_file in game_files:
            try:
                sequences = self._extract_decision_sequences(game_file)
                self.sequence_db.extend(sequences)
            except Exception as e:
                logger.warning(f"Error processing {game_file}: {e}")
        
        logger.info(f"📊 Extracted {len(self.sequence_db)} decision sequences")
        
        # Analyze different pattern types
        self._analyze_opening_patterns()
        self._analyze_constraint_satisfaction_patterns()
        self._analyze_endgame_patterns()
        self._analyze_crisis_management_patterns()
        self._analyze_efficiency_patterns()
        
        # Generate actionable insights
        insights = self._generate_insights()
        
        # Create comprehensive report
        report = self._create_pattern_report(insights)
        
        return report
    
    def _extract_decision_sequences(self, game_file: Path) -> List[DecisionSequence]:
        """Extract meaningful decision sequences from a game."""
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        decisions = game_data.get('decisions', [])
        if not decisions:
            return []
        
        strategy = game_data.get('strategy', 'unknown')
        final_rejections = game_data.get('rejected_count', float('inf'))
        game_id = game_file.stem
        
        # Calculate outcome quality (lower rejections = higher quality)
        outcome_quality = max(0, 1000 - final_rejections) / 1000
        
        sequences = []
        
        # Extract different types of sequences
        sequences.extend(self._extract_opening_sequence(game_id, strategy, decisions, outcome_quality, final_rejections))
        sequences.extend(self._extract_constraint_sequences(game_id, strategy, decisions, outcome_quality, final_rejections))
        sequences.extend(self._extract_endgame_sequence(game_id, strategy, decisions, outcome_quality, final_rejections))
        sequences.extend(self._extract_crisis_sequences(game_id, strategy, decisions, outcome_quality, final_rejections))
        
        return sequences
    
    def _extract_opening_sequence(self, game_id: str, strategy: str, decisions: List[Dict], 
                                 outcome_quality: float, final_rejections: int) -> List[DecisionSequence]:
        """Extract opening game sequences (first 100 decisions)."""
        if len(decisions) < 50:
            return []
        
        opening_decisions = decisions[:100]
        
        # Analyze opening efficiency
        admits = sum(1 for d in opening_decisions if d.get('decision', False))
        rejects = len(opening_decisions) - admits
        
        context = {
            'phase': 'opening',
            'total_decisions': len(opening_decisions),
            'admits': admits,
            'rejects': rejects,
            'admit_rate': admits / len(opening_decisions),
            'efficiency_metric': admits / max(rejects, 1)
        }
        
        return [DecisionSequence(
            game_id=game_id,
            strategy=strategy,
            start_index=0,
            end_index=99,
            decisions=opening_decisions,
            context=context,
            outcome_quality=outcome_quality,
            final_rejections=final_rejections
        )]
    
    def _extract_constraint_sequences(self, game_id: str, strategy: str, decisions: List[Dict],
                                     outcome_quality: float, final_rejections: int) -> List[DecisionSequence]:
        """Extract sequences focused on constraint satisfaction."""
        sequences = []
        
        # Find sequences where constraint-related reasoning dominates
        window_size = 50
        for i in range(0, len(decisions) - window_size, 25):
            window = decisions[i:i + window_size]
            
            # Check if this window is constraint-focused
            constraint_decisions = [d for d in window if 'constraint' in d.get('reasoning', '').lower()]
            
            if len(constraint_decisions) >= window_size * 0.3:  # 30% constraint-focused
                admits = sum(1 for d in window if d.get('decision', False))
                
                context = {
                    'phase': 'constraint_satisfaction',
                    'window_start': i,
                    'constraint_focus_rate': len(constraint_decisions) / len(window),
                    'admit_rate': admits / len(window),
                    'reasoning_patterns': self._analyze_reasoning_patterns(window)
                }
                
                sequences.append(DecisionSequence(
                    game_id=game_id,
                    strategy=strategy,
                    start_index=i,
                    end_index=i + window_size - 1,
                    decisions=window,
                    context=context,
                    outcome_quality=outcome_quality,
                    final_rejections=final_rejections
                ))
        
        return sequences
    
    def _extract_endgame_sequence(self, game_id: str, strategy: str, decisions: List[Dict],
                                 outcome_quality: float, final_rejections: int) -> List[DecisionSequence]:
        """Extract endgame sequences (last 100 decisions)."""
        if len(decisions) < 100:
            return []
        
        endgame_decisions = decisions[-100:]
        
        admits = sum(1 for d in endgame_decisions if d.get('decision', False))
        rejects = len(endgame_decisions) - admits
        
        context = {
            'phase': 'endgame',
            'total_decisions': len(endgame_decisions),
            'admits': admits,
            'rejects': rejects,
            'admit_rate': admits / len(endgame_decisions),
            'selectivity': rejects / len(endgame_decisions)
        }
        
        return [DecisionSequence(
            game_id=game_id,
            strategy=strategy,
            start_index=len(decisions) - 100,
            end_index=len(decisions) - 1,
            decisions=endgame_decisions,
            context=context,
            outcome_quality=outcome_quality,
            final_rejections=final_rejections
        )]
    
    def _extract_crisis_sequences(self, game_id: str, strategy: str, decisions: List[Dict],
                                 outcome_quality: float, final_rejections: int) -> List[DecisionSequence]:
        """Extract sequences that handle crisis situations (constraint deficits)."""
        sequences = []
        
        # Look for sequences with high reject rates followed by recovery
        window_size = 30
        for i in range(0, len(decisions) - window_size, 10):
            window = decisions[i:i + window_size]
            
            rejects = sum(1 for d in window if not d.get('decision', True))
            reject_rate = rejects / len(window)
            
            # High rejection rate indicates potential crisis management
            if reject_rate >= 0.7:
                reasoning_patterns = self._analyze_reasoning_patterns(window)
                
                context = {
                    'phase': 'crisis_management',
                    'window_start': i,
                    'reject_rate': reject_rate,
                    'crisis_type': self._classify_crisis_type(reasoning_patterns),
                    'reasoning_patterns': reasoning_patterns
                }
                
                sequences.append(DecisionSequence(
                    game_id=game_id,
                    strategy=strategy,
                    start_index=i,
                    end_index=i + window_size - 1,
                    decisions=window,
                    context=context,
                    outcome_quality=outcome_quality,
                    final_rejections=final_rejections
                ))
        
        return sequences
    
    def _analyze_reasoning_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze reasoning patterns in a sequence of decisions."""
        reasoning_counts = defaultdict(int)
        admit_by_reason = defaultdict(int)
        reject_by_reason = defaultdict(int)
        
        for decision in decisions:
            reasoning = decision.get('reasoning', 'unknown')
            admit = decision.get('decision', False)
            
            reasoning_counts[reasoning] += 1
            
            if admit:
                admit_by_reason[reasoning] += 1
            else:
                reject_by_reason[reasoning] += 1
        
        return {
            'reasoning_distribution': dict(reasoning_counts),
            'admit_by_reasoning': dict(admit_by_reason),
            'reject_by_reasoning': dict(reject_by_reason),
            'dominant_reasoning': max(reasoning_counts.items(), key=lambda x: x[1])[0] if reasoning_counts else None
        }
    
    def _classify_crisis_type(self, reasoning_patterns: Dict[str, Any]) -> str:
        """Classify the type of crisis based on reasoning patterns."""
        reasoning_dist = reasoning_patterns.get('reasoning_distribution', {})
        
        if not reasoning_dist:
            return 'unknown'
        
        top_reason = max(reasoning_dist.items(), key=lambda x: x[1])[0]
        
        if 'constraint' in top_reason.lower():
            return 'constraint_deficit'
        elif 'capacity' in top_reason.lower() or 'limit' in top_reason.lower():
            return 'capacity_management'
        elif 'efficiency' in top_reason.lower():
            return 'efficiency_optimization'
        else:
            return 'general_selectivity'
    
    def _analyze_opening_patterns(self) -> None:
        """Analyze opening game patterns across elite games."""
        opening_sequences = [seq for seq in self.sequence_db if seq.context.get('phase') == 'opening']
        
        if not opening_sequences:
            return
        
        # Group by strategy
        by_strategy = defaultdict(list)
        for seq in opening_sequences:
            by_strategy[seq.strategy].append(seq)
        
        # Find best opening patterns
        for strategy, sequences in by_strategy.items():
            # Sort by outcome quality
            best_sequences = sorted(sequences, key=lambda x: x.outcome_quality, reverse=True)[:5]
            
            # Analyze common patterns in best openings
            admit_rates = [seq.context['admit_rate'] for seq in best_sequences]
            efficiency_metrics = [seq.context['efficiency_metric'] for seq in best_sequences]
            
            pattern_info = {
                'strategy': strategy,
                'sample_size': len(sequences),
                'best_admit_rate': np.mean(admit_rates),
                'best_efficiency': np.mean(efficiency_metrics),
                'top_games': [seq.game_id for seq in best_sequences],
                'avg_final_rejections': np.mean([seq.final_rejections for seq in best_sequences])
            }
            
            self.patterns['opening_patterns'].append(pattern_info)
    
    def _analyze_constraint_satisfaction_patterns(self) -> None:
        """Analyze constraint satisfaction patterns."""
        constraint_sequences = [seq for seq in self.sequence_db 
                               if seq.context.get('phase') == 'constraint_satisfaction']
        
        if not constraint_sequences:
            return
        
        # Find most effective constraint satisfaction approaches
        by_performance = sorted(constraint_sequences, key=lambda x: x.outcome_quality, reverse=True)
        top_performers = by_performance[:20]  # Top 20 sequences
        
        # Analyze common characteristics
        focus_rates = [seq.context['constraint_focus_rate'] for seq in top_performers]
        admit_rates = [seq.context['admit_rate'] for seq in top_performers]
        
        pattern_info = {
            'sample_size': len(constraint_sequences),
            'top_performers_count': len(top_performers),
            'optimal_constraint_focus_rate': np.mean(focus_rates),
            'optimal_admit_rate_during_constraints': np.mean(admit_rates),
            'best_strategies': [seq.strategy for seq in top_performers],
            'avg_final_rejections': np.mean([seq.final_rejections for seq in top_performers])
        }
        
        self.patterns['constraint_patterns'].append(pattern_info)
    
    def _analyze_endgame_patterns(self) -> None:
        """Analyze endgame optimization patterns."""
        endgame_sequences = [seq for seq in self.sequence_db if seq.context.get('phase') == 'endgame']
        
        if not endgame_sequences:
            return
        
        # Find optimal endgame strategies
        by_performance = sorted(endgame_sequences, key=lambda x: x.outcome_quality, reverse=True)
        top_endgames = by_performance[:15]
        
        selectivity_rates = [seq.context['selectivity'] for seq in top_endgames]
        admit_rates = [seq.context['admit_rate'] for seq in top_endgames]
        
        pattern_info = {
            'sample_size': len(endgame_sequences),
            'optimal_endgame_selectivity': np.mean(selectivity_rates),
            'optimal_endgame_admit_rate': np.mean(admit_rates),
            'best_endgame_strategies': [seq.strategy for seq in top_endgames],
            'avg_final_rejections': np.mean([seq.final_rejections for seq in top_endgames])
        }
        
        self.patterns['endgame_patterns'].append(pattern_info)
    
    def _analyze_crisis_management_patterns(self) -> None:
        """Analyze crisis management patterns."""
        crisis_sequences = [seq for seq in self.sequence_db 
                           if seq.context.get('phase') == 'crisis_management']
        
        if not crisis_sequences:
            return
        
        # Group by crisis type
        by_crisis_type = defaultdict(list)
        for seq in crisis_sequences:
            crisis_type = seq.context.get('crisis_type', 'unknown')
            by_crisis_type[crisis_type].append(seq)
        
        crisis_analysis = {}
        for crisis_type, sequences in by_crisis_type.items():
            best_sequences = sorted(sequences, key=lambda x: x.outcome_quality, reverse=True)[:10]
            
            if best_sequences:
                crisis_analysis[crisis_type] = {
                    'sample_size': len(sequences),
                    'best_recovery_rate': np.mean([seq.outcome_quality for seq in best_sequences]),
                    'optimal_reject_rate': np.mean([seq.context['reject_rate'] for seq in best_sequences]),
                    'successful_strategies': [seq.strategy for seq in best_sequences],
                    'avg_final_rejections': np.mean([seq.final_rejections for seq in best_sequences])
                }
        
        self.patterns['crisis_patterns'].append(crisis_analysis)
    
    def _analyze_efficiency_patterns(self) -> None:
        """Analyze overall efficiency patterns."""
        # Calculate efficiency metrics for all sequences
        efficiency_data = []
        
        for seq in self.sequence_db:
            if seq.context.get('phase') in ['opening', 'constraint_satisfaction', 'endgame']:
                efficiency_score = seq.context.get('admit_rate', 0) / max(1 - seq.context.get('admit_rate', 0), 0.1)
                
                efficiency_data.append({
                    'game_id': seq.game_id,
                    'strategy': seq.strategy,
                    'phase': seq.context['phase'],
                    'efficiency_score': efficiency_score,
                    'final_rejections': seq.final_rejections,
                    'outcome_quality': seq.outcome_quality
                })
        
        # Find most efficient approaches by phase
        by_phase = defaultdict(list)
        for data in efficiency_data:
            by_phase[data['phase']].append(data)
        
        efficiency_analysis = {}
        for phase, data_list in by_phase.items():
            top_efficient = sorted(data_list, key=lambda x: x['efficiency_score'], reverse=True)[:10]
            
            efficiency_analysis[phase] = {
                'sample_size': len(data_list),
                'top_efficiency_score': np.mean([d['efficiency_score'] for d in top_efficient]),
                'best_strategies': [d['strategy'] for d in top_efficient],
                'avg_final_rejections': np.mean([d['final_rejections'] for d in top_efficient])
            }
        
        self.patterns['efficiency_patterns'].append(efficiency_analysis)
    
    def _generate_insights(self) -> List[PatternInsight]:
        """Generate actionable insights from pattern analysis."""
        insights = []
        
        # Opening strategy insights
        if 'opening_patterns' in self.patterns:
            for pattern in self.patterns['opening_patterns']:
                if pattern['best_efficiency'] > 0.5:
                    insights.append(PatternInsight(
                        pattern_type='opening_strategy',
                        description=f"{pattern['strategy']} shows excellent opening efficiency ({pattern['best_efficiency']:.2f})",
                        frequency=pattern['sample_size'],
                        avg_performance=pattern['avg_final_rejections'],
                        examples=pattern['top_games'][:3],
                        recommendation=f"Use {pattern['strategy']} opening patterns for early game transformer training"
                    ))
        
        # Constraint satisfaction insights
        if 'constraint_patterns' in self.patterns:
            for pattern in self.patterns['constraint_patterns']:
                insights.append(PatternInsight(
                    pattern_type='constraint_satisfaction',
                    description=f"Optimal constraint focus rate: {pattern['optimal_constraint_focus_rate']:.2f}",
                    frequency=pattern['sample_size'],
                    avg_performance=pattern['avg_final_rejections'],
                    examples=pattern['best_strategies'][:3],
                    recommendation="Train constraint-focused transformer head with these parameters"
                ))
        
        # Crisis management insights
        if 'crisis_patterns' in self.patterns:
            for crisis_data in self.patterns['crisis_patterns']:
                for crisis_type, analysis in crisis_data.items():
                    insights.append(PatternInsight(
                        pattern_type='crisis_management',
                        description=f"{crisis_type} recovery with {analysis['optimal_reject_rate']:.2f} reject rate",
                        frequency=analysis['sample_size'],
                        avg_performance=analysis['avg_final_rejections'],
                        examples=analysis['successful_strategies'][:3],
                        recommendation=f"Create specialized transformer for {crisis_type} situations"
                    ))
        
        return insights
    
    def _create_pattern_report(self, insights: List[PatternInsight]) -> Dict[str, Any]:
        """Create comprehensive pattern analysis report."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_sequences_analyzed': len(self.sequence_db),
            'games_analyzed': len(set(seq.game_id for seq in self.sequence_db)),
            'patterns_discovered': self.patterns,
            'actionable_insights': [
                {
                    'type': insight.pattern_type,
                    'description': insight.description,
                    'frequency': insight.frequency,
                    'performance': insight.avg_performance,
                    'examples': insight.examples,
                    'recommendation': insight.recommendation
                } for insight in insights
            ],
            'transformer_architecture_recommendations': self._generate_architecture_recommendations(insights),
            'synthetic_data_generation_targets': self._generate_synthetic_targets(insights)
        }
    
    def _generate_architecture_recommendations(self, insights: List[PatternInsight]) -> List[str]:
        """Generate transformer architecture recommendations based on patterns."""
        recommendations = []
        
        pattern_types = set(insight.pattern_type for insight in insights)
        
        if 'opening_strategy' in pattern_types:
            recommendations.append("Implement early-game specialized transformer head")
        
        if 'constraint_satisfaction' in pattern_types:
            recommendations.append("Add constraint-focused attention mechanism")
        
        if 'crisis_management' in pattern_types:
            recommendations.append("Create crisis detection and recovery modules")
        
        if len(pattern_types) >= 3:
            recommendations.append("Design ensemble architecture with specialized experts")
        
        return recommendations
    
    def _generate_synthetic_targets(self, insights: List[PatternInsight]) -> List[Dict[str, Any]]:
        """Generate targets for synthetic data generation."""
        targets = []
        
        for insight in insights:
            if insight.avg_performance < 800:  # High-quality patterns
                targets.append({
                    'pattern_type': insight.pattern_type,
                    'target_performance': insight.avg_performance,
                    'frequency_boost_needed': max(100 - insight.frequency, 0),
                    'source_examples': insight.examples,
                    'generation_priority': 'high' if insight.avg_performance < 750 else 'medium'
                })
        
        return targets

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Decision pattern analyzer")
    parser.add_argument('--elite-dir', default='ultra_elite_games', help='Elite games directory')
    parser.add_argument('--output-report', default='pattern_analysis_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    analyzer = DecisionPatternAnalyzer(args.elite_dir)
    report = analyzer.analyze_patterns()
    
    # Save report
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n🎯 DECISION PATTERN ANALYSIS COMPLETE")
    print("=" * 50)
    
    insights = report.get('actionable_insights', [])
    print(f"📊 Total insights discovered: {len(insights)}")
    
    for insight in insights[:5]:  # Top 5 insights
        print(f"\n🔍 {insight['type'].upper()}")
        print(f"   {insight['description']}")
        print(f"   Performance: {insight['performance']:.1f} rejections")
        print(f"   Recommendation: {insight['recommendation']}")
    
    print(f"\n📁 Full report saved to: {args.output_report}")

if __name__ == "__main__":
    main()