#!/usr/bin/env python3
"""
ABOUTME: Advanced multi-tier game data analyzer and filter for transformer training optimization
ABOUTME: Creates quality-based tiers and analyzes patterns in elite performance games
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class GameAnalysis:
    """Detailed analysis of a single game."""
    file_path: str
    strategy: str
    rejections: int
    admitted: int
    success: bool
    efficiency_score: float
    constraint_satisfaction: Dict[str, Any]
    decision_patterns: Dict[str, Any]
    tier: str

class MassDataAnalyzer:
    """Advanced analyzer for processing thousands of games and creating training tiers."""
    
    # Quality tier definitions based on MASTERPLAN
    TIERS = {
        'ultra_elite': {'max_rejections': 750, 'description': 'Ultra-elite games for premium training'},
        'elite': {'max_rejections': 800, 'description': 'Elite games for core training'}, 
        'good': {'max_rejections': 850, 'description': 'Good games for diverse training'},
        'average': {'max_rejections': 950, 'description': 'Average games for baseline training'},
        'below_average': {'max_rejections': float('inf'), 'description': 'Below average games for analysis only'}
    }
    
    def __init__(self, game_logs_dir: str = "game_logs", output_base_dir: str = "training_tiers"):
        self.game_logs_dir = Path(game_logs_dir)
        self.output_base_dir = Path(output_base_dir)
        self.games_by_tier = defaultdict(list)
        self.strategy_performance = defaultdict(list)
        self.pattern_analysis = defaultdict(list)
        
    def analyze_all_games(self, date_filter: str = None) -> Dict[str, Any]:
        """
        Analyze all games and categorize into training tiers.
        
        Args:
            date_filter: Optional date filter (YYYYMMDD) to only process today's games
            
        Returns:
            Comprehensive analysis report
        """
        logger.info("🔍 Starting comprehensive game analysis...")
        
        # Find all game files
        if date_filter:
            pattern = f"game_*{date_filter}*.json"
        else:
            pattern = "game_*.json"
            
        game_files = list(self.game_logs_dir.glob(pattern))
        logger.info(f"📁 Found {len(game_files)} game files to analyze")
        
        if not game_files:
            logger.warning(f"No game files found matching pattern: {pattern}")
            return {}
        
        # Process each game
        analyses = []
        for game_file in game_files:
            try:
                analysis = self._analyze_single_game(game_file)
                if analysis:
                    analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Error analyzing {game_file}: {e}")
        
        logger.info(f"✅ Successfully analyzed {len(analyses)} games")
        
        # Categorize into tiers
        self._categorize_by_tiers(analyses)
        
        # Generate comprehensive report
        report = self._generate_analysis_report(analyses)
        
        # Save filtered games to tier directories
        self._save_tier_games()
        
        return report
    
    def _analyze_single_game(self, game_file: Path) -> GameAnalysis:
        """Perform detailed analysis of a single game."""
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        # Extract core metrics
        strategy = game_data.get('strategy', 'unknown')
        rejections = game_data.get('rejected_count', float('inf'))
        admitted = game_data.get('admitted_count', 0)
        success = game_data.get('success', False)
        
        # Calculate efficiency score (lower rejections = higher efficiency)
        if rejections == float('inf'):
            efficiency_score = 0.0
        else:
            # Efficiency: admitted people per rejection (higher is better)
            efficiency_score = admitted / max(rejections, 1) * 100
        
        # Analyze constraint satisfaction
        constraints = game_data.get('final_constraints', {})
        constraint_analysis = self._analyze_constraints(constraints)
        
        # Analyze decision patterns
        decisions = game_data.get('decisions', [])
        decision_patterns = self._analyze_decision_patterns(decisions, strategy)
        
        # Determine tier
        tier = self._determine_tier(rejections)
        
        return GameAnalysis(
            file_path=str(game_file),
            strategy=strategy,
            rejections=rejections,
            admitted=admitted,
            success=success,
            efficiency_score=efficiency_score,
            constraint_satisfaction=constraint_analysis,
            decision_patterns=decision_patterns,
            tier=tier
        )
    
    def _analyze_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well constraints were satisfied."""
        analysis = {}
        
        for constraint_name, data in constraints.items():
            if isinstance(data, dict):
                current = data.get('current', 0)
                required = data.get('required', 0)
                satisfaction_rate = current / max(required, 1) * 100
                analysis[constraint_name] = {
                    'current': current,
                    'required': required,
                    'satisfaction_rate': satisfaction_rate,
                    'met': current >= required
                }
        
        return analysis
    
    def _analyze_decision_patterns(self, decisions: List[Dict], strategy: str) -> Dict[str, Any]:
        """Analyze decision patterns for insights."""
        if not decisions:
            return {}
        
        # Basic decision stats
        admit_count = sum(1 for d in decisions if d.get('decision', False))
        reject_count = len(decisions) - admit_count
        admit_rate = admit_count / len(decisions) * 100
        
        # Reasoning pattern analysis
        reasoning_counts = Counter(d.get('reasoning', 'unknown') for d in decisions)
        
        # Temporal patterns (early vs late game decisions)
        early_decisions = decisions[:len(decisions)//3]
        late_decisions = decisions[-len(decisions)//3:]
        
        early_admit_rate = sum(1 for d in early_decisions if d.get('decision', False)) / len(early_decisions) * 100 if early_decisions else 0
        late_admit_rate = sum(1 for d in late_decisions if d.get('decision', False)) / len(late_decisions) * 100 if late_decisions else 0
        
        return {
            'total_decisions': len(decisions),
            'admit_count': admit_count,
            'reject_count': reject_count,
            'admit_rate': admit_rate,
            'reasoning_patterns': dict(reasoning_counts),
            'early_admit_rate': early_admit_rate,
            'late_admit_rate': late_admit_rate,
            'strategy_focus': self._classify_strategy_focus(reasoning_counts, strategy)
        }
    
    def _classify_strategy_focus(self, reasoning_counts: Counter, strategy: str) -> str:
        """Classify the primary focus of the strategy based on reasoning patterns."""
        if not reasoning_counts:
            return 'unknown'
        
        top_reasons = reasoning_counts.most_common(3)
        
        # Classify based on most common reasoning patterns
        if any('constraint' in reason[0].lower() for reason in top_reasons):
            return 'constraint_focused'
        elif any('efficiency' in reason[0].lower() or 'optimal' in reason[0].lower() for reason in top_reasons):
            return 'efficiency_focused'
        elif any('balanced' in reason[0].lower() for reason in top_reasons):
            return 'balanced'
        else:
            return 'pattern_based'
    
    def _determine_tier(self, rejections: int) -> str:
        """Determine quality tier based on rejection count."""
        for tier_name, tier_info in self.TIERS.items():
            if rejections <= tier_info['max_rejections']:
                return tier_name
        return 'below_average'
    
    def _categorize_by_tiers(self, analyses: List[GameAnalysis]) -> None:
        """Categorize games by quality tiers."""
        self.games_by_tier.clear()
        
        for analysis in analyses:
            self.games_by_tier[analysis.tier].append(analysis)
            self.strategy_performance[analysis.strategy].append(analysis.rejections)
    
    def _generate_analysis_report(self, analyses: List[GameAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        total_games = len(analyses)
        
        # Tier distribution
        tier_stats = {}
        for tier_name, tier_info in self.TIERS.items():
            tier_games = self.games_by_tier.get(tier_name, [])
            if tier_games:
                rejections = [game.rejections for game in tier_games]
                tier_stats[tier_name] = {
                    'count': len(tier_games),
                    'percentage': len(tier_games) / total_games * 100,
                    'avg_rejections': np.mean(rejections),
                    'min_rejections': min(rejections),
                    'max_rejections': max(rejections),
                    'strategies': Counter(game.strategy for game in tier_games)
                }
            else:
                tier_stats[tier_name] = {
                    'count': 0,
                    'percentage': 0,
                    'avg_rejections': 0,
                    'min_rejections': 0,
                    'max_rejections': 0,
                    'strategies': {}
                }
        
        # Strategy performance analysis
        strategy_stats = {}
        for strategy, rejections_list in self.strategy_performance.items():
            if rejections_list:
                strategy_stats[strategy] = {
                    'total_games': len(rejections_list),
                    'avg_rejections': np.mean(rejections_list),
                    'min_rejections': min(rejections_list),
                    'best_performances': sorted(rejections_list)[:5],  # Top 5 games
                    'success_rate': sum(1 for analysis in analyses 
                                       if analysis.strategy == strategy and analysis.success) / len(rejections_list) * 100
                }
        
        # Overall performance insights
        all_rejections = [a.rejections for a in analyses if a.rejections != float('inf')]
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_games_analyzed': total_games,
            'tier_distribution': tier_stats,
            'strategy_performance': strategy_stats,
            'overall_stats': {
                'avg_rejections': np.mean(all_rejections) if all_rejections else 0,
                'min_rejections': min(all_rejections) if all_rejections else 0,
                'ultra_elite_target_met': len(self.games_by_tier.get('ultra_elite', [])) >= 500,
                'record_beat_games': len([r for r in all_rejections if r < 716])
            },
            'training_recommendations': self._generate_training_recommendations(tier_stats, strategy_stats)
        }
        
        return report
    
    def _generate_training_recommendations(self, tier_stats: Dict, strategy_stats: Dict) -> Dict[str, Any]:
        """Generate recommendations for transformer training."""
        recommendations = {
            'primary_training_data': [],
            'augmentation_targets': [],
            'pattern_analysis': [],
            'architecture_suggestions': []
        }
        
        # Primary training data recommendations
        ultra_elite_count = tier_stats.get('ultra_elite', {}).get('count', 0)
        elite_count = tier_stats.get('elite', {}).get('count', 0)
        
        if ultra_elite_count >= 100:
            recommendations['primary_training_data'].append(f"Use {ultra_elite_count} ultra-elite games as primary training set")
        
        if elite_count >= 500:
            recommendations['primary_training_data'].append(f"Supplement with {elite_count} elite games for diversity")
        
        # Find best performing strategies
        best_strategies = sorted(strategy_stats.items(), key=lambda x: x[1].get('avg_rejections', float('inf')))[:3]
        for strategy, stats in best_strategies:
            if stats.get('avg_rejections', 0) < 850:
                recommendations['augmentation_targets'].append(f"Generate synthetic variations of {strategy} strategy")
        
        return recommendations
    
    def _save_tier_games(self) -> None:
        """Save tier metadata and file lists (no copying to save space)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for tier_name, games in self.games_by_tier.items():
            if not games:
                continue
                
            tier_dir = self.output_base_dir / f"{tier_name}_{timestamp}"
            tier_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file list instead of copying
            file_list = [game.file_path for game in games]
            
            # Save tier metadata with file references
            metadata = {
                'tier': tier_name,
                'description': self.TIERS[tier_name]['description'],
                'max_rejections': self.TIERS[tier_name]['max_rejections'],
                'game_count': len(games),
                'avg_rejections': np.mean([g.rejections for g in games]),
                'strategies': Counter(g.strategy for g in games),
                'file_paths': file_list,
                'best_games': sorted(file_list, key=lambda x: next(g.rejections for g in games if g.file_path == x))[:10],
                'created': datetime.now().isoformat()
            }
            
            with open(tier_dir / 'tier_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save just the file list for easy access
            with open(tier_dir / 'game_files.txt', 'w') as f:
                for file_path in file_list:
                    f.write(f"{file_path}\n")
            
            logger.info(f"📁 Created {tier_name} tier with {len(games)} games at {tier_dir}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced mass game data analyzer")
    parser.add_argument('--game-logs-dir', default='game_logs', help='Game logs directory')
    parser.add_argument('--output-dir', default='training_tiers', help='Output directory for tiers')
    parser.add_argument('--date-filter', help='Filter by date (YYYYMMDD)')
    parser.add_argument('--save-report', help='Save analysis report to file')
    
    args = parser.parse_args()
    
    analyzer = MassDataAnalyzer(args.game_logs_dir, args.output_dir)
    report = analyzer.analyze_all_games(args.date_filter)
    
    # Print summary
    print("\n🎯 MASS DATA ANALYSIS COMPLETE")
    print("=" * 50)
    
    tier_dist = report.get('tier_distribution', {})
    for tier_name, stats in tier_dist.items():
        if stats['count'] > 0:
            print(f"🏆 {tier_name.upper()}: {stats['count']} games "
                  f"(avg: {stats['avg_rejections']:.1f} rejections)")
    
    # Best performances
    overall = report.get('overall_stats', {})
    min_rejections = overall.get('min_rejections', 0)
    record_beat = overall.get('record_beat_games', 0)
    
    print(f"\n🥇 BEST PERFORMANCE: {min_rejections} rejections")
    print(f"🎖️  RECORD-BEATING GAMES: {record_beat}")
    
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📊 Full report saved to: {args.save_report}")

if __name__ == "__main__":
    main()