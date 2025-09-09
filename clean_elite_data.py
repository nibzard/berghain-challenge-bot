#!/usr/bin/env python3
"""
ABOUTME: Clean and filter only successful elite games for accurate transformer training
ABOUTME: Removes invalid/failed games and creates verified training tiers based on actual performance
"""

import json
import glob
from pathlib import Path
import shutil
import re
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ValidGameData:
    """Represents a validated successful game."""
    file_path: str
    strategy: str
    rejections: int
    admitted: int
    success: bool
    efficiency_score: float

class EliteDataCleaner:
    """Clean and organize elite training data with verified successful games only."""
    
    # Updated tier definitions based on actual successful game performance
    TIERS = {
        'ultra_elite': {'max_rejections': 780, 'min_admitted': 900, 'description': 'Ultra-elite successful games'},
        'elite': {'max_rejections': 820, 'min_admitted': 900, 'description': 'Elite successful games'}, 
        'good': {'max_rejections': 850, 'min_admitted': 900, 'description': 'Good successful games'},
    }
    
    def __init__(self, game_logs_dir: str = "game_logs", output_dir: str = "verified_elite_data"):
        self.game_logs_dir = Path(game_logs_dir)
        self.output_dir = Path(output_dir)
        self.valid_games = []
        self.invalid_games = []
        
    def clean_and_filter(self) -> Dict[str, Any]:
        """Clean data and create verified training tiers."""
        logger.info("🧹 Starting elite data cleaning and filtering...")
        
        # Process all game files
        game_files = list(self.game_logs_dir.glob("game_*.json"))
        logger.info(f"📁 Processing {len(game_files)} game files...")
        
        for game_file in game_files:
            try:
                game_data = self._validate_game(game_file)
                if game_data:
                    self.valid_games.append(game_data)
                else:
                    self.invalid_games.append(str(game_file))
            except Exception as e:
                logger.warning(f"Error processing {game_file}: {e}")
                self.invalid_games.append(str(game_file))
        
        logger.info(f"✅ Found {len(self.valid_games)} valid games")
        logger.info(f"❌ Found {len(self.invalid_games)} invalid/failed games")
        
        # Create verified tiers
        tiers = self._create_verified_tiers()
        
        # Save clean data
        self._save_clean_tiers(tiers)
        
        # Generate report
        report = self._generate_cleaning_report(tiers)
        
        return report
    
    def _validate_game(self, game_file: Path) -> ValidGameData:
        """Validate a single game and return structured data if valid."""
        with open(game_file, 'r') as f:
            data = json.load(f)
        
        # Extract basic metrics
        success = data.get('success', False)
        admitted = data.get('admitted_count', 0)
        rejections = data.get('rejected_count', float('inf'))
        
        # Must be successful and meet minimum constraints
        if not success:
            return None
        if admitted < 900:  # Must admit substantial number of people
            return None
        if rejections == float('inf') or rejections > 1000:  # Reasonable rejection count
            return None
            
        # Extract strategy from filename
        strategy = self._extract_strategy(game_file)
        
        # Calculate efficiency score
        efficiency_score = admitted / max(rejections, 1)
        
        return ValidGameData(
            file_path=str(game_file),
            strategy=strategy,
            rejections=rejections,
            admitted=admitted,
            success=success,
            efficiency_score=efficiency_score
        )
    
    def _extract_strategy(self, game_file: Path) -> str:
        """Extract strategy name from filename."""
        fname = game_file.name
        match = re.match(r'game_([^_]+)_', fname)
        if match:
            strategy = match.group(1)
            # Clean up strategy names
            if strategy.startswith('elite'):
                return 'elite_lstm'
            elif strategy.startswith('ultra'):
                return 'ultra_elite_lstm'  
            elif strategy.startswith('constraint'):
                return 'constraint_focused_lstm'
            return strategy
        return 'unknown'
    
    def _create_verified_tiers(self) -> Dict[str, List[ValidGameData]]:
        """Create verified tiers based on successful game performance."""
        tiers = {tier_name: [] for tier_name in self.TIERS.keys()}
        
        for game in self.valid_games:
            for tier_name, tier_info in self.TIERS.items():
                if (game.rejections <= tier_info['max_rejections'] and 
                    game.admitted >= tier_info['min_admitted']):
                    tiers[tier_name].append(game)
                    break
        
        return tiers
    
    def _save_clean_tiers(self, tiers: Dict[str, List[ValidGameData]]) -> None:
        """Save verified clean tiers."""
        self.output_dir.mkdir(exist_ok=True)
        
        for tier_name, games in tiers.items():
            if not games:
                continue
                
            tier_dir = self.output_dir / tier_name
            tier_dir.mkdir(exist_ok=True)
            
            # Copy valid games
            copied_files = []
            for game in games:
                src = Path(game.file_path)
                if src.exists():
                    dst = tier_dir / src.name
                    shutil.copy2(src, dst)
                    copied_files.append(str(dst))
            
            # Save tier metadata
            tier_metadata = {
                'tier': tier_name,
                'description': self.TIERS[tier_name]['description'],
                'max_rejections': self.TIERS[tier_name]['max_rejections'],
                'min_admitted': self.TIERS[tier_name]['min_admitted'],
                'game_count': len(games),
                'avg_rejections': sum(g.rejections for g in games) / len(games),
                'min_rejections': min(g.rejections for g in games),
                'max_rejections': max(g.rejections for g in games),
                'avg_admitted': sum(g.admitted for g in games) / len(games),
                'strategies': dict(Counter(g.strategy for g in games)),
                'file_paths': copied_files,
                'best_games': [g.file_path for g in sorted(games, key=lambda x: x.rejections)[:10]]
            }
            
            with open(tier_dir / 'tier_metadata.json', 'w') as f:
                json.dump(tier_metadata, f, indent=2)
            
            logger.info(f"📁 Created {tier_name} tier: {len(games)} games (best: {min(g.rejections for g in games)})")
    
    def _generate_cleaning_report(self, tiers: Dict[str, List[ValidGameData]]) -> Dict[str, Any]:
        """Generate comprehensive cleaning report."""
        # Strategy performance analysis
        strategy_performance = defaultdict(list)
        for game in self.valid_games:
            strategy_performance[game.strategy].append(game.rejections)
        
        strategy_stats = {}
        for strategy, rejections in strategy_performance.items():
            strategy_stats[strategy] = {
                'game_count': len(rejections),
                'avg_rejections': sum(rejections) / len(rejections),
                'min_rejections': min(rejections),
                'max_rejections': max(rejections),
                'sub_750_count': sum(1 for r in rejections if r < 750),
                'record_potential': min(rejections) < 750
            }
        
        # Tier analysis
        tier_stats = {}
        for tier_name, games in tiers.items():
            if games:
                tier_stats[tier_name] = {
                    'count': len(games),
                    'avg_rejections': sum(g.rejections for g in games) / len(games),
                    'best_rejection': min(g.rejections for g in games),
                    'strategies': dict(Counter(g.strategy for g in games)),
                    'training_potential': 'high' if len(games) >= 50 else 'medium' if len(games) >= 10 else 'low'
                }
        
        report = {
            'total_files_processed': len(self.valid_games) + len(self.invalid_games),
            'valid_successful_games': len(self.valid_games),
            'invalid_failed_games': len(self.invalid_games),
            'best_performance': min(g.rejections for g in self.valid_games) if self.valid_games else None,
            'record_beating_potential': any(g.rejections < 750 for g in self.valid_games),
            'tier_distribution': tier_stats,
            'strategy_performance': strategy_stats,
            'training_recommendations': self._generate_training_recommendations(tier_stats, strategy_stats)
        }
        
        return report
    
    def _generate_training_recommendations(self, tier_stats: Dict, strategy_stats: Dict) -> List[str]:
        """Generate specific training recommendations."""
        recommendations = []
        
        # Check ultra-elite tier
        ultra_elite = tier_stats.get('ultra_elite', {})
        if ultra_elite.get('count', 0) >= 20:
            recommendations.append(f"Use {ultra_elite['count']} ultra-elite games for primary transformer training")
            recommendations.append(f"Focus on best performer: {ultra_elite['best_rejection']} rejections")
        
        # Identify best strategies
        best_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['min_rejections'])[:3]
        for strategy, stats in best_strategies:
            if stats['min_rejections'] < 780:
                recommendations.append(f"Prioritize {strategy} strategy (best: {stats['min_rejections']} rejections)")
        
        # Record-beating potential
        record_candidates = [s for s, stats in strategy_stats.items() if stats['min_rejections'] < 750]
        if record_candidates:
            recommendations.append(f"Focus on record-beating strategies: {', '.join(record_candidates)}")
        else:
            recommendations.append("Need synthetic data generation - closest to record is 750 rejections")
        
        return recommendations

def main():
    """Main execution."""
    cleaner = EliteDataCleaner()
    report = cleaner.clean_and_filter()
    
    # Save comprehensive report
    with open('elite_data_cleaning_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n🧹 ELITE DATA CLEANING COMPLETE")
    print("=" * 50)
    print(f"✅ Valid successful games: {report['valid_successful_games']}")
    print(f"❌ Invalid/failed games: {report['invalid_failed_games']}")
    print(f"🏆 Best performance: {report['best_performance']} rejections")
    print(f"🎯 Record beating potential: {report['record_beating_potential']}")
    
    print(f"\n📊 TIER DISTRIBUTION:")
    for tier_name, stats in report['tier_distribution'].items():
        print(f"  {tier_name:12s}: {stats['count']:3d} games (best: {stats['best_rejection']:3d}, avg: {stats['avg_rejections']:.1f})")
    
    print(f"\n🚀 TOP STRATEGIES:")
    strategy_items = sorted(report['strategy_performance'].items(), key=lambda x: x[1]['min_rejections'])
    for strategy, stats in strategy_items[:8]:
        potential = "🎖️" if stats['min_rejections'] < 750 else "🏅" if stats['min_rejections'] < 780 else "📈"
        print(f"  {potential} {strategy:20s}: {stats['game_count']:3d} games, best: {stats['min_rejections']:3d}, avg: {stats['avg_rejections']:.1f}")
    
    print(f"\n💡 TRAINING RECOMMENDATIONS:")
    for rec in report['training_recommendations']:
        print(f"  • {rec}")
    
    print(f"\n📄 Full report saved to: elite_data_cleaning_report.json")

if __name__ == "__main__":
    main()