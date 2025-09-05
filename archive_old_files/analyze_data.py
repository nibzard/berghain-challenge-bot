# ABOUTME: Analysis tool for game logs to discover patterns and reverse engineer mechanics
# ABOUTME: Compares claimed vs actual distributions and finds exploitable patterns

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
import seaborn as sns

class GameDataAnalyzer:
    def __init__(self, data_dir: str = "game_logs"):
        self.data_dir = Path(data_dir)
        self.games_data = []
        self.load_all_games()
    
    def load_all_games(self):
        """Load all game log files."""
        if not self.data_dir.exists():
            print(f"❌ Data directory {self.data_dir} does not exist. Run data_collector.py first.")
            return
        
        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            print(f"❌ No game log files found in {self.data_dir}")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.load(f)
                    self.games_data.append(game_data)
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
        
        print(f"📁 Loaded {len(self.games_data)} game log files")
    
    def analyze_claimed_vs_actual_frequencies(self):
        """Compare claimed frequencies with observed frequencies."""
        print("\n📊 CLAIMED vs ACTUAL FREQUENCY ANALYSIS")
        print("="*50)
        
        by_scenario = defaultdict(lambda: {
            'claimed': defaultdict(list),
            'actual': defaultdict(list),
            'games': []
        })
        
        for game in self.games_data:
            scenario = game['scenario']
            by_scenario[scenario]['games'].append(game)
            
            # Claimed frequencies
            for attr, freq in game['attribute_frequencies'].items():
                by_scenario[scenario]['claimed'][attr].append(freq)
            
            # Calculate actual frequencies
            total_people = len(game['people'])
            if total_people > 0:
                attr_counts = defaultdict(int)
                for person in game['people']:
                    for attr, has_attr in person['attributes'].items():
                        if has_attr:
                            attr_counts[attr] += 1
                
                for attr, count in attr_counts.items():
                    actual_freq = count / total_people
                    by_scenario[scenario]['actual'][attr].append(actual_freq)
        
        # Print analysis
        for scenario in sorted(by_scenario.keys()):
            data = by_scenario[scenario]
            print(f"\n🎯 Scenario {scenario} ({len(data['games'])} games):")
            print("-" * 30)
            
            all_attrs = set(data['claimed'].keys()) | set(data['actual'].keys())
            
            for attr in sorted(all_attrs):
                claimed_vals = data['claimed'].get(attr, [])
                actual_vals = data['actual'].get(attr, [])
                
                if claimed_vals and actual_vals:
                    claimed_avg = np.mean(claimed_vals)
                    actual_avg = np.mean(actual_vals)
                    difference = actual_avg - claimed_avg
                    percent_diff = (difference / claimed_avg) * 100 if claimed_avg > 0 else 0
                    
                    print(f"  {attr:15} | Claimed: {claimed_avg:.3f} | Actual: {actual_avg:.3f} | "
                          f"Diff: {percent_diff:+.1f}%")
                    
                    # Statistical test
                    if len(actual_vals) > 1:
                        # Test if actual frequency significantly differs from claimed
                        t_stat, p_value = stats.ttest_1samp(actual_vals, claimed_avg)
                        if p_value < 0.05:
                            print(f"                  | ⚠️  SIGNIFICANT DIFFERENCE (p={p_value:.3f})")
    
    def analyze_temporal_patterns(self):
        """Look for patterns in how attributes appear over time."""
        print("\n⏰ TEMPORAL PATTERN ANALYSIS")
        print("="*40)
        
        for scenario in [1, 2, 3]:
            scenario_games = [g for g in self.games_data if g['scenario'] == scenario]
            if not scenario_games:
                continue
                
            print(f"\n🎯 Scenario {scenario}:")
            
            # Analyze if certain attributes become more/less common over time
            for game in scenario_games[:2]:  # Analyze first 2 games for brevity
                people = game['people']
                if len(people) < 100:
                    continue
                
                print(f"  Game {game['game_id'][:8]}:")
                
                # Split game into early, middle, late phases
                early = people[:len(people)//3]
                middle = people[len(people)//3:2*len(people)//3]
                late = people[2*len(people)//3:]
                
                phases = {'Early': early, 'Middle': middle, 'Late': late}
                
                for attr in game['attribute_frequencies'].keys():
                    phase_freqs = []
                    for phase_name, phase_people in phases.items():
                        if phase_people:
                            count = sum(1 for p in phase_people if p['attributes'].get(attr, False))
                            freq = count / len(phase_people)
                            phase_freqs.append((phase_name, freq))
                    
                    # Check for trend
                    freqs = [f[1] for f in phase_freqs]
                    if len(freqs) == 3:
                        early_late_diff = freqs[2] - freqs[0]
                        if abs(early_late_diff) > 0.05:  # 5% difference
                            trend = "↗️" if early_late_diff > 0 else "↘️"
                            print(f"    {attr:15} | {trend} {phase_freqs[0][1]:.3f} → {phase_freqs[2][1]:.3f}")
    
    def analyze_decision_patterns(self):
        """Analyze patterns in our random decisions."""
        print("\n🎲 DECISION PATTERN ANALYSIS")
        print("="*35)
        
        by_scenario = defaultdict(lambda: {
            'total_people': 0,
            'total_accepted': 0,
            'attr_accept_rates': defaultdict(list)
        })
        
        for game in self.games_data:
            scenario = game['scenario']
            data = by_scenario[scenario]
            
            for person in game['people']:
                data['total_people'] += 1
                if person['decision']:
                    data['total_accepted'] += 1
                
                # Track accept rate by attributes
                person_attrs = [attr for attr, has_attr in person['attributes'].items() if has_attr]
                attr_key = tuple(sorted(person_attrs)) if person_attrs else ('none',)
                
                data['attr_accept_rates'][attr_key].append(person['decision'])
        
        for scenario in sorted(by_scenario.keys()):
            data = by_scenario[scenario]
            overall_rate = data['total_accepted'] / data['total_people'] if data['total_people'] > 0 else 0
            
            print(f"\n🎯 Scenario {scenario}:")
            print(f"  Overall accept rate: {overall_rate:.3f}")
            print(f"  Total people seen: {data['total_people']}")
            
            # Show accept rates by attribute combinations
            print("  Accept rates by attributes:")
            for attr_combo, decisions in data['attr_accept_rates'].items():
                if len(decisions) >= 10:  # Only show combos with enough data
                    accept_rate = np.mean(decisions)
                    count = len(decisions)
                    attrs_str = ', '.join(attr_combo) if attr_combo != ('none',) else 'none'
                    print(f"    {attrs_str:25} | {accept_rate:.3f} ({count} people)")
    
    def find_winning_patterns(self):
        """Identify patterns in successful vs failed games."""
        print("\n🏆 WINNING PATTERN ANALYSIS")
        print("="*35)
        
        by_scenario = defaultdict(lambda: {
            'successful': [],
            'failed': []
        })
        
        for game in self.games_data:
            scenario = game['scenario']
            if game['final_status'] == 'completed':
                by_scenario[scenario]['successful'].append(game)
            else:
                by_scenario[scenario]['failed'].append(game)
        
        for scenario in sorted(by_scenario.keys()):
            data = by_scenario[scenario]
            successful = data['successful']
            failed = data['failed']
            
            print(f"\n🎯 Scenario {scenario}:")
            print(f"  Successful: {len(successful)} games")
            print(f"  Failed: {len(failed)} games")
            
            if successful and failed:
                # Compare rejection counts
                successful_rejections = [g['final_rejected_count'] for g in successful]
                failed_rejections = [g['final_rejected_count'] for g in failed]
                
                print(f"  Avg rejections (successful): {np.mean(successful_rejections):.1f}")
                print(f"  Avg rejections (failed): {np.mean(failed_rejections):.1f}")
                
                # Compare early game behavior
                for game_type, games in [('Successful', successful), ('Failed', failed)]:
                    if games:
                        # Analyze first 100 decisions
                        early_accept_rates = []
                        for game in games:
                            early_people = game['people'][:100]
                            if len(early_people) >= 50:
                                accept_count = sum(1 for p in early_people if p['decision'])
                                rate = accept_count / len(early_people)
                                early_accept_rates.append(rate)
                        
                        if early_accept_rates:
                            avg_early_rate = np.mean(early_accept_rates)
                            print(f"  {game_type} early accept rate: {avg_early_rate:.3f}")
    
    def generate_insights(self):
        """Generate actionable insights from the analysis."""
        print("\n💡 KEY INSIGHTS & RECOMMENDATIONS")
        print("="*45)
        
        # Check if we have enough data
        if len(self.games_data) < 5:
            print("⚠️  Need more data for reliable insights. Run data_collector.py with more games.")
            return
        
        # Insight 1: Frequency accuracy
        print("\n1. 🎯 FREQUENCY ACCURACY:")
        all_diffs = []
        for game in self.games_data:
            total_people = len(game['people'])
            if total_people > 50:  # Only analyze games with reasonable sample size
                for attr, claimed_freq in game['attribute_frequencies'].items():
                    actual_count = sum(1 for p in game['people'] if p['attributes'].get(attr, False))
                    actual_freq = actual_count / total_people
                    diff = abs(actual_freq - claimed_freq)
                    all_diffs.append(diff)
        
        if all_diffs:
            avg_diff = np.mean(all_diffs)
            max_diff = max(all_diffs)
            print(f"   Average frequency difference: {avg_diff:.3f}")
            print(f"   Maximum frequency difference: {max_diff:.3f}")
            if avg_diff > 0.05:
                print("   🚨 SIGNIFICANT: Claimed frequencies may be inaccurate!")
            else:
                print("   ✅ Claimed frequencies appear reasonably accurate")
        
        # Insight 2: Success patterns
        print("\n2. 🏆 SUCCESS PATTERNS:")
        successful_games = [g for g in self.games_data if g['final_status'] == 'completed']
        if successful_games:
            avg_successful_rejections = np.mean([g['final_rejected_count'] for g in successful_games])
            print(f"   Average rejections in successful games: {avg_successful_rejections:.1f}")
            print(f"   Success rate: {len(successful_games)}/{len(self.games_data)} ({100*len(successful_games)/len(self.games_data):.1f}%)")
        
        # Insight 3: Next steps
        print("\n3. 🚀 RECOMMENDED STRATEGY:")
        print("   - Collect more data (target: 20+ games per scenario)")
        print("   - Focus on early-game selectivity patterns")
        print("   - Test if certain attribute combinations are more valuable")
        print("   - Implement adaptive strategy based on empirical frequencies")

def main():
    analyzer = GameDataAnalyzer()
    
    if not analyzer.games_data:
        print("No data to analyze. Run data_collector.py first!")
        return
    
    analyzer.analyze_claimed_vs_actual_frequencies()
    analyzer.analyze_temporal_patterns()
    analyzer.analyze_decision_patterns()
    analyzer.find_winning_patterns()
    analyzer.generate_insights()

if __name__ == "__main__":
    main()