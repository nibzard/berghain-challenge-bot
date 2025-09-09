# ABOUTME: Create simplified elite training data directly from high-performing game logs
# ABOUTME: Focus on strategy patterns from best games without complex decision analysis

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleTrainingExample:
    """Simplified training example from elite games"""
    game_id: str
    original_strategy: str
    state_features: Dict[str, float]
    target_strategy: str
    performance_weight: float
    final_rejections: int

def get_elite_games(max_rejections: int = 850) -> List[Dict]:
    """Get elite games with low rejection counts"""
    logs_path = Path("game_logs")
    elite_games = []
    
    logger.info(f"Scanning for elite games with < {max_rejections} rejections...")
    
    for log_file in logs_path.glob("game_*.json"):
        try:
            with open(log_file, 'r') as f:
                game_data = json.load(f)
            
            rejected_count = game_data.get('rejected_count', float('inf'))
            success = game_data.get('success', False)
            
            if success and rejected_count < max_rejections:
                elite_games.append(game_data)
                
        except Exception as e:
            logger.debug(f"Failed to process {log_file}: {e}")
            continue
    
    logger.info(f"Found {len(elite_games)} elite games")
    
    # Show distribution by strategy
    strategy_counts = defaultdict(list)
    for game in elite_games:
        strategy = game.get('strategy_name', 'unknown')
        rejections = game.get('rejected_count', 0)
        strategy_counts[strategy].append(rejections)
    
    logger.info("Elite games by strategy:")
    for strategy, rejections in strategy_counts.items():
        avg_rejections = sum(rejections) / len(rejections)
        logger.info(f"  {strategy}: {len(rejections)} games, avg {avg_rejections:.1f} rejections")
    
    return elite_games

def extract_game_phases(decisions: List[Dict]) -> List[Tuple[str, Dict]]:
    """Extract key game phases from decisions"""
    phases = []
    total_decisions = len(decisions)
    
    # Early game (first 30% of decisions)
    early_end = int(0.3 * total_decisions)
    if early_end > 10:
        early_decisions = decisions[:early_end]
        early_features = calculate_phase_features(early_decisions, 'early')
        phases.append(('early', early_features))
    
    # Mid game (30-70% of decisions)
    mid_start = int(0.3 * total_decisions)
    mid_end = int(0.7 * total_decisions)
    if mid_end - mid_start > 10:
        mid_decisions = decisions[mid_start:mid_end]
        mid_features = calculate_phase_features(mid_decisions, 'mid')
        phases.append(('mid', mid_features))
    
    # Late game (last 30% of decisions)
    late_start = int(0.7 * total_decisions)
    if total_decisions - late_start > 10:
        late_decisions = decisions[late_start:]
        late_features = calculate_phase_features(late_decisions, 'late')
        phases.append(('late', late_features))
    
    return phases

def calculate_phase_features(decisions: List[Dict], phase: str) -> Dict[str, float]:
    """Calculate features for a game phase"""
    if not decisions:
        return {}
    
    # Count acceptances and rejections
    acceptances = sum(1 for d in decisions if d.get('decision', False))
    rejections = len(decisions) - acceptances
    acceptance_rate = acceptances / len(decisions) if decisions else 0
    
    # Count attribute progress
    young_accepted = sum(1 for d in decisions 
                        if d.get('decision', False) and d.get('attributes', {}).get('young', False))
    well_dressed_accepted = sum(1 for d in decisions 
                               if d.get('decision', False) and d.get('attributes', {}).get('well_dressed', False))
    
    # Calculate efficiency metrics
    multi_attr_accepted = sum(1 for d in decisions 
                             if d.get('decision', False) 
                             and d.get('attributes', {}).get('young', False)
                             and d.get('attributes', {}).get('well_dressed', False))
    
    efficiency = multi_attr_accepted / max(acceptances, 1)
    
    # Game progress features
    avg_person_index = sum(d.get('person_index', 0) for d in decisions) / len(decisions)
    capacity_progress = avg_person_index / 1000.0  # Rough capacity estimate
    
    return {
        'phase': {'early': 0, 'mid': 1, 'late': 2}.get(phase, 0),
        'acceptance_rate': acceptance_rate,
        'young_rate': young_accepted / len(decisions),
        'well_dressed_rate': well_dressed_accepted / len(decisions),
        'multi_attr_efficiency': efficiency,
        'capacity_progress': min(capacity_progress, 1.0),
        'decision_pressure': rejections / len(decisions),
        'selectivity': 1.0 - acceptance_rate,
        'constraint_focus': (young_accepted + well_dressed_accepted) / max(acceptances * 2, 1)
    }

def create_training_examples(elite_games: List[Dict]) -> List[SimpleTrainingExample]:
    """Create training examples from elite games"""
    training_examples = []
    
    logger.info(f"Creating training examples from {len(elite_games)} elite games...")
    
    for game in elite_games:
        game_id = game.get('game_id', 'unknown')
        strategy = game.get('strategy_name', 'unknown')
        rejections = game.get('rejected_count', 0)
        decisions = game.get('decisions', [])
        
        if not decisions:
            continue
        
        # Calculate performance weight (lower rejections = higher weight)
        if rejections <= 780:
            weight = 3.0  # Exceptional performance
        elif rejections <= 820:
            weight = 2.5  # Very good performance
        elif rejections <= 850:
            weight = 2.0  # Good performance
        else:
            weight = 1.5
        
        # Extract phases from the game
        phases = extract_game_phases(decisions)
        
        for phase_name, features in phases:
            # Map strategy names to model vocabulary
            target_strategy = map_strategy_to_vocab(strategy)
            
            example = SimpleTrainingExample(
                game_id=f"{game_id}_{phase_name}",
                original_strategy=strategy,
                state_features=features,
                target_strategy=target_strategy,
                performance_weight=weight,
                final_rejections=rejections
            )
            
            training_examples.append(example)
    
    logger.info(f"Created {len(training_examples)} training examples")
    
    # Add RBCR2 bias for early/mid game
    rbcr2_examples = [ex for ex in training_examples 
                     if 'rbcr2' in ex.original_strategy.lower()
                     and ex.state_features.get('phase', 0) < 2  # early or mid
                     and ex.final_rejections < 830]
    
    # Duplicate best RBCR2 examples
    for example in rbcr2_examples:
        boosted_example = SimpleTrainingExample(
            game_id=f"{example.game_id}_rbcr2_boost",
            original_strategy=example.original_strategy,
            state_features=example.state_features,
            target_strategy="rbcr2",
            performance_weight=example.performance_weight * 2.0,
            final_rejections=example.final_rejections
        )
        training_examples.append(boosted_example)
    
    logger.info(f"Final dataset: {len(training_examples)} examples with RBCR2 bias")
    return training_examples

def map_strategy_to_vocab(strategy_name: str) -> str:
    """Map strategy names to model vocabulary"""
    strategy_mapping = {
        'rbcr2': 'rbcr2',
        'rbcr': 'rbcr',
        'ultra_elite_lstm': 'ultra_elite_lstm',
        'constraint_focused_lstm': 'constraint_focused_lstm',
        'ultimate3h': 'ultimate3h',
        'ultimate3': 'ultimate3',
        'ultimate2': 'ultimate2',
        'perfect': 'perfect',
        'dual_deficit': 'dual_deficit',
        'dualdeficit': 'dual_deficit',
        'apex': 'rbcr2',  # Map apex to rbcr2 since it's not in our vocab
        'rl_lstm_hybrid': 'ultra_elite_lstm',  # Map to similar strategy
        'elite_lstm': 'ultra_elite_lstm',  # Map to similar strategy
        'rarityweighted': 'rbcr2'  # Map to rbcr2
    }
    
    strategy_lower = strategy_name.lower().replace(' ', '_').replace('-', '_')
    return strategy_mapping.get(strategy_lower, 'rbcr2')  # Default to rbcr2

def save_simple_training_data(examples: List[SimpleTrainingExample], 
                             output_path: str = "training_data/simple_elite_training.json"):
    """Save training examples in format expected by trainer"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to expected format
    training_data = []
    
    for example in examples:
        # Create state sequence (simplified)
        state_sequence = [example.state_features]
        
        # Map strategy to index
        strategy_vocab = ['rbcr2', 'ultra_elite_lstm', 'constraint_focused_lstm', 
                         'perfect', 'ultimate3', 'ultimate3h', 'dual_deficit', 'rbcr']
        
        try:
            strategy_target = strategy_vocab.index(example.target_strategy)
        except ValueError:
            strategy_target = 0  # Default to rbcr2
        
        # Create parameter targets (simplified - all zeros for now)
        parameter_target = [0.0, 0.0, 0.0, 0.0]  # Placeholder values
        
        training_example = {
            'game_id': example.game_id,
            'original_strategy': example.original_strategy,
            'game_phase': ['early', 'mid', 'late'][int(example.state_features.get('phase', 0))],
            'state_sequence': state_sequence,
            'strategy_decision': example.target_strategy,
            'performance_weight': example.performance_weight,
            'final_rejections': example.final_rejections,
            'strategy_target': strategy_target,
            'parameter_target': parameter_target
        }
        
        training_data.append(training_example)
    
    # Add metadata
    metadata = {
        "total_examples": len(training_data),
        "weight_distribution": {
            "high_weight": sum(1 for ex in training_data if ex['performance_weight'] > 2.5),
            "medium_weight": sum(1 for ex in training_data if 2.0 <= ex['performance_weight'] <= 2.5),
            "standard_weight": sum(1 for ex in training_data if ex['performance_weight'] < 2.0)
        },
        "strategy_distribution": {},
        "performance_stats": {
            "avg_rejections": sum(ex['final_rejections'] for ex in training_data) / len(training_data),
            "best_rejection_count": min(ex['final_rejections'] for ex in training_data),
            "worst_rejection_count": max(ex['final_rejections'] for ex in training_data)
        }
    }
    
    # Calculate strategy distribution
    strategy_counts = defaultdict(int)
    for ex in training_data:
        strategy_counts[ex['strategy_decision']] += 1
    metadata["strategy_distribution"] = dict(strategy_counts)
    
    output_data = {
        "metadata": metadata,
        "training_examples": training_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved elite training data to {output_path}")
    logger.info(f"Strategy distribution: {dict(strategy_counts)}")
    logger.info(f"Performance range: {metadata['performance_stats']['best_rejection_count']}-{metadata['performance_stats']['worst_rejection_count']} rejections")
    
    return output_path

def main():
    """Create elite training data"""
    print("🎯 Creating Elite Training Data for Performance Improvement")
    print("=" * 60)
    
    # Get elite games (< 850 rejections)
    elite_games = get_elite_games(max_rejections=850)
    
    if not elite_games:
        print("❌ No elite games found!")
        return
    
    # Create training examples
    training_examples = create_training_examples(elite_games)
    
    if not training_examples:
        print("❌ No training examples created!")
        return
    
    # Save training data
    output_path = save_simple_training_data(training_examples)
    
    print(f"\n✅ Elite training data created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Examples: {len(training_examples)}")
    print(f"   Source: {len(elite_games)} elite games")

if __name__ == "__main__":
    main()