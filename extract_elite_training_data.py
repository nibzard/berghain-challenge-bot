# ABOUTME: Extract training data from only the highest-performing games for improved transformer training
# ABOUTME: Filters games to < 850 rejections and weights examples by performance

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from berghain.analysis.strategic_decision_analyzer import StrategicDecisionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EliteTrainingExample:
    """Training example with performance weighting"""
    game_id: str
    original_strategy: str
    game_phase: str
    state_sequence: List[Dict[str, Any]]
    strategy_decision: str
    performance_weight: float  # Higher for better performing games
    final_rejections: int

def filter_elite_games(game_logs_dir: str = "game_logs", max_rejections: int = 850) -> Dict[str, Dict]:
    """Find games with exceptional performance (low rejection count)"""
    logs_path = Path(game_logs_dir)
    elite_games = {}
    
    logger.info(f"Scanning for elite games with < {max_rejections} rejections...")
    
    # Look for completed game logs
    for log_file in logs_path.glob("game_*.json"):
        try:
            with open(log_file, 'r') as f:
                game_data = json.load(f)
            
            # Extract final game state (different format)
            rejected_count = game_data.get('rejected_count', float('inf'))
            game_status = game_data.get('status', '')
            success = game_data.get('success', False)
            
            # Only include successful games with low rejections
            if success and rejected_count < max_rejections:
                game_id = game_data.get('game_id', log_file.stem)
                strategy = game_data.get('strategy_name', 'unknown')
                
                elite_games[game_id] = {
                    'file_path': str(log_file),
                    'strategy': strategy,
                    'rejections': rejected_count,
                    'game_data': game_data
                }
                
        except Exception as e:
            logger.debug(f"Failed to process {log_file}: {e}")
            continue
    
    logger.info(f"Found {len(elite_games)} elite games")
    
    # Show distribution by strategy
    strategy_counts = defaultdict(list)
    for game_id, data in elite_games.items():
        strategy_counts[data['strategy']].append(data['rejections'])
    
    logger.info("Elite games by strategy:")
    for strategy, rejections in strategy_counts.items():
        avg_rejections = sum(rejections) / len(rejections)
        logger.info(f"  {strategy}: {len(rejections)} games, avg {avg_rejections:.1f} rejections")
    
    return elite_games

def calculate_performance_weight(rejections: int, best_rejection_count: int = 750) -> float:
    """Calculate performance weight - lower rejections get higher weight"""
    if rejections <= best_rejection_count:
        return 3.0  # Maximum weight for exceptional performance
    elif rejections <= 800:
        return 2.0  # High weight for very good performance
    elif rejections <= 850:
        return 1.5  # Medium weight for good performance
    else:
        return 1.0  # Standard weight

def extract_elite_training_examples(elite_games: Dict[str, Dict]) -> List[EliteTrainingExample]:
    """Extract training examples from elite games with performance weighting"""
    analyzer = StrategicDecisionAnalyzer()
    training_examples = []
    
    logger.info("Extracting training examples from elite games...")
    
    for game_id, game_info in elite_games.items():
        game_data = game_info['game_data']
        rejections = game_info['rejections']
        strategy = game_info['strategy']
        
        try:
            # Extract strategic decisions from this game
            decisions = analyzer.extract_strategic_decisions([game_data])
            
            # Convert to training examples
            for decision in decisions:
                # Build state sequence leading to this decision
                state_sequence = analyzer._build_state_sequence_for_decision(game_data, decision.person_index)
                
                # Determine game phase
                capacity_ratio = decision.person_index / 1000.0  # Rough approximation
                if capacity_ratio < 0.3:
                    game_phase = 'early'
                elif capacity_ratio < 0.7:
                    game_phase = 'mid'
                else:
                    game_phase = 'late'
                
                # Calculate performance weight
                performance_weight = calculate_performance_weight(rejections)
                
                # Create training example
                example = EliteTrainingExample(
                    game_id=game_id,
                    original_strategy=strategy,
                    game_phase=game_phase,
                    state_sequence=state_sequence,
                    strategy_decision=strategy,  # Keep the winning strategy
                    performance_weight=performance_weight,
                    final_rejections=rejections
                )
                
                training_examples.append(example)
                
        except Exception as e:
            logger.warning(f"Failed to extract decisions from game {game_id}: {e}")
            continue
    
    logger.info(f"Extracted {len(training_examples)} elite training examples")
    
    # Show weight distribution
    weights = [ex.performance_weight for ex in training_examples]
    avg_weight = sum(weights) / len(weights) if weights else 0
    high_weight_count = sum(1 for w in weights if w > 2.0)
    
    logger.info(f"Performance weights: avg={avg_weight:.2f}, high-weight={high_weight_count}")
    
    return training_examples

def augment_with_rbcr2_bias(training_examples: List[EliteTrainingExample]) -> List[EliteTrainingExample]:
    """Add bias towards RBCR2 for early/mid game phases"""
    augmented_examples = training_examples.copy()
    
    # Find RBCR2 examples in early/mid game
    rbcr2_examples = [ex for ex in training_examples 
                     if 'rbcr2' in ex.original_strategy.lower() 
                     and ex.game_phase in ['early', 'mid']
                     and ex.final_rejections < 900]
    
    logger.info(f"Found {len(rbcr2_examples)} RBCR2 early/mid game examples to boost")
    
    # Duplicate best RBCR2 examples with higher weight
    for example in rbcr2_examples:
        if example.final_rejections < 850:
            # Create boosted copy
            boosted_example = EliteTrainingExample(
                game_id=f"{example.game_id}_rbcr2_boost",
                original_strategy=example.original_strategy,
                game_phase=example.game_phase,
                state_sequence=example.state_sequence,
                strategy_decision="rbcr2",  # Explicitly favor RBCR2
                performance_weight=example.performance_weight * 2.0,  # Double the weight
                final_rejections=example.final_rejections
            )
            augmented_examples.append(boosted_example)
    
    logger.info(f"Augmented dataset: {len(training_examples)} → {len(augmented_examples)} examples")
    return augmented_examples

def save_elite_training_data(training_examples: List[EliteTrainingExample], 
                           output_path: str = "training_data/elite_strategy_controller_training.json"):
    """Save elite training data to file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    training_data = []
    for example in training_examples:
        example_dict = asdict(example)
        training_data.append(example_dict)
    
    # Add metadata
    metadata = {
        "total_examples": len(training_data),
        "weight_distribution": {
            "high_weight": sum(1 for ex in training_data if ex['performance_weight'] > 2.0),
            "medium_weight": sum(1 for ex in training_data if 1.5 <= ex['performance_weight'] <= 2.0),
            "standard_weight": sum(1 for ex in training_data if ex['performance_weight'] < 1.5)
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
        strategy_counts[ex['original_strategy']] += 1
    metadata["strategy_distribution"] = dict(strategy_counts)
    
    # Save data
    output_data = {
        "metadata": metadata,
        "training_examples": training_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved elite training data to {output_path}")
    logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    return output_path

def main():
    """Extract elite training data for improved transformer performance"""
    print("🔍 Extracting Elite Training Data for Performance Improvement")
    print("=" * 60)
    
    # Step 1: Find elite games
    elite_games = filter_elite_games(max_rejections=850)
    
    if not elite_games:
        print("❌ No elite games found! Consider raising the rejection threshold.")
        return
    
    # Step 2: Extract training examples
    training_examples = extract_elite_training_examples(elite_games)
    
    if not training_examples:
        print("❌ No training examples extracted!")
        return
    
    # Step 3: Add RBCR2 bias
    augmented_examples = augment_with_rbcr2_bias(training_examples)
    
    # Step 4: Save elite training data
    output_path = save_elite_training_data(augmented_examples)
    
    print(f"\n✅ Elite training data extraction complete!")
    print(f"   Output: {output_path}")
    print(f"   Examples: {len(augmented_examples)}")
    print(f"   Ready for improved transformer training!")

if __name__ == "__main__":
    main()