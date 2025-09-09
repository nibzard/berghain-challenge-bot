# ABOUTME: Data augmentation pipeline for creating synthetic elite games from existing ultra-elite games
# ABOUTME: Generates additional training data through strategic perturbation and trajectory optimization

import json
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from copy import deepcopy
import uuid

logger = logging.getLogger(__name__)


class EliteGameAugmenter:
    """
    Advanced data augmentation for elite games using strategic perturbations.
    
    Techniques:
    1. Decision flipping with strategic constraints
    2. Trajectory splicing from multiple games  
    3. Noise injection in attribute patterns
    4. Strategy mixing from different elite games
    5. Temporal reordering of non-critical decisions
    """
    
    def __init__(self, preserve_quality_threshold: float = 0.95):
        """
        Args:
            preserve_quality_threshold: Minimum quality to preserve when augmenting
        """
        self.preserve_threshold = preserve_quality_threshold
        self.random = random.Random(42)  # Reproducible augmentation
        
    def load_ultra_elite_games(self, ultra_elite_dir: str) -> List[Dict[str, Any]]:
        """Load ultra-elite games for augmentation."""
        games = []
        for json_file in Path(ultra_elite_dir).glob("elite_*.json"):
            if 'stats' not in json_file.name:
                try:
                    with open(json_file, 'r') as f:
                        game_data = json.load(f)
                    games.append(game_data)
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(games)} ultra-elite games for augmentation")
        return games
    
    def analyze_game_patterns(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in a game to guide augmentation."""
        decisions = game['decisions']
        
        # Decision timing patterns
        accept_positions = [i for i, d in enumerate(decisions) if d['accepted']]
        reject_positions = [i for i, d in enumerate(decisions) if not d['accepted']]
        
        # Attribute patterns
        young_accepts = []
        well_dressed_accepts = []
        dual_accepts = []
        
        for i, decision in enumerate(decisions):
            if decision['accepted']:
                attrs = decision['person']['attributes']
                if attrs.get('young', False):
                    young_accepts.append(i)
                if attrs.get('well_dressed', False):
                    well_dressed_accepts.append(i)
                if attrs.get('young', False) and attrs.get('well_dressed', False):
                    dual_accepts.append(i)
        
        # Strategy phases (early, mid, late game)
        total_decisions = len(decisions)
        early_phase = decisions[:total_decisions//3]
        mid_phase = decisions[total_decisions//3:2*total_decisions//3]
        late_phase = decisions[2*total_decisions//3:]
        
        early_accept_rate = np.mean([d['accepted'] for d in early_phase])
        mid_accept_rate = np.mean([d['accepted'] for d in mid_phase])
        late_accept_rate = np.mean([d['accepted'] for d in late_phase])
        
        return {
            'accept_positions': accept_positions,
            'reject_positions': reject_positions,
            'young_accepts': young_accepts,
            'well_dressed_accepts': well_dressed_accepts,
            'dual_accepts': dual_accepts,
            'phase_accept_rates': [early_accept_rate, mid_accept_rate, late_accept_rate],
            'total_decisions': total_decisions,
            'admitted_count': game['admitted_count'],
            'rejected_count': game['rejected_count']
        }
    
    def strategic_decision_flip(self, game: Dict[str, Any], flip_probability: float = 0.05) -> Dict[str, Any]:
        """
        Strategically flip some decisions while maintaining game constraints.
        Only flips decisions that don't violate constraints.
        """
        augmented_game = deepcopy(game)
        decisions = augmented_game['decisions']
        patterns = self.analyze_game_patterns(game)
        
        # Track current state
        current_young = 0
        current_well_dressed = 0
        current_admitted = 0
        
        # Constraints
        young_target = 600
        well_dressed_target = 600
        max_capacity = 1000
        
        flipped_count = 0
        
        for i, decision in enumerate(decisions):
            attrs = decision['person']['attributes']
            has_young = attrs.get('young', False)
            has_well_dressed = attrs.get('well_dressed', False)
            
            # Should we attempt to flip this decision?
            if self.random.random() < flip_probability:
                
                if decision['accepted']:
                    # Try to flip accept -> reject
                    # Only if we can still meet constraints later
                    remaining_decisions = len(decisions) - i - 1
                    
                    young_deficit_after = max(young_target - (current_young - (1 if has_young else 0)), 0)
                    well_dressed_deficit_after = max(well_dressed_target - (current_well_dressed - (1 if has_well_dressed else 0)), 0)
                    
                    # Estimate if we can still meet constraints
                    expected_young_remaining = remaining_decisions * 0.323  # Typical frequency
                    expected_well_dressed_remaining = remaining_decisions * 0.323
                    
                    if (young_deficit_after <= expected_young_remaining and 
                        well_dressed_deficit_after <= expected_well_dressed_remaining):
                        
                        decision['accepted'] = False
                        decision['reasoning'] = f"augmented_reject_{decision['reasoning']}"
                        current_admitted -= 1
                        if has_young:
                            current_young -= 1
                        if has_well_dressed:
                            current_well_dressed -= 1
                        flipped_count += 1
                
                else:
                    # Try to flip reject -> accept
                    # Only if we have capacity and it helps with constraints
                    if current_admitted < max_capacity:
                        young_needed = current_young < young_target
                        well_dressed_needed = current_well_dressed < well_dressed_target
                        
                        # Flip if person has needed attributes
                        if (young_needed and has_young) or (well_dressed_needed and has_well_dressed):
                            decision['accepted'] = True
                            decision['reasoning'] = f"augmented_accept_{decision['reasoning']}"
                            current_admitted += 1
                            if has_young:
                                current_young += 1
                            if has_well_dressed:
                                current_well_dressed += 1
                            flipped_count += 1
            
            # Update running totals
            if decision['accepted']:
                current_admitted += (1 if i == 0 or not decisions[i-1].get('_counted', False) else 0)
                decision['_counted'] = True
        
        # Update game metadata
        augmented_game['admitted_count'] = current_admitted
        augmented_game['rejected_count'] = len(decisions) - current_admitted
        augmented_game['game_id'] = str(uuid.uuid4())
        
        # Clean up temporary markers
        for decision in decisions:
            if '_counted' in decision:
                del decision['_counted']
        
        logger.debug(f"Flipped {flipped_count} decisions in strategic flip augmentation")
        return augmented_game
    
    def trajectory_splice(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create new game by splicing trajectories from multiple elite games.
        Takes early decisions from one game, middle from another, etc.
        """
        if len(games) < 2:
            return deepcopy(games[0])
        
        # Select 2-3 games to splice
        selected_games = self.random.sample(games, min(3, len(games)))
        base_game = deepcopy(selected_games[0])
        
        # Determine splice points
        total_decisions = len(base_game['decisions'])
        splice_point_1 = total_decisions // 3
        splice_point_2 = 2 * total_decisions // 3
        
        # Replace middle section with decisions from another game
        if len(selected_games) > 1:
            donor_game = selected_games[1]
            donor_decisions = donor_game['decisions']
            
            # Find corresponding section in donor game
            donor_total = len(donor_decisions)
            donor_start = donor_total // 3
            donor_end = 2 * donor_total // 3
            donor_section = donor_decisions[donor_start:donor_end]
            
            # Adjust section length to match target
            target_length = splice_point_2 - splice_point_1
            if len(donor_section) > target_length:
                donor_section = donor_section[:target_length]
            elif len(donor_section) < target_length:
                # Repeat some decisions to fill
                while len(donor_section) < target_length:
                    donor_section.append(self.random.choice(donor_section))
            
            # Update reasoning to show splicing
            for decision in donor_section:
                decision['reasoning'] = f"spliced_{decision['reasoning']}"
            
            # Replace the section
            base_game['decisions'][splice_point_1:splice_point_2] = donor_section
        
        # Update metadata
        base_game['game_id'] = str(uuid.uuid4())
        
        # Recalculate game statistics
        admitted_count = sum(1 for d in base_game['decisions'] if d['accepted'])
        base_game['admitted_count'] = admitted_count
        base_game['rejected_count'] = len(base_game['decisions']) - admitted_count
        
        return base_game
    
    def noise_injection(self, game: Dict[str, Any], noise_level: float = 0.02) -> Dict[str, Any]:
        """
        Inject controlled noise by slightly modifying non-critical decisions.
        """
        augmented_game = deepcopy(game)
        decisions = augmented_game['decisions']
        patterns = self.analyze_game_patterns(game)
        
        modified_count = 0
        
        for i, decision in enumerate(decisions):
            # Only modify decisions with some probability
            if self.random.random() < noise_level:
                
                # Identify if this is a "critical" decision (has needed attributes)
                attrs = decision['person']['attributes']
                has_needed_attrs = attrs.get('young', False) or attrs.get('well_dressed', False)
                
                # Be more careful with critical decisions
                modification_prob = 0.3 if has_needed_attrs else 0.7
                
                if self.random.random() < modification_prob:
                    # Flip the decision
                    original_decision = decision['accepted']
                    decision['accepted'] = not original_decision
                    decision['reasoning'] = f"noised_{decision['reasoning']}"
                    modified_count += 1
        
        # Update game metadata
        admitted_count = sum(1 for d in decisions if d['accepted'])
        augmented_game['admitted_count'] = admitted_count
        augmented_game['rejected_count'] = len(decisions) - admitted_count
        augmented_game['game_id'] = str(uuid.uuid4())
        
        logger.debug(f"Applied noise to {modified_count} decisions")
        return augmented_game
    
    def strategy_mixing(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mix strategies from different games by combining their decision patterns.
        """
        if len(games) < 2:
            return deepcopy(games[0])
        
        # Select base game and strategy donor
        base_game = deepcopy(self.random.choice(games))
        strategy_donors = [g for g in games if g['game_id'] != base_game['game_id']]
        donor_game = self.random.choice(strategy_donors)
        
        # Analyze both games' strategies
        base_patterns = self.analyze_game_patterns(base_game)
        donor_patterns = self.analyze_game_patterns(donor_game)
        
        # Mix acceptance rates by phase
        mixed_rates = []
        for base_rate, donor_rate in zip(base_patterns['phase_accept_rates'], donor_patterns['phase_accept_rates']):
            # Weighted average with some randomness
            weight = self.random.uniform(0.3, 0.7)
            mixed_rate = weight * base_rate + (1 - weight) * donor_rate
            mixed_rates.append(mixed_rate)
        
        # Apply mixed strategy to base game
        decisions = base_game['decisions']
        total_decisions = len(decisions)
        
        # Divide into phases
        phase_boundaries = [total_decisions//3, 2*total_decisions//3, total_decisions]
        current_phase = 0
        
        for i, decision in enumerate(decisions):
            # Update phase
            while current_phase < len(phase_boundaries) and i >= phase_boundaries[current_phase]:
                current_phase += 1
            
            target_rate = mixed_rates[min(current_phase, len(mixed_rates)-1)]
            
            # Probabilistically adjust decision based on mixed strategy
            if self.random.random() < 0.1:  # 10% chance to apply strategy mixing
                attrs = decision['person']['attributes']
                has_needed = attrs.get('young', False) or attrs.get('well_dressed', False)
                
                # Apply strategy-based decision
                if has_needed and self.random.random() < target_rate:
                    if not decision['accepted']:
                        decision['accepted'] = True
                        decision['reasoning'] = f"mixed_strategy_accept_{decision['reasoning']}"
                elif not has_needed and self.random.random() < (1 - target_rate):
                    if decision['accepted']:
                        decision['accepted'] = False
                        decision['reasoning'] = f"mixed_strategy_reject_{decision['reasoning']}"
        
        # Update metadata
        admitted_count = sum(1 for d in decisions if d['accepted'])
        base_game['admitted_count'] = admitted_count
        base_game['rejected_count'] = len(decisions) - admitted_count
        base_game['game_id'] = str(uuid.uuid4())
        
        return base_game
    
    def generate_augmented_games(
        self, 
        ultra_elite_games: List[Dict[str, Any]], 
        augmentation_factor: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate augmented games using all techniques.
        
        Args:
            ultra_elite_games: List of original ultra-elite games
            augmentation_factor: How many augmented games per original game
            
        Returns:
            List of augmented games
        """
        augmented_games = []
        
        for original_game in ultra_elite_games:
            logger.info(f"Augmenting game {original_game['game_id'][:8]}...")
            
            for aug_idx in range(augmentation_factor):
                # Choose augmentation technique
                technique = self.random.choice([
                    'strategic_flip', 'trajectory_splice', 'noise_injection', 'strategy_mixing'
                ])
                
                if technique == 'strategic_flip':
                    aug_game = self.strategic_decision_flip(original_game, flip_probability=0.03)
                elif technique == 'trajectory_splice':
                    aug_game = self.trajectory_splice(ultra_elite_games)
                elif technique == 'noise_injection':
                    aug_game = self.noise_injection(original_game, noise_level=0.02)
                elif technique == 'strategy_mixing':
                    aug_game = self.strategy_mixing(ultra_elite_games)
                
                # Add augmentation metadata
                aug_game['augmented'] = True
                aug_game['augmentation_technique'] = technique
                aug_game['original_game_id'] = original_game['game_id']
                
                augmented_games.append(aug_game)
        
        logger.info(f"Generated {len(augmented_games)} augmented games from {len(ultra_elite_games)} originals")
        return augmented_games
    
    def save_augmented_games(
        self, 
        augmented_games: List[Dict[str, Any]], 
        output_dir: str = "augmented_elite_games"
    ) -> None:
        """Save augmented games to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, game in enumerate(augmented_games):
            technique = game.get('augmentation_technique', 'unknown')
            rejections = game.get('rejected_count', 0)
            filename = f"augmented_{technique}_{rejections}rej_{i:03d}_{game['game_id'][:8]}.json"
            
            with open(output_path / filename, 'w') as f:
                json.dump(game, f, indent=2)
        
        # Save summary statistics
        stats = {
            'total_augmented_games': len(augmented_games),
            'techniques_used': list(set(g.get('augmentation_technique', 'unknown') for g in augmented_games)),
            'avg_rejections': np.mean([g.get('rejected_count', 0) for g in augmented_games]),
            'rejection_range': [
                min(g.get('rejected_count', 0) for g in augmented_games),
                max(g.get('rejected_count', 0) for g in augmented_games)
            ]
        }
        
        with open(output_path / "augmentation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved {len(augmented_games)} augmented games to {output_dir}/")


def create_augmented_dataset(
    ultra_elite_dir: str = "ultra_elite_games",
    output_dir: str = "augmented_elite_games", 
    augmentation_factor: int = 3
) -> int:
    """
    Main function to create augmented dataset.
    
    Returns:
        Number of augmented games created
    """
    augmenter = EliteGameAugmenter()
    
    # Load ultra-elite games
    ultra_elite_games = augmenter.load_ultra_elite_games(ultra_elite_dir)
    
    if not ultra_elite_games:
        raise ValueError(f"No ultra-elite games found in {ultra_elite_dir}")
    
    # Generate augmented games
    augmented_games = augmenter.generate_augmented_games(
        ultra_elite_games, 
        augmentation_factor=augmentation_factor
    )
    
    # Save augmented games
    augmenter.save_augmented_games(augmented_games, output_dir)
    
    return len(augmented_games)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create augmented elite game dataset")
    parser.add_argument('--ultra-elite-dir', default='ultra_elite_games', help='Ultra-elite games directory')
    parser.add_argument('--output-dir', default='augmented_elite_games', help='Output directory for augmented games')
    parser.add_argument('--augmentation-factor', type=int, default=3, help='Augmentation factor per original game')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    num_augmented = create_augmented_dataset(
        args.ultra_elite_dir,
        args.output_dir, 
        args.augmentation_factor
    )
    
    print(f"\n✨ Successfully created {num_augmented} augmented elite games!")