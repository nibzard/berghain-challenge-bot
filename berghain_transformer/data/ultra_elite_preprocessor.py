"""
ABOUTME: Ultra-elite game data preprocessor for dual-head transformer training
ABOUTME: Extracts optimal decision patterns with specialized constraint and efficiency labeling
"""

import json
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, NamedTuple
from collections import defaultdict, deque
import torch
from dataclasses import dataclass
import pickle
import logging

logger = logging.getLogger(__name__)

class DecisionSequence(NamedTuple):
    """Structured decision sequence for training."""
    states: np.ndarray        # Game state features
    actions: np.ndarray       # Decision actions (0/1)
    constraint_labels: np.ndarray  # Constraint-focused labels  
    efficiency_labels: np.ndarray  # Efficiency-focused labels
    rewards: np.ndarray       # Immediate rewards
    next_states: np.ndarray   # Next game states
    metadata: Dict[str, Any]  # Additional info

@dataclass
class GameFeatures:
    """Extracted features from a game state."""
    # Constraint satisfaction features
    young_current: int
    young_needed: int
    well_dressed_current: int
    well_dressed_needed: int
    constraint_pressure: float  # How urgent constraint satisfaction is
    
    # Efficiency features
    total_admitted: int
    total_rejected: int
    rejection_rate: float
    efficiency_trend: float     # Recent efficiency trend
    
    # Game progress features
    game_progress: float        # 0 to 1 based on admits
    time_pressure: float        # Urgency based on remaining capacity
    
    # Person-specific features
    person_young: bool
    person_well_dressed: bool
    person_value: float         # How valuable this person is
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.young_current,
            self.young_needed,
            self.well_dressed_current,
            self.well_dressed_needed,
            self.constraint_pressure,
            self.total_admitted,
            self.total_rejected,
            self.rejection_rate,
            self.efficiency_trend,
            self.game_progress,
            self.time_pressure,
            float(self.person_young),
            float(self.person_well_dressed),
            self.person_value
        ])

class UltraElitePreprocessor:
    """Specialized preprocessor for ultra-elite games with dual-head training data."""
    
    def __init__(
        self,
        elite_data_dir: str = "../ultra_elite_training",
        sequence_length: int = 50,
        overlap_ratio: float = 0.8,
        constraint_focus_threshold: float = 0.7,
        efficiency_focus_threshold: float = 0.3
    ):
        self.elite_data_dir = Path(elite_data_dir)
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio
        self.constraint_focus_threshold = constraint_focus_threshold
        self.efficiency_focus_threshold = efficiency_focus_threshold
        
        # Feature dimension (should match GameFeatures.to_array())
        self.feature_dim = 14
        
        # Training sequences
        self.sequences = []
        
    def process_all_tiers(self) -> List[DecisionSequence]:
        """Process all elite games and extract training sequences."""
        logger.info("🎯 Processing ultra-elite games for dual-head training...")
        
        if not self.elite_data_dir.exists():
            logger.error(f"Elite data directory not found: {self.elite_data_dir}")
            return []
        
        # Process all games in the directory
        all_sequences = self._process_elite_games()
        
        logger.info(f"✅ Total sequences extracted: {len(all_sequences)}")
        self.sequences = all_sequences
        return all_sequences
    
    def _process_elite_games(self) -> List[DecisionSequence]:
        """Process all elite games in the directory."""
        sequences = []
        
        game_files = list(self.elite_data_dir.glob("elite_*.json"))
        logger.info(f"📁 Found {len(game_files)} elite game files")
        
        for game_file in game_files:
            try:
                # Determine tier based on rejection count in filename
                rejection_match = re.search(r'(\d+)rej_', game_file.name)
                rejections = int(rejection_match.group(1)) if rejection_match else 999
                
                if rejections <= 780:
                    tier_name = 'ultra_elite'
                elif rejections <= 820:
                    tier_name = 'elite'
                else:
                    tier_name = 'good'
                
                game_sequences = self._extract_sequences_from_game(game_file, tier_name)
                sequences.extend(game_sequences)
                
                if len(game_sequences) > 0:
                    logger.debug(f"📊 {game_file.name}: {len(game_sequences)} sequences ({tier_name})")
                    
            except Exception as e:
                logger.warning(f"Error processing {game_file}: {e}")
        
        return sequences
    
    def _process_tier(self, tier_dir: Path, tier_name: str) -> List[DecisionSequence]:
        """Process a specific tier directory."""
        sequences = []
        
        game_files = list(tier_dir.glob("game_*.json"))
        for game_file in game_files:
            try:
                game_sequences = self._extract_sequences_from_game(game_file, tier_name)
                sequences.extend(game_sequences)
            except Exception as e:
                logger.warning(f"Error processing {game_file}: {e}")
        
        return sequences
    
    def _extract_sequences_from_game(self, game_file: Path, tier_name: str) -> List[DecisionSequence]:
        """Extract training sequences from a single game."""
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        decisions = game_data.get('decisions', [])
        if len(decisions) < self.sequence_length:
            return []
        
        # Extract game metadata
        strategy = game_data.get('strategy', 'unknown')
        final_rejections = game_data.get('rejected_count', 0)
        final_admitted = game_data.get('admitted_count', 0)
        
        sequences = []
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for start_idx in range(0, len(decisions) - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            
            sequence = self._create_training_sequence(
                decisions[start_idx:end_idx],
                start_idx,
                game_data,
                tier_name,
                strategy,
                final_rejections
            )
            
            if sequence:
                sequences.append(sequence)
        
        return sequences
    
    def _create_training_sequence(
        self,
        decisions: List[Dict],
        start_idx: int,
        game_data: Dict,
        tier_name: str,
        strategy: str,
        final_rejections: int
    ) -> Optional[DecisionSequence]:
        """Create a training sequence from a window of decisions."""
        
        # Initialize tracking variables
        young_count = 0
        well_dressed_count = 0
        total_admitted = start_idx  # Approximate starting point
        total_rejected = 0
        
        # Extract features and labels for each decision
        states = []
        actions = []
        constraint_labels = []
        efficiency_labels = []
        rewards = []
        next_states = []
        
        for i, decision in enumerate(decisions):
            # Extract person attributes
            attributes = decision.get('attributes', {})
            person_young = attributes.get('young', False)
            person_well_dressed = attributes.get('well_dressed', False)
            decision_made = decision.get('decision', False)
            reasoning = decision.get('reasoning', '')
            
            # Calculate current game state features
            game_features = self._calculate_game_features(
                young_count, well_dressed_count,
                total_admitted, total_rejected,
                person_young, person_well_dressed,
                i / len(decisions)  # progress within sequence
            )
            
            states.append(game_features.to_array())
            actions.append(1 if decision_made else 0)
            
            # Generate specialized labels for dual heads
            constraint_label, efficiency_label = self._generate_dual_labels(
                game_features, person_young, person_well_dressed,
                reasoning, decision_made, tier_name
            )
            
            constraint_labels.append(constraint_label)
            efficiency_labels.append(efficiency_label)
            
            # Calculate reward (efficiency-based)
            reward = self._calculate_reward(game_features, decision_made)
            rewards.append(reward)
            
            # Update counters
            if decision_made:
                total_admitted += 1
                if person_young:
                    young_count += 1
                if person_well_dressed:
                    well_dressed_count += 1
            else:
                total_rejected += 1
            
            # Calculate next state features (for next iteration)
            if i < len(decisions) - 1:
                next_features = self._calculate_game_features(
                    young_count, well_dressed_count,
                    total_admitted, total_rejected,
                    person_young, person_well_dressed,
                    (i + 1) / len(decisions)
                )
                next_states.append(next_features.to_array())
            else:
                # For last decision, use current state
                next_states.append(game_features.to_array())
        
        # Convert to numpy arrays
        states_array = np.array(states, dtype=np.float32)
        actions_array = np.array(actions, dtype=np.int64)
        constraint_labels_array = np.array(constraint_labels, dtype=np.int64)
        efficiency_labels_array = np.array(efficiency_labels, dtype=np.int64)
        rewards_array = np.array(rewards, dtype=np.float32)
        next_states_array = np.array(next_states, dtype=np.float32)
        
        metadata = {
            'tier': tier_name,
            'strategy': strategy,
            'final_rejections': final_rejections,
            'sequence_start_idx': start_idx,
            'game_file': str(game_data.get('game_id', 'unknown'))
        }
        
        return DecisionSequence(
            states=states_array,
            actions=actions_array,
            constraint_labels=constraint_labels_array,
            efficiency_labels=efficiency_labels_array,
            rewards=rewards_array,
            next_states=next_states_array,
            metadata=metadata
        )
    
    def _calculate_game_features(
        self,
        young_count: int,
        well_dressed_count: int,
        total_admitted: int,
        total_rejected: int,
        person_young: bool,
        person_well_dressed: bool,
        sequence_progress: float
    ) -> GameFeatures:
        """Calculate comprehensive game state features."""
        
        # Constraint requirements (scenario 1: 600 young + 600 well_dressed)
        young_needed = max(0, 600 - young_count)
        well_dressed_needed = max(0, 600 - well_dressed_count)
        
        # Calculate constraint pressure (how urgent satisfaction is)
        remaining_capacity = max(1, 1000 - total_admitted)
        constraint_pressure = (young_needed + well_dressed_needed) / remaining_capacity
        
        # Efficiency metrics
        total_decisions = total_admitted + total_rejected
        rejection_rate = total_rejected / max(1, total_decisions)
        
        # Calculate efficiency trend (simplified)
        efficiency_trend = 1.0 - rejection_rate  # Higher when rejecting less
        
        # Game progress and time pressure
        game_progress = total_admitted / 1000.0  # Progress toward 1000 admits
        time_pressure = min(1.0, total_decisions / 20000.0)  # Approaching rejection limit
        
        # Person value calculation
        person_value = 0.0
        if person_young and young_needed > 0:
            person_value += 0.5
        if person_well_dressed and well_dressed_needed > 0:
            person_value += 0.5
        if person_young and person_well_dressed:
            person_value += 0.2  # Bonus for dual attributes
        
        return GameFeatures(
            young_current=young_count,
            young_needed=young_needed,
            well_dressed_current=well_dressed_count,
            well_dressed_needed=well_dressed_needed,
            constraint_pressure=constraint_pressure,
            total_admitted=total_admitted,
            total_rejected=total_rejected,
            rejection_rate=rejection_rate,
            efficiency_trend=efficiency_trend,
            game_progress=game_progress,
            time_pressure=time_pressure,
            person_young=person_young,
            person_well_dressed=person_well_dressed,
            person_value=person_value
        )
    
    def _generate_dual_labels(
        self,
        features: GameFeatures,
        person_young: bool,
        person_well_dressed: bool,
        reasoning: str,
        actual_decision: bool,
        tier_name: str
    ) -> Tuple[int, int]:
        """Generate specialized labels for constraint and efficiency heads."""
        
        # Constraint head label (focused on meeting requirements)
        constraint_label = 0  # Default: reject
        
        # Strong preference to admit if person helps with constraints
        if features.young_needed > 0 and person_young:
            constraint_label = 1
        elif features.well_dressed_needed > 0 and person_well_dressed:
            constraint_label = 1
        elif person_young and person_well_dressed:
            constraint_label = 1  # Always valuable
        elif features.constraint_pressure < self.constraint_focus_threshold:
            # Constraints not critical, can be more selective
            constraint_label = 1 if features.person_value > 0.3 else 0
        
        # Efficiency head label (focused on minimizing rejections)
        efficiency_label = 0  # Default: reject
        
        # Efficiency-focused decisions
        if features.person_value > 0.6:  # High-value person
            efficiency_label = 1
        elif features.rejection_rate > self.efficiency_focus_threshold:
            # Currently rejecting too much, be more accepting
            if features.person_value > 0.2:
                efficiency_label = 1
        elif tier_name == 'ultra_elite':
            # For ultra-elite games, trust the actual decision more
            efficiency_label = 1 if actual_decision else 0
        
        return constraint_label, efficiency_label
    
    def _calculate_reward(self, features: GameFeatures, decision_made: bool) -> float:
        """Calculate immediate reward for the decision."""
        
        if decision_made:
            # Reward for admitting
            reward = features.person_value  # Base reward for person value
            
            # Bonus for constraint satisfaction
            if features.young_needed > 0 and features.person_young:
                reward += 0.3
            if features.well_dressed_needed > 0 and features.person_well_dressed:
                reward += 0.3
                
            # Penalty for over-admitting when constraints are met
            if features.young_needed == 0 and features.well_dressed_needed == 0:
                reward -= 0.1
                
        else:
            # Reward for rejecting
            reward = 0.1  # Small positive reward for selectivity
            
            # Penalty for rejecting valuable people
            if features.person_value > 0.5:
                reward -= features.person_value
            
            # Penalty for rejecting when constraints need attention
            if features.constraint_pressure > 0.8:
                if features.person_young and features.young_needed > 0:
                    reward -= 0.5
                if features.person_well_dressed and features.well_dressed_needed > 0:
                    reward -= 0.5
        
        return np.clip(reward, -1.0, 1.0)
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed sequences to disk."""
        data = {
            'sequences': self.sequences,
            'feature_dim': self.feature_dim,
            'sequence_length': self.sequence_length,
            'total_sequences': len(self.sequences),
            'metadata': {
                'constraint_focus_threshold': self.constraint_focus_threshold,
                'efficiency_focus_threshold': self.efficiency_focus_threshold,
                'tiers_processed': ['ultra_elite', 'elite', 'good']
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"💾 Saved {len(self.sequences)} sequences to {output_path}")
    
    def create_pytorch_dataset(self) -> 'UltraEliteDataset':
        """Create PyTorch dataset from processed sequences."""
        return UltraEliteDataset(self.sequences)


class UltraEliteDataset(torch.utils.data.Dataset):
    """PyTorch dataset for dual-head transformer training."""
    
    def __init__(self, sequences: List[DecisionSequence]):
        self.sequences = sequences
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        return {
            'states': torch.FloatTensor(seq.states),
            'actions': torch.LongTensor(seq.actions),
            'constraint_labels': torch.LongTensor(seq.constraint_labels),
            'efficiency_labels': torch.LongTensor(seq.efficiency_labels),
            'rewards': torch.FloatTensor(seq.rewards),
            'next_states': torch.FloatTensor(seq.next_states),
            'metadata': seq.metadata
        }


def main():
    """Main processing function."""
    preprocessor = UltraElitePreprocessor()
    sequences = preprocessor.process_all_tiers()
    
    if sequences:
        preprocessor.save_processed_data('ultra_elite_training_data.pkl')
        
        print(f"\n🎯 ULTRA-ELITE PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"📊 Total sequences: {len(sequences)}")
        print(f"🎮 Feature dimension: {preprocessor.feature_dim}")
        print(f"📏 Sequence length: {preprocessor.sequence_length}")
        
        # Show tier distribution
        tier_counts = defaultdict(int)
        for seq in sequences:
            tier_counts[seq.metadata['tier']] += 1
        
        print(f"\n📈 TIER DISTRIBUTION:")
        for tier, count in tier_counts.items():
            print(f"  {tier:12s}: {count:4d} sequences")
        
        print(f"\n💾 Data saved to: ultra_elite_training_data.pkl")
    else:
        print("❌ No sequences extracted!")

if __name__ == "__main__":
    main()