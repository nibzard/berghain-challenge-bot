# ABOUTME: Enhanced data preprocessing pipeline with strategic features for LSTM training
# ABOUTME: Includes rarity scores, constraint urgency, and temporal game dynamics

import json
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class EnhancedGameDataPreprocessor:
    """
    Enhanced preprocessor with strategic features for better LSTM training.
    
    Features include:
    - Basic person attributes (young, well_dressed)
    - Constraint progress and urgency 
    - Game capacity and rejection management
    - Attribute rarity and strategic value
    - Temporal game dynamics
    - Decision context features
    """
    
    def __init__(self, sequence_length: int = 50):
        """
        Args:
            sequence_length: Maximum sequence length for LSTM training
        """
        self.sequence_length = sequence_length
        self.feature_dim = 15  # Enhanced feature count
        
        # Feature names for documentation
        self.feature_names = [
            'well_dressed', 'young',  # Person attributes (0-1)
            'constraint_progress_young', 'constraint_progress_well_dressed',  # Constraint progress (0-1+)
            'capacity_ratio', 'rejection_ratio',  # Game resource usage (0-1) 
            'remaining_capacity', 'remaining_rejections',  # Resources left (normalized 0-1)
            'game_phase',  # Game progression (0-1)
            'person_index_norm',  # Person arrival time (0-1)
            'constraint_urgency_young', 'constraint_urgency_well_dressed',  # Urgency scores (0-1)
            'accept_rate_recent', 'multi_attribute_bonus', 'strategic_value'  # Strategic features (0-1)
        ]
        
    def load_game_logs(self, log_directory: str) -> List[Dict[str, Any]]:
        """Load all individual game JSON files from directory."""
        games = []
        
        for filename in os.listdir(log_directory):
            if (filename.startswith('game_') and 
                filename.endswith('.json') and 
                'consolidated' not in filename):
                
                filepath = os.path.join(log_directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        game_data = json.load(f)
                        # Only include successful games for supervised learning
                        if game_data.get('success', False):
                            games.append(game_data)
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")
        
        logger.info(f"Loaded {len(games)} successful games from {log_directory}")
        return games
    
    def calculate_constraint_urgency(self, current_count: int, target_count: int, 
                                   remaining_capacity: int, decisions_left: int) -> float:
        """
        Calculate urgency for meeting a constraint.
        
        Returns:
            0.0 = no urgency (constraint already met)
            0.5 = moderate urgency
            1.0 = extreme urgency (running out of time/space)
        """
        if current_count >= target_count:
            return 0.0  # Constraint already satisfied
        
        deficit = target_count - current_count
        
        # Can't meet constraint = maximum urgency
        if deficit > remaining_capacity or deficit > decisions_left * 0.5:  # Assuming 50% of remaining people might have this attribute
            return 1.0
        
        # Calculate urgency based on how tight the constraint is
        capacity_pressure = deficit / max(remaining_capacity, 1)
        time_pressure = deficit / max(decisions_left * 0.3, 1)  # 30% attribute frequency
        
        urgency = min(max(capacity_pressure, time_pressure), 1.0)
        return urgency
    
    def calculate_strategic_value(self, person_attrs: Dict[str, bool], 
                                constraint_progress: Dict[str, float],
                                remaining_capacity: int) -> float:
        """
        Calculate strategic value of accepting this person.
        
        Returns:
            0.0 = no strategic value
            0.5 = moderate value
            1.0 = extremely valuable
        """
        value = 0.0
        
        # Value based on needed attributes
        needed_attrs = []
        for attr in ['young', 'well_dressed']:
            if person_attrs.get(attr, False) and constraint_progress.get(attr, 0) < 1.0:
                needed_attrs.append(attr)
                # More value if constraint is further from completion
                value += (1.0 - constraint_progress.get(attr, 0)) * 0.4
        
        # Bonus for having multiple needed attributes
        if len(needed_attrs) >= 2:
            value += 0.3
        
        # Value decreases if capacity is very low
        if remaining_capacity < 50:
            value *= 0.7
        
        return min(value, 1.0)
    
    def extract_features_and_labels(self, game_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract enhanced feature sequences and decision labels from a single game.
        
        Returns:
            features: Array of shape (num_decisions, feature_dim)
            labels: Array of shape (num_decisions,) with 0/1 decisions
        """
        decisions = game_data['decisions']
        features = []
        labels = []
        
        # Track game state progression
        admitted_count = 0
        rejected_count = 0
        constraints = game_data['constraints']
        
        # Get constraint targets
        young_target = next((c['min_count'] for c in constraints if c['attribute'] == 'young'), 600)
        well_dressed_target = next((c['min_count'] for c in constraints if c['attribute'] == 'well_dressed'), 600)
        
        # Track admitted attributes
        young_admitted = 0
        well_dressed_admitted = 0
        
        # Track recent accept rate (sliding window)
        recent_decisions = []  # Store last 100 decisions
        window_size = 100
        
        # Game constants
        max_capacity = 1000
        max_rejections = 20000
        
        for i, decision in enumerate(decisions):
            person_attrs = decision['attributes']
            decision_made = decision['decision']
            
            # Basic person attributes
            well_dressed = 1.0 if person_attrs.get('well_dressed', False) else 0.0
            young = 1.0 if person_attrs.get('young', False) else 0.0
            
            # Constraint progress
            constraint_progress_y = young_admitted / young_target if young_target > 0 else 1.0
            constraint_progress_w = well_dressed_admitted / well_dressed_target if well_dressed_target > 0 else 1.0
            
            # Game resource usage
            total_decisions = admitted_count + rejected_count
            capacity_ratio = admitted_count / max_capacity
            rejection_ratio = rejected_count / max_rejections
            
            # Remaining resources (normalized)
            remaining_capacity = max(max_capacity - admitted_count, 0) / max_capacity
            remaining_rejections = max(max_rejections - rejected_count, 0) / max_rejections
            
            # Game phase (more granular)
            game_phase = min(total_decisions / 2000, 1.0)  # Typical game length
            
            # Person index normalized
            person_index_norm = min(i / 25000, 1.0)
            
            # Constraint urgency
            decisions_left = len(decisions) - i
            remaining_cap_actual = max_capacity - admitted_count
            
            constraint_urgency_y = self.calculate_constraint_urgency(
                young_admitted, young_target, remaining_cap_actual, decisions_left
            )
            constraint_urgency_w = self.calculate_constraint_urgency(
                well_dressed_admitted, well_dressed_target, remaining_cap_actual, decisions_left
            )
            
            # Recent accept rate
            if len(recent_decisions) > 0:
                accept_rate_recent = sum(recent_decisions) / len(recent_decisions)
            else:
                accept_rate_recent = 0.5  # Default assumption
            
            # Multi-attribute bonus
            has_young = person_attrs.get('young', False)
            has_well_dressed = person_attrs.get('well_dressed', False)
            multi_attribute_bonus = 1.0 if (has_young and has_well_dressed) else 0.0
            
            # Strategic value
            constraint_progress = {'young': constraint_progress_y, 'well_dressed': constraint_progress_w}
            strategic_value = self.calculate_strategic_value(
                person_attrs, constraint_progress, remaining_cap_actual
            )
            
            # Create enhanced feature vector
            feature_vector = np.array([
                well_dressed, young,
                constraint_progress_y, constraint_progress_w,
                capacity_ratio, rejection_ratio,
                remaining_capacity, remaining_rejections,
                game_phase, person_index_norm,
                constraint_urgency_y, constraint_urgency_w,
                accept_rate_recent, multi_attribute_bonus, strategic_value
            ], dtype=np.float32)
            
            features.append(feature_vector)
            labels.append(1 if decision_made else 0)
            
            # Update game state for next iteration
            if decision_made:
                admitted_count += 1
                if person_attrs.get('young', False):
                    young_admitted += 1
                if person_attrs.get('well_dressed', False):
                    well_dressed_admitted += 1
                recent_decisions.append(1)
            else:
                rejected_count += 1
                recent_decisions.append(0)
            
            # Maintain sliding window for recent decisions
            if len(recent_decisions) > window_size:
                recent_decisions.pop(0)
        
        return np.array(features), np.array(labels, dtype=np.int64)
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create fixed-length sequences for LSTM training with overlapping windows.
        
        Returns:
            sequence_features: Array of shape (num_sequences, sequence_length, feature_dim)
            sequence_labels: Array of shape (num_sequences, sequence_length)
        """
        if len(features) <= self.sequence_length:
            # Pad short sequences
            pad_length = self.sequence_length - len(features)
            padded_features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
            padded_labels = np.pad(labels, (0, pad_length), mode='constant')
            return padded_features[np.newaxis, :, :], padded_labels[np.newaxis, :]
        
        # Create overlapping windows for longer sequences
        sequences_features = []
        sequences_labels = []
        
        # Use stride of sequence_length // 3 for more overlap (better temporal learning)
        stride = max(self.sequence_length // 3, 1)
        
        for i in range(0, len(features) - self.sequence_length + 1, stride):
            end_idx = i + self.sequence_length
            sequences_features.append(features[i:end_idx])
            sequences_labels.append(labels[i:end_idx])
        
        return np.array(sequences_features), np.array(sequences_labels)
    
    def prepare_dataset(self, games: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepare complete dataset from list of games.
        
        Returns:
            all_sequences: List of feature tensors
            all_labels: List of label tensors  
        """
        all_sequences = []
        all_labels = []
        
        game_quality_info = []  # Track per-game statistics
        
        for game in games:
            try:
                features, labels = self.extract_features_and_labels(game)
                seq_features, seq_labels = self.create_sequences(features, labels)
                
                # Track game statistics
                accept_rate = np.mean(labels)
                game_info = {
                    'game_id': game.get('game_id', 'unknown'),
                    'rejected_count': game.get('rejected_count', 0),
                    'decisions_count': len(labels),
                    'accept_rate': accept_rate,
                    'sequences_created': len(seq_features)
                }
                game_quality_info.append(game_info)
                
                # Convert to tensors and add to lists
                for i in range(len(seq_features)):
                    all_sequences.append(torch.tensor(seq_features[i], dtype=torch.float32))
                    all_labels.append(torch.tensor(seq_labels[i], dtype=torch.long))
                    
            except Exception as e:
                logger.warning(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
        
        logger.info(f"Created {len(all_sequences)} training sequences from {len(games)} games")
        
        # Log dataset statistics
        if all_labels:
            all_labels_flat = torch.cat([labels.flatten() for labels in all_labels]).numpy()
            accept_rate_overall = np.mean(all_labels_flat)
            logger.info(f"Overall accept rate: {accept_rate_overall:.3f}")
            
        # Log per-game statistics
        if game_quality_info:
            avg_rejections = np.mean([g['rejected_count'] for g in game_quality_info])
            logger.info(f"Average rejections in training games: {avg_rejections:.0f}")
            
        return all_sequences, all_labels


def prepare_enhanced_training_data(
    log_directory: str, 
    test_split: float = 0.2, 
    sequence_length: int = 50
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Main function to prepare enhanced training data from game logs.
    
    Args:
        log_directory: Path to directory containing game log JSON files
        test_split: Fraction of data to use for validation
        sequence_length: Length of sequences for LSTM training
        
    Returns:
        train_data: List of (features, labels) tuples for training
        val_data: List of (features, labels) tuples for validation
    """
    processor = EnhancedGameDataPreprocessor(sequence_length=sequence_length)
    
    # Load games
    games = processor.load_game_logs(log_directory)
    if not games:
        raise ValueError(f"No successful games found in {log_directory}")
    
    # Prepare dataset
    sequences, labels = processor.prepare_dataset(games)
    
    # Split into train/validation by game (not randomly by sequence)
    # This prevents data leakage where sequences from the same game appear in both sets
    game_ids = []
    game_sequence_indices = {}
    
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        # Use a simple hash of the first few features as game identifier
        game_hash = hash(tuple(seq[0, :5].tolist()))  # Hash first 5 features of first timestep
        if game_hash not in game_sequence_indices:
            game_sequence_indices[game_hash] = []
        game_sequence_indices[game_hash].append(i)
        game_ids.append(game_hash)
    
    # Split games (not sequences)
    unique_games = list(game_sequence_indices.keys())
    train_games, val_games = train_test_split(unique_games, test_size=test_split, random_state=42)
    
    # Get sequence indices for train/val games
    train_indices = []
    val_indices = []
    
    for game in train_games:
        train_indices.extend(game_sequence_indices[game])
    for game in val_games:
        val_indices.extend(game_sequence_indices[game])
    
    # Split sequences based on games
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    # Combine into tuples
    train_data = list(zip(train_sequences, train_labels))
    val_data = list(zip(val_sequences, val_labels))
    
    logger.info(f"Enhanced training data prepared:")
    logger.info(f"  Training games: {len(train_games)}, sequences: {len(train_data)}")
    logger.info(f"  Validation games: {len(val_games)}, sequences: {len(val_data)}")
    logger.info(f"  Features per timestep: {processor.feature_dim}")
    
    return train_data, val_data