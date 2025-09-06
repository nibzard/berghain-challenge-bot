# ABOUTME: Data preprocessing pipeline for supervised learning from game logs
# ABOUTME: Converts historical decision data into training sequences for LSTM models

import json
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class GameDataPreprocessor:
    """
    Preprocesses game log data for supervised learning.
    
    Converts decision sequences from game logs into training data where:
    - X: Sequential game state features
    - y: Expert decision labels (0=reject, 1=accept)
    """
    
    def __init__(self, sequence_length: int = 50):
        """
        Args:
            sequence_length: Maximum sequence length for LSTM training
        """
        self.sequence_length = sequence_length
        self.feature_dim = 8  # Matches StateEncoder feature count
        
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
    
    def extract_features_and_labels(self, game_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature sequences and decision labels from a single game.
        
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
        
        for i, decision in enumerate(decisions):
            person_attrs = decision['attributes']
            decision_made = decision['decision']
            
            # Person attributes
            well_dressed = 1.0 if person_attrs.get('well_dressed', False) else 0.0
            young = 1.0 if person_attrs.get('young', False) else 0.0
            
            # Constraint progress
            constraint_progress_y = min(young_admitted / young_target, 1.0) if young_target > 0 else 1.0
            constraint_progress_w = min(well_dressed_admitted / well_dressed_target, 1.0) if well_dressed_target > 0 else 1.0
            
            # Capacity and rejection ratios
            total_decisions = admitted_count + rejected_count
            capacity_ratio = admitted_count / 1000.0  # Max capacity
            rejection_ratio = rejected_count / 20000.0 if total_decisions > 0 else 0.0  # Max rejections
            
            # Game phase
            if admitted_count < 300:
                game_phase = 0.0  # Early
            elif admitted_count < 700:
                game_phase = 0.5  # Mid
            else:
                game_phase = 1.0  # Late
            
            # Person index normalized
            person_index_norm = min(i / 25000, 1.0)
            
            # Create feature vector
            feature_vector = np.array([
                well_dressed, young, constraint_progress_y, constraint_progress_w,
                capacity_ratio, rejection_ratio, game_phase, person_index_norm
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
            else:
                rejected_count += 1
        
        return np.array(features), np.array(labels, dtype=np.int64)
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create fixed-length sequences for LSTM training.
        
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
        
        for i in range(0, len(features) - self.sequence_length + 1, self.sequence_length // 2):
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
        
        for game in games:
            try:
                features, labels = self.extract_features_and_labels(game)
                seq_features, seq_labels = self.create_sequences(features, labels)
                
                # Convert to tensors and add to lists
                for i in range(len(seq_features)):
                    all_sequences.append(torch.tensor(seq_features[i], dtype=torch.float32))
                    all_labels.append(torch.tensor(seq_labels[i], dtype=torch.long))
                    
            except Exception as e:
                logger.warning(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
        
        logger.info(f"Created {len(all_sequences)} training sequences")
        return all_sequences, all_labels


def prepare_training_data(log_directory: str, test_split: float = 0.2, sequence_length: int = 50) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Main function to prepare training data from game logs.
    
    Args:
        log_directory: Path to directory containing game log JSON files
        test_split: Fraction of data to use for validation
        sequence_length: Length of sequences for LSTM training
        
    Returns:
        train_data: List of (features, labels) tuples for training
        val_data: List of (features, labels) tuples for validation
    """
    processor = GameDataPreprocessor(sequence_length=sequence_length)
    
    # Load games
    games = processor.load_game_logs(log_directory)
    if not games:
        raise ValueError(f"No successful games found in {log_directory}")
    
    # Prepare dataset
    sequences, labels = processor.prepare_dataset(games)
    
    # Split into train/validation
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=test_split, random_state=42, shuffle=True
    )
    
    # Combine into tuples
    train_data = list(zip(train_sequences, train_labels))
    val_data = list(zip(val_sequences, val_labels))
    
    logger.info(f"Training data: {len(train_data)} sequences")
    logger.info(f"Validation data: {len(val_data)} sequences")
    
    return train_data, val_data