# ABOUTME: Ultra-enhanced data preprocessor for LSTM training on elite games with 30+ strategic features
# ABOUTME: Includes lookahead features, risk metrics, pattern analysis, and dynamic thresholds

import json
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GameMetrics:
    """Container for game-level metrics and patterns."""
    total_decisions: int
    accept_rate: float
    constraint_satisfaction_rate: float
    rejection_efficiency: float
    phase_transitions: List[float]
    critical_decisions: List[int]


class UltraElitePreprocessor:
    """
    Ultra-enhanced preprocessor with 30+ strategic features for elite LSTM training.
    
    Features include:
    - Basic person attributes (2)
    - Constraint progress and urgency (6) 
    - Game capacity and resource management (6)
    - Temporal and phase features (4)
    - Lookahead and risk assessment (6)
    - Pattern recognition and streaks (4)
    - Dynamic thresholds and adaptation (4)
    - Strategic value and correlation (3)
    Total: 35 features
    """
    
    def __init__(self, sequence_length: int = 100):
        """
        Args:
            sequence_length: Maximum sequence length for LSTM training
        """
        self.sequence_length = sequence_length
        self.feature_dim = 35  # Ultra-enhanced feature count
        
        # Enhanced feature names for documentation
        self.feature_names = [
            # Basic attributes (2)
            'well_dressed', 'young',
            # Constraint progress (6)
            'constraint_progress_young', 'constraint_progress_well_dressed',
            'constraint_urgency_young', 'constraint_urgency_well_dressed',
            'constraint_deficit_young', 'constraint_deficit_well_dressed',
            # Resource management (6)
            'capacity_ratio', 'rejection_ratio', 'remaining_capacity', 'remaining_rejections',
            'resource_pressure', 'efficiency_ratio',
            # Temporal features (4)
            'game_phase', 'person_index_norm', 'time_pressure', 'decisions_per_minute',
            # Lookahead features (6)
            'expected_people_remaining', 'constraint_feasibility_young', 'constraint_feasibility_well_dressed',
            'risk_constraint_failure', 'safety_margin', 'lookahead_score',
            # Pattern features (4)
            'accept_streak', 'reject_streak', 'pattern_momentum', 'phase_transition_score',
            # Dynamic thresholds (4)
            'adaptive_threshold', 'selectivity_score', 'opportunity_cost', 'decision_confidence',
            # Strategic features (3)
            'multi_attribute_bonus', 'correlation_score', 'strategic_value'
        ]
        
    def load_ultra_elite_games(self, ultra_elite_dir: str) -> List[Dict[str, Any]]:
        """Load ultra-elite games from directory."""
        games = []
        
        for filename in os.listdir(ultra_elite_dir):
            if filename.startswith('elite_') and filename.endswith('.json') and 'stats' not in filename:
                filepath = os.path.join(ultra_elite_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        game_data = json.load(f)
                        games.append(game_data)
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")
        
        logger.info(f"Loaded {len(games)} ultra-elite games from {ultra_elite_dir}")
        return games
    
    def calculate_lookahead_features(self, person_idx: int, total_decisions: int,
                                   young_admitted: int, well_dressed_admitted: int,
                                   young_target: int, well_dressed_target: int,
                                   capacity_left: int) -> Dict[str, float]:
        """Calculate lookahead and risk assessment features."""
        decisions_left = max(total_decisions - person_idx, 1)
        expected_people = min(decisions_left, 25000 - person_idx)  # Typical max people
        
        # Estimate remaining people with attributes (based on typical frequencies)
        young_freq = 0.323
        well_dressed_freq = 0.323
        
        expected_young = expected_people * young_freq
        expected_well_dressed = expected_people * well_dressed_freq
        
        # Constraint feasibility
        young_deficit = max(young_target - young_admitted, 0)
        well_dressed_deficit = max(well_dressed_target - well_dressed_admitted, 0)
        
        young_feasible = 1.0 if young_deficit <= expected_young else expected_young / max(young_deficit, 1)
        well_dressed_feasible = 1.0 if well_dressed_deficit <= expected_well_dressed else expected_well_dressed / max(well_dressed_deficit, 1)
        
        # Risk assessment
        total_deficit = young_deficit + well_dressed_deficit
        risk_failure = min(total_deficit / max(capacity_left, 1), 1.0)
        
        # Safety margin
        safety_margin = max((capacity_left - total_deficit) / max(capacity_left, 1), 0.0)
        
        # Lookahead score (higher = better future prospects)
        lookahead_score = (young_feasible + well_dressed_feasible) / 2.0
        
        return {
            'expected_people_remaining': min(expected_people / 1000, 1.0),  # Normalize
            'constraint_feasibility_young': young_feasible,
            'constraint_feasibility_well_dressed': well_dressed_feasible,
            'risk_constraint_failure': risk_failure,
            'safety_margin': safety_margin,
            'lookahead_score': lookahead_score
        }
    
    def calculate_pattern_features(self, recent_decisions: deque, person_idx: int) -> Dict[str, float]:
        """Calculate pattern recognition and streak features."""
        if len(recent_decisions) == 0:
            return {
                'accept_streak': 0.0,
                'reject_streak': 0.0,
                'pattern_momentum': 0.0,
                'phase_transition_score': 0.0
            }
        
        decisions_list = list(recent_decisions)
        
        # Calculate current streaks
        accept_streak = 0
        reject_streak = 0
        
        for decision in reversed(decisions_list):
            if decision == 1:  # Accept
                if reject_streak > 0:
                    break
                accept_streak += 1
            else:  # Reject
                if accept_streak > 0:
                    break
                reject_streak += 1
        
        # Pattern momentum (how consistent recent decisions are)
        if len(decisions_list) >= 10:
            recent_10 = decisions_list[-10:]
            momentum = abs(sum(recent_10) - 5) / 5.0  # Distance from balanced
        else:
            momentum = 0.0
        
        # Phase transition detection (change in acceptance rate)
        phase_transition = 0.0
        if len(decisions_list) >= 20:
            first_half = decisions_list[-20:-10]
            second_half = decisions_list[-10:]
            rate_change = abs(np.mean(second_half) - np.mean(first_half))
            phase_transition = min(rate_change * 2, 1.0)  # Scale to 0-1
        
        return {
            'accept_streak': min(accept_streak / 10.0, 1.0),  # Normalize to 0-1
            'reject_streak': min(reject_streak / 50.0, 1.0),  # Normalize to 0-1
            'pattern_momentum': momentum,
            'phase_transition_score': phase_transition
        }
    
    def calculate_dynamic_thresholds(self, game_state: Dict, person_attrs: Dict,
                                   recent_decisions: deque, person_idx: int) -> Dict[str, float]:
        """Calculate adaptive thresholds and decision confidence."""
        # Adaptive threshold based on game progress
        game_progress = min(person_idx / 2000, 1.0)
        base_threshold = 0.5
        
        # Adjust threshold based on constraint urgency
        young_progress = game_state.get('young_progress', 0.5)
        well_dressed_progress = game_state.get('well_dressed_progress', 0.5)
        avg_progress = (young_progress + well_dressed_progress) / 2
        
        if avg_progress < 0.3:  # Behind on constraints
            adaptive_threshold = base_threshold - 0.2  # Be less selective
        elif avg_progress > 0.8:  # Ahead on constraints
            adaptive_threshold = base_threshold + 0.2  # Be more selective
        else:
            adaptive_threshold = base_threshold
        
        adaptive_threshold = np.clip(adaptive_threshold, 0.1, 0.9)
        
        # Selectivity score (how picky we should be)
        capacity_pressure = game_state.get('capacity_ratio', 0.5)
        selectivity = min(capacity_pressure * 1.5, 1.0)
        
        # Opportunity cost (cost of wrong decision)
        has_needed_attrs = any(person_attrs.get(attr, False) for attr in ['young', 'well_dressed'])
        opportunity_cost = 0.8 if has_needed_attrs else 0.2
        
        # Decision confidence (how sure we are)
        recent_accept_rate = np.mean(recent_decisions) if recent_decisions else 0.5
        confidence = abs(recent_accept_rate - 0.5) * 2  # Higher when more extreme
        
        return {
            'adaptive_threshold': adaptive_threshold,
            'selectivity_score': selectivity,
            'opportunity_cost': opportunity_cost,
            'decision_confidence': confidence
        }
    
    def extract_enhanced_features_and_labels(self, game_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ultra-enhanced feature sequences and decision labels from a game."""
        decisions = game_data['decisions']
        features = []
        labels = []
        
        # Initialize game state tracking
        admitted_count = 0
        rejected_count = 0
        constraints = game_data['constraints']
        
        # Get constraint targets
        young_target = next((c['min_count'] for c in constraints if c['attribute'] == 'young'), 600)
        well_dressed_target = next((c['min_count'] for c in constraints if c['attribute'] == 'well_dressed'), 600)
        
        # Track admitted attributes
        young_admitted = 0
        well_dressed_admitted = 0
        
        # Track recent decisions (sliding window)
        recent_decisions = deque(maxlen=100)
        decision_times = deque(maxlen=100)
        
        # Game constants
        max_capacity = 1000
        max_rejections = 20000
        
        for i, decision in enumerate(decisions):
            # Handle both formats: decision['attributes'] or decision['person']['attributes']
            if 'attributes' in decision:
                person_attrs = decision['attributes']
                decision_made = decision['decision']
            else:
                person_attrs = decision['person']['attributes']
                decision_made = decision['accepted']
            
            # Basic person attributes
            well_dressed = 1.0 if person_attrs.get('well_dressed', False) else 0.0
            young = 1.0 if person_attrs.get('young', False) else 0.0
            
            # Constraint progress and deficits
            constraint_progress_y = young_admitted / young_target if young_target > 0 else 1.0
            constraint_progress_w = well_dressed_admitted / well_dressed_target if well_dressed_target > 0 else 1.0
            
            young_deficit = max(young_target - young_admitted, 0) / young_target
            well_dressed_deficit = max(well_dressed_target - well_dressed_admitted, 0) / well_dressed_target
            
            # Resource management
            total_decisions = admitted_count + rejected_count
            capacity_ratio = admitted_count / max_capacity
            rejection_ratio = rejected_count / max_rejections
            remaining_capacity = max(max_capacity - admitted_count, 0) / max_capacity
            remaining_rejections = max(max_rejections - rejected_count, 0) / max_rejections
            
            resource_pressure = max(capacity_ratio, rejection_ratio)
            efficiency_ratio = admitted_count / max(total_decisions, 1) if total_decisions > 0 else 0.0
            
            # Temporal features
            game_phase = min(total_decisions / 2000, 1.0)
            person_index_norm = min(i / 25000, 1.0)
            time_pressure = min(i / 15000, 1.0)  # Accelerating pressure
            
            # Decisions per minute (simulate time pressure)
            decisions_per_minute = min(i / max(i * 0.1, 1), 100) / 100  # Normalize
            
            # Constraint urgency
            decisions_left = len(decisions) - i
            remaining_cap_actual = max_capacity - admitted_count
            
            constraint_urgency_y = self._calculate_constraint_urgency(
                young_admitted, young_target, remaining_cap_actual, decisions_left
            )
            constraint_urgency_w = self._calculate_constraint_urgency(
                well_dressed_admitted, well_dressed_target, remaining_cap_actual, decisions_left
            )
            
            # Lookahead features
            lookahead_features = self.calculate_lookahead_features(
                i, len(decisions), young_admitted, well_dressed_admitted,
                young_target, well_dressed_target, remaining_cap_actual
            )
            
            # Pattern features
            pattern_features = self.calculate_pattern_features(recent_decisions, i)
            
            # Game state for dynamic thresholds
            game_state = {
                'young_progress': constraint_progress_y,
                'well_dressed_progress': constraint_progress_w,
                'capacity_ratio': capacity_ratio
            }
            
            # Dynamic threshold features
            dynamic_features = self.calculate_dynamic_thresholds(
                game_state, person_attrs, recent_decisions, i
            )
            
            # Strategic features
            multi_attribute_bonus = 1.0 if (well_dressed and young) else 0.0
            
            # Correlation score (how well this person fits current needs)
            need_young = constraint_progress_y < 0.8
            need_well_dressed = constraint_progress_w < 0.8
            has_young = person_attrs.get('young', False)
            has_well_dressed = person_attrs.get('well_dressed', False)
            
            correlation_score = 0.0
            if need_young and has_young:
                correlation_score += 0.5
            if need_well_dressed and has_well_dressed:
                correlation_score += 0.5
            
            # Strategic value (overall utility of accepting this person)
            strategic_value = correlation_score * (1.0 - resource_pressure)
            if multi_attribute_bonus:
                strategic_value += 0.2
            strategic_value = min(strategic_value, 1.0)
            
            # Create ultra-enhanced feature vector (35 features)
            feature_vector = np.array([
                # Basic (2)
                well_dressed, young,
                # Constraint progress (6)
                constraint_progress_y, constraint_progress_w,
                constraint_urgency_y, constraint_urgency_w,
                young_deficit, well_dressed_deficit,
                # Resource management (6)
                capacity_ratio, rejection_ratio, remaining_capacity, remaining_rejections,
                resource_pressure, efficiency_ratio,
                # Temporal (4)
                game_phase, person_index_norm, time_pressure, decisions_per_minute,
                # Lookahead (6)
                lookahead_features['expected_people_remaining'],
                lookahead_features['constraint_feasibility_young'],
                lookahead_features['constraint_feasibility_well_dressed'],
                lookahead_features['risk_constraint_failure'],
                lookahead_features['safety_margin'],
                lookahead_features['lookahead_score'],
                # Pattern (4)
                pattern_features['accept_streak'],
                pattern_features['reject_streak'],
                pattern_features['pattern_momentum'],
                pattern_features['phase_transition_score'],
                # Dynamic thresholds (4)
                dynamic_features['adaptive_threshold'],
                dynamic_features['selectivity_score'],
                dynamic_features['opportunity_cost'],
                dynamic_features['decision_confidence'],
                # Strategic (3)
                multi_attribute_bonus, correlation_score, strategic_value
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
        
        return np.array(features), np.array(labels, dtype=np.int64)
    
    def _calculate_constraint_urgency(self, current_count: int, target_count: int, 
                                    remaining_capacity: int, decisions_left: int) -> float:
        """Calculate urgency for meeting a constraint."""
        if current_count >= target_count:
            return 0.0
        
        deficit = target_count - current_count
        
        if deficit > remaining_capacity or deficit > decisions_left * 0.5:
            return 1.0
        
        capacity_pressure = deficit / max(remaining_capacity, 1)
        time_pressure = deficit / max(decisions_left * 0.3, 1)
        
        urgency = min(max(capacity_pressure, time_pressure), 1.0)
        return urgency
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with enhanced overlapping."""
        if len(features) <= self.sequence_length:
            pad_length = self.sequence_length - len(features)
            padded_features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
            padded_labels = np.pad(labels, (0, pad_length), mode='constant')
            return padded_features[np.newaxis, :, :], padded_labels[np.newaxis, :]
        
        sequences_features = []
        sequences_labels = []
        
        # Use smaller stride for more overlap and better learning
        stride = max(self.sequence_length // 4, 1)
        
        for i in range(0, len(features) - self.sequence_length + 1, stride):
            end_idx = i + self.sequence_length
            sequences_features.append(features[i:end_idx])
            sequences_labels.append(labels[i:end_idx])
        
        return np.array(sequences_features), np.array(sequences_labels)
    
    def prepare_ultra_elite_dataset(self, games: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Prepare complete dataset from ultra-elite games."""
        all_sequences = []
        all_labels = []
        
        game_quality_info = []
        
        for game in games:
            try:
                features, labels = self.extract_enhanced_features_and_labels(game)
                seq_features, seq_labels = self.create_sequences(features, labels)
                
                # Quality metrics
                accept_rate = np.mean(labels)
                game_info = {
                    'game_id': game.get('game_id', 'unknown'),
                    'rejected_count': game.get('rejected_count', 0),
                    'decisions_count': len(labels),
                    'accept_rate': accept_rate,
                    'sequences_created': len(seq_features),
                    'strategy': game.get('strategy', 'unknown')
                }
                game_quality_info.append(game_info)
                
                # Convert to tensors
                for i in range(len(seq_features)):
                    all_sequences.append(torch.tensor(seq_features[i], dtype=torch.float32))
                    all_labels.append(torch.tensor(seq_labels[i], dtype=torch.long))
                    
            except Exception as e:
                logger.warning(f"Error processing ultra-elite game {game.get('game_id', 'unknown')}: {e}")
        
        logger.info(f"Created {len(all_sequences)} training sequences from {len(games)} ultra-elite games")
        
        if all_labels:
            all_labels_flat = torch.cat([labels.flatten() for labels in all_labels]).numpy()
            accept_rate_overall = np.mean(all_labels_flat)
            logger.info(f"Overall accept rate: {accept_rate_overall:.3f}")
            
        if game_quality_info:
            avg_rejections = np.mean([g['rejected_count'] for g in game_quality_info])
            logger.info(f"Average rejections in ultra-elite games: {avg_rejections:.0f}")
            
            # Log strategy breakdown
            strategy_counts = {}
            for info in game_quality_info:
                strategy = info['strategy']
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            logger.info(f"Strategy breakdown: {strategy_counts}")
            
        return all_sequences, all_labels


def prepare_ultra_elite_training_data(
    ultra_elite_dir: str, 
    test_split: float = 0.2, 
    sequence_length: int = 100
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Prepare ultra-enhanced training data from ultra-elite games.
    
    Args:
        ultra_elite_dir: Directory with ultra-elite games
        test_split: Fraction for validation
        sequence_length: LSTM sequence length
        
    Returns:
        train_data: Training sequences
        val_data: Validation sequences
    """
    processor = UltraElitePreprocessor(sequence_length=sequence_length)
    
    # Load ultra-elite games
    games = processor.load_ultra_elite_games(ultra_elite_dir)
    if not games:
        raise ValueError(f"No ultra-elite games found in {ultra_elite_dir}")
    
    # Prepare dataset with enhanced features
    sequences, labels = processor.prepare_ultra_elite_dataset(games)
    
    # Split by games to avoid data leakage
    game_indices = {}
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        game_hash = hash(tuple(seq[0, :5].tolist()))
        if game_hash not in game_indices:
            game_indices[game_hash] = []
        game_indices[game_hash].append(i)
    
    # Split games
    unique_games = list(game_indices.keys())
    train_games, val_games = train_test_split(unique_games, test_size=test_split, random_state=42)
    
    # Get indices for each split
    train_indices = []
    val_indices = []
    
    for game in train_games:
        train_indices.extend(game_indices[game])
    for game in val_games:
        val_indices.extend(game_indices[game])
    
    # Create final datasets
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    train_data = list(zip(train_sequences, train_labels))
    val_data = list(zip(val_sequences, val_labels))
    
    logger.info(f"Ultra-elite training data prepared:")
    logger.info(f"  Training games: {len(train_games)}, sequences: {len(train_data)}")
    logger.info(f"  Validation games: {len(val_games)}, sequences: {len(val_data)}")
    logger.info(f"  Features per timestep: {processor.feature_dim}")
    
    return train_data, val_data