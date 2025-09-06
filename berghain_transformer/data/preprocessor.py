"""Data preprocessing for transformer training on Berghain game logs."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import pickle


class GameStateEncoder:
    """Encodes game states into fixed-size feature vectors."""
    
    def __init__(self, scenario: int = 1):
        self.scenario = scenario
        self.feature_dim = 128
        
        # Known attributes from scenario 1
        self.primary_attributes = ['young', 'well_dressed']
        self.all_attributes = set()
        self.attribute_to_idx = {}
        
    def fit(self, game_logs_path: Path):
        """Learn attribute mappings from game logs."""
        attributes = set()
        
        for log_file in game_logs_path.glob("events_*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    event = json.loads(line)
                    if event['event_type'] == 'person_evaluated':
                        person = event['person']
                        attributes.update(person.get('attributes', []))
        
        self.all_attributes = sorted(attributes)
        self.attribute_to_idx = {attr: idx for idx, attr in enumerate(self.all_attributes)}
        self.n_attributes = len(self.all_attributes)
        
    def encode_state(self, event: Dict[str, Any], game_state: Dict[str, Any]) -> np.ndarray:
        """Encode a single game state into a feature vector."""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0
        
        # Current person attributes (one-hot encoding)
        if event['event_type'] == 'person_evaluated':
            person = event['person']
            for attr in person.get('attributes', []):
                if attr in self.attribute_to_idx:
                    attr_idx = self.attribute_to_idx[attr]
                    if idx + attr_idx < self.feature_dim:
                        features[idx + attr_idx] = 1.0
        idx += min(self.n_attributes, self.feature_dim - idx)
        
        # Game progress features
        if idx < self.feature_dim:
            features[idx] = game_state['total_admitted'] / 1000.0
            idx += 1
        if idx < self.feature_dim:
            features[idx] = game_state['total_rejected'] / 20000.0
            idx += 1
        
        # Constraint satisfaction
        for constraint in game_state.get('constraints', []):
            if idx >= self.feature_dim:
                break
            features[idx] = constraint['current'] / max(constraint['required'], 1)
            idx += 1
            features[idx] = (constraint['required'] - constraint['current']) / max(constraint['required'], 1)
            idx += 1
        
        # Attribute frequencies seen so far
        attr_counts = game_state.get('attribute_counts', {})
        total_seen = max(sum(attr_counts.values()), 1)
        
        for attr in self.primary_attributes:
            if idx >= self.feature_dim:
                break
            features[idx] = attr_counts.get(attr, 0) / total_seen
            idx += 1
        
        # Recent decision history (last 10 decisions)
        recent_decisions = game_state.get('recent_decisions', [])
        for i in range(min(10, len(recent_decisions))):
            if idx >= self.feature_dim:
                break
            features[idx] = 1.0 if recent_decisions[-(i+1)] else -1.0
            idx += 1
        
        return features


class BerghainSequenceDataset(Dataset):
    """Dataset for transformer training on game sequences."""
    
    def __init__(
        self,
        game_logs_path: Path,
        encoder: GameStateEncoder,
        seq_length: int = 100,
        stride: int = 50,
        elite_only: bool = False,
        scenario: int = 1
    ):
        self.encoder = encoder
        self.seq_length = seq_length
        self.stride = stride
        self.sequences = []
        
        # Load and process game logs
        self._load_sequences(game_logs_path, elite_only, scenario)
    
    def _load_sequences(self, game_logs_path: Path, elite_only: bool, scenario: int):
        """Load and preprocess game sequences."""
        target_path = game_logs_path / "elite_games" if elite_only else game_logs_path / "game_logs_filtered"
        
        if not target_path.exists():
            target_path = game_logs_path / "game_logs"
        
        pattern = f"events_scenario_{scenario}_*.jsonl"
        log_files = list(target_path.glob(pattern))
        
        for log_file in log_files:
            # Parse game log
            states, actions, rewards = self._parse_game_log(log_file)
            
            if len(states) < self.seq_length:
                continue
            
            # Create overlapping sequences
            for start_idx in range(0, len(states) - self.seq_length + 1, self.stride):
                end_idx = start_idx + self.seq_length
                seq_states = states[start_idx:end_idx]
                seq_actions = actions[start_idx:end_idx]
                seq_rewards = rewards[start_idx:end_idx]
                
                # Calculate returns-to-go
                rtg = self._calculate_returns_to_go(seq_rewards)
                
                self.sequences.append({
                    'states': torch.FloatTensor(seq_states),
                    'actions': torch.LongTensor(seq_actions),
                    'rewards': torch.FloatTensor(seq_rewards),
                    'returns_to_go': torch.FloatTensor(rtg),
                    'mask': torch.ones(self.seq_length, dtype=torch.bool)
                })
    
    def _parse_game_log(self, log_file: Path) -> Tuple[List, List, List]:
        """Parse a single game log file."""
        states = []
        actions = []
        rewards = []
        
        game_state = {
            'total_admitted': 0,
            'total_rejected': 0,
            'constraints': [],
            'attribute_counts': defaultdict(int),
            'recent_decisions': []
        }
        
        with open(log_file) as f:
            for line in f:
                event = json.loads(line)
                
                if event['event_type'] == 'game_started':
                    # Initialize constraints
                    game_state['constraints'] = [
                        {'attribute': c['attribute'], 'required': c['required'], 'current': 0}
                        for c in event.get('constraints', [])
                    ]
                
                elif event['event_type'] == 'person_evaluated':
                    # Encode current state
                    state_vector = self.encoder.encode_state(event, game_state)
                    states.append(state_vector)
                    
                    # Record action (0 = reject, 1 = admit)
                    action = 1 if event['decision']['admitted'] else 0
                    actions.append(action)
                    
                    # Calculate immediate reward
                    reward = self._calculate_reward(event, game_state)
                    rewards.append(reward)
                    
                    # Update game state
                    person = event['person']
                    for attr in person.get('attributes', []):
                        game_state['attribute_counts'][attr] += 1
                    
                    if action == 1:
                        game_state['total_admitted'] += 1
                        for constraint in game_state['constraints']:
                            if constraint['attribute'] in person.get('attributes', []):
                                constraint['current'] += 1
                    else:
                        game_state['total_rejected'] += 1
                    
                    game_state['recent_decisions'].append(action == 1)
                    if len(game_state['recent_decisions']) > 10:
                        game_state['recent_decisions'].pop(0)
        
        return states, actions, rewards
    
    def _calculate_reward(self, event: Dict, game_state: Dict) -> float:
        """Calculate immediate reward for an action."""
        admitted = event['decision']['admitted']
        person = event['person']
        
        reward = 0.0
        
        # Positive reward for admitting people with needed attributes
        if admitted:
            for constraint in game_state['constraints']:
                if constraint['current'] < constraint['required']:
                    if constraint['attribute'] in person.get('attributes', []):
                        # Higher reward for scarce attributes
                        deficit_ratio = 1 - (constraint['current'] / constraint['required'])
                        reward += 1.0 + deficit_ratio
        
        # Penalty for approaching limits
        if game_state['total_admitted'] > 900:
            reward -= 0.5
        if game_state['total_rejected'] > 18000:
            reward -= 0.5
        
        # Small penalty for any action (encourages efficiency)
        reward -= 0.01
        
        return reward
    
    def _calculate_returns_to_go(self, rewards: List[float]) -> List[float]:
        """Calculate returns-to-go for a sequence of rewards."""
        rtg = []
        cumsum = 0
        for r in reversed(rewards):
            cumsum += r
            rtg.append(cumsum)
        return list(reversed(rtg))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def create_dataloaders(
    game_logs_path: Path,
    batch_size: int = 32,
    seq_length: int = 100,
    train_split: float = 0.9,
    elite_only: bool = False,
    scenario: int = 1
) -> Tuple[DataLoader, DataLoader, GameStateEncoder]:
    """Create train and validation dataloaders."""
    
    # Initialize encoder
    encoder = GameStateEncoder(scenario=scenario)
    encoder.fit(game_logs_path / "game_logs")
    
    # Create dataset
    dataset = BerghainSequenceDataset(
        game_logs_path,
        encoder,
        seq_length=seq_length,
        elite_only=elite_only,
        scenario=scenario
    )
    
    # Split into train and validation
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, encoder


def save_encoder(encoder: GameStateEncoder, path: Path):
    """Save encoder for later use during inference."""
    with open(path, 'wb') as f:
        pickle.dump({
            'attribute_to_idx': encoder.attribute_to_idx,
            'all_attributes': encoder.all_attributes,
            'n_attributes': encoder.n_attributes,
            'feature_dim': encoder.feature_dim,
            'primary_attributes': encoder.primary_attributes
        }, f)


def load_encoder(path: Path) -> GameStateEncoder:
    """Load a saved encoder."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    encoder = GameStateEncoder()
    encoder.attribute_to_idx = data['attribute_to_idx']
    encoder.all_attributes = data['all_attributes']
    encoder.n_attributes = data['n_attributes']
    encoder.feature_dim = data['feature_dim']
    encoder.primary_attributes = data['primary_attributes']
    
    return encoder