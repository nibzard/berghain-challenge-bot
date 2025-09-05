# ABOUTME: LSTM-based policy network for RL-driven Berghain admission decisions
# ABOUTME: Dual-headed architecture with policy and value estimation for PPO training

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class LSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network for sequential decision making in Berghain game.
    
    Architecture:
    - LSTM layers for temporal modeling
    - Policy head for action probabilities (accept/reject)
    - Value head for state value estimation (for PPO)
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # [well_dressed, young, constraint_progress_y, constraint_progress_w, capacity_ratio, rejection_ratio, game_phase, person_index_norm]
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # LSTM backbone for sequential modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Binary: [reject_prob, accept_prob]
            nn.Softmax(dim=-1)
        )
        
        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single scalar value
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Optional hidden state tuple (h0, c0)
        
        Returns:
            policy: Action probabilities (batch_size, sequence_length, 2)
            value: State values (batch_size, sequence_length, 1)
            hidden: Updated hidden state tuple
        """
        # LSTM forward pass
        lstm_out, hidden_new = self.lstm(x, hidden)
        
        # Apply heads
        policy = self.policy_head(lstm_out)
        value = self.value_head(lstm_out)
        
        return policy, value, hidden_new
    
    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action, log probability, and value for a given state.
        
        Args:
            x: Input state tensor
            hidden: LSTM hidden state
            deterministic: If True, take argmax action; if False, sample from policy
        
        Returns:
            action: Selected action (0=reject, 1=accept)
            log_prob: Log probability of the action
            value: State value estimate
            hidden: Updated hidden state
        """
        policy, value, hidden_new = self.forward(x, hidden)
        
        if deterministic:
            action = torch.argmax(policy, dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(policy.squeeze(1))
            action = dist.sample()
        
        # Calculate log probability
        log_prob = torch.log(policy.squeeze(1).gather(1, action.unsqueeze(1))).squeeze(1)
        
        return action, log_prob, value.squeeze(-1).squeeze(1), hidden_new


class StateEncoder:
    """
    Encodes game state and person attributes into neural network input format.
    """
    
    def __init__(self):
        self.feature_names = [
            'well_dressed', 'young', 'constraint_progress_y', 'constraint_progress_w',
            'capacity_ratio', 'rejection_ratio', 'game_phase', 'person_index_norm'
        ]
    
    def encode_state(self, person, game_state) -> np.ndarray:
        """
        Encode current person and game state into feature vector.
        
        Args:
            person: Person object with attributes
            game_state: GameState object
            
        Returns:
            np.ndarray: Feature vector of shape (input_dim,)
        """
        # Person attributes
        well_dressed = float(person.has_attribute('well_dressed'))
        young = float(person.has_attribute('young'))
        
        # Constraint progress (0.0 to 1.0+)
        constraint_progress = game_state.constraint_progress()
        constraint_progress_y = constraint_progress.get('young', 0.0)
        constraint_progress_w = constraint_progress.get('well_dressed', 0.0)
        
        # Capacity usage
        capacity_ratio = game_state.capacity_ratio
        rejection_ratio = game_state.rejection_ratio
        
        # Game phase (early/mid/late based on admitted count)
        if game_state.admitted_count < 300:
            game_phase = 0.0  # Early
        elif game_state.admitted_count < 700:
            game_phase = 0.5  # Mid
        else:
            game_phase = 1.0  # Late
        
        # Normalized person index (temporal position)
        person_index_norm = min(person.index / 25000, 1.0)  # Cap at reasonable game length
        
        return np.array([
            well_dressed, young, constraint_progress_y, constraint_progress_w,
            capacity_ratio, rejection_ratio, game_phase, person_index_norm
        ], dtype=np.float32)
    
    def encode_sequence(self, decisions_history) -> np.ndarray:
        """
        Encode a sequence of decisions into input format for training.
        
        Args:
            decisions_history: List of (person, game_state) tuples
            
        Returns:
            np.ndarray: Sequence tensor of shape (sequence_length, input_dim)
        """
        sequence = []
        for person, game_state in decisions_history:
            features = self.encode_state(person, game_state)
            sequence.append(features)
        
        return np.array(sequence, dtype=np.float32)


class PolicyInference:
    """
    Helper class for using trained policy for inference in the game environment.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = LSTMPolicyNetwork()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.encoder = StateEncoder()
        self.hidden_state = None
        
    def reset(self):
        """Reset hidden state for new game."""
        self.hidden_state = None
    
    def get_action(self, person, game_state, deterministic: bool = True) -> Tuple[bool, str]:
        """
        Get action for current person using trained policy.
        
        Args:
            person: Person object
            game_state: GameState object
            deterministic: Whether to use deterministic policy
            
        Returns:
            bool: Accept (True) or reject (False)
            str: Reasoning string
        """
        # Encode state
        features = self.encoder.encode_state(person, game_state)
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value, self.hidden_state = self.model.get_action_and_value(
                x, self.hidden_state, deterministic=deterministic
            )
        
        accept = bool(action.item() == 1)
        
        # Generate reasoning based on action and state
        constraint_progress = game_state.constraint_progress()
        reasoning_parts = []
        
        if accept:
            reasoning_parts.append("rl_accept")
            if person.has_attribute('young') and constraint_progress.get('young', 0) < 1.0:
                reasoning_parts.append("helps_young")
            if person.has_attribute('well_dressed') and constraint_progress.get('well_dressed', 0) < 1.0:
                reasoning_parts.append("helps_well_dressed")
        else:
            reasoning_parts.append("rl_reject")
            if not person.has_attribute('young') and not person.has_attribute('well_dressed'):
                reasoning_parts.append("no_needed_attributes")
        
        reasoning = "_".join(reasoning_parts)
        
        return accept, reasoning