"""Transformer model for Berghain game decision making."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class BerghainTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 2,
        n_layers: int = 6,
        n_heads: int = 8,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 1000,
        use_value_head: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.use_value_head = use_value_head
        
        # Input embeddings
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.action_embedding = nn.Embedding(action_dim, d_model)
        self.reward_embedding = nn.Linear(1, d_model)
        
        # Token type embeddings (state, action, reward)
        self.token_type_embedding = nn.Embedding(3, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length * 3)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, action_dim)
        )
        
        if use_value_head:
            self.value_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_values: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = states.shape[:2]
        device = states.device
        
        # Embed states
        state_embeddings = self.state_embedding(states)
        state_type = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        state_embeddings += self.token_type_embedding(state_type)
        
        # Build sequence: s_0, a_0, r_0, s_1, a_1, r_1, ...
        if actions is not None and rewards is not None:
            # Training mode with full sequences
            action_embeddings = self.action_embedding(actions)
            action_type = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            action_embeddings += self.token_type_embedding(action_type)
            
            reward_embeddings = self.reward_embedding(rewards.unsqueeze(-1))
            reward_type = torch.full((batch_size, seq_len), 2, dtype=torch.long, device=device)
            reward_embeddings += self.token_type_embedding(reward_type)
            
            # Interleave: s, a, r, s, a, r, ...
            total_seq_len = seq_len * 3
            embeddings = torch.zeros(batch_size, total_seq_len, self.d_model, device=device)
            embeddings[:, 0::3] = state_embeddings
            embeddings[:, 1::3] = action_embeddings
            embeddings[:, 2::3] = reward_embeddings
        else:
            # Inference mode - only states
            embeddings = state_embeddings
            total_seq_len = seq_len
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create causal mask for autoregressive generation
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(total_seq_len, total_seq_len, device=device) * float('-inf'),
                diagonal=1
            )
        
        # Apply transformer
        hidden = self.transformer(embeddings, mask=attention_mask)
        
        # Get predictions from state positions
        if actions is not None and rewards is not None:
            # Extract hidden states at state positions
            state_hidden = hidden[:, 0::3]
        else:
            state_hidden = hidden
        
        # Action predictions
        action_logits = self.action_head(state_hidden)
        
        # Value predictions if needed
        values = None
        if return_values and self.use_value_head:
            values = self.value_head(state_hidden).squeeze(-1)
        
        return action_logits, values


class DecisionTransformer(BerghainTransformer):
    """Decision Transformer variant for offline RL."""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_value_head'] = False  # DT doesn't use value function
        super().__init__(*args, **kwargs)
        
        # Additional return-to-go embedding
        self.rtg_embedding = nn.Linear(1, self.d_model)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = states.shape[:2]
        device = states.device
        
        # Embed inputs
        state_embeddings = self.state_embedding(states)
        
        if returns_to_go is not None:
            rtg_embeddings = self.rtg_embedding(returns_to_go.unsqueeze(-1))
            state_embeddings = state_embeddings + rtg_embeddings
        
        if actions is not None:
            action_embeddings = self.action_embedding(actions)
            
            # Interleave state and action embeddings
            embeddings = torch.zeros(
                batch_size, seq_len * 2, self.d_model, device=device
            )
            embeddings[:, 0::2] = state_embeddings
            embeddings[:, 1::2] = action_embeddings
            total_seq_len = seq_len * 2
        else:
            embeddings = state_embeddings
            total_seq_len = seq_len
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(total_seq_len, total_seq_len, device=device) * float('-inf'),
                diagonal=1
            )
        
        # Apply transformer
        hidden = self.transformer(embeddings, mask=attention_mask)
        
        # Get action predictions from state positions
        if actions is not None:
            state_hidden = hidden[:, 0::2]
        else:
            state_hidden = hidden
        
        action_logits = self.action_head(state_hidden)
        
        return action_logits