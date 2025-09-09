"""
ABOUTME: Dual-head transformer architecture for Berghain Challenge optimization
ABOUTME: Implements separate heads for constraint satisfaction and efficiency optimization as per MASTERPLAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class DualHeadOutput:
    """Output from dual-head transformer containing both constraint and efficiency predictions."""
    constraint_logits: torch.Tensor  # Focus on meeting requirements
    efficiency_logits: torch.Tensor  # Focus on minimizing rejections  
    combined_logits: torch.Tensor    # Final decision logits
    attention_weights: torch.Tensor  # Attention visualization
    constraint_confidence: torch.Tensor  # Confidence in constraint satisfaction
    efficiency_confidence: torch.Tensor  # Confidence in efficiency optimization
    head_weights: torch.Tensor       # Dynamic weighting between heads

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism optimized for Berghain decision making."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        return output, attention_weights.mean(dim=1)  # Average over heads for visualization

class TransformerBlock(nn.Module):
    """Transformer block with residual connections and layer normalization."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class ConstraintHead(nn.Module):
    """Specialized head for constraint satisfaction decisions."""
    
    def __init__(self, d_model: int, constraint_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.constraint_projection = nn.Linear(d_model, constraint_dim)
        self.constraint_attention = nn.MultiheadAttention(constraint_dim, num_heads=4, dropout=dropout)
        
        self.decision_net = nn.Sequential(
            nn.Linear(constraint_dim, constraint_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(constraint_dim // 2, constraint_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(constraint_dim // 4, 2)  # admit/reject
        )
        
        self.confidence_net = nn.Sequential(
            nn.Linear(constraint_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project to constraint-specific space
        constraint_features = self.constraint_projection(x)
        
        # Self-attention over constraint features
        attn_output, _ = self.constraint_attention(
            constraint_features.transpose(0, 1), 
            constraint_features.transpose(0, 1),
            constraint_features.transpose(0, 1)
        )
        constraint_context = attn_output.transpose(0, 1)
        
        # Use only the last timestep for decision (current person)
        current_features = constraint_context[:, -1, :]
        
        # Generate decision logits and confidence
        decision_logits = self.decision_net(current_features)
        confidence = self.confidence_net(current_features)
        
        return decision_logits, confidence

class EfficiencyHead(nn.Module):
    """Specialized head for efficiency optimization (minimizing rejections)."""
    
    def __init__(self, d_model: int, efficiency_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.efficiency_projection = nn.Linear(d_model, efficiency_dim)
        self.temporal_conv = nn.Conv1d(efficiency_dim, efficiency_dim, kernel_size=3, padding=1)
        
        self.efficiency_net = nn.Sequential(
            nn.Linear(efficiency_dim, efficiency_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(efficiency_dim // 2, efficiency_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(efficiency_dim // 4, 2)  # admit/reject
        )
        
        self.confidence_net = nn.Sequential(
            nn.Linear(efficiency_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project to efficiency-specific space
        efficiency_features = self.efficiency_projection(x)
        
        # Apply temporal convolution to capture efficiency patterns
        conv_input = efficiency_features.transpose(1, 2)  # (batch, features, seq)
        conv_output = F.relu(self.temporal_conv(conv_input))
        efficiency_context = conv_output.transpose(1, 2)  # (batch, seq, features)
        
        # Use only the last timestep for decision
        current_features = efficiency_context[:, -1, :]
        
        # Generate decision logits and confidence
        decision_logits = self.efficiency_net(current_features)
        confidence = self.confidence_net(current_features)
        
        return decision_logits, confidence

class GameStateEncoder(nn.Module):
    """Encode game state information into transformer-ready format."""
    
    def __init__(self, state_dim: int, d_model: int, max_seq_length: int = 1000):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(max_seq_length, d_model)
        )
        self.dropout = nn.Dropout(0.1)
        
        # Initialize positional encoding
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Embed state features
        embedded = self.state_embedding(x)
        
        # Add positional encoding
        embedded = embedded + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        return self.dropout(embedded)

class DualHeadTransformer(nn.Module):
    """
    Dual-head transformer for Berghain Challenge as specified in MASTERPLAN.
    
    Head 1: Constraint satisfaction (focus on meeting requirements)
    Head 2: Efficiency optimization (focus on minimizing rejections)
    Combines both heads for final decision with dynamic weighting.
    """
    
    def __init__(
        self,
        state_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        constraint_dim: int = 64,
        efficiency_dim: int = 64,
        max_seq_length: int = 1000,
        dropout: float = 0.1,
        use_dynamic_weighting: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # Input encoding
        self.state_encoder = GameStateEncoder(state_dim, d_model, max_seq_length)
        
        # Transformer backbone
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Specialized heads
        self.constraint_head = ConstraintHead(d_model, constraint_dim, dropout)
        self.efficiency_head = EfficiencyHead(d_model, efficiency_dim, dropout)
        
        # Dynamic head weighting network
        if use_dynamic_weighting:
            self.head_weighting_net = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2),  # weights for [constraint_head, efficiency_head]
                nn.Softmax(dim=-1)
            )
        else:
            # Static weighting (equal importance)
            self.register_buffer('static_weights', torch.tensor([0.5, 0.5]))
        
        # Final combination layer
        self.combination_layer = nn.Sequential(
            nn.Linear(4, 8),  # 2 heads * 2 logits each = 4 inputs
            nn.ReLU(),
            nn.Linear(8, 2)   # final admit/reject decision
        )
        
    def forward(
        self, 
        state_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> DualHeadOutput:
        """
        Forward pass through dual-head transformer.
        
        Args:
            state_sequence: (batch_size, seq_len, state_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            DualHeadOutput with specialized predictions
        """
        batch_size, seq_len, _ = state_sequence.shape
        
        # Encode input sequence
        x = self.state_encoder(state_sequence)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn = layer(x, mask)
            attention_weights.append(attn)
        
        # Get predictions from specialized heads
        constraint_logits, constraint_confidence = self.constraint_head(x)
        efficiency_logits, efficiency_confidence = self.efficiency_head(x)
        
        # Dynamic head weighting based on current context
        if self.use_dynamic_weighting:
            current_context = x[:, -1, :]  # Last timestep
            head_weights = self.head_weighting_net(current_context)
        else:
            head_weights = self.static_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Combine head outputs
        combined_input = torch.cat([
            constraint_logits * head_weights[:, 0:1],
            efficiency_logits * head_weights[:, 1:2]
        ], dim=-1)
        
        combined_logits = self.combination_layer(combined_input)
        
        # Aggregate attention weights
        avg_attention = torch.stack(attention_weights).mean(dim=0) if return_attention else None
        
        return DualHeadOutput(
            constraint_logits=constraint_logits,
            efficiency_logits=efficiency_logits,
            combined_logits=combined_logits,
            attention_weights=avg_attention,
            constraint_confidence=constraint_confidence,
            efficiency_confidence=efficiency_confidence,
            head_weights=head_weights
        )
    
    def get_decision_explanation(self, output: DualHeadOutput) -> Dict[str, Any]:
        """
        Generate human-readable explanation of the decision process.
        
        Args:
            output: DualHeadOutput from forward pass
            
        Returns:
            Dictionary with decision explanation
        """
        constraint_probs = F.softmax(output.constraint_logits, dim=-1)
        efficiency_probs = F.softmax(output.efficiency_logits, dim=-1)
        combined_probs = F.softmax(output.combined_logits, dim=-1)
        
        explanation = {
            'final_decision': 'admit' if combined_probs[0, 1] > 0.5 else 'reject',
            'final_confidence': combined_probs[0].max().item(),
            'constraint_head': {
                'recommendation': 'admit' if constraint_probs[0, 1] > 0.5 else 'reject',
                'confidence': constraint_probs[0].max().item(),
                'admit_prob': constraint_probs[0, 1].item(),
                'internal_confidence': output.constraint_confidence[0].item()
            },
            'efficiency_head': {
                'recommendation': 'admit' if efficiency_probs[0, 1] > 0.5 else 'reject', 
                'confidence': efficiency_probs[0].max().item(),
                'admit_prob': efficiency_probs[0, 1].item(),
                'internal_confidence': output.efficiency_confidence[0].item()
            },
            'head_weights': {
                'constraint_weight': output.head_weights[0, 0].item(),
                'efficiency_weight': output.head_weights[0, 1].item()
            },
            'reasoning': self._generate_reasoning(output)
        }
        
        return explanation
    
    def _generate_reasoning(self, output: DualHeadOutput) -> str:
        """Generate human-readable reasoning for the decision."""
        constraint_admit = F.softmax(output.constraint_logits, dim=-1)[0, 1] > 0.5
        efficiency_admit = F.softmax(output.efficiency_logits, dim=-1)[0, 1] > 0.5
        final_admit = F.softmax(output.combined_logits, dim=-1)[0, 1] > 0.5
        
        constraint_weight = output.head_weights[0, 0].item()
        efficiency_weight = output.head_weights[0, 1].item()
        
        if constraint_admit and efficiency_admit:
            if final_admit:
                return f"dual_head_agreement_admit (c:{constraint_weight:.2f}, e:{efficiency_weight:.2f})"
            else:
                return f"dual_head_conflict_reject (c:{constraint_weight:.2f}, e:{efficiency_weight:.2f})"
        elif not constraint_admit and not efficiency_admit:
            return f"dual_head_agreement_reject (c:{constraint_weight:.2f}, e:{efficiency_weight:.2f})"
        else:
            primary_head = "constraint" if constraint_weight > efficiency_weight else "efficiency"
            action = "admit" if final_admit else "reject"
            return f"dual_head_conflict_{action}_{primary_head}_dominant (c:{constraint_weight:.2f}, e:{efficiency_weight:.2f})"

class DualHeadTrainer:
    """Specialized trainer for dual-head transformer with multi-objective optimization."""
    
    def __init__(
        self,
        model: DualHeadTransformer,
        constraint_loss_weight: float = 0.4,
        efficiency_loss_weight: float = 0.4,
        combined_loss_weight: float = 0.2,
        confidence_loss_weight: float = 0.1
    ):
        self.model = model
        self.constraint_loss_weight = constraint_loss_weight
        self.efficiency_loss_weight = efficiency_loss_weight
        self.combined_loss_weight = combined_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        
    def compute_loss(
        self,
        output: DualHeadOutput,
        targets: torch.Tensor,
        constraint_targets: Optional[torch.Tensor] = None,
        efficiency_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss for dual-head training.
        
        Args:
            output: DualHeadOutput from model
            targets: Ground truth decisions (0/1)
            constraint_targets: Optional constraint-specific targets
            efficiency_targets: Optional efficiency-specific targets
            
        Returns:
            Dictionary of loss components
        """
        # Use main targets for all heads if specialized targets not provided
        if constraint_targets is None:
            constraint_targets = targets
        if efficiency_targets is None:
            efficiency_targets = targets
            
        # Compute individual losses
        constraint_loss = F.cross_entropy(output.constraint_logits, constraint_targets)
        efficiency_loss = F.cross_entropy(output.efficiency_logits, efficiency_targets)
        combined_loss = F.cross_entropy(output.combined_logits, targets)
        
        # Confidence regularization (encourage high confidence when correct)
        constraint_correct = (output.constraint_logits.argmax(-1) == constraint_targets).float()
        efficiency_correct = (output.efficiency_logits.argmax(-1) == efficiency_targets).float()
        
        constraint_confidence_loss = F.mse_loss(
            output.constraint_confidence.squeeze(), constraint_correct
        )
        efficiency_confidence_loss = F.mse_loss(
            output.efficiency_confidence.squeeze(), efficiency_correct
        )
        
        confidence_loss = (constraint_confidence_loss + efficiency_confidence_loss) / 2
        
        # Total loss
        total_loss = (
            self.constraint_loss_weight * constraint_loss +
            self.efficiency_loss_weight * efficiency_loss +
            self.combined_loss_weight * combined_loss +
            self.confidence_loss_weight * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'constraint_loss': constraint_loss,
            'efficiency_loss': efficiency_loss,
            'combined_loss': combined_loss,
            'confidence_loss': confidence_loss
        }