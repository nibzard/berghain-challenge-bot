# ABOUTME: Enhanced LSTM architectures with attention, bidirectional processing, and ensemble capabilities
# ABOUTME: Advanced models for ultra-elite game strategy learning with sequence-to-sequence processing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for game phase awareness."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class AttentionMechanism(nn.Module):
    """Attention mechanism for focusing on critical decisions."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.output_proj(attended), attention_weights


class UltraEliteLSTMNetwork(nn.Module):
    """
    Ultra-enhanced LSTM network with:
    - Bidirectional LSTM processing
    - Multi-head attention mechanism
    - Positional encoding for game phase
    - Residual connections
    - Dynamic feature scaling
    """
    
    def __init__(
        self, 
        input_dim: int = 35, 
        hidden_dim: int = 512, 
        num_layers: int = 3, 
        num_heads: int = 8,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection and normalization
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionMechanism(hidden_dim * 2, num_heads)  # *2 for bidirectional
        
        # Feature importance learning
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Sigmoid()
        )
        
        # Enhanced output layers with residual connections
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 2)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 4)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(3)
        ])
        
        self.relu = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif len(param.shape) >= 2 and 'weight' in name:
                torch.nn.init.kaiming_normal_(param.data, nonlinearity='relu')
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Input projection and normalization
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq, hidden*2)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out)
            lstm_out = lstm_out + attended_out  # Residual connection
        
        # Feature importance gating
        gate = self.feature_gate(lstm_out)
        lstm_out = lstm_out * gate
        
        # Apply output layers with residual connections
        out = lstm_out
        for i, (linear, norm, dropout) in enumerate(zip(self.output_layers[:-1], self.layer_norms, self.dropouts)):
            residual = out
            out = linear(out)
            out = norm(out)
            out = self.relu(out)
            out = dropout(out)
            
            # Add residual connection if dimensions match
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
        
        # Final output layer
        out = self.output_layers[-1](out)
        
        if return_attention:
            return out, attention_weights
        return out


class EnsembleLSTMNetwork(nn.Module):
    """Ensemble of multiple LSTM networks for robust predictions."""
    
    def __init__(
        self, 
        input_dim: int = 35,
        num_models: int = 3,
        hidden_dims: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 384, 256]
        
        self.num_models = num_models
        self.models = nn.ModuleList()
        
        # Create ensemble of models with different architectures
        for i in range(num_models):
            hidden_dim = hidden_dims[i % len(hidden_dims)]
            model = UltraEliteLSTMNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=3,
                num_heads=8,
                dropout=dropout,
                use_attention=i < 2,  # Use attention for first 2 models
                use_positional_encoding=i != 1  # Skip pos encoding for middle model
            )
            self.models.append(model)
        
        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x, return_individual=False):
        batch_size, seq_len, _ = x.shape
        outputs = []
        
        # Get predictions from all models
        for model in self.models:
            out = model(x)
            outputs.append(out)
        
        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=-1)  # (batch, seq, 2, num_models)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = torch.sum(stacked_outputs * weights, dim=-1)
        
        if return_individual:
            return ensemble_output, outputs
        return ensemble_output


class QualityWeightedLoss(nn.Module):
    """Custom loss function weighted by game outcome quality."""
    
    def __init__(self, base_weight: float = 1.0, quality_scaling: float = 2.0):
        super().__init__()
        self.base_weight = base_weight
        self.quality_scaling = quality_scaling
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets, quality_scores=None):
        """
        Args:
            predictions: Model predictions (batch, seq, 2)
            targets: Ground truth labels (batch, seq)
            quality_scores: Quality scores for each game (batch,)
        """
        batch_size, seq_len, _ = predictions.shape
        
        # Reshape for loss calculation
        pred_flat = predictions.view(-1, 2)
        target_flat = targets.view(-1)
        
        # Base cross-entropy loss
        losses = self.ce_loss(pred_flat, target_flat)
        losses = losses.view(batch_size, seq_len)
        
        # Apply quality weighting if provided
        if quality_scores is not None:
            quality_scores = quality_scores.unsqueeze(1)  # (batch, 1)
            # Higher quality games get more weight
            quality_weights = 1.0 + (quality_scores * self.quality_scaling)
            losses = losses * quality_weights
        
        return losses.mean()


class ConstraintAwareLoss(nn.Module):
    """Loss function that penalizes constraint violations."""
    
    def __init__(self, violation_penalty: float = 2.0):
        super().__init__()
        self.violation_penalty = violation_penalty
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets, constraint_violations=None):
        """
        Args:
            predictions: Model predictions (batch, seq, 2)
            targets: Ground truth labels (batch, seq)
            constraint_violations: Binary mask for constraint violations (batch, seq)
        """
        batch_size, seq_len, _ = predictions.shape
        
        pred_flat = predictions.view(-1, 2)
        target_flat = targets.view(-1)
        
        losses = self.ce_loss(pred_flat, target_flat)
        losses = losses.view(batch_size, seq_len)
        
        # Penalize decisions that lead to constraint violations
        if constraint_violations is not None:
            penalty_mask = constraint_violations.float()
            losses = losses * (1.0 + penalty_mask * self.violation_penalty)
        
        return losses.mean()


def create_model(
    model_type: str = "ultra_elite",
    input_dim: int = 35,
    **kwargs
) -> nn.Module:
    """Factory function to create different model types."""
    
    if model_type == "ultra_elite":
        return UltraEliteLSTMNetwork(input_dim=input_dim, **kwargs)
    elif model_type == "ensemble":
        return EnsembleLSTMNetwork(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_loss_function(
    loss_type: str = "quality_weighted",
    **kwargs
) -> nn.Module:
    """Factory function to create different loss functions."""
    
    if loss_type == "quality_weighted":
        return QualityWeightedLoss(**kwargs)
    elif loss_type == "constraint_aware":
        return ConstraintAwareLoss(**kwargs)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class ModelTrainer:
    """Enhanced trainer with advanced features."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.train_history = {
            'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []
        }
    
    def train_epoch(self, train_loader, quality_scores=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features)
            
            # Calculate loss
            if isinstance(self.loss_fn, (QualityWeightedLoss, ConstraintAwareLoss)):
                batch_quality = quality_scores[batch_idx] if quality_scores else None
                loss = self.loss_fn(outputs, labels, batch_quality)
            else:
                outputs_flat = outputs.view(-1, 2)
                labels_flat = labels.view(-1)
                loss = self.loss_fn(outputs_flat, labels_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.argmax(outputs.view(-1, 2), dim=1)
            total_correct += (predicted == labels.view(-1)).sum().item()
            total_samples += labels.view(-1).size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                
                outputs_flat = outputs.view(-1, 2)
                labels_flat = labels.view(-1)
                loss = nn.CrossEntropyLoss()(outputs_flat, labels_flat)
                
                total_loss += loss.item()
                
                predicted = torch.argmax(outputs_flat, dim=1)
                total_correct += (predicted == labels_flat).sum().item()
                total_samples += labels_flat.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int, patience: int = 10):
        """Full training loop."""
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.train_history