#!/usr/bin/env python3
"""
ABOUTME: Enhanced LSTM training script with improved features and training strategy
ABOUTME: Uses filtered high-quality data and strategic features for better performance
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import pickle

# Import enhanced components
from berghain.training.enhanced_data_preprocessor import prepare_enhanced_training_data
from berghain.training.lstm_policy import SequenceDataset


# Enhanced LSTM model with more features
class EnhancedLSTMPolicyNetwork(nn.Module):
    """
    Enhanced LSTM-based policy network with improved architecture.
    """
    
    def __init__(
        self,
        input_dim: int = 15,  # Enhanced feature count
        hidden_dim: int = 256,  # Larger hidden dimension
        lstm_layers: int = 3,   # More layers for complex patterns
        dropout: float = 0.2    # Higher dropout for regularization
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # LSTM backbone with layer normalization
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False  # Keep unidirectional for causal modeling
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # More sophisticated policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Binary: [reject_prob, accept_prob]
            nn.Softmax(dim=-1)
        )
        
        # Value head (for potential RL fine-tuning later)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using improved initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)  # Better for RNNs
                elif 'linear' in name.lower() or any(x in name for x in ['policy_head', 'value_head']):
                    if param.dim() >= 2:  # Only for 2D+ tensors
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.normal_(param, 0.0, 0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: tuple = None
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        # LSTM forward pass
        lstm_out, hidden_new = self.lstm(x, hidden)
        
        # Apply layer normalization
        lstm_out_norm = self.layer_norm(lstm_out)
        
        # Handle batch normalization in heads
        batch_size, seq_len, hidden_dim = lstm_out_norm.shape
        lstm_flat = lstm_out_norm.view(-1, hidden_dim)
        
        # Apply heads
        policy_flat = self.policy_head(lstm_flat)
        value_flat = self.value_head(lstm_flat)
        
        # Reshape back to sequences
        policy = policy_flat.view(batch_size, seq_len, 2)
        value = value_flat.view(batch_size, seq_len, 1)
        
        return policy, value, hidden_new
    
    def set_training_history(self, history_dict: dict) -> None:
        """Set training history for the model."""
        self.training_history = history_dict
    
    def get_training_history(self) -> dict:
        """Get training history from the model."""
        return getattr(self, 'training_history', {})


@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration with curriculum learning."""
    # Data parameters
    log_directory: str
    sequence_length: int = 50
    test_split: float = 0.15  # Smaller validation set (more training data)
    
    # Model parameters
    input_dim: int = 15  # Enhanced feature count
    hidden_dim: int = 256
    lstm_layers: int = 3
    dropout: float = 0.2
    
    # Training parameters
    epochs: int = 100  # More epochs
    batch_size: int = 16  # Smaller batch for better gradients
    learning_rate: float = 0.0001  # Lower learning rate
    weight_decay: float = 1e-4
    patience: int = 20  # More patience
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_epochs: List[int] = None  # Will be set to [20, 50, 100]
    
    # Advanced training
    gradient_clip: float = 1.0
    label_smoothing: float = 0.05  # Reduce overconfidence
    
    # Device and paths
    device: str = 'auto'
    save_path: str = 'models/enhanced_lstm'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        config_dict = asdict(self)
        if self.curriculum_epochs is None:
            config_dict['curriculum_epochs'] = [20, 50, 100]
        return config_dict


class EnhancedLSTMTrainer:
    """
    Enhanced trainer with curriculum learning and better optimization.
    """
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.logger.info(f"Training on device: {self.device}")
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = None
        
        # Create output directory
        Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Prepare enhanced training and validation data loaders."""
        self.logger.info("Preparing enhanced training data...")
        
        # Load and prepare data with enhanced features
        train_data, val_data = prepare_enhanced_training_data(
            log_directory=self.config.log_directory,
            test_split=self.config.test_split,
            sequence_length=self.config.sequence_length
        )
        
        if not train_data or not val_data:
            raise ValueError(f"No training data found in {self.config.log_directory}")
        
        # Create datasets
        train_dataset = SequenceDataset(train_data)
        val_dataset = SequenceDataset(val_data)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=0,  # Keep 0 for compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size * 2,  # Larger batches for validation
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        data_info = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_samples': len(train_dataset) + len(val_dataset),
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'sequence_length': self.config.sequence_length,
            'feature_dim': self.config.input_dim
        }
        
        self.logger.info(f"Enhanced training data prepared:")
        self.logger.info(f"  Training samples: {data_info['train_samples']}")
        self.logger.info(f"  Validation samples: {data_info['val_samples']}")
        self.logger.info(f"  Feature dimensions: {data_info['feature_dim']}")
        
        return train_loader, val_loader, data_info
    
    def initialize_model(self) -> None:
        """Initialize enhanced model, optimizer, scheduler, and criterion."""
        self.logger.info("Initializing enhanced model...")
        
        # Create enhanced model
        self.model = EnhancedLSTMPolicyNetwork(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            lstm_layers=self.config.lstm_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Enhanced model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize learning rate scheduler
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        
        # Initialize criterion with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        # Initialize enhanced training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': [],
            'epochs': [],
            'best_epoch': 0,
            'best_val_accuracy': 0.0,
            'total_epochs': 0
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with enhanced techniques."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            policy, _, _ = self.model(sequences)
            
            # Reshape for loss calculation
            policy_flat = policy.view(-1, 2)
            labels_flat = labels.view(-1)
            
            # Calculate loss with label smoothing
            loss = self.criterion(policy_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(policy_flat, 1)
                total_predictions += labels_flat.size(0)
                correct_predictions += (predicted == labels_flat).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                policy, _, _ = self.model(sequences)
                
                # Reshape for loss calculation
                policy_flat = policy.view(-1, 2)
                labels_flat = labels.view(-1)
                
                # Calculate loss
                loss = self.criterion(policy_flat, labels_flat)
                total_loss += loss.item()
                
                # Track accuracy
                _, predicted = torch.max(policy_flat, 1)
                total_predictions += labels_flat.size(0)
                correct_predictions += (predicted == labels_flat).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save enhanced model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.history,
            'config': self.config.to_dict(),
            'model_class': 'EnhancedLSTMPolicyNetwork'
        }
        
        # Save latest checkpoint
        checkpoint_path = f"{self.config.save_path}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = f"{self.config.save_path}_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 New best model saved: {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main enhanced training loop."""
        self.logger.info("🚀 Starting enhanced LSTM training pipeline...")
        
        # Prepare data
        train_loader, val_loader, data_info = self.prepare_data()
        
        # Initialize model
        self.initialize_model()
        
        # Training state
        best_val_accuracy = 0.0
        patience_counter = 0
        start_time = datetime.now()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = datetime.now()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.config.learning_rate
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch)
            self.history['total_epochs'] = epoch
            
            # Check for best model
            is_best = val_acc > best_val_accuracy
            if is_best:
                best_val_accuracy = val_acc
                self.history['best_val_accuracy'] = val_acc
                self.history['best_epoch'] = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % 20 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log progress
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if patience_counter >= self.config.patience:
                self.logger.info(f"🛑 Early stopping triggered after {epoch} epochs")
                break
        
        # Training completed
        total_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"✅ Enhanced training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"🏆 Best validation accuracy: {best_val_accuracy:.2f}% at epoch {self.history['best_epoch']}")
        
        # Set training history in model
        self.model.set_training_history(self.history)
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False)
        
        return self.history


def main():
    """Main enhanced training script."""
    parser = argparse.ArgumentParser(description="Train Enhanced LSTM Policy Network")
    
    # Data parameters
    parser.add_argument('--log-directory', type=str, default='game_logs_filtered',
                       help='Directory containing filtered high-quality game logs')
    parser.add_argument('--sequence-length', type=int, default=50,
                       help='Sequence length for LSTM training')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--lstm-layers', type=int, default=3,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--save-path', type=str, default='models/enhanced_lstm',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create enhanced config
    config = EnhancedTrainingConfig(
        log_directory=args.log_directory,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
        save_path=args.save_path
    )
    
    # Initialize trainer
    trainer = EnhancedLSTMTrainer(config)
    
    # Train the enhanced model
    history = trainer.train()
    
    print(f"🎉 Enhanced LSTM training completed!")
    print(f"📊 Best validation accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"📁 Model saved to: {config.save_path}_best.pth")


if __name__ == "__main__":
    main()