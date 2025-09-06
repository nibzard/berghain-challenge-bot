#!/usr/bin/env python3
"""
ABOUTME: Comprehensive supervised learning trainer for LSTM policy network
ABOUTME: Includes training history tracking, visualization, and model checkpointing
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

from berghain.training.lstm_policy import LSTMPolicyNetwork, SequenceDataset
from berghain.training.data_preprocessor import prepare_training_data


@dataclass
class TrainingHistory:
    """Training history data structure."""
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    epochs: List[int]
    best_epoch: int
    best_val_accuracy: float
    total_epochs: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingHistory':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data parameters
    log_directory: str
    sequence_length: int = 50
    test_split: float = 0.2
    
    # Model parameters
    input_dim: int = 8
    hidden_dim: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10  # Early stopping patience
    
    # Device and paths
    device: str = 'auto'  # 'auto', 'cpu', or 'cuda'
    save_path: str = 'models/lstm_supervised'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SupervisedLSTMTrainer:
    """
    Comprehensive trainer for LSTM policy network using supervised learning.
    
    Features:
    - Training history tracking
    - Early stopping with patience
    - Model checkpointing
    - Real-time visualization
    - Comprehensive logging
    """
    
    def __init__(self, config: TrainingConfig):
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
        self.criterion = None
        self.history = None
        
        # Create output directory
        Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create console handler if not already exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """
        Prepare training and validation data loaders.
        
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
            data_info: Information about the dataset
        """
        self.logger.info("Preparing training data...")
        
        # Load and prepare data
        train_data, val_data = prepare_training_data(
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
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        data_info = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_samples': len(train_dataset) + len(val_dataset),
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'sequence_length': self.config.sequence_length
        }
        
        self.logger.info(f"Training samples: {data_info['train_samples']}")
        self.logger.info(f"Validation samples: {data_info['val_samples']}")
        
        return train_loader, val_loader, data_info
    
    def initialize_model(self) -> None:
        """Initialize model, optimizer, and criterion."""
        self.logger.info("Initializing model...")
        
        # Create model
        self.model = LSTMPolicyNetwork(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            lstm_layers=self.config.lstm_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize training history
        self.history = TrainingHistory(
            train_loss=[],
            val_loss=[],
            train_accuracy=[],
            val_accuracy=[],
            epochs=[],
            best_epoch=0,
            best_val_accuracy=0.0,
            total_epochs=0
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            policy, _, _ = self.model(sequences)
            
            # Reshape for loss calculation
            policy_flat = policy.view(-1, 2)  # (batch * seq_len, 2)
            labels_flat = labels.view(-1)  # (batch * seq_len,)
            
            # Calculate loss
            loss = self.criterion(policy_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(policy_flat, 1)
            total_predictions += labels_flat.size(0)
            correct_predictions += (predicted == labels_flat).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
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
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.history.to_dict(),
            'config': self.config.to_dict()
        }
        
        # Save latest checkpoint
        checkpoint_path = f"{self.config.save_path}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = f"{self.config.save_path}_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: {best_path}")
    
    def plot_training_history(self, save_path: str = None) -> None:
        """Plot and optionally save training history."""
        if not self.history.epochs:
            self.logger.warning("No training history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.history.epochs
        
        # Plot losses
        ax1.plot(epochs, self.history.train_loss, label='Train Loss', color='blue')
        ax1.plot(epochs, self.history.val_loss, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(epochs, self.history.train_accuracy, label='Train Accuracy', color='green')
        ax2.plot(epochs, self.history.val_accuracy, label='Validation Accuracy', color='orange')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot loss difference
        loss_diff = [v - t for t, v in zip(self.history.train_loss, self.history.val_loss)]
        ax3.plot(epochs, loss_diff, label='Val - Train Loss', color='purple')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Overfitting Indicator (Val Loss - Train Loss)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot learning rate (if available)
        ax4.text(0.1, 0.8, f"Best Epoch: {self.history.best_epoch}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f"Best Val Accuracy: {self.history.best_val_accuracy:.2f}%", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Final Train Loss: {self.history.train_loss[-1]:.4f}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f"Final Val Loss: {self.history.val_loss[-1]:.4f}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f"Total Epochs: {self.history.total_epochs}", transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to: {save_path}")
        
        plt.show()
    
    def train(self, plot_during_training: bool = True) -> TrainingHistory:
        """
        Main training loop.
        
        Args:
            plot_during_training: Whether to plot progress during training
            
        Returns:
            Training history
        """
        self.logger.info("Starting LSTM training pipeline...")
        
        # Prepare data
        train_loader, val_loader, data_info = self.prepare_data()
        
        # Initialize model
        self.initialize_model()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = datetime.now()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = datetime.now()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_accuracy.append(train_acc)
            self.history.val_accuracy.append(val_acc)
            self.history.epochs.append(epoch)
            self.history.total_epochs = epoch
            
            # Check for best model
            is_best = val_acc > self.history.best_val_accuracy
            if is_best:
                self.history.best_val_accuracy = val_acc
                self.history.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log progress
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Plot progress periodically
            if plot_during_training and epoch % 10 == 0:
                try:
                    self.plot_training_history()
                except Exception as e:
                    self.logger.warning(f"Could not plot during training: {e}")
        
        # Training completed
        total_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation accuracy: {self.history.best_val_accuracy:.2f}% at epoch {self.history.best_epoch}")
        
        # Save final model and history
        self.save_checkpoint(epoch, is_best=False)
        self._save_training_history()
        
        # Final plot
        try:
            plot_path = f"{self.config.save_path}_training_history.png"
            self.plot_training_history(save_path=plot_path)
        except Exception as e:
            self.logger.warning(f"Could not save final plot: {e}")
        
        return self.history
    
    def _save_training_history(self) -> None:
        """Save training history to file."""
        history_path = f"{self.config.save_path}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history.to_dict(), f, indent=2)
        self.logger.info(f"Training history saved to: {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = TrainingHistory.from_dict(checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    def test_model(self, test_samples: int = 100) -> Dict[str, float]:
        """Test the trained model on a small sample."""
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a checkpoint first.")
        
        # Prepare test data
        train_loader, val_loader, _ = self.prepare_data()
        
        self.model.eval()
        test_results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'samples_tested': 0
        }
        
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for batch_idx, (sequences, labels) in enumerate(val_loader):
                if total_samples >= test_samples:
                    break
                
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                policy, _, _ = self.model(sequences)
                policy_flat = policy.view(-1, 2)
                labels_flat = labels.view(-1)
                
                _, predicted = torch.max(policy_flat, 1)
                
                # Calculate metrics
                total_correct += (predicted == labels_flat).sum().item()
                total_samples += labels_flat.size(0)
                
                # Calculate precision/recall for accept class (class 1)
                true_positives += ((predicted == 1) & (labels_flat == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels_flat == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels_flat == 1)).sum().item()
        
        test_results['accuracy'] = 100.0 * total_correct / total_samples
        test_results['samples_tested'] = total_samples
        
        if true_positives + false_positives > 0:
            test_results['precision'] = 100.0 * true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives > 0:
            test_results['recall'] = 100.0 * true_positives / (true_positives + false_negatives)
        
        self.logger.info(f"Test Results: Accuracy: {test_results['accuracy']:.2f}%, "
                        f"Precision: {test_results['precision']:.2f}%, "
                        f"Recall: {test_results['recall']:.2f}%")
        
        return test_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train LSTM Policy Network with Supervised Learning")
    
    # Data parameters
    parser.add_argument('--log-directory', type=str, default='game_logs',
                       help='Directory containing game log JSON files')
    parser.add_argument('--sequence-length', type=int, default=50,
                       help='Sequence length for LSTM training')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for validation')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='LSTM hidden dimension')
    parser.add_argument('--lstm-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--save-path', type=str, default='models/lstm_supervised',
                       help='Path to save trained model')
    
    # Execution options
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting during training')
    parser.add_argument('--test-only', type=str,
                       help='Path to checkpoint for testing only')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        log_directory=args.log_directory,
        sequence_length=args.sequence_length,
        test_split=args.test_split,
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
    trainer = SupervisedLSTMTrainer(config)
    
    if args.test_only:
        # Test mode only
        trainer.load_checkpoint(args.test_only)
        results = trainer.test_model()
        print(f"Test results: {results}")
    else:
        # Training mode
        history = trainer.train(plot_during_training=not args.no_plot)
        
        # Test the trained model
        results = trainer.test_model()
        print(f"Final test results: {results}")


if __name__ == "__main__":
    main()