#!/usr/bin/env python3
"""
ABOUTME: Training script for dual-head transformer on ultra-elite Berghain game data
ABOUTME: Implements multi-objective training for constraint satisfaction and efficiency optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pickle
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.dual_head_transformer import DualHeadTransformer, DualHeadTrainer, DualHeadOutput
from data.ultra_elite_preprocessor import UltraEliteDataset, DecisionSequence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DualHeadTrainingManager:
    """Manages the complete training process for dual-head transformer."""
    
    def __init__(
        self,
        data_path: str = "berghain_transformer/ultra_elite_training_data.pkl",
        model_config: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        output_dir: str = "trained_models"
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default model configuration
        self.model_config = model_config or {
            'state_dim': 14,  # From GameFeatures
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'constraint_dim': 64,
            'efficiency_dim': 64,
            'max_seq_length': 50,
            'dropout': 0.1,
            'use_dynamic_weighting': True
        }
        
        # Default training configuration
        self.training_config = training_config or {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'weight_decay': 1e-5,
            'gradient_clip_norm': 1.0,
            'validation_split': 0.15,
            'early_stopping_patience': 10,
            'scheduler_patience': 5,
            'scheduler_factor': 0.7,
            'constraint_loss_weight': 0.4,
            'efficiency_loss_weight': 0.4,
            'combined_loss_weight': 0.2,
            'confidence_loss_weight': 0.1
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Training state
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        
    def load_data(self) -> None:
        """Load and prepare training data."""
        logger.info("📊 Loading ultra-elite training data...")
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        logger.info(f"📈 Loaded {len(sequences)} training sequences")
        
        # Create dataset
        dataset = UltraEliteDataset(sequences)
        
        # Split into train/validation
        val_size = int(len(dataset) * self.training_config['validation_split'])
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"✅ Training set: {len(train_dataset)} sequences")
        logger.info(f"✅ Validation set: {len(val_dataset)} sequences")
        
    def create_model(self) -> None:
        """Create and initialize the dual-head transformer model."""
        logger.info("🏗️  Creating dual-head transformer model...")
        
        self.model = DualHeadTransformer(**self.model_config).to(self.device)
        
        # Create specialized trainer
        self.trainer = DualHeadTrainer(
            self.model,
            constraint_loss_weight=self.training_config['constraint_loss_weight'],
            efficiency_loss_weight=self.training_config['efficiency_loss_weight'],
            combined_loss_weight=self.training_config['combined_loss_weight'],
            confidence_loss_weight=self.training_config['confidence_loss_weight']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        
    def create_optimizer(self) -> None:
        """Create optimizer and learning rate scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.training_config['scheduler_factor'],
            patience=self.training_config['scheduler_patience']
        )
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'constraint_loss': 0.0,
            'efficiency_loss': 0.0,
            'combined_loss': 0.0,
            'confidence_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            states = batch['states'].to(self.device)  # (batch, seq_len, feature_dim)
            actions = batch['actions'].to(self.device)  # (batch, seq_len)
            constraint_labels = batch['constraint_labels'].to(self.device)
            efficiency_labels = batch['efficiency_labels'].to(self.device)
            
            # Forward pass
            output = self.model(states)
            
            # Calculate losses
            losses = self.trainer.compute_loss(
                output,
                actions[:, -1],  # Only predict last decision
                constraint_labels[:, -1],
                efficiency_labels[:, -1]
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config['gradient_clip_norm']
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                           f"Loss: {losses['total_loss'].item():.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'constraint_loss': 0.0,
            'efficiency_loss': 0.0,
            'combined_loss': 0.0,
            'confidence_loss': 0.0
        }
        
        # Accuracy tracking
        correct_predictions = 0
        constraint_correct = 0
        efficiency_correct = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                constraint_labels = batch['constraint_labels'].to(self.device)
                efficiency_labels = batch['efficiency_labels'].to(self.device)
                
                # Forward pass
                output = self.model(states)
                
                # Calculate losses
                losses = self.trainer.compute_loss(
                    output,
                    actions[:, -1],
                    constraint_labels[:, -1],
                    efficiency_labels[:, -1]
                )
                
                # Accumulate losses
                for key, value in losses.items():
                    val_losses[key] += value.item()
                
                # Calculate accuracies
                predicted_actions = output.combined_logits.argmax(-1)
                constraint_predictions = output.constraint_logits.argmax(-1)
                efficiency_predictions = output.efficiency_logits.argmax(-1)
                
                correct_predictions += (predicted_actions == actions[:, -1]).sum().item()
                constraint_correct += (constraint_predictions == constraint_labels[:, -1]).sum().item()
                efficiency_correct += (efficiency_predictions == efficiency_labels[:, -1]).sum().item()
                total_predictions += actions.size(0)
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Calculate accuracies
        accuracies = {
            'combined_accuracy': correct_predictions / total_predictions,
            'constraint_accuracy': constraint_correct / total_predictions,
            'efficiency_accuracy': efficiency_correct / total_predictions
        }
        
        return val_losses, accuracies
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("🚀 Starting dual-head transformer training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.training_config['num_epochs']):
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_losses, val_accuracies = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total_loss'])
            
            # Record history
            epoch_history = {
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: "
                       f"Train Loss: {train_losses['total_loss']:.4f}, "
                       f"Val Loss: {val_losses['total_loss']:.4f}, "
                       f"Val Acc: {val_accuracies['combined_accuracy']:.3f}")
            
            # Early stopping and model saving
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_losses['total_loss'])
                logger.info(f"💾 Saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_losses['total_loss'])
            
            # Early stopping
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"🛑 Early stopping at epoch {epoch} (patience: {patience_counter})")
                break
        
        logger.info("✅ Training completed!")
        
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, self.output_dir / filename)
        
    def save_training_artifacts(self) -> None:
        """Save training history and configuration."""
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save configuration
        config = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'device': str(self.device),
            'data_path': str(self.data_path)
        }
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create training plots
        self.create_training_plots()
        
    def create_training_plots(self) -> None:
        """Create training visualization plots."""
        if not self.training_history:
            return
            
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_losses']['total_loss'] for h in self.training_history]
        val_losses = [h['val_losses']['total_loss'] for h in self.training_history]
        val_accuracies = [h['val_accuracies']['combined_accuracy'] for h in self.training_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training and validation loss
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(epochs, val_accuracies, label='Combined Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Loss components
        if len(self.training_history) > 0:
            constraint_losses = [h['val_losses']['constraint_loss'] for h in self.training_history]
            efficiency_losses = [h['val_losses']['efficiency_loss'] for h in self.training_history]
            
            ax3.plot(epochs, constraint_losses, label='Constraint Loss', color='purple')
            ax3.plot(epochs, efficiency_losses, label='Efficiency Loss', color='orange')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Head-Specific Losses')
            ax3.legend()
            ax3.grid(True)
        
        # Head accuracies
        if len(self.training_history) > 0:
            constraint_accs = [h['val_accuracies']['constraint_accuracy'] for h in self.training_history]
            efficiency_accs = [h['val_accuracies']['efficiency_accuracy'] for h in self.training_history]
            
            ax4.plot(epochs, constraint_accs, label='Constraint Accuracy', color='purple')
            ax4.plot(epochs, efficiency_accs, label='Efficiency Accuracy', color='orange')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Head-Specific Accuracies')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training plots saved to {self.output_dir}/training_history.png")

def main():
    """Main training execution."""
    # Create training manager
    trainer_manager = DualHeadTrainingManager()
    
    # Load data
    trainer_manager.load_data()
    
    # Create model
    trainer_manager.create_model()
    
    # Create optimizer
    trainer_manager.create_optimizer()
    
    # Train model
    trainer_manager.train()
    
    # Save artifacts
    trainer_manager.save_training_artifacts()
    
    print("\n🎯 DUAL-HEAD TRANSFORMER TRAINING COMPLETE")
    print("=" * 50)
    print(f"📊 Training sequences processed: {len(trainer_manager.train_loader.dataset):,}")
    print(f"💾 Model saved to: {trainer_manager.output_dir}/best_model.pt")
    print(f"📈 Training history: {trainer_manager.output_dir}/training_history.json")
    
    # Final validation performance
    if trainer_manager.training_history:
        best_epoch = min(trainer_manager.training_history, key=lambda x: x['val_losses']['total_loss'])
        print(f"🏆 Best epoch: {best_epoch['epoch']}")
        print(f"📉 Best val loss: {best_epoch['val_losses']['total_loss']:.4f}")
        print(f"🎯 Best val accuracy: {best_epoch['val_accuracies']['combined_accuracy']:.3f}")

if __name__ == "__main__":
    main()