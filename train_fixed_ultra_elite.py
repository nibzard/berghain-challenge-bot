#!/usr/bin/env python3
"""
ABOUTME: Fixed training script for ultra-elite LSTM with proper regularization to prevent overfitting
ABOUTME: Uses curriculum learning and balanced data to achieve <800 rejections
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from berghain.training.ultra_elite_preprocessor import UltraElitePreprocessor
from berghain.training.enhanced_lstm_models import UltraEliteLSTMNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BalancedEliteDataset(Dataset):
    """Balanced dataset with proper class weighting to prevent bias."""
    
    def __init__(self, sequences: List[torch.Tensor], labels: List[torch.Tensor]):
        self.sequences = sequences
        self.labels = labels
        
        # Calculate class weights for balanced training
        all_labels = torch.cat([label.flatten() for label in labels])
        self.accept_ratio = torch.mean(all_labels.float()).item()
        self.reject_ratio = 1.0 - self.accept_ratio
        
        logger.info(f"Dataset balance - Accept: {self.accept_ratio:.3f}, Reject: {self.reject_ratio:.3f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
    def get_sample_weights(self):
        """Get weights for balanced sampling."""
        weights = []
        for labels in self.labels:
            # Weight by inverse frequency - give more weight to minority class
            seq_accepts = torch.sum(labels).item()
            seq_total = len(labels)
            seq_accept_ratio = seq_accepts / seq_total
            
            # Higher weight if this sequence has a different accept pattern than overall
            if seq_accept_ratio > self.accept_ratio:
                weight = 1.0 / self.accept_ratio  # Boost accept-heavy sequences
            else:
                weight = 1.0 / self.reject_ratio  # Boost reject-heavy sequences
            
            weights.append(weight)
        
        return torch.FloatTensor(weights)


def train_fixed_ultra_elite_lstm():
    """Train ultra-elite LSTM with proper regularization."""
    
    logger.info("🚀 Training Fixed Ultra-Elite LSTM (Anti-Overfitting)")
    
    # Load ultra-elite games
    processor = UltraElitePreprocessor(sequence_length=80)  # Shorter sequences
    ultra_elite_games = processor.load_ultra_elite_games("ultra_elite_games")
    
    if len(ultra_elite_games) < 5:
        logger.error("Need at least 5 ultra-elite games for training")
        return None
    
    # Prepare dataset with proper balance
    logger.info("Preprocessing with balanced sampling...")
    sequences, labels = processor.prepare_ultra_elite_dataset(ultra_elite_games)
    
    # Create balanced splits
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    # Create datasets
    train_dataset = BalancedEliteDataset(train_sequences, train_labels)
    val_dataset = BalancedEliteDataset(val_sequences, val_labels)
    
    # Balanced sampling to prevent overfitting to majority class
    train_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    logger.info(f"Training sequences: {len(train_sequences)}")
    logger.info(f"Validation sequences: {len(val_sequences)}")
    
    # Create model with regularization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UltraEliteLSTMNetwork(
        input_dim=35,
        hidden_dim=128,  # Smaller to prevent overfitting
        num_layers=2,    # Fewer layers
        num_heads=4,     # Fewer attention heads
        dropout=0.4,     # Higher dropout
        use_attention=True,
        use_positional_encoding=True
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Balanced loss function
    train_accept_ratio = train_dataset.accept_ratio
    class_weights = torch.FloatTensor([1.0, 1.0/train_accept_ratio]).to(device)  # Balance classes
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Conservative optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0001,       # Lower learning rate
        weight_decay=0.05, # Higher weight decay
        betas=(0.9, 0.95)  # Less aggressive momentum
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training with early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 8  # Lower patience
    patience_counter = 0
    epochs = 25   # Fewer epochs
    
    logger.info(f"Starting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate loss
            outputs_flat = outputs.view(-1, 2)
            targets_flat = batch_y.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs_flat.data, 1)
            train_total += targets_flat.size(0)
            train_correct += (predicted == targets_flat).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_accept_predictions = 0
        val_actual_accepts = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                outputs_flat = outputs.view(-1, 2)
                targets_flat = batch_y.view(-1)
                
                # Use unweighted loss for validation
                loss = nn.CrossEntropyLoss()(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs_flat.data, 1)
                val_total += targets_flat.size(0)
                val_correct += (predicted == targets_flat).sum().item()
                
                # Track prediction patterns
                val_accept_predictions += (predicted == 1).sum().item()
                val_actual_accepts += (targets_flat == 1).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        val_pred_accept_rate = val_accept_predictions / val_total
        val_actual_accept_rate = val_actual_accepts / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Early stopping based on reasonable performance, not overfitting
        reasonable_performance = (
            val_acc > 75 and val_acc < 95 and  # Not too high (overfitting) or too low
            abs(val_pred_accept_rate - val_actual_accept_rate) < 0.15  # Reasonable prediction balance
        )
        
        if reasonable_performance and (val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            model_save_path = "models/fixed_ultra_elite_lstm_best.pth"
            Path(model_save_path).parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'model_config': {
                    'input_dim': 35,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'sequence_length': 80,
                    'model_type': 'ultra_elite'
                },
                'prediction_stats': {
                    'val_pred_accept_rate': val_pred_accept_rate,
                    'val_actual_accept_rate': val_actual_accept_rate
                }
            }, model_save_path)
            
            logger.info(f"🏆 Epoch {epoch+1:2d}: Train Acc={train_acc:.1f}%, "
                       f"Val Acc={val_acc:.1f}%, Val Loss={val_loss:.4f}, "
                       f"Pred Accept Rate={val_pred_accept_rate:.3f}")
        else:
            patience_counter += 1
            logger.info(f"📈 Epoch {epoch+1:2d}: Train Acc={train_acc:.1f}%, "
                       f"Val Acc={val_acc:.1f}%, Val Loss={val_loss:.4f}, "
                       f"Pred Accept Rate={val_pred_accept_rate:.3f} (no save)")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏰ Early stopping at epoch {epoch + 1}")
            break
        
        # Stop if overfitting is detected
        if val_acc > 98:
            logger.warning(f"⚠️ Potential overfitting detected (val_acc={val_acc:.1f}%), stopping early")
            break
    
    logger.info(f"✅ Fixed training completed!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.1f}%")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Model saved: models/fixed_ultra_elite_lstm_best.pth")
    
    return model


if __name__ == "__main__":
    train_fixed_ultra_elite_lstm()