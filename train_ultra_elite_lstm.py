#!/usr/bin/env python3
"""
ABOUTME: Enhanced training script for ultra-elite LSTM using 35 strategic features and advanced architectures
ABOUTME: Combines ultra-elite games, augmented data, attention mechanisms, and quality-weighted loss functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from berghain.training.ultra_elite_preprocessor import UltraElitePreprocessor, prepare_ultra_elite_training_data
from berghain.training.enhanced_lstm_models import (
    UltraEliteLSTMNetwork, EnsembleLSTMNetwork, QualityWeightedLoss, 
    ConstraintAwareLoss, create_model, create_loss_function
)
from berghain.training.data_augmentation import create_augmented_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraEliteDataset(Dataset):
    """PyTorch Dataset for ultra-elite game training with quality scores."""
    
    def __init__(self, sequences: List[torch.Tensor], labels: List[torch.Tensor], quality_scores: List[float] = None):
        self.sequences = sequences
        self.labels = labels
        self.quality_scores = quality_scores or [1.0] * len(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.quality_scores[idx]


def load_combined_elite_games(
    ultra_elite_dir: str = "ultra_elite_games",
    augmented_dir: str = "augmented_elite_games",
    use_augmented: bool = True
) -> List[Dict[str, Any]]:
    """Load both original ultra-elite and augmented games."""
    games = []
    
    # Load original ultra-elite games
    logger.info("Loading original ultra-elite games...")
    processor = UltraElitePreprocessor()
    ultra_games = processor.load_ultra_elite_games(ultra_elite_dir)
    
    for game in ultra_games:
        game['is_augmented'] = False
        game['quality_score'] = 1.0  # Perfect quality for originals
        games.append(game)
    
    # Load augmented games if requested
    if use_augmented and Path(augmented_dir).exists():
        logger.info("Loading augmented elite games...")
        augmented_games = processor.load_ultra_elite_games(augmented_dir)
        
        for game in augmented_games:
            game['is_augmented'] = True
            # Slightly lower quality score for augmented games
            game['quality_score'] = 0.8
            games.append(game)
        
        logger.info(f"Loaded {len(augmented_games)} augmented games")
    
    logger.info(f"Total games for training: {len(games)} (original: {len(ultra_games)})")
    return games


def calculate_game_quality_scores(games: List[Dict[str, Any]]) -> List[float]:
    """Calculate quality scores for games based on performance metrics."""
    quality_scores = []
    
    # Find best performance for normalization
    best_rejection_count = min(game.get('rejected_count', float('inf')) for game in games)
    
    for game in games:
        rejected_count = game.get('rejected_count', 1000)
        admitted_count = game.get('admitted_count', 0)
        is_augmented = game.get('is_augmented', False)
        
        # Base quality from rejection efficiency (lower rejections = higher quality)
        rejection_quality = best_rejection_count / max(rejected_count, 1)
        
        # Admission efficiency (higher admissions within constraints = better)
        admission_quality = min(admitted_count / 1000, 1.0)
        
        # Combined quality score
        base_quality = (rejection_quality * 0.7 + admission_quality * 0.3)
        
        # Penalty for augmented games
        if is_augmented:
            base_quality *= 0.9
        
        quality_scores.append(base_quality)
    
    # Normalize to [0.5, 1.0] range
    min_score = min(quality_scores)
    max_score = max(quality_scores)
    
    if max_score > min_score:
        normalized_scores = [0.5 + 0.5 * (score - min_score) / (max_score - min_score) 
                           for score in quality_scores]
    else:
        normalized_scores = [1.0] * len(quality_scores)
    
    logger.info(f"Quality scores range: {min(normalized_scores):.3f} - {max(normalized_scores):.3f}")
    return normalized_scores


def prepare_enhanced_dataset(
    ultra_elite_dir: str = "ultra_elite_games",
    augmented_dir: str = "augmented_elite_games", 
    sequence_length: int = 150,
    test_split: float = 0.2,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Prepare enhanced dataset with quality scores."""
    
    # Load all games
    games = load_combined_elite_games(ultra_elite_dir, augmented_dir, use_augmentation)
    
    if len(games) == 0:
        raise ValueError("No games loaded for training")
    
    # Calculate quality scores
    quality_scores = calculate_game_quality_scores(games)
    
    # Prepare data using ultra-elite preprocessor
    logger.info("Preprocessing with 35 strategic features...")
    processor = UltraElitePreprocessor(sequence_length=sequence_length)
    sequences, labels = processor.prepare_ultra_elite_dataset(games)
    
    if len(sequences) == 0:
        raise ValueError("No training sequences generated")
    
    # Assign quality scores to sequences based on games
    sequence_quality_scores = []
    game_seq_count = 0
    
    for i, game in enumerate(games):
        try:
            # Get number of sequences this game produced
            game_features, game_labels = processor.extract_enhanced_features_and_labels(game)
            game_sequences, _ = processor.create_sequences(game_features, game_labels)
            num_sequences = len(game_sequences)
            
            # Assign game's quality score to all its sequences
            for _ in range(num_sequences):
                sequence_quality_scores.append(quality_scores[i])
                
            game_seq_count += num_sequences
            
        except Exception as e:
            logger.warning(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
    
    # Ensure we have quality scores for all sequences
    while len(sequence_quality_scores) < len(sequences):
        sequence_quality_scores.append(0.8)  # Default quality score
    sequence_quality_scores = sequence_quality_scores[:len(sequences)]
    
    # Split data by games to prevent leakage
    game_indices = {}
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        game_hash = hash(tuple(seq[0, :5].tolist()))
        if game_hash not in game_indices:
            game_indices[game_hash] = []
        game_indices[game_hash].append(i)
    
    # Split games
    unique_games = list(game_indices.keys())
    split_idx = int(len(unique_games) * (1 - test_split))
    train_games = unique_games[:split_idx]
    val_games = unique_games[split_idx:]
    
    # Get sequence indices
    train_indices = []
    val_indices = []
    
    for game in train_games:
        train_indices.extend(game_indices[game])
    for game in val_games:
        val_indices.extend(game_indices[game])
    
    # Create final datasets
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_quality = [sequence_quality_scores[i] for i in train_indices]
    
    val_sequences = [sequences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_quality = [sequence_quality_scores[i] for i in val_indices]
    
    # Create datasets and dataloaders
    train_dataset = UltraEliteDataset(train_sequences, train_labels, train_quality)
    val_dataset = UltraEliteDataset(val_sequences, val_labels, val_quality)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
    
    # Dataset statistics
    dataset_stats = {
        'total_games': len(games),
        'original_games': len([g for g in games if not g.get('is_augmented', False)]),
        'augmented_games': len([g for g in games if g.get('is_augmented', False)]),
        'total_sequences': len(sequences),
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'feature_dim': processor.feature_dim,
        'sequence_length': sequence_length
    }
    
    logger.info(f"Dataset prepared:")
    logger.info(f"  Total games: {dataset_stats['total_games']} (original: {dataset_stats['original_games']}, augmented: {dataset_stats['augmented_games']})")
    logger.info(f"  Training sequences: {dataset_stats['train_sequences']}")
    logger.info(f"  Validation sequences: {dataset_stats['val_sequences']}")
    logger.info(f"  Features per timestep: {dataset_stats['feature_dim']}")
    
    return train_loader, val_loader, dataset_stats


def train_ultra_elite_lstm(
    ultra_elite_dir: str = "ultra_elite_games",
    augmented_dir: str = "augmented_elite_games",
    model_save_path: str = "models/ultra_elite_lstm_best.pth",
    model_type: str = "ultra_elite",
    loss_type: str = "quality_weighted",
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.0001,
    sequence_length: int = 150,
    test_split: float = 0.2,
    use_augmentation: bool = True
):
    """Train ultra-elite LSTM with enhanced features and architectures."""
    
    logger.info("🚀 Starting Ultra-Elite LSTM Training")
    
    # Create augmented dataset if it doesn't exist
    if use_augmentation and not Path(augmented_dir).exists():
        logger.info("Creating augmented dataset...")
        create_augmented_dataset(ultra_elite_dir, augmented_dir, augmentation_factor=3)
    
    # Prepare enhanced dataset
    train_loader, val_loader, dataset_stats = prepare_enhanced_dataset(
        ultra_elite_dir, augmented_dir, sequence_length, test_split, use_augmentation
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "ensemble":
        model = EnsembleLSTMNetwork(
            input_dim=35, 
            num_models=3,
            hidden_dims=[512, 384, 256],
            dropout=0.2
        )
    else:
        model = UltraEliteLSTMNetwork(
            input_dim=35,
            hidden_dim=512,
            num_layers=3,
            num_heads=8,
            dropout=0.2,
            use_attention=True,
            use_positional_encoding=True
        )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model_type}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Training on: {device}")
    
    # Create loss function
    if loss_type == "quality_weighted":
        criterion = QualityWeightedLoss(base_weight=1.0, quality_scaling=1.5)
    elif loss_type == "constraint_aware":
        criterion = ConstraintAwareLoss(violation_penalty=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=8,
        min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'epochs': [], 'lr': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    logger.info(f"Starting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y, batch_quality in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_quality = batch_quality.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate loss
            if isinstance(criterion, QualityWeightedLoss):
                loss = criterion(outputs, batch_y, batch_quality)
            else:
                outputs_flat = outputs.view(-1, 2)
                targets_flat = batch_y.view(-1)
                loss = criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.view(-1, 2).data, 1)
            train_total += batch_y.view(-1).size(0)
            train_correct += (predicted == batch_y.view(-1)).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                outputs_flat = outputs.view(-1, 2)
                targets_flat = batch_y.view(-1)
                
                loss = nn.CrossEntropyLoss()(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs_flat.data, 1)
                val_total += targets_flat.size(0)
                val_correct += (predicted == targets_flat).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        history['lr'].append(current_lr)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model based on validation accuracy AND loss
        improved = False
        if val_acc > best_val_acc or (val_acc >= best_val_acc - 0.1 and val_loss < best_val_loss):
            best_val_acc = max(val_acc, best_val_acc)
            best_val_loss = min(val_loss, best_val_loss)
            patience_counter = 0
            improved = True
            
            # Save model
            Path(model_save_path).parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'model_config': {
                    'model_type': model_type,
                    'input_dim': 35,
                    'hidden_dim': 512 if model_type != "ensemble" else "variable",
                    'num_layers': 3,
                    'sequence_length': sequence_length
                },
                'dataset_stats': dataset_stats,
                'history': history
            }, model_save_path)
        else:
            patience_counter += 1
        
        # Log progress
        status_emoji = "🏆" if improved else "📈"
        logger.info(f"{status_emoji} Epoch {epoch+1:3d}/{epochs} | "
                   f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                   f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | "
                   f"LR={current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏰ Early stopping triggered after {epoch + 1} epochs (patience={patience})")
            break
    
    # Save final training artifacts
    history_path = model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create enhanced training plots
    create_training_plots(history, model_save_path)
    
    logger.info(f"✅ Ultra-Elite LSTM training completed!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Model saved: {model_save_path}")
    logger.info(f"   Training history: {history_path}")
    
    return model, history


def create_training_plots(history: Dict, model_save_path: str):
    """Create comprehensive training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epochs']
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Ultra-Elite LSTM Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Ultra-Elite LSTM Training Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'lr' in history:
        axes[1, 0].semilogy(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate (log scale)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference plot
    loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
    axes[1, 1].plot(epochs, loss_diff, 'm-', linewidth=2)
    axes[1, 1].set_title('Train-Val Loss Difference', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = model_save_path.replace('.pth', '_training_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   Training plots: {plot_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ultra-Elite LSTM with enhanced features")
    parser.add_argument('--ultra-elite-dir', default='ultra_elite_games', help='Ultra-elite games directory')
    parser.add_argument('--augmented-dir', default='augmented_elite_games', help='Augmented games directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=150, help='Sequence length')
    parser.add_argument('--model-path', default='models/ultra_elite_lstm_best.pth', help='Model save path')
    parser.add_argument('--model-type', choices=['ultra_elite', 'ensemble'], default='ultra_elite', help='Model architecture')
    parser.add_argument('--loss-type', choices=['quality_weighted', 'constraint_aware', 'cross_entropy'], default='quality_weighted', help='Loss function')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    train_ultra_elite_lstm(
        ultra_elite_dir=args.ultra_elite_dir,
        augmented_dir=args.augmented_dir,
        model_save_path=args.model_path,
        model_type=args.model_type,
        loss_type=args.loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        use_augmentation=not args.no_augmentation
    )