#!/usr/bin/env python3
"""
ABOUTME: Train enhanced LSTM model on elite games collected by the Elite Game Hunter
ABOUTME: Uses the high-quality elite game dataset for superior performance
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

from berghain.training.enhanced_data_preprocessor import EnhancedGameDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EliteGameDataset(Dataset):
    """PyTorch Dataset for elite game training data."""
    
    def __init__(self, sequences: List[torch.Tensor], labels: List[torch.Tensor]):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class EnhancedLSTMPolicyNetwork(nn.Module):
    """Enhanced LSTM network for learning elite game strategies."""
    
    def __init__(self, input_dim=15, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Enhanced LSTM with more capacity
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-layer output with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, 2)  # Binary classification
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 4)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif len(param.shape) >= 2:  # Linear layer weights
                torch.nn.init.kaiming_normal_(param.data, nonlinearity='relu')
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use the output from all timesteps, not just the last
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        
        # Apply fully connected layers to each timestep
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.layer_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out  # Shape: (batch_size, sequence_length, 2)


def load_elite_games(elite_games_dir: str = "elite_games") -> List[Dict[str, Any]]:
    """Load all elite games from the directory."""
    elite_games = []
    elite_dir = Path(elite_games_dir)
    
    if not elite_dir.exists():
        raise ValueError(f"Elite games directory not found: {elite_games_dir}")
    
    for json_file in elite_dir.glob("elite_*.json"):
        try:
            with open(json_file, 'r') as f:
                game_data = json.load(f)
            
            # Convert decisions format from elite games to preprocessor format
            converted_decisions = []
            for decision in game_data['decisions']:
                converted_decision = {
                    'attributes': decision['person']['attributes'],  # Flatten attributes to top level
                    'decision': decision['accepted'],  # Convert 'accepted' to 'decision'
                    'reasoning': decision.get('reasoning', ''),
                    'timestamp': decision.get('timestamp', '')
                }
                converted_decisions.append(converted_decision)
            
            # Convert to the format expected by EnhancedGameDataPreprocessor
            converted_game = {
                'game_id': game_data['game_id'],
                'success': game_data['success'],
                'rejected_count': game_data['rejected_count'],
                'admitted_count': game_data['admitted_count'],
                'constraints': game_data['constraints'],
                'decisions': converted_decisions
            }
            
            elite_games.append(converted_game)
            
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(elite_games)} elite games")
    return elite_games


def train_elite_lstm(
    elite_games_dir: str = "elite_games",
    model_save_path: str = "models/elite_lstm_best.pth",
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.0005,
    sequence_length: int = 100,
    test_split: float = 0.2
):
    """Train enhanced LSTM on elite games."""
    
    logger.info("🎯 Starting Elite LSTM Training")
    
    # Load elite games
    logger.info("Loading elite games...")
    elite_games = load_elite_games(elite_games_dir)
    
    if len(elite_games) < 10:
        raise ValueError(f"Not enough elite games for training: {len(elite_games)}. Need at least 10.")
    
    # Prepare data using enhanced preprocessor
    logger.info("Preprocessing elite game data...")
    preprocessor = EnhancedGameDataPreprocessor(sequence_length=sequence_length)
    sequences, labels = preprocessor.prepare_dataset(elite_games)
    
    if len(sequences) == 0:
        raise ValueError("No training sequences generated from elite games")
    
    # Split data
    split_idx = int(len(sequences) * (1 - test_split))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    logger.info(f"Training sequences: {len(train_sequences)}")
    logger.info(f"Validation sequences: {len(val_sequences)}")
    
    # Create datasets
    train_dataset = EliteGameDataset(train_sequences, train_labels)
    val_dataset = EliteGameDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedLSTMPolicyNetwork(input_dim=15, hidden_dim=256, num_layers=3, dropout=0.3)
    model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training on: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs...")
    
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
            outputs = model(batch_x)  # Shape: (batch_size, seq_len, 2)
            
            # Reshape for loss calculation
            outputs_flat = outputs.view(-1, 2)  # (batch_size * seq_len, 2)
            targets_flat = batch_y.view(-1)     # (batch_size * seq_len,)
            
            loss = criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                outputs_flat = outputs.view(-1, 2)
                targets_flat = batch_y.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs_flat.data, 1)
                val_total += targets_flat.size(0)
                val_correct += (predicted == targets_flat).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            Path(model_save_path).parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'model_config': {
                    'input_dim': 15,
                    'hidden_dim': 256,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            }, model_save_path)
            
            logger.info(f"🏆 New best model saved: {model_save_path}")
        else:
            patience_counter += 1
        
        # Log progress
        logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_save_path = model_save_path.replace('_best.pth', '_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'val_acc': val_acc,
        'train_acc': train_acc,
        'model_config': {
            'input_dim': 15,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.3
        }
    }, final_save_path)
    
    # Save training history
    history_path = model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create training plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Elite LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Elite LSTM Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = model_save_path.replace('.pth', '_training_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Training completed!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"   Model saved: {model_save_path}")
    logger.info(f"   Training history: {history_path}")
    logger.info(f"   Training plot: {plot_path}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train enhanced LSTM on elite games")
    parser.add_argument('--elite-dir', default='elite_games', help='Directory with elite games')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--model-path', default='models/elite_lstm_best.pth', help='Model save path')
    
    args = parser.parse_args()
    
    train_elite_lstm(
        elite_games_dir=args.elite_dir,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length
    )