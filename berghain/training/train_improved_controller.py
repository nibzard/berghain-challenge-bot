# ABOUTME: Improved training script for strategy controller with performance-weighted loss
# ABOUTME: Uses elite training data and focuses on learning from the best performers

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from .strategy_controller import StrategyControllerTransformer, create_strategy_controller

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImprovedTrainingConfig:
    """Configuration for improved training"""
    num_epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 5e-5  # Lower LR for stability
    weight_decay: float = 0.01
    patience: int = 8
    device: str = 'cpu'
    
    # Performance-weighted loss settings
    use_performance_weighting: bool = True
    weight_smoothing: float = 0.1  # Smooth extreme weights
    
    # Elite data focus
    min_performance_weight: float = 2.0  # Only use high-performing examples
    strategy_balance_factor: float = 0.8  # Balance strategy distribution

class ImprovedStrategyControllerTrainer:
    """Improved trainer with performance weighting"""
    
    def __init__(self, config: ImprovedTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        self.model = create_strategy_controller()
        self.model.to(self.device)
        
        # Set up optimizer with lower learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=4
        )
        
        # Strategy vocabulary
        self.strategy_vocab = ['rbcr2', 'ultra_elite_lstm', 'constraint_focused_lstm',
                              'perfect', 'ultimate3', 'ultimate3h', 'dual_deficit', 'rbcr']
    
    def load_elite_training_data(self, data_path: str) -> List[Dict]:
        """Load and filter elite training data"""
        logger.info(f"Loading elite training data from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        training_examples = data.get('training_examples', [])
        metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(training_examples)} examples")
        logger.info(f"Performance range: {metadata.get('performance_stats', {}).get('best_rejection_count', 0)}-{metadata.get('performance_stats', {}).get('worst_rejection_count', 0)} rejections")
        
        # Filter by performance weight
        if self.config.use_performance_weighting:
            filtered_examples = [ex for ex in training_examples 
                               if ex.get('performance_weight', 1.0) >= self.config.min_performance_weight]
            logger.info(f"Filtered to {len(filtered_examples)} high-performance examples")
            training_examples = filtered_examples
        
        # Balance strategy distribution
        training_examples = self._balance_strategy_distribution(training_examples)
        
        return training_examples
    
    def _balance_strategy_distribution(self, examples: List[Dict]) -> List[Dict]:
        """Balance strategy distribution to prevent bias"""
        strategy_groups = defaultdict(list)
        for ex in examples:
            strategy = ex.get('strategy_decision', 'rbcr2')
            strategy_groups[strategy].append(ex)
        
        # Find target count (weighted average)
        total_count = len(examples)
        target_per_strategy = total_count // len(strategy_groups)
        
        balanced_examples = []
        for strategy, group in strategy_groups.items():
            # If strategy has too many examples, sample
            if len(group) > target_per_strategy * 1.5:
                # Prefer higher-weight examples
                group.sort(key=lambda x: x.get('performance_weight', 1.0), reverse=True)
                selected = group[:int(target_per_strategy * 1.2)]
            else:
                selected = group
            
            balanced_examples.extend(selected)
        
        logger.info(f"Balanced dataset: {len(examples)} → {len(balanced_examples)} examples")
        
        # Show final distribution
        final_dist = defaultdict(int)
        for ex in balanced_examples:
            final_dist[ex.get('strategy_decision', 'rbcr2')] += 1
        logger.info(f"Final distribution: {dict(final_dist)}")
        
        return balanced_examples
    
    def create_performance_weighted_loss(self) -> callable:
        """Create performance-weighted loss function"""
        def weighted_loss(strategy_logits, strategy_targets, param_outputs, param_targets, weights=None):
            # Base losses
            strategy_loss = nn.CrossEntropyLoss(reduction='none')(strategy_logits, strategy_targets)
            param_loss = nn.MSELoss(reduction='none')(param_outputs, param_targets).mean(dim=-1)
            
            if weights is not None:
                # Smooth extreme weights to prevent instability
                smoothed_weights = torch.clamp(weights, min=1.0, max=4.0)
                smoothed_weights = smoothed_weights / smoothed_weights.mean()  # Normalize
                
                # Apply weights
                strategy_loss = strategy_loss * smoothed_weights
                param_loss = param_loss * smoothed_weights
            
            # Strategy selection is more important than parameter tuning for elite games
            total_loss = 0.8 * strategy_loss.mean() + 0.2 * param_loss.mean()
            
            return total_loss, strategy_loss.mean(), param_loss.mean()
        
        return weighted_loss
    
    def prepare_batch(self, examples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch for training"""
        input_sequences = []
        strategy_targets = []
        param_targets = []
        weights = []
        
        for example in examples:
            # State sequence
            state_seq = example.get('state_sequence', [{}])
            input_sequences.append(state_seq)
            
            # Strategy target
            strategy_target = example.get('strategy_target', 0)
            strategy_targets.append(strategy_target)
            
            # Parameter target (simplified)
            param_target = example.get('parameter_target', [0.0, 0.0, 0.0, 0.0])
            param_targets.append(param_target)
            
            # Performance weight
            weight = example.get('performance_weight', 1.0)
            weights.append(weight)
        
        # Convert to tensors
        strategy_targets = torch.tensor(strategy_targets, dtype=torch.long).to(self.device)
        param_targets = torch.tensor(param_targets, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        return input_sequences, strategy_targets, param_targets, weights
    
    def train_epoch(self, training_examples: List[Dict]) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        epoch_losses = {'total': 0.0, 'strategy': 0.0, 'param': 0.0}
        num_batches = 0
        
        # Create performance-weighted loss function
        loss_fn = self.create_performance_weighted_loss()
        
        # Shuffle examples
        np.random.shuffle(training_examples)
        
        # Process in batches
        for i in range(0, len(training_examples), self.config.batch_size):
            batch_examples = training_examples[i:i + self.config.batch_size]
            
            try:
                # Prepare batch
                input_sequences, strategy_targets, param_targets, weights = self.prepare_batch(batch_examples)
                
                # Forward pass
                self.optimizer.zero_grad()
                strategy_logits, param_outputs = self.model(input_sequences)
                
                # Compute loss
                total_loss, strategy_loss, param_loss = loss_fn(
                    strategy_logits, strategy_targets, param_targets, weights
                )
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Accumulate losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['strategy'] += strategy_loss.item()
                epoch_losses['param'] += param_loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, validation_examples: List[Dict]) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {'total': 0.0, 'strategy': 0.0, 'param': 0.0}
        num_batches = 0
        
        loss_fn = self.create_performance_weighted_loss()
        
        with torch.no_grad():
            for i in range(0, len(validation_examples), self.config.batch_size):
                batch_examples = validation_examples[i:i + self.config.batch_size]
                
                try:
                    input_sequences, strategy_targets, param_targets, weights = self.prepare_batch(batch_examples)
                    
                    strategy_logits, param_outputs = self.model(input_sequences)
                    total_loss, strategy_loss, param_loss = loss_fn(
                        strategy_logits, strategy_targets, param_targets, weights
                    )
                    
                    val_losses['total'] += total_loss.item()
                    val_losses['strategy'] += strategy_loss.item()
                    val_losses['param'] += param_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    continue
        
        # Average losses
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, training_data_path: str = "training_data/simple_elite_training.json"):
        """Train the improved model"""
        logger.info("Starting improved strategy controller training")
        
        # Load data
        training_examples = self.load_elite_training_data(training_data_path)
        
        if len(training_examples) < 50:
            raise ValueError(f"Insufficient training data: {len(training_examples)} examples")
        
        # Split train/validation
        val_split = 0.15
        val_size = int(len(training_examples) * val_split)
        np.random.shuffle(training_examples)
        
        val_examples = training_examples[:val_size]
        train_examples = training_examples[val_size:]
        
        logger.info(f"Training set: {len(train_examples)} examples")
        logger.info(f"Validation set: {len(val_examples)} examples")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_losses = self.train_epoch(train_examples)
            
            # Validate
            val_losses = self.validate(val_examples) if val_examples else train_losses
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"  Train Loss: {train_losses['total']:.4f} (strategy: {train_losses['strategy']:.4f}, param: {train_losses['param']:.4f})")
            logger.info(f"  Val Loss:   {val_losses['total']:.4f} (strategy: {val_losses['strategy']:.4f}, param: {val_losses['param']:.4f})")
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'lr': self.scheduler.optimizer.param_groups[0]['lr']
            })
            
            # Early stopping
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # Save best model
                self.save_model("models/strategy_controller/improved_strategy_controller.pt", 
                              training_history)
                logger.info(f"  New best model saved! Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return training_history
    
    def save_model(self, path: str, training_history: List[Dict]):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'strategy_vocab': self.strategy_vocab,
            'config': {
                'training_config': self.config.__dict__,
                'model_params': {
                    'state_dim': 64,
                    'n_strategies': len(self.strategy_vocab),
                    'n_layers': 6,
                    'n_heads': 8,
                    'hidden_dim': 256
                }
            },
            'training_history': training_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

def main():
    """Main training function"""
    config = ImprovedTrainingConfig(
        num_epochs=25,
        batch_size=32,
        learning_rate=5e-5,
        patience=8,
        use_performance_weighting=True,
        min_performance_weight=1.5  # Include more examples
    )
    
    trainer = ImprovedStrategyControllerTrainer(config)
    
    try:
        history = trainer.train()
        print("🎉 Improved controller training completed successfully!")
        print(f"   Best validation loss: {min(h['val_loss'] for h in history):.4f}")
        print(f"   Model saved to: models/strategy_controller/improved_strategy_controller.pt")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()