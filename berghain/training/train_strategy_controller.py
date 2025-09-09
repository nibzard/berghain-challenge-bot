# ABOUTME: Training pipeline for strategy controller transformer using extracted game data
# ABOUTME: Implements reinforcement learning with behavioral cloning to learn optimal strategy orchestration

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import random
from collections import defaultdict, deque

from .strategy_controller import (
    StrategyControllerTransformer, 
    StrategyControllerTrainer, 
    create_strategy_controller
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for strategy controller"""
    # Model parameters
    model_name: str = "strategy_controller_v1"
    device: str = "cpu"
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 100
    
    # Data parameters
    train_split: float = 0.8
    validation_split: float = 0.2
    sequence_length: int = 10
    
    # Loss weights
    strategy_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    risk_loss_weight: float = 0.5
    parameter_loss_weight: float = 0.3
    
    # Regularization
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Logging
    log_interval: int = 10
    save_interval: int = 25
    use_wandb: bool = False
    
    # Output paths
    model_output_dir: str = "models/strategy_controller"
    checkpoint_dir: str = "checkpoints/strategy_controller"


class StrategyControllerDataset:
    """Dataset for training strategy controller"""
    
    def __init__(self, training_data: List[Dict], config: TrainingConfig, strategy_vocab: List[str]):
        self.training_data = training_data
        self.config = config
        self.strategy_vocab = strategy_vocab
        self.strategy_to_idx = {strategy: idx for idx, strategy in enumerate(strategy_vocab)}
        
        # Process and validate data
        self.processed_data = self._process_training_data()
        self.train_data, self.val_data = self._split_data()
        
        logger.info(f"Dataset created: {len(self.train_data)} training, {len(self.val_data)} validation examples")
    
    def _process_training_data(self) -> List[Dict]:
        """Process raw training data into model-ready format"""
        processed = []
        
        for example in self.training_data:
            # Skip examples with unknown strategies
            if example['strategy_decision'] not in self.strategy_to_idx:
                continue
            
            # Convert state sequence to tensor format
            state_sequence = []
            for state_dict in example['state_sequence']:
                processed_state = self._process_state_dict(state_dict, example)
                state_sequence.append(processed_state)
            
            # Pad or truncate sequence to desired length
            if len(state_sequence) < self.config.sequence_length:
                # Pad with last state
                last_state = state_sequence[-1] if state_sequence else self._create_default_state()
                while len(state_sequence) < self.config.sequence_length:
                    state_sequence.append(last_state)
            else:
                # Take last N states
                state_sequence = state_sequence[-self.config.sequence_length:]
            
            # Create target values
            target_strategy = self.strategy_to_idx[example['strategy_decision']]
            target_confidence = float(example['outcome_quality'])
            target_risk = max(0.0, 1.0 - target_confidence)
            
            # Process parameter adjustments
            target_parameters = {}
            param_adjustments = example.get('parameter_adjustments', {})
            
            # Normalize parameters to [0, 1] range for training
            if 'ultra_rare_threshold' in param_adjustments:
                target_parameters['ultra_rare_threshold'] = (param_adjustments['ultra_rare_threshold'] - 0.01) / 0.09
            if 'deficit_panic_threshold' in param_adjustments:
                target_parameters['deficit_panic_threshold'] = (param_adjustments['deficit_panic_threshold'] - 50) / 250
            if 'phase1_multi_attr_only' in param_adjustments:
                target_parameters['phase1_multi_attr_only'] = float(param_adjustments['phase1_multi_attr_only'])
            if 'adaptation_rate' in param_adjustments:
                target_parameters['adaptation_rate'] = (param_adjustments['adaptation_rate'] - 0.01) / 0.49
            
            processed_example = {
                'state_sequence': state_sequence,
                'target_strategy': target_strategy,
                'target_confidence': target_confidence,
                'target_risk': target_risk,
                'target_parameters': target_parameters,
                'game_id': example['game_id'],
                'original_strategy': example['original_strategy'],
                'final_success': example['final_success'],
                'final_rejections': example.get('final_rejections', 0)
            }
            
            processed.append(processed_example)
        
        logger.info(f"Processed {len(processed)} valid training examples")
        return processed
    
    def _process_state_dict(self, state_dict: Dict, example: Dict) -> Dict[str, float]:
        """Process a single state dictionary"""
        # Extract and normalize all features
        processed = {
            # Person attributes (binary features)
            'has_young': float(state_dict.get('person_attributes', {}).get('young', False)),
            'has_well_dressed': float(state_dict.get('person_attributes', {}).get('well_dressed', False)),
            
            # Constraint progress
            'young_progress': float(state_dict.get('young_progress', 0.0)),
            'well_dressed_progress': float(state_dict.get('well_dressed_progress', 0.0)),
            'min_constraint_progress': float(state_dict.get('young_progress', 0.0)) if 'young_progress' in state_dict else 0.0,
            'max_constraint_progress': float(state_dict.get('well_dressed_progress', 0.0)) if 'well_dressed_progress' in state_dict else 0.0,
            
            # Capacity and rejection ratios
            'capacity_ratio': float(state_dict.get('capacity_ratio', 0.0)),
            'rejection_ratio': float(state_dict.get('rejection_ratio', 0.0)),
            
            # Risk assessments
            'constraint_risk': float(state_dict.get('constraint_risk', 0.0)),
            'time_pressure': float(state_dict.get('rejection_ratio', 0.0)),  # Use rejection ratio as proxy
            
            # Game phase (one-hot)
            'phase_early': 1.0 if state_dict.get('game_phase', 'mid') == 'early' else 0.0,
            'phase_mid': 1.0 if state_dict.get('game_phase', 'mid') == 'mid' else 0.0,
            'phase_late': 1.0 if state_dict.get('game_phase', 'mid') == 'late' else 0.0,
            
            # Strategy performance indicators
            'recent_acceptance_rate': float(state_dict.get('recent_acceptance_rate', 0.5)),
            'strategy_performance': float(example.get('outcome_quality', 0.5)),
            'decisions_count': float(state_dict.get('decisions_since_switch', 0)) / 100.0,  # Normalize
        }
        
        # Calculate min/max constraint progress
        processed['min_constraint_progress'] = min(processed['young_progress'], processed['well_dressed_progress'])
        processed['max_constraint_progress'] = max(processed['young_progress'], processed['well_dressed_progress'])
        
        return processed
    
    def _create_default_state(self) -> Dict[str, float]:
        """Create default state for padding"""
        return {
            'has_young': 0.0, 'has_well_dressed': 0.0,
            'young_progress': 0.0, 'well_dressed_progress': 0.0,
            'min_constraint_progress': 0.0, 'max_constraint_progress': 0.0,
            'capacity_ratio': 0.0, 'rejection_ratio': 0.0,
            'constraint_risk': 0.0, 'time_pressure': 0.0,
            'phase_early': 1.0, 'phase_mid': 0.0, 'phase_late': 0.0,
            'recent_acceptance_rate': 0.5, 'strategy_performance': 0.5,
            'decisions_count': 0.0
        }
    
    def _split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train/validation sets"""
        random.shuffle(self.processed_data)
        
        split_idx = int(len(self.processed_data) * self.config.train_split)
        train_data = self.processed_data[:split_idx]
        val_data = self.processed_data[split_idx:]
        
        return train_data, val_data
    
    def get_batch(self, data: List[Dict], batch_size: int) -> Dict[str, torch.Tensor]:
        """Get a batch of training data"""
        batch_indices = random.sample(range(len(data)), min(batch_size, len(data)))
        batch_data = [data[i] for i in batch_indices]
        
        # Convert to tensor format expected by model
        batch_sequences = []
        target_strategies = []
        target_confidences = []
        target_risks = []
        target_params = defaultdict(list)
        
        for example in batch_data:
            # Convert state sequence to tensor format
            tensor_sequence = []
            for state_dict in example['state_sequence']:
                tensor_state = self._state_dict_to_tensors(state_dict)
                tensor_sequence.append(tensor_state)
            
            batch_sequences.append(tensor_sequence)
            target_strategies.append(example['target_strategy'])
            target_confidences.append(example['target_confidence'])
            target_risks.append(example['target_risk'])
            
            # Collect parameter targets
            for param_name in ['ultra_rare_threshold', 'deficit_panic_threshold', 'phase1_multi_attr_only', 'adaptation_rate']:
                target_params[param_name].append(example['target_parameters'].get(param_name, 0.5))
        
        return {
            'state_sequences': batch_sequences,
            'target_strategies': torch.LongTensor(target_strategies),
            'target_confidence': torch.FloatTensor(target_confidences),
            'target_risk': torch.FloatTensor(target_risks),
            'target_parameters': {k: torch.FloatTensor(v) for k, v in target_params.items()}
        }
    
    def _state_dict_to_tensors(self, state_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Convert state dict to tensor format expected by model"""
        # Person features (2 features: young, well_dressed)
        person_features = torch.tensor([
            state_dict['has_young'],
            state_dict['has_well_dressed']
        ] + [0.0] * 8, dtype=torch.float32)  # Pad to 10 features
        
        # Constraint features (6 features)
        constraint_features = torch.tensor([
            state_dict['young_progress'],
            state_dict['well_dressed_progress'],
            state_dict.get('young_deficit', 0.0) / 600.0,  # Normalized deficit
            state_dict.get('well_dressed_deficit', 0.0) / 600.0,
            state_dict['min_constraint_progress'],
            state_dict['max_constraint_progress']
        ], dtype=torch.float32)
        
        # Capacity features (4 features)
        capacity_features = torch.tensor([
            state_dict.get('admitted_count', 0.0) / 1000.0,
            state_dict.get('rejected_count', 0.0) / 20000.0,
            state_dict['capacity_ratio'],
            state_dict['rejection_ratio']
        ], dtype=torch.float32)
        
        # Phase features (3 features)
        phase_features = torch.tensor([
            state_dict['phase_early'],
            state_dict['phase_mid'],
            state_dict['phase_late']
        ], dtype=torch.float32)
        
        # Risk features (4 features)
        risk_features = torch.tensor([
            state_dict['constraint_risk'],
            state_dict['rejection_ratio'],
            state_dict['time_pressure'],
            0.0  # Placeholder for uncertainty
        ], dtype=torch.float32)
        
        # Strategy features (8 features)
        strategy_features = torch.tensor([
            state_dict['recent_acceptance_rate'],
            state_dict['strategy_performance'],
            state_dict['decisions_count'],
            0.5,  # strategy_confidence placeholder
            0.5,  # parameter_effectiveness placeholder
            0.5,  # constraint_focus_score placeholder
            0.5,  # efficiency_score placeholder
            0.5   # adaptability_score placeholder
        ], dtype=torch.float32)
        
        return {
            'person_features': person_features,
            'constraint_features': constraint_features,
            'capacity_features': capacity_features,
            'phase_features': phase_features,
            'risk_features': risk_features,
            'strategy_features': strategy_features
        }


class AdvancedStrategyControllerTrainer:
    """Advanced trainer with learning rate scheduling, early stopping, and logging"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        self.model = create_strategy_controller()
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=config.learning_rate * 0.01
        )
        
        # Loss functions
        self.strategy_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Create output directories
        Path(config.model_output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="berghain-strategy-controller",
                config=vars(config),
                name=config.model_name
            )
    
    def train(self, dataset: StrategyControllerDataset):
        """Main training loop"""
        logger.info(f"Starting training with {len(dataset.train_data)} training examples")
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_losses = self._train_epoch(dataset)
            
            # Validation phase
            val_losses = self._validate_epoch(dataset)
            
            # Logging
            self._log_epoch(epoch, train_losses, val_losses)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping check
            if self._check_early_stopping(val_losses['total']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch, val_losses)
        
        # Save final model
        self._save_final_model()
        
        if self.config.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")
    
    def _train_epoch(self, dataset: StrategyControllerDataset) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        # Calculate number of batches per epoch
        batches_per_epoch = len(dataset.train_data) // self.config.batch_size
        
        progress_bar = tqdm(range(batches_per_epoch), desc="Training")
        
        for batch_idx in progress_bar:
            # Get batch
            batch = dataset.get_batch(dataset.train_data, self.config.batch_size)
            
            # Forward pass
            losses = self._train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{losses['total']:.4f}"})
            
            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_interval == 0:
                wandb.log({f"train/{k}": v for k, v in losses.items()}, step=self.global_step)
        
        # Average losses over batches
        return {key: value / num_batches for key, value in epoch_losses.items()}
    
    def _validate_epoch(self, dataset: StrategyControllerDataset) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            # Calculate number of validation batches
            val_batches = max(1, len(dataset.val_data) // self.config.batch_size)
            
            for _ in range(val_batches):
                # Get validation batch
                batch = dataset.get_batch(dataset.val_data, self.config.batch_size)
                
                # Forward pass
                losses = self._validation_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value
                
                num_batches += 1
        
        # Average losses over batches
        return {key: value / num_batches for key, value in epoch_losses.items()}
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch['state_sequences'])
        
        # Calculate losses
        losses = self._calculate_losses(outputs, batch)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        
        # Update weights
        self.optimizer.step()
        
        return {key: loss.item() for key, loss in losses.items()}
    
    def _validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step"""
        # Forward pass
        outputs = self.model(batch['state_sequences'])
        
        # Calculate losses
        losses = self._calculate_losses(outputs, batch)
        
        return {key: loss.item() for key, loss in losses.items()}
    
    def _calculate_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate all loss components"""
        # Strategy classification loss
        strategy_loss = self.strategy_criterion(
            outputs['strategy_logits'],
            batch['target_strategies'].to(self.device)
        )
        
        # Confidence regression loss
        confidence_loss = self.regression_criterion(
            outputs['confidence'].squeeze(),
            batch['target_confidence'].to(self.device)
        )
        
        # Risk regression loss
        risk_loss = self.regression_criterion(
            outputs['risk_assessment'].squeeze(),
            batch['target_risk'].to(self.device)
        )
        
        # Parameter regression losses
        param_losses = []
        for param_name, param_output in outputs['parameter_adjustments'].items():
            if param_name in batch['target_parameters']:
                param_loss = self.regression_criterion(
                    param_output.squeeze(),
                    batch['target_parameters'][param_name].to(self.device)
                )
                param_losses.append(param_loss)
        
        total_param_loss = sum(param_losses) if param_losses else torch.tensor(0.0, device=self.device)
        
        # Total weighted loss
        total_loss = (
            self.config.strategy_loss_weight * strategy_loss +
            self.config.confidence_loss_weight * confidence_loss +
            self.config.risk_loss_weight * risk_loss +
            self.config.parameter_loss_weight * total_param_loss
        )
        
        return {
            'total': total_loss,
            'strategy': strategy_loss,
            'confidence': confidence_loss,
            'risk': risk_loss,
            'parameters': total_param_loss
        }
    
    def _log_epoch(self, epoch: int, train_losses: Dict[str, float], val_losses: Dict[str, float]):
        """Log epoch results"""
        logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
        logger.info(f"Train - Total: {train_losses['total']:.4f}, Strategy: {train_losses['strategy']:.4f}, "
                   f"Confidence: {train_losses['confidence']:.4f}, Risk: {train_losses['risk']:.4f}")
        logger.info(f"Val   - Total: {val_losses['total']:.4f}, Strategy: {val_losses['strategy']:.4f}, "
                   f"Confidence: {val_losses['confidence']:.4f}, Risk: {val_losses['risk']:.4f}")
        
        # Store training history
        self.training_history.append({
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch,
                **{f"train/{k}": v for k, v in train_losses.items()},
                **{f"val/{k}": v for k, v in val_losses.items()},
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping criteria"""
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def _save_checkpoint(self, epoch: int, val_losses: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': vars(self.config)
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_losses['total'] <= self.best_val_loss:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def _save_final_model(self):
        """Save final trained model"""
        final_model_path = Path(self.config.model_output_dir) / "trained_strategy_controller.pt"
        
        # Save model state dict only
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'strategy_vocab': self.model.strategy_vocab,
            'config': vars(self.config),
            'training_history': self.training_history
        }, final_model_path)
        
        logger.info(f"Final model saved to {final_model_path}")


def main():
    """Main training function"""
    # Load training data
    with open('strategy_controller_training_data.json', 'r') as f:
        training_data = json.load(f)
    
    # Create training config
    config = TrainingConfig(
        model_name="strategy_controller_trained",
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=50,
        use_wandb=False,
        device="cpu"
    )
    
    # Create model to get strategy vocabulary
    model = create_strategy_controller()
    strategy_vocab = model.strategy_vocab
    
    # Create dataset
    dataset = StrategyControllerDataset(
        training_data['training_examples'],
        config,
        strategy_vocab
    )
    
    # Create trainer
    trainer = AdvancedStrategyControllerTrainer(config)
    
    # Train model
    trainer.train(dataset)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final training loss: {trainer.training_history[-1]['train_losses']['total']:.4f}")
    print(f"📊 Final validation loss: {trainer.training_history[-1]['val_losses']['total']:.4f}")
    print(f"💾 Model saved to: {config.model_output_dir}/trained_strategy_controller.pt")


if __name__ == "__main__":
    main()