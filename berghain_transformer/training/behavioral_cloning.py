"""Behavioral cloning trainer for Berghain Transformer."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import wandb
from tqdm import tqdm
import numpy as np

from ..models.transformer_model import BerghainTransformer, DecisionTransformer
from ..data.preprocessor import create_dataloaders, save_encoder


class BehavioralCloningTrainer:
    """Trains transformer model via behavioral cloning on expert demonstrations."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.warmup_steps = warmup_steps
        self.scheduler = self._create_scheduler(learning_rate)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def _create_scheduler(self, base_lr: float):
        """Create learning rate scheduler with linear warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            returns_to_go = batch['returns_to_go'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            if isinstance(self.model, DecisionTransformer):
                action_logits = self.model(
                    states=states,
                    actions=actions[:, :-1],  # Don't include last action for input
                    rewards=rewards[:, :-1],
                    returns_to_go=returns_to_go
                )
            else:
                action_logits, _ = self.model(
                    states=states,
                    actions=actions[:, :-1],
                    rewards=rewards[:, :-1]
                )
            
            # Calculate loss (predict next action)
            target_actions = actions[:, 1:]  # Shift actions by 1 for targets
            loss = self.criterion(
                action_logits.reshape(-1, action_logits.size(-1)),
                target_actions.reshape(-1)
            )
            
            # Calculate accuracy
            predictions = torch.argmax(action_logits, dim=-1)
            correct = (predictions == target_actions).float()
            accuracy = correct.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += correct.sum().item()
            total_samples += target_actions.numel()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': accuracy.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': accuracy.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': epoch * len(train_loader) + batch_idx
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                returns_to_go = batch['returns_to_go'].to(self.device)
                
                # Forward pass
                if isinstance(self.model, DecisionTransformer):
                    action_logits = self.model(
                        states=states,
                        actions=actions[:, :-1],
                        rewards=rewards[:, :-1],
                        returns_to_go=returns_to_go
                    )
                else:
                    action_logits, _ = self.model(
                        states=states,
                        actions=actions[:, :-1],
                        rewards=rewards[:, :-1]
                    )
                
                # Calculate loss
                target_actions = actions[:, 1:]
                loss = self.criterion(
                    action_logits.reshape(-1, action_logits.size(-1)),
                    target_actions.reshape(-1)
                )
                
                # Calculate accuracy
                predictions = torch.argmax(action_logits, dim=-1)
                correct = (predictions == target_actions).float()
                
                total_loss += loss.item()
                total_correct += correct.sum().item()
                total_samples += target_actions.numel()
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Path,
        save_frequency: int = 5,
        early_stopping_patience: int = 10
    ):
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_accuracy': train_acc,
                    'val/epoch_loss': val_loss,
                    'val/epoch_accuracy': val_acc
                })
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                self._save_checkpoint(save_dir, epoch, val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(save_dir, epoch, val_loss, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        print("\nTraining completed!")
        self._save_training_history(save_dir)
    
    def _save_checkpoint(
        self,
        save_dir: Path,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if is_best:
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def _save_training_history(self, save_dir: Path):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


def train_behavioral_cloning(
    game_logs_path: Path,
    save_dir: Path,
    model_type: str = 'decision_transformer',
    num_epochs: int = 50,
    batch_size: int = 32,
    seq_length: int = 100,
    learning_rate: float = 1e-4,
    elite_only: bool = True,
    scenario: int = 1,
    use_wandb: bool = False,
    wandb_project: str = 'berghain-transformer'
):
    """Main function to train a model via behavioral cloning."""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={
                'model_type': model_type,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'seq_length': seq_length,
                'learning_rate': learning_rate,
                'elite_only': elite_only,
                'scenario': scenario
            }
        )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, encoder = create_dataloaders(
        game_logs_path,
        batch_size=batch_size,
        seq_length=seq_length,
        elite_only=elite_only,
        scenario=scenario
    )
    
    # Save encoder
    save_dir.mkdir(parents=True, exist_ok=True)
    save_encoder(encoder, save_dir / 'encoder.pkl')
    
    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'decision_transformer':
        model = DecisionTransformer(
            state_dim=encoder.feature_dim,
            action_dim=2,
            n_layers=6,
            n_heads=8,
            d_model=256,
            d_ff=1024,
            dropout=0.1,
            max_seq_length=seq_length
        )
    else:
        model = BerghainTransformer(
            state_dim=encoder.feature_dim,
            action_dim=2,
            n_layers=6,
            n_heads=8,
            d_model=256,
            d_ff=1024,
            dropout=0.1,
            max_seq_length=seq_length,
            use_value_head=False
        )
    
    # Create trainer
    trainer = BehavioralCloningTrainer(
        model=model,
        learning_rate=learning_rate,
        use_wandb=use_wandb
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=save_dir,
        save_frequency=5,
        early_stopping_patience=10
    )
    
    if use_wandb:
        wandb.finish()
    
    return model, encoder