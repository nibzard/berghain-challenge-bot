#!/usr/bin/env python3
"""
ABOUTME: Main training script for Berghain RL agent
ABOUTME: Supports PPO training, behavioral cloning, and hyperparameter tuning
"""

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from berghain.training.lstm_policy import LSTMPolicyNetwork
from berghain.training.ppo_trainer import PPOTrainer, PPOConfig
from berghain.training.rl_environment import BerghainRLEnvironment
from berghain.training.data_collector import ExpertDataCollector, create_behavioral_cloning_dataset


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def behavioral_cloning_pretraining(
    model: LSTMPolicyNetwork,
    expert_dataset_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cpu'
) -> None:
    """
    Pre-train the model using behavioral cloning on expert demonstrations.
    
    Args:
        model: LSTM policy network to train
        expert_dataset_path: Path to expert dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting behavioral cloning pre-training for {epochs} epochs")
    
    # Load expert dataset
    collector = ExpertDataCollector()
    trajectories, metadata = collector.load_expert_dataset(expert_dataset_path)
    
    if not trajectories:
        logger.error("No expert trajectories found!")
        return
    
    logger.info(f"Loaded {len(trajectories)} expert trajectories")
    
    # Prepare training data
    states, actions = [], []
    for trajectory in trajectories:
        for experience in trajectory:
            states.append(experience.state)
            actions.append(experience.action)
    
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    dataset_size = len(states)
    num_batches = (dataset_size + batch_size - 1) // batch_size
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        
        # Shuffle data
        indices = torch.randperm(dataset_size)
        states_shuffled = states[indices]
        actions_shuffled = actions[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)
            
            batch_states = states_shuffled[start_idx:end_idx].unsqueeze(1)  # Add sequence dimension
            batch_actions = actions_shuffled[start_idx:end_idx]
            
            # Forward pass
            policy, _, _ = model(batch_states)
            policy = policy.squeeze(1)  # Remove sequence dimension
            
            # Calculate loss
            loss = criterion(policy, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(policy, dim=1)
            correct_predictions += (predictions == batch_actions).sum().item()
        
        # Log progress
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / dataset_size
        
        if epoch % 10 == 0:
            logger.info(f"BC Epoch {epoch}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
    
    logger.info(f"Behavioral cloning completed. Final accuracy: {accuracy:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train Berghain RL Agent")
    
    # Training mode
    parser.add_argument('--mode', type=str, choices=['ppo', 'bc', 'collect'], default='ppo',
                       help='Training mode: ppo (reinforcement learning), bc (behavioral cloning), collect (collect expert data)')
    
    # Model parameters
    parser.add_argument('--input-dim', type=int, default=8, help='Input dimension for LSTM')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--scenario', type=int, default=1, help='Game scenario to train on')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (cpu/cuda)')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    
    # Behavioral cloning parameters
    parser.add_argument('--bc-epochs', type=int, default=50, help='Behavioral cloning epochs')
    parser.add_argument('--bc-dataset', type=str, default='data/expert_trajectories.pkl', 
                       help='Path to expert dataset for behavioral cloning')
    parser.add_argument('--pretrain-bc', action='store_true', help='Pre-train with behavioral cloning before PPO')
    
    # Data collection parameters
    parser.add_argument('--game-logs-dir', type=str, default='game_logs', help='Directory with game logs')
    parser.add_argument('--min-success-rate', type=float, default=0.8, help='Minimum success rate for expert strategies')
    parser.add_argument('--max-trajectories', type=int, default=500, help='Maximum trajectories to collect')
    
    # Output parameters
    parser.add_argument('--save-path', type=str, default='models/berghain_rl', help='Model save path')
    parser.add_argument('--wandb-project', type=str, help='Weights & Biases project name')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directories
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Log configuration
    logger.info(f"Starting Berghain RL training with configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Scenario: {args.scenario}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Model: {args.hidden_dim}x{args.lstm_layers} LSTM")
    
    if args.mode == 'collect':
        # Collect expert data
        logger.info("Collecting expert demonstrations...")
        create_behavioral_cloning_dataset(
            game_logs_dir=args.game_logs_dir,
            output_path=args.bc_dataset,
            min_success_rate=args.min_success_rate,
            max_trajectories=args.max_trajectories
        )
        logger.info("Expert data collection completed!")
        return
    
    # Initialize model
    device = torch.device(args.device)
    model = LSTMPolicyNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.mode == 'bc':
        # Behavioral cloning only
        if not Path(args.bc_dataset).exists():
            logger.error(f"Expert dataset not found: {args.bc_dataset}")
            logger.info("Run with --mode collect first to create expert dataset")
            return
        
        behavioral_cloning_pretraining(
            model=model,
            expert_dataset_path=args.bc_dataset,
            epochs=args.bc_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        # Save BC model
        bc_path = f"{args.save_path}_bc.pth"
        torch.save(model.state_dict(), bc_path)
        logger.info(f"Behavioral cloning model saved to {bc_path}")
        
    elif args.mode == 'ppo':
        # PPO training (with optional BC pretraining)
        if args.pretrain_bc:
            if Path(args.bc_dataset).exists():
                logger.info("Pre-training with behavioral cloning...")
                behavioral_cloning_pretraining(
                    model=model,
                    expert_dataset_path=args.bc_dataset,
                    epochs=args.bc_epochs // 2,  # Fewer epochs for pretraining
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device
                )
                logger.info("Behavioral cloning pre-training completed")
            else:
                logger.warning(f"BC dataset not found: {args.bc_dataset}. Skipping pre-training.")
        
        # Setup PPO configuration
        config = PPOConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps
        )
        
        # Initialize trainer
        trainer = PPOTrainer(
            config=config,
            model=model,
            device=args.device,
            wandb_project=args.wandb_project
        )
        
        # Start training
        logger.info("Starting PPO training...")
        trainer.train(scenario=args.scenario, save_path=args.save_path)
        
        logger.info("Training completed!")
    
    # Test the trained model
    logger.info("Testing trained model...")
    try:
        test_env = BerghainRLEnvironment(scenario=args.scenario, use_simulator=True)
        
        model.eval()
        with torch.no_grad():
            state = test_env.reset()
            total_reward = 0
            steps = 0
            hidden = None
            
            while not test_env.done and steps < 1000:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                policy, value, hidden = model(state_tensor, hidden)
                action = torch.argmax(policy, dim=-1).item()
                
                state, reward, done, info = test_env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            summary = test_env.get_episode_summary()
            logger.info(f"Test episode: Reward={total_reward:.2f}, Steps={steps}, Success={summary['success']}")
            logger.info(f"Final state: Admitted={summary['admitted_count']}, Rejected={summary['rejected_count']}")
    
    except Exception as e:
        logger.error(f"Error during testing: {e}")
    
    logger.info("Training script completed!")


if __name__ == "__main__":
    main()