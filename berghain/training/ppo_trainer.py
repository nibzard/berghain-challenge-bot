# ABOUTME: PPO (Proximal Policy Optimization) trainer for Berghain RL agent
# ABOUTME: Implements stable policy gradient training with clipped objective and GAE

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional
import wandb
from collections import deque
from dataclasses import dataclass
import time

from .lstm_policy import LSTMPolicyNetwork
from .rl_environment import BerghainRLEnvironment, Experience

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # PPO parameters
    clip_ratio: float = 0.2  # PPO clip ratio
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    
    # Training parameters
    batch_size: int = 64
    mini_batch_size: int = 16
    epochs_per_update: int = 4
    max_grad_norm: float = 0.5
    
    # Environment parameters
    num_envs: int = 4  # Number of parallel environments
    steps_per_update: int = 1024  # Steps to collect before update
    
    # Training schedule
    total_timesteps: int = 1000000
    eval_frequency: int = 10000  # Evaluate every N steps
    save_frequency: int = 50000  # Save model every N steps
    
    # Early stopping
    early_stopping_patience: int = 10
    target_success_rate: float = 0.8


class PPOTrainer:
    """
    PPO trainer for the Berghain RL agent.
    
    Implements Proximal Policy Optimization with:
    - Generalized Advantage Estimation (GAE)
    - Clipped objective function
    - Value function learning
    - Entropy regularization
    """
    
    def __init__(
        self,
        config: PPOConfig,
        model: LSTMPolicyNetwork,
        device: str = 'cpu',
        wandb_project: Optional[str] = None
    ):
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_success_rate = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=20)
        
        # Wandb logging
        if wandb_project:
            wandb.init(project=wandb_project, config=config.__dict__)
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def compute_advantages(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update step.
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'clip_fraction': 0.0,
            'kl_divergence': 0.0
        }
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create mini-batches
        batch_size = len(states)
        mini_batch_size = self.config.mini_batch_size
        indices = np.arange(batch_size)
        
        for epoch in range(self.config.epochs_per_update):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]
                
                # Forward pass
                # Add sequence dimension for LSTM
                mb_states_seq = mb_states.unsqueeze(1)  # (batch, 1, features)
                policy, values, _ = self.model(mb_states_seq)
                policy = policy.squeeze(1)  # Remove sequence dimension
                values = values.squeeze(-1).squeeze(1)  # Remove extra dimensions
                
                # Calculate new log probabilities
                action_probs = policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                new_log_probs = torch.log(action_probs + 1e-8)
                
                # Calculate entropy
                entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()
                
                # Calculate policy ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Calculate clipped policy loss
                policy_loss1 = ratio * mb_advantages
                policy_loss2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                
                # Calculate value loss
                if self.config.value_loss_coef > 0:
                    value_pred_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values, -self.config.clip_ratio, self.config.clip_ratio
                    )
                    value_losses = (values - mb_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss -
                    self.config.entropy_coef * entropy
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                    kl_div = ((new_log_probs - mb_old_log_probs) ** 2).mean() * 0.5
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['clip_fraction'] += clip_fraction.item()
                metrics['kl_divergence'] += kl_div.item()
        
        # Average metrics over all updates
        num_updates = self.config.epochs_per_update * (batch_size // mini_batch_size)
        for key in metrics:
            metrics[key] /= num_updates
        
        return metrics
    
    def collect_rollouts(self, envs: List[BerghainRLEnvironment]) -> Tuple[List[Experience], Dict[str, float]]:
        """
        Collect rollouts from multiple environments.
        
        Args:
            envs: List of environment instances
            
        Returns:
            experiences: List of experiences
            rollout_info: Dictionary of rollout statistics
        """
        experiences = []
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # Reset environments
        states = []
        hidden_states = []
        for env in envs:
            state = env.reset()
            states.append(state)
            hidden_states.append(None)
        
        steps_collected = 0
        
        while steps_collected < self.config.steps_per_update:
            # Convert states to tensor
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            states_tensor = states_tensor.unsqueeze(1)  # Add sequence dimension
            
            # Get actions and values from policy
            with torch.no_grad():
                policy, values, new_hidden_states = self.model(states_tensor)
                policy = policy.squeeze(1)
                values = values.squeeze(-1).squeeze(1)
                
                # Sample actions
                dist = torch.distributions.Categorical(policy)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            # Execute actions in environments
            next_states = []
            for i, env in enumerate(envs):
                action = actions[i].item()
                log_prob = log_probs[i].item()
                value = values[i].item()
                
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                experience = Experience(
                    state=states[i].copy(),
                    action=action,
                    reward=reward,
                    next_state=next_state.copy() if not done else None,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    person_index=info.get('episode_steps', 0),
                    game_state_snapshot=info
                )
                experiences.append(experience)
                
                if done:
                    # Episode finished
                    episode_summary = env.get_episode_summary()
                    episode_rewards.append(episode_summary['total_reward'])
                    episode_lengths.append(episode_summary['episode_length'])
                    if episode_summary['success']:
                        success_count += 1
                    
                    # Reset environment
                    next_state = env.reset()
                    hidden_states[i] = None
                else:
                    hidden_states[i] = (new_hidden_states[0][:, i:i+1, :].clone(), 
                                       new_hidden_states[1][:, i:i+1, :].clone())
                
                next_states.append(next_state)
                steps_collected += 1
            
            states = next_states
        
        rollout_info = {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'success_rate': success_count / max(len(episode_rewards), 1),
            'episodes_completed': len(episode_rewards)
        }
        
        return experiences, rollout_info
    
    def train(self, scenario: int = 1, save_path: str = 'models/berghain_rl') -> None:
        """
        Main training loop.
        
        Args:
            scenario: Game scenario to train on
            save_path: Path to save trained models
        """
        logger.info(f"Starting PPO training for scenario {scenario}")
        
        # Create environments
        envs = []
        for _ in range(self.config.num_envs):
            env = BerghainRLEnvironment(scenario=scenario, use_simulator=True)
            envs.append(env)
        
        start_time = time.time()
        
        while self.global_step < self.config.total_timesteps:
            # Collect rollouts
            experiences, rollout_info = self.collect_rollouts(envs)
            
            # Convert experiences to tensors
            states = torch.tensor([exp.state for exp in experiences], dtype=torch.float32, device=self.device)
            actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=self.device)
            rewards = np.array([exp.reward for exp in experiences])
            old_log_probs = torch.tensor([exp.log_prob for exp in experiences], dtype=torch.float32, device=self.device)
            old_values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32, device=self.device)
            dones = np.array([exp.done for exp in experiences], dtype=np.float32)
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(rewards, old_values.cpu().numpy(), dones)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # PPO update
            training_metrics = self.ppo_update(states, actions, old_log_probs, advantages, returns, old_values)
            
            # Update tracking
            self.global_step += len(experiences)
            self.episode_rewards.extend([rollout_info['mean_episode_reward']] * rollout_info['episodes_completed'])
            self.episode_lengths.extend([rollout_info['mean_episode_length']] * rollout_info['episodes_completed'])
            self.success_rates.append(rollout_info['success_rate'])
            
            # Logging
            if self.global_step % 10000 == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
                avg_success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
                elapsed_time = time.time() - start_time
                
                logger.info(
                    f"Step {self.global_step:,}/{self.config.total_timesteps:,} | "
                    f"Reward: {avg_reward:.2f} | "
                    f"Success Rate: {avg_success_rate:.2%} | "
                    f"Time: {elapsed_time/60:.1f}min"
                )
                
                if self.use_wandb:
                    wandb.log({
                        'global_step': self.global_step,
                        'rollout/mean_reward': rollout_info['mean_episode_reward'],
                        'rollout/mean_length': rollout_info['mean_episode_length'],
                        'rollout/success_rate': rollout_info['success_rate'],
                        'train/policy_loss': training_metrics['policy_loss'],
                        'train/value_loss': training_metrics['value_loss'],
                        'train/entropy': training_metrics['entropy'],
                        'train/clip_fraction': training_metrics['clip_fraction'],
                        'train/kl_divergence': training_metrics['kl_divergence'],
                        'metrics/avg_reward_100': avg_reward,
                        'metrics/avg_success_rate_20': avg_success_rate
                    })
            
            # Model saving
            if self.global_step % self.config.save_frequency == 0:
                model_path = f"{save_path}_step_{self.global_step}.pth"
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Early stopping check
            current_success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
                self.patience_counter = 0
                # Save best model
                best_model_path = f"{save_path}_best.pth"
                torch.save(self.model.state_dict(), best_model_path)
            else:
                self.patience_counter += 1
            
            if (current_success_rate >= self.config.target_success_rate and
                self.patience_counter >= self.config.early_stopping_patience):
                logger.info(f"Early stopping triggered. Target success rate {self.config.target_success_rate:.2%} reached.")
                break
        
        # Final model save
        final_model_path = f"{save_path}_final.pth"
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"Training completed. Final model saved to {final_model_path}")
        
        if self.use_wandb:
            wandb.finish()