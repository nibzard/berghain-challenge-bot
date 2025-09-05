# ABOUTME: RL environment wrapper for Berghain game training
# ABOUTME: Provides gym-like interface for collecting experience and computing rewards

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy

from ..core import GameState, Person, Decision, BerghainAPIClient, GameResult, GameStatus
from ..core.local_simulator import LocalSimulatorClient
from .lstm_policy import StateEncoder

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single step of experience for RL training."""
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    log_prob: float
    value: float
    person_index: int
    game_state_snapshot: Dict[str, Any]


class BerghainRLEnvironment:
    """
    RL Environment wrapper for the Berghain admission game.
    
    Provides a standardized interface for collecting experience data
    and computing rewards for training reinforcement learning agents.
    """
    
    def __init__(
        self,
        scenario: int = 1,
        use_simulator: bool = True,  # Use local simulator for faster training
        api_client: Optional[BerghainAPIClient] = None,
        reward_config: Optional[Dict[str, float]] = None
    ):
        self.scenario = scenario
        self.use_simulator = use_simulator
        
        if use_simulator:
            self.simulator = LocalSimulatorClient()
            self.api_client = None
        else:
            self.api_client = api_client or BerghainAPIClient()
            self.simulator = None
        
        self.encoder = StateEncoder()
        
        # Reward configuration
        default_rewards = {
            'constraint_help': 1.0,        # +1 for accepting person who helps unmet constraint
            'dual_bonus': 0.5,             # +0.5 extra for dual-attribute people
            'over_acceptance_penalty': -0.5, # -0.5 for accepting when constraint already met
            'filler_rejection_bonus': 0.1,  # +0.1 for rejecting people with no needed attributes
            'game_completion_bonus': 10.0,  # +10 for successfully completing game
            'game_failure_penalty': -10.0,  # -10 for failing the game
            'efficiency_bonus_scale': 2.0,  # Scale factor for efficiency-based rewards
            'constraint_balance_bonus': 1.0, # Bonus for balanced constraint progress
        }
        if reward_config:
            default_rewards.update(reward_config)
        self.reward_config = default_rewards
        
        # Episode state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        try:
            if self.use_simulator:
                self.game_state = self.simulator.start_new_game(self.scenario)
                self.current_person = self.simulator.get_next_person(self.game_state, 0)
            else:
                self.game_state = self.api_client.start_new_game(self.scenario)
                self.current_person = self.api_client.get_next_person(self.game_state, 0)
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise
        
        self.episode_steps = 0
        self.episode_rewards = []
        self.decisions_made = []
        self.done = False
        
        if self.current_person is None:
            raise ValueError("No initial person available - check game setup")
        
        initial_state = self.encoder.encode_state(self.current_person, self.game_state)
        return initial_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = reject, 1 = accept
            
        Returns:
            next_state: Next observation
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information
        """
        if self.done or self.current_person is None:
            raise ValueError("Environment is done or person is None - call reset()")
        
        accept = bool(action == 1)
        
        # Calculate reward before state update
        reward = self._calculate_reward(self.current_person, self.game_state, accept)
        
        # Create decision record
        decision = Decision(self.current_person, accept, f"rl_action_{action}")
        self.decisions_made.append(decision)
        
        # Store pre-action state for reward calculation
        old_game_state = copy.deepcopy(self.game_state)
        
        # Execute action and get next person
        try:
            if self.use_simulator:
                response = self.simulator.submit_decision(self.game_state, self.current_person, accept)
                self.game_state.update_decision(decision)
                
                if response["status"] == "running" and "nextPerson" in response:
                    person_data = response["nextPerson"]
                    self.current_person = Person(person_data["personIndex"], person_data["attributes"])
                else:
                    self.current_person = None
                    if response["status"] != "running":
                        status_map = {"completed": GameStatus.COMPLETED, "failed": GameStatus.FAILED}
                        self.game_state.complete_game(status_map.get(response["status"], GameStatus.FAILED))
            else:
                response = self.api_client.submit_decision(self.game_state, self.current_person, accept)
                self.game_state.update_decision(decision)
                
                if response["status"] == "running" and "nextPerson" in response:
                    person_data = response["nextPerson"]
                    self.current_person = Person(person_data["personIndex"], person_data["attributes"])
                else:
                    self.current_person = None
                    if response["status"] != "running":
                        status_map = {"completed": GameStatus.COMPLETED, "failed": GameStatus.FAILED}
                        self.game_state.complete_game(status_map.get(response["status"], GameStatus.FAILED))
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            self.done = True
            self.current_person = None
        
        # Check if episode is done
        self.done = (
            self.current_person is None or
            not self.game_state.can_continue() or
            self.game_state.status != GameStatus.RUNNING
        )
        
        # Calculate final reward bonus/penalty if episode ended
        if self.done:
            reward += self._calculate_terminal_reward()
        
        # Get next state
        if self.current_person is not None and not self.done:
            next_state = self.encoder.encode_state(self.current_person, self.game_state)
        else:
            next_state = np.zeros(self.encoder.encode_state(Person(0, {}), self.game_state).shape, dtype=np.float32)
        
        self.episode_steps += 1
        self.episode_rewards.append(reward)
        
        # Info dictionary
        info = {
            'episode_steps': self.episode_steps,
            'admitted_count': self.game_state.admitted_count,
            'rejected_count': self.game_state.rejected_count,
            'constraint_progress': self.game_state.constraint_progress(),
            'game_status': self.game_state.status.value if self.done else 'running',
            'success': self.game_state.status == GameStatus.COMPLETED if self.done else False,
            'total_reward': sum(self.episode_rewards),
            'person_attributes': self.current_person.attributes if self.current_person else {}
        }
        
        return next_state, reward, self.done, info
    
    def _calculate_reward(self, person: Person, game_state: GameState, accept: bool) -> float:
        """Calculate immediate reward for the action taken."""
        reward = 0.0
        
        # Get constraint information
        constraint_progress = game_state.constraint_progress()
        constraint_shortage = game_state.constraint_shortage()
        
        needed_attributes = []
        for attr in ['young', 'well_dressed']:
            if person.has_attribute(attr) and constraint_shortage.get(attr, 0) > 0:
                needed_attributes.append(attr)
        
        if accept:
            # Reward for accepting someone who helps with unmet constraints
            if needed_attributes:
                reward += self.reward_config['constraint_help'] * len(needed_attributes)
                
                # Bonus for dual-attribute people
                if len(needed_attributes) >= 2:
                    reward += self.reward_config['dual_bonus']
            
            # Penalty for accepting when constraints are already met
            has_unneeded_attrs = any(
                person.has_attribute(attr) and constraint_shortage.get(attr, 0) == 0 
                for attr in ['young', 'well_dressed']
            )
            if has_unneeded_attrs and not needed_attributes:
                reward += self.reward_config['over_acceptance_penalty']
            
            # Encourage balanced constraint progress
            if len(needed_attributes) == 1:
                progress_values = [constraint_progress.get(attr, 0) for attr in ['young', 'well_dressed']]
                progress_imbalance = abs(progress_values[0] - progress_values[1])
                if progress_imbalance > 0.2:  # If one constraint is 20%+ ahead
                    reward += self.reward_config['constraint_balance_bonus'] * (1 - progress_imbalance)
        
        else:  # reject
            # Small bonus for rejecting people who don't help with constraints
            if not needed_attributes:
                reward += self.reward_config['filler_rejection_bonus']
        
        return reward
    
    def _calculate_terminal_reward(self) -> float:
        """Calculate terminal reward based on final game outcome."""
        if self.game_state.status == GameStatus.COMPLETED:
            # Success bonus
            reward = self.reward_config['game_completion_bonus']
            
            # Efficiency bonus based on rejection count (fewer rejections = better)
            efficiency = 1.0 - (self.game_state.rejected_count / self.game_state.max_rejections)
            reward += self.reward_config['efficiency_bonus_scale'] * efficiency
            
            return reward
        
        elif self.game_state.status == GameStatus.FAILED:
            return self.reward_config['game_failure_penalty']
        
        return 0.0
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the completed episode."""
        return {
            'episode_length': self.episode_steps,
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'admitted_count': self.game_state.admitted_count,
            'rejected_count': self.game_state.rejected_count,
            'success': self.game_state.status == GameStatus.COMPLETED,
            'constraint_progress': self.game_state.constraint_progress(),
            'constraint_satisfaction': self.game_state.are_all_constraints_satisfied(),
            'game_status': self.game_state.status.value,
            'decisions_made': len(self.decisions_made)
        }
    
    def collect_trajectories(self, policy_fn, num_episodes: int = 1) -> List[List[Experience]]:
        """
        Collect trajectory data using a policy function.
        
        Args:
            policy_fn: Function that takes (state, info) and returns (action, log_prob, value)
            num_episodes: Number of episodes to collect
            
        Returns:
            List of trajectories, where each trajectory is a list of Experience objects
        """
        trajectories = []
        
        for episode in range(num_episodes):
            trajectory = []
            state = self.reset()
            done = False
            
            while not done:
                # Get action from policy
                action, log_prob, value = policy_fn(state, {})
                
                # Execute step
                next_state, reward, done, info = self.step(action)
                
                # Store experience
                experience = Experience(
                    state=state.copy(),
                    action=action,
                    reward=reward,
                    next_state=next_state.copy() if not done else None,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    person_index=info.get('episode_steps', 0),
                    game_state_snapshot={
                        'admitted_count': info['admitted_count'],
                        'rejected_count': info['rejected_count'],
                        'constraint_progress': info['constraint_progress'].copy()
                    }
                )
                trajectory.append(experience)
                
                state = next_state
            
            trajectories.append(trajectory)
            logger.info(f"Episode {episode + 1}/{num_episodes}: "
                       f"Length={len(trajectory)}, "
                       f"Reward={sum(exp.reward for exp in trajectory):.2f}, "
                       f"Success={info.get('success', False)}")
        
        return trajectories