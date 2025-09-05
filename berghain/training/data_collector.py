# ABOUTME: Offline data collection utility for RL training from existing strategies
# ABOUTME: Converts game logs and expert demonstrations into training data

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import pickle

from ..core import GameState, Person, Decision, Constraint, AttributeStatistics
from ..core.domain import GameStatus
from .lstm_policy import StateEncoder
from .rl_environment import Experience

logger = logging.getLogger(__name__)


class ExpertDataCollector:
    """
    Collects expert demonstrations from game logs for offline RL training.
    
    Converts successful game trajectories into training data that can be used for:
    1. Behavioral cloning initialization
    2. Offline RL training
    3. Reward shaping validation
    """
    
    def __init__(self, game_logs_dir: str = "game_logs"):
        self.game_logs_dir = Path(game_logs_dir)
        self.encoder = StateEncoder()
    
    def load_game_log(self, log_file: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a game log file."""
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {log_file}: {e}")
            return None
    
    def reconstruct_game_state(
        self, 
        game_log: Dict[str, Any], 
        step_index: int
    ) -> Tuple[GameState, Person]:
        """
        Reconstruct game state at a specific step from game log.
        
        Args:
            game_log: Complete game log
            step_index: Index of the decision step
            
        Returns:
            GameState and Person at that step
        """
        # Extract basic game info
        game_id = game_log["game_id"]
        # Handle both old format (scenario) and new format (scenario_id)
        scenario = game_log.get("scenario") or game_log.get("scenario_id")
        
        # Reconstruct constraints
        constraints = []
        for constraint_data in game_log["constraints"]:
            # Handle both old format (minCount) and new format (min_count)
            min_count = constraint_data.get("minCount") or constraint_data.get("min_count")
            constraints.append(
                Constraint(constraint_data["attribute"], min_count)
            )
        
        # Reconstruct statistics
        statistics = AttributeStatistics(
            frequencies=game_log["attribute_frequencies"],
            correlations=game_log["attribute_correlations"]
        )
        
        # Create initial game state
        game_state = GameState(
            game_id=game_id,
            scenario=scenario,
            constraints=constraints,
            statistics=statistics
        )
        
        # Handle both old format (people) and new format (decisions)
        people_data = game_log.get("people") or game_log.get("decisions", [])
        
        # Replay decisions up to step_index to reconstruct state
        for i in range(step_index):
            person_data = people_data[i]
            person = Person(person_data["person_index"], person_data["attributes"])
            decision = Decision(person, person_data["decision"], "expert_replay")
            game_state.update_decision(decision)
        
        # Get current person
        if step_index < len(people_data):
            person_data = people_data[step_index]
            current_person = Person(person_data["person_index"], person_data["attributes"])
        else:
            # End of game
            current_person = Person(0, {})
        
        return game_state, current_person
    
    def extract_trajectory_from_log(self, game_log: Dict[str, Any]) -> Optional[List[Experience]]:
        """
        Extract a trajectory (sequence of experiences) from a game log.
        
        Args:
            game_log: Complete game log
            
        Returns:
            List of Experience objects representing the trajectory
        """
        # Handle both old format (people) and new format (decisions)
        people_data = game_log.get("people") or game_log.get("decisions", [])
        if not people_data:
            return None
        
        trajectory = []
        
        for step_index in range(len(people_data)):
            try:
                # Reconstruct game state at this step
                game_state, person = self.reconstruct_game_state(game_log, step_index)
                
                # Get decision made by expert
                person_data = people_data[step_index]
                expert_action = 1 if person_data["decision"] else 0
                
                # Encode state
                state = self.encoder.encode_state(person, game_state)
                
                # Calculate reward (retrospectively)
                reward = self._calculate_expert_reward(person, game_state, expert_action)
                
                # Determine if this is the last step
                is_done = (step_index == len(people_data) - 1)
                
                # Create next state if not done
                if not is_done:
                    next_game_state, next_person = self.reconstruct_game_state(game_log, step_index + 1)
                    next_state = self.encoder.encode_state(next_person, next_game_state)
                else:
                    next_state = None
                
                # Create experience
                experience = Experience(
                    state=state,
                    action=expert_action,
                    reward=reward,
                    next_state=next_state,
                    done=is_done,
                    log_prob=0.0,  # Not available from expert
                    value=0.0,     # Not available from expert
                    person_index=person.index,
                    game_state_snapshot={
                        'admitted_count': game_state.admitted_count,
                        'rejected_count': game_state.rejected_count,
                        'constraint_progress': game_state.constraint_progress()
                    }
                )
                
                trajectory.append(experience)
                
            except Exception as e:
                logger.error(f"Error processing step {step_index}: {e}")
                return None
        
        return trajectory
    
    def _calculate_expert_reward(self, person: Person, game_state: GameState, action: int) -> float:
        """Calculate reward for expert action (same logic as training environment)."""
        reward = 0.0
        accept = bool(action == 1)
        
        # Get constraint information
        constraint_shortage = game_state.constraint_shortage()
        
        needed_attributes = []
        for attr in ['young', 'well_dressed']:
            if person.has_attribute(attr) and constraint_shortage.get(attr, 0) > 0:
                needed_attributes.append(attr)
        
        if accept:
            # Reward for accepting someone who helps with unmet constraints
            if needed_attributes:
                reward += 1.0 * len(needed_attributes)
                # Bonus for dual-attribute people
                if len(needed_attributes) >= 2:
                    reward += 0.5
            else:
                # Penalty for accepting when constraints are already met
                reward -= 0.5
        else:  # reject
            # Small bonus for rejecting people who don't help with constraints
            if not needed_attributes:
                reward += 0.1
        
        return reward
    
    def collect_expert_trajectories(
        self, 
        min_success_rate: float = 0.8,
        max_trajectories: int = 1000,
        strategy_filters: Optional[List[str]] = None
    ) -> List[List[Experience]]:
        """
        Collect expert trajectories from successful game logs.
        
        Args:
            min_success_rate: Minimum success rate to consider a strategy as expert
            max_trajectories: Maximum number of trajectories to collect
            strategy_filters: List of strategy names to include (None for all)
            
        Returns:
            List of expert trajectories
        """
        logger.info(f"Collecting expert trajectories from {self.game_logs_dir}")
        
        if not self.game_logs_dir.exists():
            logger.error(f"Game logs directory {self.game_logs_dir} does not exist")
            return []
        
        # Find all game log files
        log_files = list(self.game_logs_dir.glob("*.json"))
        logger.info(f"Found {len(log_files)} log files")
        
        trajectories = []
        strategy_success_counts = {}
        strategy_total_counts = {}
        
        for log_file in log_files:
            game_log = self.load_game_log(log_file)
            if not game_log:
                continue
            
            # Extract strategy info from filename or metadata
            strategy_name = self._extract_strategy_name(log_file, game_log)
            
            # Track strategy statistics
            if strategy_name not in strategy_success_counts:
                strategy_success_counts[strategy_name] = 0
                strategy_total_counts[strategy_name] = 0
            
            strategy_total_counts[strategy_name] += 1
            
            # Check if this was a successful game
            game_successful = self._is_successful_game(game_log)
            if game_successful:
                strategy_success_counts[strategy_name] += 1
            
            # Apply strategy filter
            if strategy_filters and strategy_name not in strategy_filters:
                continue
            
            # Only collect from successful games
            if not game_successful:
                continue
            
            # Extract trajectory
            trajectory = self.extract_trajectory_from_log(game_log)
            if trajectory:
                trajectories.append(trajectory)
                
                if len(trajectories) >= max_trajectories:
                    break
        
        # Filter by success rate
        filtered_trajectories = []
        for log_file in log_files:
            game_log = self.load_game_log(log_file)
            if not game_log:
                continue
            
            strategy_name = self._extract_strategy_name(log_file, game_log)
            
            # Calculate success rate
            if strategy_total_counts.get(strategy_name, 0) > 0:
                success_rate = strategy_success_counts[strategy_name] / strategy_total_counts[strategy_name]
            else:
                success_rate = 0.0
            
            # Skip strategies with low success rate
            if success_rate < min_success_rate:
                continue
            
            # Add to filtered trajectories
            if self._is_successful_game(game_log):
                trajectory = self.extract_trajectory_from_log(game_log)
                if trajectory:
                    filtered_trajectories.append(trajectory)
                    
                    if len(filtered_trajectories) >= max_trajectories:
                        break
        
        # Report statistics
        logger.info(f"Strategy success rates:")
        for strategy, count in strategy_total_counts.items():
            success_count = strategy_success_counts.get(strategy, 0)
            success_rate = success_count / count if count > 0 else 0.0
            logger.info(f"  {strategy}: {success_rate:.2%} ({success_count}/{count})")
        
        logger.info(f"Collected {len(filtered_trajectories)} expert trajectories")
        return filtered_trajectories
    
    def _extract_strategy_name(self, log_file: Path, game_log: Dict[str, Any]) -> str:
        """Extract strategy name from filename or log metadata."""
        # Try to get from game log metadata
        if "strategy_params" in game_log and "strategy_name" in game_log["strategy_params"]:
            return game_log["strategy_params"]["strategy_name"]
        
        # Try to parse from filename
        filename = log_file.stem
        for strategy in ["ogds", "ultimate", "rbcr", "pec", "greedy", "adaptive", "balanced"]:
            if strategy in filename.lower():
                return strategy
        
        return "unknown"
    
    def _is_successful_game(self, game_log: Dict[str, Any]) -> bool:
        """Check if a game was successful based on the log."""
        # Check status field (new format) or final_status (old format)
        if "status" in game_log:
            return game_log["status"] == "completed"
        if "final_status" in game_log:
            return game_log["final_status"] == "completed"
        
        # Handle both old format (people) and new format (decisions)
        people_data = game_log.get("people") or game_log.get("decisions", [])
        
        # Try to infer success from constraints and game outcome
        if "constraints" not in game_log or not people_data:
            return False
        
        # Count final admitted attributes
        final_counts = {}
        for person_data in people_data:
            if person_data["decision"]:  # Was accepted
                for attr, value in person_data["attributes"].items():
                    if value:
                        final_counts[attr] = final_counts.get(attr, 0) + 1
        
        # Check if all constraints were satisfied
        for constraint in game_log["constraints"]:
            attr = constraint["attribute"]
            # Handle both old format (minCount) and new format (min_count)
            required = constraint.get("minCount") or constraint.get("min_count")
            actual = final_counts.get(attr, 0)
            if actual < required:
                return False
        
        return True
    
    def save_expert_dataset(
        self, 
        trajectories: List[List[Experience]], 
        save_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save expert trajectories to disk for later use."""
        dataset = {
            'trajectories': trajectories,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'num_trajectories': len(trajectories),
            'total_steps': sum(len(traj) for traj in trajectories)
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Expert dataset saved to {save_path}")
        logger.info(f"Dataset contains {len(trajectories)} trajectories with {dataset['total_steps']} total steps")
    
    def load_expert_dataset(self, dataset_path: str) -> Tuple[List[List[Experience]], Dict[str, Any]]:
        """Load expert dataset from disk."""
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        trajectories = dataset['trajectories']
        metadata = dataset.get('metadata', {})
        
        logger.info(f"Loaded expert dataset with {len(trajectories)} trajectories")
        return trajectories, metadata


def create_behavioral_cloning_dataset(
    game_logs_dir: str = "game_logs",
    output_path: str = "data/expert_trajectories.pkl",
    min_success_rate: float = 0.8,
    max_trajectories: int = 500
) -> None:
    """
    Create a dataset for behavioral cloning from game logs.
    
    Args:
        game_logs_dir: Directory containing game logs
        output_path: Path to save the dataset
        min_success_rate: Minimum success rate for expert strategies
        max_trajectories: Maximum number of trajectories to collect
    """
    collector = ExpertDataCollector(game_logs_dir)
    
    # Collect expert trajectories
    trajectories = collector.collect_expert_trajectories(
        min_success_rate=min_success_rate,
        max_trajectories=max_trajectories
    )
    
    if not trajectories:
        logger.error("No expert trajectories collected!")
        return
    
    # Save dataset
    metadata = {
        'min_success_rate': min_success_rate,
        'max_trajectories': max_trajectories,
        'source_dir': game_logs_dir
    }
    
    collector.save_expert_dataset(trajectories, output_path, metadata)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    create_behavioral_cloning_dataset()