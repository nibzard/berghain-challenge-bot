# ABOUTME: Hybrid transformer solver that uses strategy controller to orchestrate existing algorithms
# ABOUTME: Combines transformer-based strategic coordination with proven algorithmic strategies

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
from collections import deque

from ..core.domain import Person, GameState
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver
from ..training.strategy_controller import StrategyControllerTransformer, create_strategy_controller

# Import existing strategies
from .rbcr2_solver import RBCR2Strategy
from .perfect_solver import PerfectStrategy  
from .ultimate3_solver import Ultimate3Strategy
from .ultimate3h_solver import Ultimate3HStrategy
from .dual_deficit_solver import DualDeficitController

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Tracks performance of a strategy during game"""
    strategy_name: str
    decisions_made: int
    successful_decisions: int
    constraint_progress_contribution: float
    efficiency_score: float
    last_used_at: int


class HybridTransformerStrategy(BaseDecisionStrategy):
    """Strategy that uses transformer controller to orchestrate algorithmic strategies"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 temperature: float = 0.3,
                 strategy_params: Dict[str, Any] = None):
        super().__init__(strategy_params)
        
        self.device = device
        self.temperature = temperature
        
        # Load or create strategy controller
        self.controller = self._load_controller(model_path)
        
        # Initialize component strategies
        self.strategies = {
            'rbcr2': RBCR2Strategy(),
            'perfect': PerfectStrategy(),
            'ultimate3': Ultimate3Strategy(),
            'ultimate3h': Ultimate3HStrategy(),
            'dual_deficit': DualDeficitController(),
            'ultra_elite_lstm': None,  # Will load if available
            'constraint_focused_lstm': None  # Will load if available
        }
        
        # Try to load LSTM strategies if available
        self._try_load_lstm_strategies()
        
        # Strategy tracking
        self.current_strategy = 'rbcr2'  # Default start
        self.strategy_history: List[str] = []
        self.decision_count = 0
        self.last_strategy_switch = 0
        
        # Performance tracking
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        for strategy_name in self.strategies:
            if self.strategies[strategy_name] is not None:
                self.strategy_performances[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    decisions_made=0,
                    successful_decisions=0,
                    constraint_progress_contribution=0.0,
                    efficiency_score=0.5,
                    last_used_at=0
                )
        
        # State tracking for controller
        self.state_history: deque = deque(maxlen=20)  # Keep last 20 states
        self.constraint_progress_history: deque = deque(maxlen=50)
        
    def _load_controller(self, model_path: Optional[str]) -> StrategyControllerTransformer:
        """Load strategy controller model"""
        # Default to improved trained model if no path specified
        if model_path is None:
            improved_path = "models/strategy_controller/improved_strategy_controller.pt"
            if Path(improved_path).exists():
                model_path = improved_path
            else:
                model_path = "models/strategy_controller/trained_strategy_controller.pt"
        
        if model_path and Path(model_path).exists():
            try:
                controller = create_strategy_controller()
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    controller.load_state_dict(checkpoint['model_state_dict'])
                else:
                    controller.load_state_dict(checkpoint)
                    
                controller.eval()
                logger.info(f"Loaded trained strategy controller from {model_path}")
                return controller
            except Exception as e:
                logger.warning(f"Failed to load controller from {model_path}: {e}")
        
        # Create new controller (untrained) only if trained model not found
        controller = create_strategy_controller()
        controller.eval()
        logger.info("Using untrained strategy controller - train with python -m berghain.training.train_strategy_controller")
        return controller
    
    def _try_load_lstm_strategies(self):
        """Try to load LSTM strategies if available"""
        try:
            from .ultra_elite_lstm_solver import UltraEliteLSTMStrategy
            self.strategies['ultra_elite_lstm'] = UltraEliteLSTMStrategy()
            logger.info("Loaded Ultra Elite LSTM strategy")
        except Exception as e:
            logger.debug(f"Ultra Elite LSTM strategy not available: {e}")
            self.strategies.pop('ultra_elite_lstm', None)
        
        try:
            from .constraint_focused_lstm_solver import ConstraintFocusedLSTMStrategy
            self.strategies['constraint_focused_lstm'] = ConstraintFocusedLSTMStrategy()
            logger.info("Loaded Constraint Focused LSTM strategy")
        except Exception as e:
            logger.debug(f"Constraint Focused LSTM strategy not available: {e}")
            self.strategies.pop('constraint_focused_lstm', None)
    
    @property
    def name(self) -> str:
        return "HybridTransformer"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Main decision method using strategy controller orchestration"""
        self.decision_count += 1
        
        # Build current state representation
        current_state = self._build_state_representation(person, game_state)
        self.state_history.append(current_state)
        
        # Decide if we should switch strategies
        should_switch = self._should_switch_strategy(game_state)
        
        if should_switch or len(self.strategy_history) == 0:
            new_strategy = self._select_strategy(game_state)
            if new_strategy != self.current_strategy:
                logger.debug(f"Strategy switch: {self.current_strategy} → {new_strategy} at decision {self.decision_count}")
                self.current_strategy = new_strategy
                self.last_strategy_switch = self.decision_count
                self.strategy_history.append(new_strategy)
                
                # Update strategy parameters based on controller recommendation
                self._update_strategy_parameters(new_strategy, current_state)
        
        # Get decision from current strategy
        if self.current_strategy not in self.strategies or self.strategies[self.current_strategy] is None:
            # Fallback to rbcr2 if strategy not available
            self.current_strategy = 'rbcr2'
        
        current_strategy_obj = self.strategies[self.current_strategy]
        accept, reasoning = current_strategy_obj.should_accept(person, game_state)
        
        # Update performance tracking
        self._update_performance_tracking(person, game_state, accept)
        
        # Enhanced reasoning with controller info
        controller_reasoning = f"hybrid_transformer[{self.current_strategy}]_{reasoning}"
        
        return accept, controller_reasoning
    
    def _build_state_representation(self, person: Person, game_state: GameState) -> Dict[str, Any]:
        """Build state representation for strategy controller"""
        # Basic constraint progress
        young_progress = game_state.admitted_attributes.get('young', 0) / 600.0
        well_dressed_progress = game_state.admitted_attributes.get('well_dressed', 0) / 600.0
        
        # Constraint deficits
        young_deficit = max(0, 600 - game_state.admitted_attributes.get('young', 0))
        well_dressed_deficit = max(0, 600 - game_state.admitted_attributes.get('well_dressed', 0))
        
        # Game progress
        capacity_ratio = game_state.admitted_count / game_state.target_capacity
        rejection_ratio = game_state.rejected_count / game_state.max_rejections
        
        # Risk assessment
        remaining_capacity = game_state.remaining_capacity
        max_deficit = max(young_deficit, well_dressed_deficit)
        constraint_risk = max_deficit / max(remaining_capacity, 1) if remaining_capacity > 0 else 1.0
        
        # Strategy performance
        current_perf = self.strategy_performances.get(self.current_strategy)
        recent_efficiency = current_perf.efficiency_score if current_perf else 0.5
        
        # Game phase
        if capacity_ratio < 0.3:
            game_phase = 'early'
        elif capacity_ratio < 0.7:
            game_phase = 'mid'
        else:
            game_phase = 'late'
        
        return {
            'person_attributes': person.attributes,
            'young_progress': young_progress,
            'well_dressed_progress': well_dressed_progress,
            'young_deficit': young_deficit,
            'well_dressed_deficit': well_dressed_deficit,
            'min_constraint_progress': min(young_progress, well_dressed_progress),
            'max_constraint_progress': max(young_progress, well_dressed_progress),
            'admitted_count': game_state.admitted_count,
            'rejected_count': game_state.rejected_count,
            'capacity_ratio': capacity_ratio,
            'rejection_ratio': rejection_ratio,
            'constraint_risk': constraint_risk,
            'game_phase': game_phase,
            'recent_acceptance_rate': self._calculate_recent_acceptance_rate(),
            'strategy_performance': recent_efficiency,
            'decisions_since_switch': self.decision_count - self.last_strategy_switch,
            'strategy_confidence': self._calculate_strategy_confidence(),
            'time_pressure': rejection_ratio,  # Proxy for time pressure
            'uncertainty': constraint_risk  # Proxy for decision uncertainty
        }
    
    def _should_switch_strategy(self, game_state: GameState) -> bool:
        """Determine if we should consider switching strategies"""
        # Switch every 150-200 decisions to re-evaluate (reduced frequency)
        decisions_since_switch = self.decision_count - self.last_strategy_switch
        
        # Regular re-evaluation (less frequent)
        if decisions_since_switch > 150:
            return True
        
        # Emergency switches
        constraint_progress = game_state.constraint_progress()
        min_progress = min(constraint_progress.values()) if constraint_progress else 0
        capacity_ratio = game_state.capacity_ratio
        
        # Switch if constraints are at risk
        if capacity_ratio > 0.5 and min_progress < 0.4:  # Behind on constraints mid-game
            return True
        
        # Switch if rejection rate is too high
        if game_state.rejection_ratio > 0.6:  # Using too many rejections
            return True
        
        return False
    
    def _select_strategy(self, game_state: GameState) -> str:
        """Use controller to select optimal strategy"""
        try:
            # Get recent state sequence for controller
            state_sequence = list(self.state_history) if len(self.state_history) > 0 else [
                self._build_state_representation(Person(0, {}), game_state)
            ]
            
            # Get controller recommendation
            decision = self.controller.predict_strategy(state_sequence, temperature=self.temperature)
            
            # Validate strategy exists and is available
            if decision.selected_strategy in self.strategies and self.strategies[decision.selected_strategy] is not None:
                return decision.selected_strategy
            else:
                logger.warning(f"Controller recommended unavailable strategy: {decision.selected_strategy}")
                return self._fallback_strategy_selection(game_state)
                
        except Exception as e:
            logger.warning(f"Controller strategy selection failed: {e}")
            return self._fallback_strategy_selection(game_state)
    
    def _fallback_strategy_selection(self, game_state: GameState) -> str:
        """Fallback strategy selection using heuristics with RBCR2 bias"""
        constraint_progress = game_state.constraint_progress()
        min_progress = min(constraint_progress.values()) if constraint_progress else 0
        capacity_ratio = game_state.capacity_ratio
        
        # Emergency constraint focus
        if capacity_ratio > 0.7 and min_progress < 0.8:
            return 'dual_deficit'
        
        # Late game efficiency - still prefer RBCR2 unless critical
        elif capacity_ratio > 0.85:
            return 'perfect'
        
        # Mid-game - prefer RBCR2 (proven performer)
        elif capacity_ratio > 0.3:
            return 'rbcr2'  # Changed from ultimate3h to rbcr2
        
        # Early game - always RBCR2 (best baseline performance)
        else:
            return 'rbcr2'
    
    def _update_strategy_parameters(self, strategy_name: str, current_state: Dict[str, Any]):
        """Update strategy parameters based on controller recommendations"""
        try:
            state_sequence = [current_state]
            decision = self.controller.predict_strategy(state_sequence, temperature=0.0)  # Deterministic for parameters
            
            strategy_obj = self.strategies[strategy_name]
            if hasattr(strategy_obj, 'update_params'):
                strategy_obj.update_params(decision.parameter_adjustments)
                logger.debug(f"Updated {strategy_name} parameters: {decision.parameter_adjustments}")
                
        except Exception as e:
            logger.debug(f"Failed to update strategy parameters: {e}")
    
    def _calculate_recent_acceptance_rate(self) -> float:
        """Calculate recent acceptance rate for context"""
        # This is simplified - in practice would track actual decisions
        if len(self.constraint_progress_history) < 2:
            return 0.5
        
        recent_progress = list(self.constraint_progress_history)[-10:]
        if len(recent_progress) < 2:
            return 0.5
        
        # Estimate acceptance rate from progress changes
        avg_progress_change = sum(recent_progress) / len(recent_progress)
        return max(0.1, min(0.9, avg_progress_change))
    
    def _calculate_strategy_confidence(self) -> float:
        """Calculate confidence in current strategy"""
        current_perf = self.strategy_performances.get(self.current_strategy)
        if not current_perf or current_perf.decisions_made < 5:
            return 0.5
        
        success_rate = current_perf.successful_decisions / current_perf.decisions_made
        return min(0.95, max(0.05, success_rate))
    
    def _update_performance_tracking(self, person: Person, game_state: GameState, accept: bool):
        """Update performance tracking for current strategy"""
        if self.current_strategy not in self.strategy_performances:
            return
        
        perf = self.strategy_performances[self.current_strategy]
        perf.decisions_made += 1
        perf.last_used_at = self.decision_count
        
        # Track constraint progress contribution
        if accept:
            young_contrib = 1.0 if person.has_attribute('young') else 0.0
            well_dressed_contrib = 1.0 if person.has_attribute('well_dressed') else 0.0
            constraint_contrib = (young_contrib + well_dressed_contrib) / 2.0
            
            perf.constraint_progress_contribution += constraint_contrib
            perf.successful_decisions += 1
        
        # Update efficiency score (running average)
        current_efficiency = perf.successful_decisions / perf.decisions_made
        perf.efficiency_score = 0.9 * perf.efficiency_score + 0.1 * current_efficiency
        
        # Track overall constraint progress for context
        constraint_progress = game_state.constraint_progress()
        min_progress = min(constraint_progress.values()) if constraint_progress else 0
        self.constraint_progress_history.append(min_progress)
    
    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        params = super().get_params()
        params.update({
            'strategy_type': 'hybrid_transformer',
            'current_strategy': self.current_strategy,
            'decision_count': self.decision_count,
            'strategy_history': self.strategy_history,
            'strategy_performances': {
                name: {
                    'decisions_made': perf.decisions_made,
                    'success_rate': perf.successful_decisions / max(perf.decisions_made, 1),
                    'efficiency_score': perf.efficiency_score,
                    'constraint_contribution': perf.constraint_progress_contribution
                }
                for name, perf in self.strategy_performances.items()
            },
            'controller_temperature': self.temperature,
            'available_strategies': list(self.strategies.keys())
        })
        return params


class HybridTransformerSolver(BaseSolver):
    """Solver using hybrid transformer strategy"""
    
    def __init__(self, 
                 solver_id: str = "hybrid_transformer",
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 temperature: float = 0.3,
                 enable_high_score_check: bool = True,
                 api_client = None):
        
        # Create hybrid strategy
        strategy = HybridTransformerStrategy(
            model_path=model_path,
            device=device,
            temperature=temperature
        )
        
        super().__init__(
            strategy=strategy,
            solver_id=solver_id,
            enable_high_score_check=enable_high_score_check,
            api_client=api_client
        )


def main():
    """Test hybrid transformer solver"""
    # Create solver
    solver = HybridTransformerSolver(
        solver_id="hybrid_test",
        temperature=0.5
    )
    
    print(f"Created hybrid transformer solver with strategy: {solver.strategy.name}")
    print(f"Available strategies: {list(solver.strategy.strategies.keys())}")
    print(f"Controller has {sum(p.numel() for p in solver.strategy.controller.parameters())} parameters")
    
    # Test strategy selection
    from ..core.domain import GameState, Person, Constraint, AttributeStatistics
    
    # Create dummy game state
    constraints = [
        Constraint('young', 600),
        Constraint('well_dressed', 600)
    ]
    
    stats = AttributeStatistics(
        frequencies={'young': 0.323, 'well_dressed': 0.323},
        correlations={'young': {'well_dressed': 0.18}, 'well_dressed': {'young': 0.18}}
    )
    
    game_state = GameState(
        game_id="test_game",
        scenario=1,
        constraints=constraints,
        statistics=stats
    )
    
    # Test decision making
    test_person = Person(100, {'young': True, 'well_dressed': False})
    
    accept, reasoning = solver.strategy.should_accept(test_person, game_state)
    print(f"\nTest decision: {accept} - {reasoning}")
    print(f"Current strategy: {solver.strategy.current_strategy}")


if __name__ == "__main__":
    main()