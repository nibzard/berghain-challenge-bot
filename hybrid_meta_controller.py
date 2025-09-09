#!/usr/bin/env python3
"""
ABOUTME: Hybrid meta-controller system that combines best strategies for record-breaking performance
ABOUTME: Intelligently switches between ultra_elite_lstm, rbcr2, and dual-head transformer based on game state
"""

import json
import requests
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import logging
import importlib.util

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class StrategyType(Enum):
    """Available strategy types for the meta-controller."""
    ULTRA_ELITE_LSTM = "ultra_elite_lstm"
    RBCR2 = "rbcr2" 
    DUAL_HEAD_TRANSFORMER = "dual_head_transformer"
    EMERGENCY_RBCR = "emergency_rbcr"

@dataclass
class GameStateAnalysis:
    """Analysis of current game state for strategy selection."""
    total_admitted: int
    total_rejected: int
    young_count: int
    well_dressed_count: int
    young_needed: int
    well_dressed_needed: int
    game_progress: float      # 0.0 to 1.0
    rejection_rate: float
    constraint_pressure: float  # How urgent constraint satisfaction is
    efficiency_trend: float     # Recent efficiency (admits per decision)
    phase: str                  # 'opening', 'midgame', 'endgame', 'crisis'

@dataclass
class StrategyDecision:
    """Decision made by a strategy with confidence and reasoning."""
    decision: bool
    confidence: float
    reasoning: str
    strategy_type: StrategyType
    additional_info: Dict[str, Any] = None

class HybridMetaController:
    """
    Meta-controller that intelligently combines multiple high-performance strategies.
    
    Based on our analysis:
    - ultra_elite_lstm: Best overall (750 rejections)
    - rbcr2: Consistent and reliable (771 rejections) 
    - dual_head_transformer: Specialized for constraint vs efficiency tradeoffs
    """
    
    def __init__(
        self,
        api_url: str = "https://berghain.challenges.listenlabs.ai",
        scenario: int = 1,
        enable_transformer: bool = False,  # Will enable when training completes
        voting_threshold: float = 0.7,     # Confidence threshold for single-strategy decisions
        crisis_threshold: float = 0.8      # When to enter crisis mode
    ):
        self.api_url = api_url
        self.scenario = scenario
        self.enable_transformer = enable_transformer
        self.voting_threshold = voting_threshold
        self.crisis_threshold = crisis_threshold
        
        # Game state tracking
        self.total_admitted = 0
        self.total_rejected = 0
        self.young_count = 0
        self.well_dressed_count = 0
        self.decisions_history = deque(maxlen=100)
        self.recent_decisions = deque(maxlen=20)  # For efficiency trend
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'avg_confidence': 0.0})
        
        # Load available strategies
        self.strategies = self._load_strategies()
        
        logger.info(f"🎯 Hybrid Meta-Controller initialized with {len(self.strategies)} strategies")
        
    def _load_strategies(self) -> Dict[StrategyType, Any]:
        """Load available strategy solvers."""
        strategies = {}
        
        # Try to load each strategy type
        try:
            # RBCR2 - our most reliable performer
            from berghain.solvers.rbcr2_solver import RBCR2Solver
            strategies[StrategyType.RBCR2] = RBCR2Solver(
                api_url=self.api_url,
                scenario=self.scenario
            )
            logger.info("✅ Loaded RBCR2 solver")
        except Exception as e:
            logger.warning(f"Failed to load RBCR2: {e}")
            
        try:
            # Ultra Elite LSTM - our best performer  
            from berghain.solvers.ultra_elite_lstm_solver import UltraEliteLSTMSolver
            strategies[StrategyType.ULTRA_ELITE_LSTM] = UltraEliteLSTMSolver(
                api_url=self.api_url,
                scenario=self.scenario
            )
            logger.info("✅ Loaded Ultra Elite LSTM solver")
        except Exception as e:
            logger.warning(f"Failed to load Ultra Elite LSTM: {e}")
            
        try:
            # Emergency RBCR fallback
            from berghain.solvers.rbcr_solver import RBCRSolver  
            strategies[StrategyType.EMERGENCY_RBCR] = RBCRSolver(
                api_url=self.api_url,
                scenario=self.scenario
            )
            logger.info("✅ Loaded Emergency RBCR solver")
        except Exception as e:
            logger.warning(f"Failed to load Emergency RBCR: {e}")
            
        if self.enable_transformer:
            try:
                # Dual-head transformer (when available)
                from berghain_transformer.transformer_solver import TransformerSolver
                strategies[StrategyType.DUAL_HEAD_TRANSFORMER] = TransformerSolver(
                    model_path=Path("berghain_transformer/trained_models/best_model.pt"),
                    encoder_path=Path("berghain_transformer/models/encoder.pkl"),
                    api_url=self.api_url,
                    scenario=self.scenario
                )
                logger.info("✅ Loaded Dual-Head Transformer solver")
            except Exception as e:
                logger.warning(f"Failed to load Dual-Head Transformer: {e}")
        
        return strategies
    
    def analyze_game_state(self, person: Dict[str, Any]) -> GameStateAnalysis:
        """Analyze current game state for intelligent strategy selection."""
        
        # Calculate constraint needs (scenario 1: 600 young + 600 well_dressed)
        young_needed = max(0, 600 - self.young_count)
        well_dressed_needed = max(0, 600 - self.well_dressed_count)
        
        # Calculate game progress and pressure metrics
        game_progress = self.total_admitted / 1000.0
        total_decisions = self.total_admitted + self.total_rejected
        rejection_rate = self.total_rejected / max(1, total_decisions)
        
        # Constraint pressure (how urgent constraint satisfaction is)
        remaining_capacity = max(1, 1000 - self.total_admitted)
        constraint_pressure = (young_needed + well_dressed_needed) / remaining_capacity
        
        # Recent efficiency trend
        if len(self.recent_decisions) >= 5:
            recent_admits = sum(1 for d in list(self.recent_decisions)[-10:] if d)
            efficiency_trend = recent_admits / min(10, len(self.recent_decisions))
        else:
            efficiency_trend = 0.5  # Neutral
            
        # Determine game phase
        if game_progress < 0.3:
            phase = 'opening'
        elif game_progress > 0.85:
            phase = 'endgame'
        elif constraint_pressure > self.crisis_threshold:
            phase = 'crisis'
        else:
            phase = 'midgame'
            
        return GameStateAnalysis(
            total_admitted=self.total_admitted,
            total_rejected=self.total_rejected,
            young_count=self.young_count,
            well_dressed_count=self.well_dressed_count,
            young_needed=young_needed,
            well_dressed_needed=well_dressed_needed,
            game_progress=game_progress,
            rejection_rate=rejection_rate,
            constraint_pressure=constraint_pressure,
            efficiency_trend=efficiency_trend,
            phase=phase
        )
    
    def select_primary_strategy(self, analysis: GameStateAnalysis) -> StrategyType:
        """Select the primary strategy based on game state analysis."""
        
        # Crisis mode - use most reliable strategy
        if analysis.phase == 'crisis':
            if StrategyType.ULTRA_ELITE_LSTM in self.strategies:
                return StrategyType.ULTRA_ELITE_LSTM
            else:
                return StrategyType.RBCR2
        
        # Opening game - RBCR2 excels early
        elif analysis.phase == 'opening':
            if StrategyType.RBCR2 in self.strategies:
                return StrategyType.RBCR2
            else:
                return StrategyType.ULTRA_ELITE_LSTM
                
        # Endgame - Ultra Elite LSTM for optimization
        elif analysis.phase == 'endgame':
            if StrategyType.ULTRA_ELITE_LSTM in self.strategies:
                return StrategyType.ULTRA_ELITE_LSTM
            else:
                return StrategyType.RBCR2
                
        # Midgame - Use transformer if available, otherwise Ultra Elite LSTM
        else:
            if self.enable_transformer and StrategyType.DUAL_HEAD_TRANSFORMER in self.strategies:
                return StrategyType.DUAL_HEAD_TRANSFORMER
            elif StrategyType.ULTRA_ELITE_LSTM in self.strategies:
                return StrategyType.ULTRA_ELITE_LSTM
            else:
                return StrategyType.RBCR2
    
    def get_strategy_decision(
        self, 
        strategy_type: StrategyType,
        person: Dict[str, Any]
    ) -> Optional[StrategyDecision]:
        """Get a decision from a specific strategy."""
        
        if strategy_type not in self.strategies:
            return None
            
        try:
            strategy = self.strategies[strategy_type]
            
            # Call the strategy's decision method
            # Note: This assumes all strategies implement a decide() method
            # In practice, you might need to adapt this based on actual strategy interfaces
            
            if hasattr(strategy, 'decide'):
                decision, reasoning = strategy.decide(person)
                confidence = 0.8  # Default confidence
                
                # Try to extract confidence if available
                if hasattr(strategy, 'last_confidence'):
                    confidence = strategy.last_confidence
                    
            else:
                # Fallback for strategies without decide method
                decision = strategy.evaluate_person(person)  # Adapt based on actual interface
                reasoning = f"{strategy_type.value}_decision"
                confidence = 0.7
            
            return StrategyDecision(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                strategy_type=strategy_type
            )
            
        except Exception as e:
            logger.warning(f"Error getting decision from {strategy_type.value}: {e}")
            return None
    
    def voting_ensemble_decision(
        self,
        person: Dict[str, Any],
        analysis: GameStateAnalysis
    ) -> StrategyDecision:
        """Use ensemble voting when no single strategy is confident enough."""
        
        decisions = []
        weights = {}
        
        # Get decisions from all available strategies
        for strategy_type in self.strategies.keys():
            decision = self.get_strategy_decision(strategy_type, person)
            if decision:
                decisions.append(decision)
                
                # Weight strategies based on phase and performance
                if strategy_type == StrategyType.ULTRA_ELITE_LSTM:
                    weights[strategy_type] = 1.0  # Highest weight for best performer
                elif strategy_type == StrategyType.RBCR2:
                    weights[strategy_type] = 0.9  # High weight for reliability
                elif strategy_type == StrategyType.DUAL_HEAD_TRANSFORMER:
                    weights[strategy_type] = 0.8  # High weight when available
                else:
                    weights[strategy_type] = 0.6  # Default weight
        
        if not decisions:
            # Fallback to admit if no strategies available
            return StrategyDecision(
                decision=True,
                confidence=0.1,
                reasoning="meta_controller_fallback_admit",
                strategy_type=StrategyType.ULTRA_ELITE_LSTM
            )
        
        # Weighted voting
        admit_weight = 0.0
        reject_weight = 0.0
        total_weight = 0.0
        
        for decision in decisions:
            weight = weights.get(decision.strategy_type, 0.5)
            total_weight += weight
            
            if decision.decision:
                admit_weight += weight * decision.confidence
            else:
                reject_weight += weight * decision.confidence
        
        # Make final decision
        final_decision = admit_weight > reject_weight
        confidence = max(admit_weight, reject_weight) / total_weight if total_weight > 0 else 0.5
        
        # Create reasoning string
        admit_strategies = [d.strategy_type.value for d in decisions if d.decision]
        reject_strategies = [d.strategy_type.value for d in decisions if not d.decision]
        
        reasoning = f"ensemble_vote_{'admit' if final_decision else 'reject'}"
        if admit_strategies and reject_strategies:
            reasoning += f"_conflict"
        
        return StrategyDecision(
            decision=final_decision,
            confidence=confidence,
            reasoning=reasoning,
            strategy_type=StrategyType.ULTRA_ELITE_LSTM,  # Primary for tracking
            additional_info={
                'admit_strategies': admit_strategies,
                'reject_strategies': reject_strategies,
                'admit_weight': admit_weight,
                'reject_weight': reject_weight
            }
        )
    
    def make_decision(self, person: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Make the final admission decision using the hybrid meta-controller.
        
        Returns:
            Tuple of (decision, reasoning)
        """
        
        # Analyze current game state
        analysis = self.analyze_game_state(person)
        
        # Select primary strategy
        primary_strategy = self.select_primary_strategy(analysis)
        
        # Get decision from primary strategy
        primary_decision = self.get_strategy_decision(primary_strategy, person)
        
        # If primary strategy is confident enough, use its decision
        if primary_decision and primary_decision.confidence >= self.voting_threshold:
            final_decision = primary_decision
        else:
            # Use ensemble voting for low-confidence situations
            final_decision = self.voting_ensemble_decision(person, analysis)
        
        # Update tracking
        self.recent_decisions.append(final_decision.decision)
        
        # Update counters
        if final_decision.decision:
            self.total_admitted += 1
            if person.get('young', False):
                self.young_count += 1
            if person.get('well_dressed', False):
                self.well_dressed_count += 1
        else:
            self.total_rejected += 1
        
        # Enhanced reasoning with game state context
        enhanced_reasoning = f"meta_{final_decision.reasoning}_phase_{analysis.phase}_progress_{analysis.game_progress:.2f}"
        
        return final_decision.decision, enhanced_reasoning
    
    def run_game(self) -> Dict[str, Any]:
        """Run a complete game using the hybrid meta-controller."""
        logger.info("🎮 Starting Berghain game with Hybrid Meta-Controller...")
        
        # Initialize game
        init_response = requests.post(f"{self.api_url}/init", json={"scenario": self.scenario})
        if init_response.status_code != 200:
            raise Exception(f"Failed to initialize game: {init_response.text}")
        
        game_id = init_response.json()["game_id"]
        logger.info(f"🆔 Game initialized: {game_id}")
        
        decisions_made = []
        game_running = True
        
        while game_running:
            try:
                # Get next person
                person_response = requests.get(f"{self.api_url}/person/{game_id}")
                
                if person_response.status_code == 200:
                    person = person_response.json()
                    
                    # Make decision using meta-controller
                    decision, reasoning = self.make_decision(person)
                    
                    # Submit decision
                    decision_response = requests.post(
                        f"{self.api_url}/decision",
                        json={"game_id": game_id, "decision": decision}
                    )
                    
                    # Record decision
                    decisions_made.append({
                        "person_index": len(decisions_made),
                        "attributes": person,
                        "decision": decision,
                        "reasoning": reasoning,
                        "timestamp": time.time()
                    })
                    
                    if len(decisions_made) % 100 == 0:
                        logger.info(f"⚡ Processed {len(decisions_made)} people | "
                                   f"Admitted: {self.total_admitted} | "
                                   f"Rejected: {self.total_rejected}")
                
                elif person_response.status_code == 204:
                    # Game ended
                    game_running = False
                else:
                    logger.error(f"Error getting person: {person_response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"Error during game: {e}")
                break
        
        # Get final results
        try:
            results_response = requests.get(f"{self.api_url}/results/{game_id}")
            if results_response.status_code == 200:
                results = results_response.json()
            else:
                results = {"error": "Failed to get results"}
        except Exception as e:
            results = {"error": f"Exception getting results: {e}"}
        
        # Compile comprehensive game data
        game_data = {
            "game_id": game_id,
            "strategy": "hybrid_meta_controller",
            "scenario": self.scenario,
            "decisions": decisions_made,
            "final_admitted": self.total_admitted,
            "final_rejected": self.total_rejected,
            "young_count": self.young_count,
            "well_dressed_count": self.well_dressed_count,
            "success": results.get("success", False),
            "admitted_count": results.get("admitted_count", self.total_admitted),
            "rejected_count": results.get("rejected_count", self.total_rejected),
            "results": results
        }
        
        logger.info(f"🏁 Game completed!")
        logger.info(f"✅ Success: {game_data['success']}")
        logger.info(f"👥 Admitted: {game_data['admitted_count']}")
        logger.info(f"❌ Rejected: {game_data['rejected_count']}")
        
        return game_data


def main():
    """Test the hybrid meta-controller."""
    controller = HybridMetaController(enable_transformer=False)  # Will enable when transformer is ready
    
    # Run a test game
    game_data = controller.run_game()
    
    # Save results
    output_file = f"game_hybrid_meta_{game_data['rejected_count']}rej_{int(time.time())}.json"
    with open(f"game_logs/{output_file}", 'w') as f:
        json.dump(game_data, f, indent=2)
    
    print("\n🎯 HYBRID META-CONTROLLER TEST COMPLETE")
    print("=" * 50)
    print(f"🏆 Success: {game_data['success']}")
    print(f"👥 Admitted: {game_data['admitted_count']}")
    print(f"❌ Rejected: {game_data['rejected_count']}")
    print(f"📁 Results saved to: game_logs/{output_file}")

if __name__ == "__main__":
    main()