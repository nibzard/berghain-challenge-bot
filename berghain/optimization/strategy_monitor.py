# ABOUTME: Real-time strategy performance monitoring and early termination
# ABOUTME: Kills underperforming strategies to save resources

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Real-time performance metrics for a strategy."""
    solver_id: str
    strategy_name: str
    start_time: float
    
    # Current game state
    admitted: int = 0
    rejected: int = 0
    progress_young: float = 0.0
    progress_well_dressed: float = 0.0
    
    # Performance indicators
    rejection_rate: float = 0.0  # rejections per second
    progress_rate: float = 0.0   # constraint progress per second
    efficiency_score: float = 0.0  # progress / rejections ratio
    
    # Termination flags
    terminated: bool = False
    termination_reason: str = ""
    
    def update(self, admitted: int, rejected: int, progress: Dict[str, float]):
        """Update metrics from game state."""
        self.admitted = admitted
        self.rejected = rejected
        self.progress_young = progress.get('young', 0.0)
        self.progress_well_dressed = progress.get('well_dressed', 0.0)
        
        # Calculate rates
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.rejection_rate = rejected / elapsed
            avg_progress = (self.progress_young + self.progress_well_dressed) / 2
            self.progress_rate = avg_progress / elapsed
            
            # Efficiency: how much progress per rejection
            if rejected > 0:
                self.efficiency_score = avg_progress / rejected
            else:
                self.efficiency_score = avg_progress


class StrategyPerformanceMonitor:
    """Monitors strategy performance and terminates underperformers."""
    
    def __init__(self, 
                 min_evaluation_time: int = 60,  # Don't evaluate before 60 seconds
                 efficiency_threshold: float = 0.3,  # Terminate if efficiency < 30% of best
                 rejection_rate_threshold: float = 2.0,  # Terminate if rejection rate > 2x best
                 min_strategies: int = 2):  # Always keep at least 2 strategies
        self.min_evaluation_time = min_evaluation_time
        self.efficiency_threshold = efficiency_threshold
        self.rejection_rate_threshold = rejection_rate_threshold
        self.min_strategies = min_strategies
        
        self.metrics: Dict[str, StrategyMetrics] = {}
        self.termination_callbacks: List = []
        
    def register_strategy(self, solver_id: str, strategy_name: str):
        """Register a new strategy for monitoring."""
        self.metrics[solver_id] = StrategyMetrics(
            solver_id=solver_id,
            strategy_name=strategy_name,
            start_time=time.time()
        )
        logger.info(f"📊 Monitoring strategy: {solver_id} ({strategy_name})")
    
    def update_strategy(self, solver_id: str, admitted: int, rejected: int, 
                       progress: Dict[str, float]):
        """Update strategy performance metrics."""
        if solver_id in self.metrics:
            self.metrics[solver_id].update(admitted, rejected, progress)
            
            # Check if strategy should be terminated
            if self._should_terminate(solver_id):
                self._terminate_strategy(solver_id)
    
    def _should_terminate(self, solver_id: str) -> bool:
        """Determine if a strategy should be terminated."""
        metric = self.metrics[solver_id]
        
        # Don't terminate if already terminated
        if metric.terminated:
            return False
        
        # Don't terminate in first evaluation period
        elapsed = time.time() - metric.start_time
        if elapsed < self.min_evaluation_time:
            return False
        
        # Don't terminate if too few strategies remain
        active_strategies = [m for m in self.metrics.values() if not m.terminated]
        if len(active_strategies) <= self.min_strategies:
            return False
        
        # Get best performing strategy for comparison
        best_efficiency = max(m.efficiency_score for m in active_strategies if m != metric)
        best_rejection_rate = min(m.rejection_rate for m in active_strategies if m != metric and m.rejection_rate > 0)
        
        # Terminate if efficiency is too low
        if best_efficiency > 0 and metric.efficiency_score < best_efficiency * self.efficiency_threshold:
            metric.termination_reason = f"Low efficiency: {metric.efficiency_score:.3f} vs best {best_efficiency:.3f}"
            return True
        
        # Terminate if rejection rate is too high
        if best_rejection_rate > 0 and metric.rejection_rate > best_rejection_rate * self.rejection_rate_threshold:
            metric.termination_reason = f"High rejection rate: {metric.rejection_rate:.1f}/s vs best {best_rejection_rate:.1f}/s"
            return True
        
        return False
    
    def _terminate_strategy(self, solver_id: str):
        """Mark strategy as terminated and notify callbacks."""
        metric = self.metrics[solver_id]
        metric.terminated = True
        
        logger.warning(f"🛑 Terminating underperforming strategy: {solver_id}")
        logger.warning(f"   Reason: {metric.termination_reason}")
        logger.warning(f"   Performance: {metric.efficiency_score:.3f} efficiency, {metric.rejection_rate:.1f} rejections/s")
        
        # Notify termination callbacks
        for callback in self.termination_callbacks:
            try:
                callback(solver_id, metric.termination_reason)
            except Exception as e:
                logger.error(f"Error in termination callback: {e}")
    
    def add_termination_callback(self, callback):
        """Add callback to be called when strategies are terminated."""
        self.termination_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get current performance summary for all strategies."""
        summary = {}
        for solver_id, metric in self.metrics.items():
            summary[solver_id] = {
                "strategy": metric.strategy_name,
                "admitted": metric.admitted,
                "rejected": metric.rejected,
                "efficiency": metric.efficiency_score,
                "rejection_rate": metric.rejection_rate,
                "terminated": metric.terminated,
                "termination_reason": metric.termination_reason
            }
        return summary
    
    def get_best_performers(self, n: int = 3) -> List[str]:
        """Get the top N performing strategy IDs."""
        active_metrics = [(sid, m) for sid, m in self.metrics.items() if not m.terminated]
        sorted_by_efficiency = sorted(active_metrics, key=lambda x: x[1].efficiency_score, reverse=True)
        return [sid for sid, _ in sorted_by_efficiency[:n]]
