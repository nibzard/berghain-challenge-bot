# ABOUTME: High score checking for early game termination
# ABOUTME: Prevents wasting time on games that exceed best known scores

import logging
from typing import Optional
from ..config.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class HighScoreChecker:
    """Checks if current game should be terminated based on high scores."""
    
    def __init__(self, scenario_id: int, enabled: bool = None):
        self.scenario_id = scenario_id
        self.config_manager = ConfigManager()
        
        # Override enabled setting if explicitly provided
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = self.config_manager.is_high_score_checking_enabled()
        
        # Load configuration
        self.high_score = self.config_manager.get_high_score_threshold(scenario_id)
        self.buffer_percentage = self.config_manager.get_buffer_percentage()
        
        # Calculate effective threshold
        if self.high_score is not None and self.enabled:
            self.threshold = int(self.high_score * self.buffer_percentage)
            logger.info(f"🏆 High score checker enabled for scenario {scenario_id}: "
                       f"threshold={self.threshold} (buffer={self.buffer_percentage*100:.0f}% of {self.high_score})")
        else:
            self.threshold = None
            logger.debug(f"High score checker disabled for scenario {scenario_id}")
    
    def should_terminate(self, current_rejections: int) -> bool:
        """Check if game should be terminated based on current rejection count."""
        if not self.enabled or self.threshold is None:
            return False
        
        return current_rejections >= self.threshold
    
    def get_termination_reason(self, current_rejections: int) -> str:
        """Get human-readable reason for termination."""
        if not self.should_terminate(current_rejections):
            return ""
        
        return (f"Rejections ({current_rejections}) exceeded high score threshold "
                f"({self.threshold}, {self.buffer_percentage*100:.0f}% of record {self.high_score})")
    
    def get_info(self) -> dict:
        """Get information about the high score checker state."""
        return {
            "scenario_id": self.scenario_id,
            "enabled": self.enabled,
            "high_score": self.high_score,
            "threshold": self.threshold,
            "buffer_percentage": self.buffer_percentage
        }