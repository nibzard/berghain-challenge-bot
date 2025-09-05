# ABOUTME: Configuration manager for YAML scenario and strategy files
# ABOUTME: Central config loading and validation system

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class ConfigManager:
    """Manages configuration loading from YAML files."""
    
    def __init__(self, config_dir: str = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent
        
        self.scenarios_dir = self.config_dir / "scenarios"
        self.strategies_dir = self.config_dir / "strategies"
    
    def get_scenario_config(self, scenario_id: int) -> Optional[Dict[str, Any]]:
        """Load scenario configuration by ID."""
        scenario_file = self.scenarios_dir / f"scenario_{scenario_id}.yaml"
        
        if not scenario_file.exists():
            return None
        
        try:
            with open(scenario_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add scenario ID to config
            config["scenario_id"] = scenario_id
            return config
            
        except (yaml.YAMLError, FileNotFoundError):
            return None
    
    def get_strategy_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Load strategy configuration by name."""
        strategy_file = self.strategies_dir / f"{strategy_name}.yaml"
        
        if not strategy_file.exists():
            return None
        
        try:
            with open(strategy_file, 'r') as f:
                config = yaml.safe_load(f)
            
            return config
            
        except (yaml.YAMLError, FileNotFoundError):
            return None
    
    def list_available_scenarios(self) -> List[int]:
        """List all available scenario IDs."""
        scenario_files = list(self.scenarios_dir.glob("scenario_*.yaml"))
        scenario_ids = []
        
        for file in scenario_files:
            try:
                scenario_id = int(file.stem.split('_')[1])
                scenario_ids.append(scenario_id)
            except (ValueError, IndexError):
                continue
        
        return sorted(scenario_ids)
    
    def list_available_strategies(self) -> List[str]:
        """List all available strategy names."""
        strategy_files = list(self.strategies_dir.glob("*.yaml"))
        return [f.stem for f in strategy_files]
    
    def validate_scenario_config(self, config: Dict[str, Any]) -> bool:
        """Validate scenario configuration structure."""
        required_fields = ["constraints", "expected_frequencies"]
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate constraints structure
        if not isinstance(config["constraints"], list):
            return False
        
        for constraint in config["constraints"]:
            if not isinstance(constraint, dict):
                return False
            if "attribute" not in constraint or "min_count" not in constraint:
                return False
        
        return True
    
    def validate_strategy_config(self, config: Dict[str, Any]) -> bool:
        """Validate strategy configuration structure."""
        # Strategy configs can be flexible, just check it's a dict
        return isinstance(config, dict)