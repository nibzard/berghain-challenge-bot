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
        """Load strategy configuration by name.

        Supports:
        - Top-level names (e.g., "conservative")
        - Relative subpaths (e.g., "evolved/conservative")
        - Name lookup across all subfolders (matches on filename stem)
        """
        # 1) If a relative subpath is provided, resolve directly under strategies_dir
        try_paths = []
        rel_path = Path(strategy_name)
        if not rel_path.is_absolute() and len(rel_path.parts) > 1:
            # Allow specifying subfolders, with or without .yaml
            if rel_path.suffix == '.yaml':
                try_paths.append(self.strategies_dir / rel_path)
            else:
                try_paths.append(self.strategies_dir / (rel_path.as_posix() + '.yaml'))

        # 2) Top-level file (default behavior)
        try_paths.append(self.strategies_dir / f"{strategy_name}.yaml")

        # 3) Search by stem name across subdirectories
        # Defer this search until direct paths fail
        def _load_yaml(p: Path) -> Optional[Dict[str, Any]]:
            try:
                with open(p, 'r') as f:
                    return yaml.safe_load(f)
            except (yaml.YAMLError, FileNotFoundError):
                return None

        for p in try_paths:
            if p.exists():
                cfg = _load_yaml(p)
                if cfg is not None:
                    return cfg

        # Fall back to rglob search by stem
        target_stem = strategy_name.rsplit('.', 1)[0]
        for file in self.strategies_dir.rglob('*.yaml'):
            if file.stem == target_stem:
                cfg = _load_yaml(file)
                if cfg is not None:
                    return cfg

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
        """List available strategy names (top-level + subfolders, deduplicated)."""
        names: List[str] = []
        seen = set()

        # Prefer top-level names first for stable ordering
        for f in sorted(self.strategies_dir.glob('*.yaml')):
            stem = f.stem
            if stem not in seen:
                seen.add(stem)
                names.append(stem)

        # Then include subdirectories
        for f in sorted(self.strategies_dir.rglob('*.yaml')):
            if f.parent == self.strategies_dir:
                continue  # already handled top-level
            stem = f.stem
            if stem not in seen:
                seen.add(stem)
                names.append(stem)

        return names
    
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
    
    def get_high_scores_config(self) -> Optional[Dict[str, Any]]:
        """Load high scores configuration."""
        high_scores_file = self.config_dir / "high_scores.yaml"
        
        if not high_scores_file.exists():
            return None
        
        try:
            with open(high_scores_file, 'r') as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, FileNotFoundError):
            return None
    
    def get_high_score_threshold(self, scenario_id: int) -> Optional[int]:
        """Get the high score threshold for a scenario."""
        config = self.get_high_scores_config()
        if not config:
            return None
        
        high_scores = config.get('high_scores', {})
        return high_scores.get(f'scenario_{scenario_id}')
    
    def is_high_score_checking_enabled(self) -> bool:
        """Check if high score checking is enabled."""
        config = self.get_high_scores_config()
        if not config:
            return False
        
        settings = config.get('settings', {})
        return settings.get('enabled', True)
    
    def get_buffer_percentage(self) -> float:
        """Get the buffer percentage for high score checking."""
        config = self.get_high_scores_config()
        if not config:
            return 1.0
        
        settings = config.get('settings', {})
        return settings.get('buffer_percentage', 0.95)
    
    def validate_strategy_config(self, config: Dict[str, Any]) -> bool:
        """Validate strategy configuration structure."""
        # Strategy configs can be flexible, just check it's a dict
        return isinstance(config, dict)
