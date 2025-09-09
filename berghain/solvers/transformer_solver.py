"""
ABOUTME: Dual-head transformer strategy for Berghain Challenge
ABOUTME: AI-powered decision making trained on 37,418 elite games to beat 716 rejection record
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver


class TransformerSolver(BaseSolver):
    """Transformer-based solver using dual-head architecture trained in Google Colab."""
    
    def __init__(
        self,
        solver_id: str = "transformer",
        config_manager=None,
        api_client=None,
        enable_high_score_check: bool = True,
        model_path: Optional[str] = None,
        temperature: float = 1.0
    ):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        
        # Get strategy config
        try:
            strategy_config = config_manager.get_strategy_config("transformer")
            params = strategy_config.get("parameters", {})
        except:
            params = {}
        
        # Override with provided parameters
        if model_path:
            params['model_path'] = model_path
        if temperature != 1.0:
            params['temperature'] = temperature
            
        strategy = TransformerStrategy(params)
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class TransformerStrategy(BaseDecisionStrategy):
    """Decision strategy using dual-head transformer trained on elite games."""
    
    def __init__(self, strategy_params: dict = None):
        defaults = {
            'model_path': 'berghain_transformer/models/berghain_transformer_deployment.pt',
            'temperature': 0.3,
            'confidence_threshold': 0.5,
        }
        
        if strategy_params:
            defaults.update(strategy_params)
        
        # Call parent constructor
        super().__init__(defaults)
        
        # Model parameters
        self.model_path = defaults['model_path']
        self.temperature = defaults['temperature']
        self.confidence_threshold = defaults['confidence_threshold']
        
        # Load the model
        self.model = self._load_model()
        self.decision_history = []
        
        print(f"🤖 Transformer strategy loaded: temp={self.temperature}")
    
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "TransformerAI"
    
    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Legacy interface - returns (decision, reasoning)."""
        return self.decide(person, game_state)
    
    def _load_model(self):
        """Load the dual-head transformer model."""
        # Add colab path for model import
        colab_path = Path(__file__).parent.parent.parent / 'colab'
        if str(colab_path) not in sys.path:
            sys.path.append(str(colab_path))
        
        try:
            from models.dual_head_transformer import DualHeadTransformer
        except ImportError as e:
            raise ImportError(f"Cannot import dual-head transformer: {e}. Make sure colab/ folder is present.")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # Create and load model
        model = DualHeadTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _encode_game_state(self, game_state: GameState, person: Person) -> np.ndarray:
        """Encode game state and person into 14-dimensional feature vector."""
        
        # Get constraint requirements (scenario 1: young + well_dressed)
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        young_needed = max(0, 600 - young_current)
        well_dressed_needed = max(0, 600 - well_dressed_current)
        
        # Calculate progress and pressure metrics
        total_admitted = game_state.admitted_count
        total_rejected = game_state.rejected_count
        rejection_rate = total_rejected / max(1, total_admitted + total_rejected)
        
        game_progress = min(1.0, total_admitted / 1000.0)
        remaining_capacity = max(0, 1000 - total_admitted)
        time_pressure = 1.0 - (remaining_capacity / 1000.0)
        
        constraint_pressure = (young_needed + well_dressed_needed) / max(1, remaining_capacity)
        constraint_pressure = min(1.0, constraint_pressure)
        
        efficiency_trend = max(0.0, min(1.0, 1.0 - rejection_rate))
        
        # Person attributes
        person_young = person.attributes.get('young', False)
        person_well_dressed = person.attributes.get('well_dressed', False)
        
        # Person value for constraints
        person_value = 0.0
        if person_young and young_needed > 0:
            person_value += 0.5
        if person_well_dressed and well_dressed_needed > 0:
            person_value += 0.5
        
        # 14-dimensional feature vector matching training format
        features = np.array([
            young_current,
            young_needed,
            well_dressed_current,
            well_dressed_needed,
            constraint_pressure,
            total_admitted,
            total_rejected,
            rejection_rate,
            efficiency_trend,
            game_progress,
            time_pressure,
            float(person_young),
            float(person_well_dressed),
            person_value
        ], dtype=np.float32)
        
        return features
    
    def _make_constraint_aware_decision(
        self, action_prob: float, person: Person, game_state: GameState,
        constraint_conf: float, efficiency_conf: float
    ) -> Tuple[bool, float]:
        """Make constraint-aware decision with dynamic thresholds."""
        
        # Get current constraint status
        young_current = game_state.admitted_attributes.get('young', 0)
        well_dressed_current = game_state.admitted_attributes.get('well_dressed', 0)
        young_deficit = max(0, 600 - young_current)
        well_dressed_deficit = max(0, 600 - well_dressed_current)
        
        # Calculate remaining capacity and constraint urgency
        remaining_capacity = max(0, 1000 - game_state.admitted_count)
        total_deficit = young_deficit + well_dressed_deficit
        
        # Person's constraint attributes
        is_young = person.attributes.get('young', False)
        is_well_dressed = person.attributes.get('well_dressed', False)
        is_dual = is_young and is_well_dressed
        is_single = (is_young and not is_well_dressed) or (is_well_dressed and not is_young)
        is_filler = not is_young and not is_well_dressed
        
        # Dynamic threshold based on game state
        base_threshold = 0.52
        
        # CRITICAL: If we're running out of capacity relative to deficits
        if remaining_capacity <= total_deficit + 50:  # Emergency mode
            if is_dual:
                threshold = 0.45  # Very liberal for dual-attribute people
            elif (is_young and young_deficit > 0) or (is_well_dressed and well_dressed_deficit > 0):
                threshold = 0.48  # Liberal for needed singles
            else:
                threshold = 0.65  # Strict for filler people
        
        # High deficit pressure
        elif total_deficit > 400:  # Need lots of constraints
            if is_dual:
                threshold = 0.48  # Liberal for duals
            elif (is_young and young_deficit > well_dressed_deficit) or (is_well_dressed and well_dressed_deficit > young_deficit):
                threshold = 0.50  # Moderate for most-needed attribute
            elif is_single:
                threshold = 0.53  # Slightly strict for other singles
            else:
                threshold = 0.60  # Strict for filler
        
        # Moderate deficit pressure
        elif total_deficit > 200:  # Some constraint pressure
            if is_dual:
                threshold = 0.50  # Moderate for duals
            elif is_single:
                threshold = 0.54  # Moderate-strict for singles
            else:
                threshold = 0.58  # Strict for filler
        
        # Low constraint pressure - be more selective
        else:
            if is_dual:
                threshold = 0.52  # Base threshold for duals
            elif is_single:
                threshold = 0.56  # Strict for singles
            else:
                threshold = 0.62  # Very strict for filler
        
        # Confidence boost: if model is very confident, adjust threshold
        if constraint_conf > 0.8 or efficiency_conf > 0.8:
            threshold -= 0.02  # Slightly more liberal when model is confident
        
        # Make decision
        admit = action_prob > threshold
        return admit, threshold
    
    def decide(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        """Make admission decision using dual-head transformer."""
        
        # Encode current state
        state_features = self._encode_game_state(game_state, person)
        
        # Create sequence input (50 timesteps)
        seq_len = 50
        context_length = min(len(self.decision_history), seq_len - 1)
        
        if context_length > 0:
            # Use recent history as context
            context_features = np.stack([
                hist['features'] for hist in self.decision_history[-context_length:]
            ])
            
            if context_length < seq_len - 1:
                # Pad with zeros
                padding = np.zeros((seq_len - 1 - context_length, 14), dtype=np.float32)
                sequence_features = np.vstack([padding, context_features, state_features.reshape(1, -1)])
            else:
                # Use last seq_len-1 decisions plus current
                sequence_features = np.vstack([context_features[-(seq_len-1):], state_features.reshape(1, -1)])
        else:
            # No history, pad with zeros
            padding = np.zeros((seq_len - 1, 14), dtype=np.float32)
            sequence_features = np.vstack([padding, state_features.reshape(1, -1)])
        
        # Convert to tensor and get prediction
        input_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Get probabilities from combined head (handle different tensor shapes)
            if len(output.combined_logits.shape) == 3:
                combined_logits = output.combined_logits[0, -1, :]  # Batch, sequence, classes
            else:
                combined_logits = output.combined_logits[0, :]      # Batch, classes
            
            # Apply temperature
            if self.temperature != 1.0:
                combined_logits = combined_logits / self.temperature
            
            probs = F.softmax(combined_logits, dim=-1)
            action_prob = probs[1].item()  # Probability of admission
            
            # Get additional info for reasoning (handle different tensor shapes)
            if len(output.constraint_confidence.shape) == 3:
                constraint_conf = output.constraint_confidence[0, -1].item()
                efficiency_conf = output.efficiency_confidence[0, -1].item()
            else:
                constraint_conf = output.constraint_confidence[0].item()
                efficiency_conf = output.efficiency_confidence[0].item()
                
            # Handle head_weights shape carefully
            if len(output.head_weights.shape) == 3:
                head_weights = output.head_weights[0, -1].cpu().numpy()
            elif len(output.head_weights.shape) == 2:
                head_weights = output.head_weights[0].cpu().numpy()
            else:
                head_weights = output.head_weights.cpu().numpy()
        
        # Dynamic constraint-aware decision making
        admit, threshold_used = self._make_constraint_aware_decision(
            action_prob, person, game_state, constraint_conf, efficiency_conf
        )
        confidence = max(action_prob, 1 - action_prob)
        
        # Store in history
        self.decision_history.append({
            'features': state_features,
            'admit': admit,
            'confidence': confidence,
            'action_prob': action_prob
        })
        
        # Limit history size
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-50:]
        
        # Create detailed reasoning
        attrs = []
        if person.attributes.get('young'): attrs.append('young')
        if person.attributes.get('well_dressed'): attrs.append('well_dressed')
        if person.attributes.get('creative'): attrs.append('creative')
        attr_str = ','.join(attrs) if attrs else 'none'
        
        young_need = max(0, 600 - game_state.admitted_attributes.get('young', 0))
        wd_need = max(0, 600 - game_state.admitted_attributes.get('well_dressed', 0))
        
        reasoning = f"AI: {attr_str} p={action_prob:.3f} t={threshold_used:.3f} conf={confidence:.2f}"
        reasoning += f" need:[y{young_need},wd{wd_need}]"
        reasoning += f" heads:[c{constraint_conf:.2f},e{efficiency_conf:.2f}]"
        if isinstance(head_weights, np.ndarray) and head_weights.size > 0:
            reasoning += f" w={head_weights[0] if len(head_weights) > 1 else head_weights.item():.2f}"
        else:
            reasoning += f" w={head_weights:.2f}"
        
        if action_prob > 0.8:
            reasoning += " STRONG_ADMIT"
        elif action_prob < 0.2:
            reasoning += " STRONG_REJECT"
        else:
            reasoning += " UNCERTAIN"
        
        return admit, reasoning