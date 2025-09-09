# ABOUTME: Strategy controller transformer that orchestrates existing algorithmic strategies
# ABOUTME: Uses reinforcement learning to learn optimal strategy selection and parameter tuning

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class StrategyControlDecision:
    """Output of strategy controller"""
    selected_strategy: str
    strategy_confidence: float
    parameter_adjustments: Dict[str, Any]
    risk_assessment: float
    recommended_phase: str
    reasoning: str


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class GameStateEncoder(nn.Module):
    """Encodes game state into transformer-friendly representation"""
    
    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        
        # Core game state encoding
        self.person_encoder = nn.Linear(10, 16)  # Person attributes (young, well_dressed, etc.)
        self.constraint_encoder = nn.Linear(6, 16)  # Constraint progress (young/well_dressed progress, deficits)
        self.capacity_encoder = nn.Linear(4, 8)  # Capacity info (admitted, rejected, ratios)
        self.phase_encoder = nn.Linear(3, 8)  # Game phase (early/mid/late indicators)
        self.risk_encoder = nn.Linear(4, 8)  # Risk assessment (constraint risk, rejection risk)
        
        # Strategy context encoding
        self.strategy_encoder = nn.Linear(8, 8)  # Current strategy performance
        
        # Combine all encodings
        self.state_combiner = nn.Linear(16 + 16 + 8 + 8 + 8 + 8, state_dim)
        
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode game state dictionary into fixed-size representation"""
        
        # Encode each component
        person_encoded = F.relu(self.person_encoder(state_dict['person_features']))
        constraint_encoded = F.relu(self.constraint_encoder(state_dict['constraint_features']))
        capacity_encoded = F.relu(self.capacity_encoder(state_dict['capacity_features']))
        phase_encoded = F.relu(self.phase_encoder(state_dict['phase_features']))
        risk_encoded = F.relu(self.risk_encoder(state_dict['risk_features']))
        strategy_encoded = F.relu(self.strategy_encoder(state_dict['strategy_features']))
        
        # Combine all encodings
        combined = torch.cat([
            person_encoded, constraint_encoded, capacity_encoded,
            phase_encoded, risk_encoded, strategy_encoded
        ], dim=-1)
        
        return self.state_combiner(combined)


class StrategyControllerTransformer(nn.Module):
    """Transformer model for strategy control decisions"""
    
    def __init__(
        self,
        state_dim: int = 64,
        n_strategies: int = 8,
        n_heads: int = 8,
        n_layers: int = 6,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_strategies = n_strategies
        self.d_model = d_model
        
        # State encoding
        self.state_encoder = GameStateEncoder(state_dim)
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.strategy_head = nn.Linear(d_model, n_strategies)
        self.confidence_head = nn.Linear(d_model, 1)
        self.risk_head = nn.Linear(d_model, 1)
        
        # Parameter adjustment heads (for key parameters)
        self.param_heads = nn.ModuleDict({
            'ultra_rare_threshold': nn.Linear(d_model, 1),
            'deficit_panic_threshold': nn.Linear(d_model, 1),
            'phase1_multi_attr_only': nn.Linear(d_model, 1),
            'adaptation_rate': nn.Linear(d_model, 1)
        })
        
        # Strategy vocabulary
        self.strategy_vocab = [
            'rbcr2', 'ultra_elite_lstm', 'constraint_focused_lstm',
            'perfect', 'ultimate3', 'ultimate3h', 'dual_deficit', 'rbcr'
        ]
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, state_sequences: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Forward pass through strategy controller"""
        batch_size = len(state_sequences)
        seq_length = len(state_sequences[0]) if state_sequences else 1
        
        # Encode all states in sequence
        encoded_states = []
        for batch_idx in range(batch_size):
            sequence_states = []
            for seq_idx in range(seq_length):
                state_dict = state_sequences[batch_idx][seq_idx] if seq_idx < len(state_sequences[batch_idx]) else state_sequences[batch_idx][-1]
                encoded_state = self.state_encoder(state_dict)
                sequence_states.append(encoded_state)
            encoded_states.append(torch.stack(sequence_states))
        
        # Shape: (batch_size, seq_length, state_dim)
        encoded_sequences = torch.stack(encoded_states)
        
        # Project to model dimension
        x = self.input_projection(encoded_sequences)
        x = self.dropout(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model)
        
        # Apply transformer
        transformer_output = self.transformer(x)
        
        # Use last timestep output for decisions
        final_output = transformer_output[:, -1, :]  # (batch_size, d_model)
        
        # Generate outputs
        strategy_logits = self.strategy_head(final_output)
        confidence = torch.sigmoid(self.confidence_head(final_output))
        risk_assessment = torch.sigmoid(self.risk_head(final_output))
        
        # Parameter adjustments
        parameter_outputs = {}
        for param_name, param_head in self.param_heads.items():
            parameter_outputs[param_name] = param_head(final_output)
        
        return {
            'strategy_logits': strategy_logits,
            'confidence': confidence,
            'risk_assessment': risk_assessment,
            'parameter_adjustments': parameter_outputs
        }
    
    def predict_strategy(self, state_sequence: List[Dict[str, Any]], temperature: float = 1.0) -> StrategyControlDecision:
        """Make a strategy control decision"""
        self.eval()
        
        with torch.no_grad():
            # Convert state sequence to tensors
            tensor_sequence = [self._convert_state_to_tensors(state) for state in state_sequence]
            
            # Forward pass
            outputs = self.forward([tensor_sequence])
            
            # Extract outputs
            strategy_logits = outputs['strategy_logits'][0]  # Remove batch dimension
            confidence = outputs['confidence'][0, 0].item()
            risk_assessment = outputs['risk_assessment'][0, 0].item()
            
            # Sample strategy with temperature
            if temperature > 0:
                strategy_probs = F.softmax(strategy_logits / temperature, dim=0)
                strategy_idx = torch.multinomial(strategy_probs, 1).item()
            else:
                strategy_idx = torch.argmax(strategy_logits).item()
            
            selected_strategy = self.strategy_vocab[strategy_idx]
            
            # Extract parameter adjustments
            parameter_adjustments = {}
            for param_name, param_output in outputs['parameter_adjustments'].items():
                param_value = param_output[0, 0].item()  # Remove batch and feature dimensions
                
                # Apply parameter-specific scaling
                if param_name == 'ultra_rare_threshold':
                    parameter_adjustments[param_name] = max(0.01, min(0.1, 0.05 + param_value * 0.05))
                elif param_name == 'deficit_panic_threshold':
                    parameter_adjustments[param_name] = max(50, min(300, 150 + param_value * 100))
                elif param_name == 'phase1_multi_attr_only':
                    parameter_adjustments[param_name] = param_value > 0
                elif param_name == 'adaptation_rate':
                    parameter_adjustments[param_name] = max(0.01, min(0.5, 0.1 + param_value * 0.2))
                else:
                    parameter_adjustments[param_name] = param_value
            
            # Determine game phase
            last_state = state_sequence[-1]
            capacity_ratio = last_state.get('capacity_ratio', 0.5)
            if capacity_ratio < 0.3:
                phase = 'early'
            elif capacity_ratio < 0.7:
                phase = 'mid'
            else:
                phase = 'late'
            
            # Generate reasoning
            reasoning = f"Selected {selected_strategy} (conf: {confidence:.2f}, risk: {risk_assessment:.2f}) in {phase} phase"
            
            return StrategyControlDecision(
                selected_strategy=selected_strategy,
                strategy_confidence=confidence,
                parameter_adjustments=parameter_adjustments,
                risk_assessment=risk_assessment,
                recommended_phase=phase,
                reasoning=reasoning
            )
    
    def _convert_state_to_tensors(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert state dictionary to tensor format"""
        # Person features (10 features)
        person_features = torch.zeros(10)
        person_attrs = state.get('person_attributes', {})
        person_features[0] = float(person_attrs.get('young', False))
        person_features[1] = float(person_attrs.get('well_dressed', False))
        # Add other person attributes as needed...
        
        # Constraint features (6 features)
        constraint_features = torch.tensor([
            state.get('young_progress', 0.0),
            state.get('well_dressed_progress', 0.0),
            state.get('young_deficit', 0.0) / 600.0,  # Normalize
            state.get('well_dressed_deficit', 0.0) / 600.0,
            state.get('min_constraint_progress', 0.0),
            state.get('max_constraint_progress', 0.0)
        ], dtype=torch.float32)
        
        # Capacity features (4 features)
        capacity_features = torch.tensor([
            state.get('admitted_count', 0.0) / 1000.0,  # Normalize
            state.get('rejected_count', 0.0) / 20000.0,
            state.get('capacity_ratio', 0.0),
            state.get('rejection_ratio', 0.0)
        ], dtype=torch.float32)
        
        # Phase features (3 features - one-hot encoded)
        phase_features = torch.zeros(3)
        phase = state.get('game_phase', 'mid')
        if phase == 'early':
            phase_features[0] = 1.0
        elif phase == 'mid':
            phase_features[1] = 1.0
        else:
            phase_features[2] = 1.0
        
        # Risk features (4 features)
        risk_features = torch.tensor([
            state.get('constraint_risk', 0.0),
            state.get('rejection_risk', 0.0),
            state.get('time_pressure', 0.0),
            state.get('uncertainty', 0.0)
        ], dtype=torch.float32)
        
        # Strategy features (8 features)
        strategy_features = torch.tensor([
            state.get('recent_acceptance_rate', 0.5),
            state.get('strategy_performance', 0.5),
            state.get('decisions_since_switch', 0.0) / 100.0,
            state.get('strategy_confidence', 0.5),
            state.get('parameter_effectiveness', 0.5),
            state.get('constraint_focus_score', 0.5),
            state.get('efficiency_score', 0.5),
            state.get('adaptability_score', 0.5)
        ], dtype=torch.float32)
        
        return {
            'person_features': person_features,
            'constraint_features': constraint_features,
            'capacity_features': capacity_features,
            'phase_features': phase_features,
            'risk_features': risk_features,
            'strategy_features': strategy_features
        }


class StrategyControllerTrainer:
    """Trainer for strategy controller transformer"""
    
    def __init__(
        self,
        model: StrategyControllerTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_data['state_sequences'])
        
        # Calculate losses
        strategy_loss = self.criterion(
            outputs['strategy_logits'],
            batch_data['target_strategies']
        )
        
        confidence_loss = self.mse_loss(
            outputs['confidence'].squeeze(),
            batch_data['target_confidence']
        )
        
        risk_loss = self.mse_loss(
            outputs['risk_assessment'].squeeze(),
            batch_data['target_risk']
        )
        
        # Parameter adjustment losses
        param_losses = {}
        total_param_loss = 0
        for param_name, param_output in outputs['parameter_adjustments'].items():
            if param_name in batch_data['target_parameters']:
                param_loss = self.mse_loss(
                    param_output.squeeze(),
                    batch_data['target_parameters'][param_name]
                )
                param_losses[param_name] = param_loss.item()
                total_param_loss += param_loss
        
        # Total loss
        total_loss = strategy_loss + 0.5 * confidence_loss + 0.5 * risk_loss + 0.3 * total_param_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'strategy_loss': strategy_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'risk_loss': risk_loss.item(),
            'param_loss': total_param_loss.item(),
            **param_losses
        }
    
    def load_training_data(self, data_file: str = "strategy_controller_training_data.json") -> List[Dict[str, Any]]:
        """Load training data from JSON file"""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        training_examples = []
        strategy_to_idx = {strategy: idx for idx, strategy in enumerate(self.model.strategy_vocab)}
        
        for example in data['training_examples']:
            if example['strategy_decision'] in strategy_to_idx:
                # Convert to training format
                state_sequence = []
                for state_dict in example['state_sequence']:
                    # Add missing fields with defaults
                    complete_state = {
                        'person_attributes': {},
                        'young_progress': state_dict.get('young_progress', 0.0),
                        'well_dressed_progress': state_dict.get('well_dressed_progress', 0.0),
                        'admitted_count': state_dict.get('total_admitted', 0),
                        'rejected_count': 0,  # Not available in current format
                        'capacity_ratio': state_dict.get('young_progress', 0.0),  # Approximation
                        'rejection_ratio': 0.0,
                        'constraint_risk': state_dict.get('constraint_risk', 0.0),
                        'game_phase': example['game_phase'],
                        'recent_acceptance_rate': 0.5,
                        'strategy_performance': example['outcome_quality']
                    }
                    state_sequence.append(complete_state)
                
                training_example = {
                    'state_sequence': state_sequence,
                    'target_strategy': strategy_to_idx[example['strategy_decision']],
                    'target_confidence': example['outcome_quality'],
                    'target_risk': 1.0 - example['outcome_quality'],  # Inverse relationship
                    'target_parameters': example.get('parameter_adjustments', {}),
                    'final_success': example['final_success'],
                    'final_rejections': example['final_rejections']
                }
                
                training_examples.append(training_example)
        
        logger.info(f"Loaded {len(training_examples)} training examples")
        return training_examples


def create_strategy_controller() -> StrategyControllerTransformer:
    """Create a new strategy controller transformer"""
    return StrategyControllerTransformer(
        state_dim=64,
        n_strategies=8,
        n_heads=8,
        n_layers=6,
        d_model=256,
        d_ff=1024,
        dropout=0.1,
        max_seq_length=100
    )


def main():
    """Test strategy controller creation and basic functionality"""
    # Create model
    model = create_strategy_controller()
    
    # Test with dummy data
    dummy_state = {
        'person_attributes': {'young': True, 'well_dressed': False},
        'young_progress': 0.6,
        'well_dressed_progress': 0.4,
        'admitted_count': 500,
        'rejected_count': 800,
        'capacity_ratio': 0.5,
        'rejection_ratio': 0.04,
        'constraint_risk': 0.3,
        'game_phase': 'mid',
        'recent_acceptance_rate': 0.6,
        'strategy_performance': 0.7
    }
    
    # Test prediction
    decision = model.predict_strategy([dummy_state], temperature=0.8)
    
    print(f"Strategy Controller Test:")
    print(f"Selected strategy: {decision.selected_strategy}")
    print(f"Confidence: {decision.strategy_confidence:.3f}")
    print(f"Risk assessment: {decision.risk_assessment:.3f}")
    print(f"Parameter adjustments: {decision.parameter_adjustments}")
    print(f"Reasoning: {decision.reasoning}")
    
    print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")


if __name__ == "__main__":
    main()