# Berghain Transformer Model

A Transformer-based approach for solving the Berghain Challenge using behavioral cloning and sequence modeling.

## Overview

This implementation leverages the power of Transformer architectures to learn optimal admission policies from expert demonstrations. Instead of traditional RL approaches, we treat the problem as a sequence modeling task where the model learns to predict optimal actions given the history of states and decisions.

## Key Features

- **Decision Transformer Architecture**: Implements both standard Transformer and Decision Transformer variants
- **Behavioral Cloning**: Learns from elite game demonstrations  
- **Sequence Modeling**: Captures long-range dependencies in game dynamics
- **Configurable Inference**: Temperature and top-k sampling for decision diversity
- **Comprehensive Evaluation**: Built-in evaluation and comparison tools

## Installation

```bash
cd berghain_transformer
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

Train on elite games from scenario 1:

```bash
python run_transformer.py --train
```

### 2. Run a Game

Use the trained model to play a game:

```bash
python run_transformer.py --run --debug
```

### 3. Evaluate Performance

Run comprehensive evaluation:

```bash
python run_transformer.py --evaluate --num-games 20
```

### 4. Compare Models

Compare multiple trained models:

```bash
python run_transformer.py --compare
```

## Architecture Details

### Model Components

1. **State Encoder**: Converts game states into fixed-size feature vectors
   - Person attributes (one-hot encoding)
   - Game progress metrics
   - Constraint satisfaction status
   - Recent decision history
   - Attribute frequency statistics

2. **Transformer Model**: Processes sequences of states to predict actions
   - Multi-head self-attention for capturing dependencies
   - Positional encoding for temporal information
   - Causal masking for autoregressive generation

3. **Decision Transformer Variant**: 
   - Includes returns-to-go conditioning
   - Optimized for offline RL scenarios

### Training Process

1. **Data Preprocessing**: 
   - Extracts sequences from game logs
   - Creates overlapping windows for training
   - Calculates rewards and returns-to-go

2. **Behavioral Cloning**:
   - Supervised learning on expert demonstrations
   - Cross-entropy loss for action prediction
   - Learning rate warmup and gradient clipping

3. **Evaluation**:
   - Success rate measurement
   - Constraint satisfaction analysis
   - Performance comparison with baselines

## Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  type: "decision_transformer"  # or "standard_transformer"
  n_layers: 6
  n_heads: 8
  d_model: 256

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 50
  
inference:
  temperature: 1.0  # Control randomness
  top_k: 1  # Number of actions to sample from
```

## Advanced Usage

### Custom Temperature Sweep

```bash
python -c "
from berghain_transformer.evaluation.evaluate import TransformerEvaluator
from pathlib import Path

evaluator = TransformerEvaluator(
    model_path=Path('models/latest/best_model.pt'),
    encoder_path=Path('models/latest/encoder.pkl')
)

results = evaluator.evaluate_temperature_sweep(
    temperatures=[0.5, 0.7, 1.0, 1.2, 1.5],
    games_per_temp=5
)
"
```

### Direct API Usage

```python
from berghain_transformer.transformer_solver import TransformerSolver
from pathlib import Path

solver = TransformerSolver(
    model_path=Path('models/latest/best_model.pt'),
    encoder_path=Path('models/latest/encoder.pkl'),
    temperature=0.8,
    top_k=2
)

result = solver.run_game()
print(f"Success: {result['success']}")
print(f"Admitted: {result['total_admitted']}")
```

## Model Advantages

### vs LSTM Approach

1. **Long-Range Dependencies**: Direct attention to any past event regardless of distance
2. **Parallel Processing**: All positions processed simultaneously during training
3. **Interpretability**: Attention weights show which past events influence decisions
4. **Offline Learning**: Naturally suited for learning from logged data

### vs Heuristic Solvers

1. **Adaptive Strategy**: Learns patterns from data rather than fixed rules
2. **Generalization**: Can discover novel strategies not encoded in heuristics
3. **Continuous Improvement**: Can be retrained as more data becomes available

## Performance Expectations

Based on behavioral cloning from elite games:

- **Success Rate**: 70-90% on Scenario 1
- **Average Admissions**: 650-750 (optimal range)
- **Average Rejections**: 8000-12000
- **Inference Time**: <0.01s per decision

## Troubleshooting

### Low Success Rate

- Increase training epochs
- Use more elite games for training
- Adjust temperature parameter (try 0.7-0.9)

### Out of Memory

- Reduce batch_size in config
- Decrease seq_length for shorter sequences
- Use CPU if GPU memory insufficient

### Slow Training

- Enable GPU: set device: "cuda" in config
- Reduce model size (n_layers, d_model)
- Use fewer workers for data loading

## Future Improvements

1. **Online Fine-tuning**: Adapt model during gameplay
2. **Multi-Scenario Training**: Single model for all scenarios
3. **Ensemble Methods**: Combine multiple models
4. **Attention Visualization**: Interpret decision reasoning
5. **Reinforcement Learning**: Fine-tune with PPO/SAC after behavioral cloning

## Citation

This implementation is based on:
- Decision Transformer: "Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021)
- Trajectory Transformer: "Offline Reinforcement Learning as One Big Sequence Modeling Problem" (Janner et al., 2021)