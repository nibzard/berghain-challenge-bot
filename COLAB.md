# Google Colab Training Instructions

This guide explains how to train the RL LSTM model using Google Colab's free GPU resources.

## Prerequisites

- Google account with Colab access
- Repository pushed to GitHub with Git LFS training data
- 1.2GB training dataset (1,200 games across 12 strategies)

## Setup Instructions

### 1. Create New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### 2. Enable GPU Runtime

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 3. Clone Repository and Setup Environment

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/berghain-challenge-bot.git
%cd berghain-challenge-bot

# Install Git LFS and pull training data
!git lfs install
!git lfs pull

# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install -r requirements.txt

# Verify GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 4. Verify Training Data

```python
import os
import json

# Check training data size
!ls -lh game_logs/*.json | wc -l
!du -sh game_logs/

# Sample a training file
sample_files = [f for f in os.listdir('game_logs/') if f.startswith('game_') and f.endswith('.json')]
print(f"Found {len(sample_files)} game files")

# Load a sample game
with open(f'game_logs/{sample_files[0]}', 'r') as f:
    sample_game = json.load(f)
    print(f"Sample game strategy: {sample_game['strategy_name']}")
    print(f"Total decisions: {sample_game['final_stats']['total_people_seen']}")
    print(f"Game successful: {sample_game['final_stats']['game_successful']}")
```

## Training the Model

### 5. Basic Training Script

```python
# Enable RL LSTM imports (uncomment in solvers/__init__.py if needed)
import sys
sys.path.append('/content/berghain-challenge-bot')

# Import training components
from berghain.training.lstm_policy import train_model, PolicyInference
from berghain.training.data_preprocessor import prepare_training_data

# Prepare training data
print("Preparing training data...")
train_data, val_data = prepare_training_data('game_logs/', test_split=0.2)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Train the model
print("Starting training...")
model = train_model(
    train_data=train_data,
    val_data=val_data,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    hidden_size=128,
    num_layers=2
)

# Save the trained model
model_path = 'trained_rl_lstm_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
```

### 6. Advanced Training Configuration

```python
# For larger models and longer training
model = train_model(
    train_data=train_data,
    val_data=val_data,
    epochs=100,           # More training epochs
    batch_size=64,        # Larger batch size for GPU
    learning_rate=0.0005, # Lower learning rate for stability
    hidden_size=256,      # Larger LSTM hidden size
    num_layers=3,         # Deeper network
    dropout=0.2,          # Regularization
    early_stopping=True,  # Stop if validation loss stops improving
    patience=10           # Early stopping patience
)
```

### 7. Monitor Training Progress

```python
# Training will output progress like:
# Epoch 1/50: Train Loss: 0.685, Val Loss: 0.632, Accuracy: 67.3%
# Epoch 2/50: Train Loss: 0.598, Val Loss: 0.587, Accuracy: 72.1%
# ...
# Best model saved at epoch 23 with validation accuracy: 85.7%

# Plot training history
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.show()

# plot_training_history(model.training_history)
```

## Testing the Trained Model

### 8. Test Model Performance

```python
# Load the trained model
policy = PolicyInference(model_path)

# Test on a sample game scenario
test_person = {
    'attributes': {'young': True, 'well_dressed': False},
    'game_state': {
        'young_count': 450,
        'well_dressed_count': 380,
        'total_admitted': 830,
        'total_rejected': 1200,
        'progress': 0.6
    }
}

decision, confidence = policy.predict(test_person)
print(f"Decision: {'ADMIT' if decision else 'REJECT'}")
print(f"Confidence: {confidence:.3f}")
```

### 9. Download Trained Model

```python
# Download the model to your local machine
from google.colab import files

files.download('trained_rl_lstm_model.pth')
```

## Resource Management

### Memory and Runtime Tips

- **Free Tier Limits**: 12-15GB RAM, 12-hour runtime
- **Pro Tier**: 25GB RAM, 24-hour runtime, priority GPU access
- **Monitor Usage**: Runtime → View resources

### Saving Checkpoints

```python
# Save intermediate checkpoints during training
def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# Resume from checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

## Expected Training Time

- **1.2M decision events**: ~2-4 hours training on T4 GPU
- **50 epochs**: Usually sufficient for convergence
- **Early stopping**: May complete in 20-30 epochs if model converges

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or hidden_size
2. **Git LFS timeout**: Re-run `!git lfs pull`
3. **CUDA out of memory**: Restart runtime and reduce model size
4. **Training too slow**: Enable GPU runtime if not already

### Performance Tips

```python
# Enable mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Use in training loop for 2x speedup
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Integration Back to Local

Once trained, download the model and integrate it back into your local codebase:

```python
# In your local berghain codebase
from berghain.training.lstm_policy import PolicyInference

# Load the Colab-trained model
policy = PolicyInference('path/to/trained_rl_lstm_model.pth')

# Use in your solvers
decision = policy.predict(person_data)
```

## Next Steps

1. Train the basic model first with default parameters
2. Experiment with hyperparameters for better performance
3. Compare RL LSTM performance against existing strategies
4. Deploy the best model as a new strategy in the game system

Happy training! 🚀