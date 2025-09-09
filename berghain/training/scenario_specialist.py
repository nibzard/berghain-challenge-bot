# ABOUTME: Scenario-specific model training to create specialized variants of the strategy controller  
# ABOUTME: Fine-tunes the base model for optimal performance on specific game scenarios

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .strategy_controller import StrategyControllerTransformer, create_strategy_controller
from .train_strategy_controller import StrategyControllerDataset, load_training_data
from ..core import GameState, Person

logger = logging.getLogger(__name__)

@dataclass
class ScenarioSpecialization:
    """Configuration for scenario-specific training"""
    scenario_id: int
    base_model_path: str
    specialized_model_path: str
    training_games_filter: Dict[str, Any]  # Criteria for selecting training games
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    num_epochs: int = 10
    scenario_weight: float = 2.0  # Weight boost for scenario-specific examples
    constraint_focus: List[str] = None  # Attributes to focus on for this scenario

class ScenarioSpecialistTrainer:
    """Trainer for creating scenario-specific model variants"""
    
    def __init__(self, base_model_path: str = "models/strategy_controller/trained_strategy_controller.pt"):
        self.base_model_path = base_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load base model
        self.base_model = self._load_base_model()
        
        # Scenario configurations
        self.scenario_configs = {
            1: {
                'constraint_focus': ['young', 'well_dressed'],
                'training_filter': {'min_rejections': 600, 'max_rejections': 900},
                'specialization_weight': 2.5,
                'description': 'Dual constraint optimization (young + well_dressed)'
            },
            2: {
                'constraint_focus': ['creative'],  # Assuming scenario 2 focuses on creative
                'training_filter': {'min_rejections': 500, 'max_rejections': 1000},
                'specialization_weight': 3.0,  # Higher weight for rare attribute scenarios
                'description': 'Single rare attribute focus (creative)'
            },
            3: {
                'constraint_focus': ['young', 'well_dressed', 'creative'],  # Multi-constraint
                'training_filter': {'min_rejections': 700, 'max_rejections': 1200},
                'specialization_weight': 2.0,
                'description': 'Multi-constraint optimization'
            }
        }
    
    def _load_base_model(self) -> StrategyControllerTransformer:
        """Load the base trained model"""
        if not Path(self.base_model_path).exists():
            raise FileNotFoundError(f"Base model not found at {self.base_model_path}")
        
        model = create_strategy_controller()
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        
        logger.info(f"Loaded base model from {self.base_model_path}")
        return model
    
    def _filter_training_data_for_scenario(self, 
                                          training_data: List[Dict], 
                                          scenario_id: int) -> List[Dict]:
        """Filter training data for scenario-specific training"""
        
        if scenario_id not in self.scenario_configs:
            logger.warning(f"No configuration found for scenario {scenario_id}")
            return training_data
        
        config = self.scenario_configs[scenario_id]
        training_filter = config['training_filter']
        constraint_focus = config['constraint_focus']
        
        filtered_data = []
        
        for example in training_data:
            # Parse game ID to extract scenario information if available
            game_id = example.get('game_id', '')
            
            # Filter by rejection count if available in metadata
            metadata = example.get('metadata', {})
            final_rejections = metadata.get('final_rejections')
            
            if final_rejections:
                if final_rejections < training_filter.get('min_rejections', 0):
                    continue
                if final_rejections > training_filter.get('max_rejections', float('inf')):
                    continue
            
            # Boost weight for examples that involve our target constraints
            example_copy = example.copy()
            
            # Check if this example involves our constraint focus
            state_sequence = example.get('state_sequence', [])
            involves_constraints = False
            
            for state in state_sequence:
                person_attrs = state.get('person_attributes', {})
                for constraint_attr in constraint_focus:
                    if person_attrs.get(constraint_attr, False):
                        involves_constraints = True
                        break
                if involves_constraints:
                    break
            
            # Apply scenario weight boost
            if involves_constraints:
                example_copy['weight'] = config['specialization_weight']
            else:
                example_copy['weight'] = 1.0
            
            filtered_data.append(example_copy)
        
        logger.info(f"Filtered {len(training_data)} → {len(filtered_data)} examples for scenario {scenario_id}")
        
        # Show weight distribution
        weights = [ex.get('weight', 1.0) for ex in filtered_data]
        avg_weight = sum(weights) / len(weights) if weights else 0
        high_weight_count = sum(1 for w in weights if w > 1.5)
        
        logger.info(f"Weight distribution: avg={avg_weight:.2f}, high-weight examples={high_weight_count}")
        
        return filtered_data
    
    def _create_scenario_specific_loss(self, scenario_id: int) -> callable:
        """Create scenario-specific loss function"""
        config = self.scenario_configs.get(scenario_id, {})
        constraint_focus = config.get('constraint_focus', [])
        
        def scenario_loss(strategy_logits, strategy_targets, param_outputs, param_targets, example_weights=None):
            """Scenario-specific loss with constraint focus"""
            
            # Base strategy prediction loss
            strategy_loss = nn.CrossEntropyLoss(reduction='none')(strategy_logits, strategy_targets)
            
            # Parameter adjustment loss  
            param_loss = nn.MSELoss(reduction='none')(param_outputs, param_targets).mean(dim=-1)
            
            # Apply example weights if provided
            if example_weights is not None:
                strategy_loss = strategy_loss * example_weights
                param_loss = param_loss * example_weights
            
            # Scenario-specific weighting
            # For constraint-focused scenarios, weight strategy decisions more heavily
            if len(constraint_focus) <= 2:  # Simple scenarios
                total_loss = 0.7 * strategy_loss.mean() + 0.3 * param_loss.mean()
            else:  # Complex scenarios
                total_loss = 0.6 * strategy_loss.mean() + 0.4 * param_loss.mean()
            
            return total_loss, strategy_loss.mean(), param_loss.mean()
        
        return scenario_loss
    
    async def train_scenario_specialist(self, 
                                       scenario_id: int,
                                       output_path: Optional[str] = None) -> str:
        """Train a scenario-specific model variant"""
        
        if scenario_id not in self.scenario_configs:
            raise ValueError(f"No configuration available for scenario {scenario_id}")
        
        config = self.scenario_configs[scenario_id]
        logger.info(f"Training scenario specialist for scenario {scenario_id}: {config['description']}")
        
        # Set output path
        if output_path is None:
            output_dir = Path("models/strategy_controller/specialists")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"scenario_{scenario_id}_specialist.pt")
        
        # Load and filter training data
        logger.info("Loading training data...")
        training_data = load_training_data("training_data/strategy_controller_training.json")
        
        scenario_data = self._filter_training_data_for_scenario(training_data, scenario_id)
        
        if len(scenario_data) < 10:
            logger.warning(f"Very little training data for scenario {scenario_id} ({len(scenario_data)} examples)")
        
        # Create dataset
        dataset = StrategyControllerDataset(scenario_data)
        
        # Create specialized model (copy of base model)
        specialist_model = create_strategy_controller()
        specialist_model.load_state_dict(self.base_model.state_dict())
        specialist_model.to(self.device)
        
        # Freeze early layers, only fine-tune later layers
        for name, param in specialist_model.named_parameters():
            if 'transformer_layers.0' in name or 'transformer_layers.1' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        logger.info(f"Freezing early transformer layers for fine-tuning")
        
        # Set up training
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, specialist_model.parameters()),
            lr=1e-5,  # Very low learning rate for fine-tuning
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, verbose=True
        )
        
        scenario_loss_fn = self._create_scenario_specific_loss(scenario_id)
        
        # Training loop
        num_epochs = 15  # More epochs for fine-tuning
        best_loss = float('inf')
        patience = 7
        patience_counter = 0
        
        specialist_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Simple batch processing (since dataset is small)
            for i in range(0, len(dataset), 16):  # Batch size 16
                batch_end = min(i + 16, len(dataset))
                batch_data = [dataset[j] for j in range(i, batch_end)]
                
                # Prepare batch
                strategy_targets = []
                param_targets = []
                input_sequences = []
                weights = []
                
                for item in batch_data:
                    input_sequences.append(item['state_sequence'])
                    strategy_targets.append(item['strategy_target'])
                    param_targets.append(item['parameter_target'])
                    weights.append(item.get('weight', 1.0))
                
                # Convert to tensors
                strategy_targets = torch.tensor(strategy_targets, dtype=torch.long).to(self.device)
                param_targets = torch.tensor(param_targets, dtype=torch.float32).to(self.device)
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
                    strategy_logits, param_outputs = specialist_model(input_sequences)
                    
                    # Compute loss
                    loss, strategy_loss, param_loss = scenario_loss_fn(
                        strategy_logits, strategy_targets, param_outputs, param_targets, weights
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(specialist_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            scheduler.step(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                torch.save(specialist_model.state_dict(), output_path)
                logger.info(f"New best model saved: {avg_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save training metadata
        metadata = {
            'scenario_id': scenario_id,
            'base_model_path': self.base_model_path,
            'specialist_model_path': output_path,
            'config': config,
            'training_examples': len(scenario_data),
            'final_loss': best_loss,
            'epochs_trained': epoch + 1
        }
        
        metadata_path = output_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Scenario {scenario_id} specialist training complete!")
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Final loss: {best_loss:.4f}")
        
        return output_path
    
    async def train_all_scenarios(self) -> Dict[int, str]:
        """Train specialists for all configured scenarios"""
        results = {}
        
        for scenario_id in self.scenario_configs.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training specialist for scenario {scenario_id}")
            
            try:
                model_path = await self.train_scenario_specialist(scenario_id)
                results[scenario_id] = model_path
                logger.info(f"✅ Scenario {scenario_id} specialist complete")
            except Exception as e:
                logger.error(f"❌ Failed to train scenario {scenario_id} specialist: {e}")
                results[scenario_id] = None
        
        return results

class ScenarioSpecialistEvaluator:
    """Evaluator for scenario specialist models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def compare_base_vs_specialist(self, scenario_id: int, num_games: int = 10) -> Dict:
        """Compare base model vs scenario specialist performance"""
        from berghain.runner import ParallelRunner
        from berghain.runner.parallel_runner import GameTask
        from berghain.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # Test base model
        base_config = config_manager.get_strategy_config('hybrid_transformer')
        if not base_config:
            base_config = {
                'name': 'Hybrid Transformer',
                'parameters': {
                    'model_path': None,  # Use base model
                    'device': 'cpu',
                    'temperature': 0.3
                }
            }
        
        # Test specialist model
        specialist_path = f"models/strategy_controller/specialists/scenario_{scenario_id}_specialist.pt"
        specialist_config = base_config.copy()
        specialist_config['parameters']['model_path'] = specialist_path
        specialist_config['name'] = f'Scenario {scenario_id} Specialist'
        
        # Run tests
        results = {}
        
        for model_type, config in [('base', base_config), ('specialist', specialist_config)]:
            # Set up runner
            runner = ParallelRunner(max_workers=min(num_games, 5))
            
            # Create tasks
            tasks = []
            for i in range(num_games):
                tasks.append(GameTask(
                    scenario_id=scenario_id,
                    strategy_name='hybrid_transformer',
                    solver_id=f"eval_{model_type}_{i:03d}",
                    strategy_params=config,
                    enable_high_score_check=False,
                    mode='local'
                ))
            
            # Run games
            batch_result = runner.run_batch(tasks)
            successful_results = [r for r in batch_result.results if r.success]
            
            if successful_results:
                rejections = [r.game_state.rejected_count for r in successful_results]
                avg_rejections = sum(rejections) / len(rejections)
                min_rejections = min(rejections)
                std_rejections = (sum((r - avg_rejections) ** 2 for r in rejections) / len(rejections)) ** 0.5
                
                results[model_type] = {
                    'success_rate': len(successful_results) / num_games,
                    'avg_rejections': avg_rejections,
                    'min_rejections': min_rejections,
                    'std_rejections': std_rejections,
                    'successful_games': len(successful_results)
                }
            else:
                results[model_type] = {
                    'success_rate': 0.0,
                    'avg_rejections': None,
                    'successful_games': 0
                }
        
        # Calculate improvement
        if results['base']['success_rate'] > 0 and results['specialist']['success_rate'] > 0:
            improvement = (results['base']['avg_rejections'] - results['specialist']['avg_rejections']) / results['base']['avg_rejections']
            results['improvement'] = improvement
        else:
            results['improvement'] = None
        
        return results


async def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train scenario-specific model variants")
    parser.add_argument('--scenario', type=int, help='Specific scenario to train (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', help='Train specialists for all scenarios')
    parser.add_argument('--evaluate', type=int, help='Evaluate specialist vs base for scenario')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluator = ScenarioSpecialistEvaluator()
        results = await evaluator.compare_base_vs_specialist(args.evaluate)
        
        print(f"📊 Scenario {args.evaluate} Evaluation Results:")
        print(f"Base Model: {results['base']['avg_rejections']:.1f} ± {results['base']['std_rejections']:.1f}")
        print(f"Specialist: {results['specialist']['avg_rejections']:.1f} ± {results['specialist']['std_rejections']:.1f}")
        
        if results['improvement']:
            print(f"Improvement: {results['improvement']*100:.1f}%")
        
    else:
        trainer = ScenarioSpecialistTrainer()
        
        if args.all:
            results = await trainer.train_all_scenarios()
            print("\n🎉 All scenario specialists trained!")
            for scenario_id, model_path in results.items():
                if model_path:
                    print(f"   Scenario {scenario_id}: {model_path}")
                else:
                    print(f"   Scenario {scenario_id}: FAILED")
        
        elif args.scenario:
            model_path = await trainer.train_scenario_specialist(args.scenario)
            print(f"✅ Scenario {args.scenario} specialist trained: {model_path}")
        
        else:
            print("Please specify --scenario N, --all, or --evaluate N")

if __name__ == "__main__":
    asyncio.run(main())