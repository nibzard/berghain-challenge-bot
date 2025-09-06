#!/usr/bin/env python3
"""Main script to train and run the Berghain Transformer model."""

import argparse
import yaml
import torch
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from berghain_transformer.training.behavioral_cloning import train_behavioral_cloning
from berghain_transformer.transformer_solver import TransformerSolver
from berghain_transformer.evaluation.evaluate import (
    TransformerEvaluator, 
    compare_models, 
    evaluate_against_baseline
)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_model(config: dict, args):
    """Train a new transformer model."""
    print("\n" + "="*60)
    print("Training Transformer Model for Berghain Challenge")
    print("="*60)
    
    # Set up paths
    game_logs_path = Path(config['paths']['game_logs'])
    models_dir = Path(config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['model']['type']}_{timestamp}"
    save_dir = models_dir / model_name
    
    print(f"\nModel will be saved to: {save_dir}")
    print(f"Training on scenario: {config['training']['data']['scenario']}")
    print(f"Using {'elite' if config['training']['data']['elite_only'] else 'all'} games")
    
    # Train model
    model, encoder = train_behavioral_cloning(
        game_logs_path=game_logs_path,
        save_dir=save_dir,
        model_type=config['model']['type'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        seq_length=config['training']['data']['seq_length'],
        learning_rate=config['training']['learning_rate'],
        elite_only=config['training']['data']['elite_only'],
        scenario=config['training']['data']['scenario'],
        use_wandb=config['wandb']['enabled'],
        wandb_project=config['wandb']['project']
    )
    
    print(f"\nTraining complete! Model saved to: {save_dir}")
    
    # Save config for reference
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return save_dir


def run_game(config: dict, args):
    """Run a single game with the transformer model."""
    print("\n" + "="*60)
    print("Running Berghain Challenge with Transformer Model")
    print("="*60)
    
    # Load model
    if args.model_path:
        model_path = Path(args.model_path) / 'best_model.pt'
        encoder_path = Path(args.model_path) / 'encoder.pkl'
    else:
        # Use most recent model
        models_dir = Path(config['paths']['models_dir'])
        model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
        if not model_dirs:
            print("No trained models found! Train a model first with --train")
            return
        
        latest_model_dir = model_dirs[-1]
        model_path = latest_model_dir / 'best_model.pt'
        encoder_path = latest_model_dir / 'encoder.pkl'
        print(f"Using model: {latest_model_dir.name}")
    
    # Create solver
    solver = TransformerSolver(
        model_path=model_path,
        encoder_path=encoder_path,
        scenario=args.scenario or config['api']['scenario'],
        temperature=args.temperature or config['inference']['temperature'],
        top_k=args.top_k or config['inference']['top_k'],
        debug=args.debug
    )
    
    # Run game
    print(f"\nStarting game (Scenario {solver.scenario})...")
    result = solver.run_game()
    
    # Save result
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"game_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nGame result saved to: {result_file}")
    
    return result


def evaluate_model(config: dict, args):
    """Evaluate a transformer model."""
    print("\n" + "="*60)
    print("Evaluating Transformer Model")
    print("="*60)
    
    # Load model
    if args.model_path:
        model_path = Path(args.model_path) / 'best_model.pt'
        encoder_path = Path(args.model_path) / 'encoder.pkl'
    else:
        # Use most recent model
        models_dir = Path(config['paths']['models_dir'])
        model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
        if not model_dirs:
            print("No trained models found! Train a model first with --train")
            return
        
        latest_model_dir = model_dirs[-1]
        model_path = latest_model_dir / 'best_model.pt'
        encoder_path = latest_model_dir / 'encoder.pkl'
        print(f"Using model: {latest_model_dir.name}")
    
    # Create evaluator
    evaluator = TransformerEvaluator(
        model_path=model_path,
        encoder_path=encoder_path,
        scenario=args.scenario or config['api']['scenario'],
        debug=args.debug
    )
    
    # Run evaluation
    num_games = args.num_games or config['evaluation']['num_games']
    print(f"\nRunning {num_games} evaluation games...")
    
    results = evaluator.evaluate_multiple_games(
        num_games=num_games,
        temperature=config['inference']['temperature'],
        top_k=config['inference']['top_k'],
        num_workers=config['evaluation']['num_workers']
    )
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Success Rate: {analysis['success_rate']:.2%}")
    print(f"Games Played: {analysis['num_games']}")
    print(f"Successful Games: {analysis['successful_games']}")
    
    if analysis['successful_games'] > 0:
        print(f"\nAverage Admitted: {analysis['avg_admitted']:.1f} ± {analysis['std_admitted']:.1f}")
        print(f"Average Rejected: {analysis['avg_rejected']:.1f} ± {analysis['std_rejected']:.1f}")
        print(f"Average Time: {analysis['avg_time']:.2f} seconds")
        
        print("\nConstraint Satisfaction:")
        for attr, stats in analysis['constraint_satisfaction'].items():
            print(f"  {attr}: {stats['satisfaction_rate']:.2%} success rate")
            print(f"    Average achieved: {stats['avg_achieved']:.1f}")
            print(f"    Average surplus: {stats['avg_surplus']:.1f} ± {stats['std_surplus']:.1f}")
    
    # Save results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    with open(results_dir / f"evaluation_{timestamp}.json", 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'analysis': analysis
        }, f, indent=2)
    
    # Create plots
    evaluator.plot_results(results, save_path=results_dir / f"evaluation_{timestamp}.png")
    
    print(f"\nResults saved to: {results_dir}")
    
    return analysis


def compare_models_cmd(config: dict, args):
    """Compare multiple transformer models."""
    print("\n" + "="*60)
    print("Comparing Transformer Models")
    print("="*60)
    
    models_dir = Path(config['paths']['models_dir'])
    
    # Get all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if len(model_dirs) < 2:
        print("Need at least 2 models to compare. Train more models first.")
        return
    
    # Prepare models for comparison
    models_to_compare = []
    for model_dir in model_dirs[-3:]:  # Compare last 3 models
        model_path = model_dir / 'best_model.pt'
        encoder_path = model_dir / 'encoder.pkl'
        
        if model_path.exists() and encoder_path.exists():
            models_to_compare.append((
                model_dir.name,
                model_path,
                encoder_path
            ))
    
    print(f"\nComparing {len(models_to_compare)} models:")
    for name, _, _ in models_to_compare:
        print(f"  - {name}")
    
    # Run comparison
    results_dir = Path(config['paths']['results_dir']) / "comparisons"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df = compare_models(
        models=models_to_compare,
        scenario=config['api']['scenario'],
        games_per_model=args.num_games or 10,
        save_dir=results_dir
    )
    
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    print(comparison_df.to_string())
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Berghain Transformer Model")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config' / 'config.yaml',
                       help='Path to configuration file')
    
    # Commands
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--run', action='store_true', help='Run a single game')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    
    # Model selection
    parser.add_argument('--model-path', type=Path, help='Path to specific model directory')
    
    # Game parameters
    parser.add_argument('--scenario', type=int, help='Scenario to play (1, 2, or 3)')
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, help='Top-k sampling parameter')
    
    # Evaluation parameters
    parser.add_argument('--num-games', type=int, help='Number of games to run')
    
    # Other
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.train:
        train_model(config, args)
    elif args.run:
        run_game(config, args)
    elif args.evaluate:
        evaluate_model(config, args)
    elif args.compare:
        compare_models_cmd(config, args)
    else:
        print("Please specify a command: --train, --run, --evaluate, or --compare")
        parser.print_help()


if __name__ == "__main__":
    main()