"""Evaluation scripts for transformer models."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from ..transformer_solver import TransformerSolver


class TransformerEvaluator:
    """Evaluates transformer models on the Berghain Challenge."""
    
    def __init__(
        self,
        model_path: Path,
        encoder_path: Path,
        scenario: int = 1,
        debug: bool = False
    ):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scenario = scenario
        self.debug = debug
        
    def evaluate_single_game(
        self,
        temperature: float = 1.0,
        top_k: int = 1
    ) -> Dict:
        """Run a single game evaluation."""
        solver = TransformerSolver(
            model_path=self.model_path,
            encoder_path=self.encoder_path,
            scenario=self.scenario,
            temperature=temperature,
            top_k=top_k,
            debug=self.debug
        )
        
        return solver.run_game()
    
    def evaluate_multiple_games(
        self,
        num_games: int = 10,
        temperature: float = 1.0,
        top_k: int = 1,
        num_workers: int = 4
    ) -> List[Dict]:
        """Evaluate model across multiple games."""
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.evaluate_single_game, temperature, top_k)
                for _ in range(num_games)
            ]
            
            for future in tqdm(as_completed(futures), total=num_games, desc="Running games"):
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception as e:
                    print(f"Game failed: {e}")
                    results.append({'success': False, 'error': str(e)})
        
        return results
    
    def evaluate_temperature_sweep(
        self,
        temperatures: List[float],
        games_per_temp: int = 5
    ) -> Dict[float, List[Dict]]:
        """Evaluate model across different temperature settings."""
        results = {}
        
        for temp in temperatures:
            print(f"\nEvaluating temperature={temp}")
            game_results = self.evaluate_multiple_games(
                num_games=games_per_temp,
                temperature=temp
            )
            results[temp] = game_results
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze evaluation results."""
        successful_games = [r for r in results if r.get('success', False)]
        
        if not successful_games:
            return {
                'success_rate': 0.0,
                'num_games': len(results),
                'successful_games': 0
            }
        
        analysis = {
            'success_rate': len(successful_games) / len(results),
            'num_games': len(results),
            'successful_games': len(successful_games),
            'avg_admitted': np.mean([r['total_admitted'] for r in successful_games]),
            'std_admitted': np.std([r['total_admitted'] for r in successful_games]),
            'avg_rejected': np.mean([r['total_rejected'] for r in successful_games]),
            'std_rejected': np.std([r['total_rejected'] for r in successful_games]),
            'avg_time': np.mean([r['elapsed_time'] for r in successful_games]),
            'constraint_satisfaction': self._analyze_constraints(successful_games)
        }
        
        return analysis
    
    def _analyze_constraints(self, results: List[Dict]) -> Dict:
        """Analyze constraint satisfaction patterns."""
        all_constraints = []
        
        for result in results:
            for constraint in result.get('constraint_status', []):
                all_constraints.append({
                    'attribute': constraint['attribute'],
                    'required': constraint['required'],
                    'achieved': constraint['current'],
                    'satisfied': constraint['satisfied'],
                    'surplus': constraint['current'] - constraint['required']
                })
        
        df = pd.DataFrame(all_constraints)
        
        if df.empty:
            return {}
        
        constraint_stats = {}
        for attr in df['attribute'].unique():
            attr_df = df[df['attribute'] == attr]
            constraint_stats[attr] = {
                'satisfaction_rate': attr_df['satisfied'].mean(),
                'avg_achieved': attr_df['achieved'].mean(),
                'avg_surplus': attr_df['surplus'].mean(),
                'std_surplus': attr_df['surplus'].std()
            }
        
        return constraint_stats
    
    def plot_results(self, results: Dict, save_path: Optional[Path] = None):
        """Create visualization of evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Success rate over games
        if isinstance(results, list):
            successes = [1 if r.get('success', False) else 0 for r in results]
            cumulative_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
            
            axes[0, 0].plot(cumulative_success)
            axes[0, 0].set_xlabel('Game Number')
            axes[0, 0].set_ylabel('Cumulative Success Rate')
            axes[0, 0].set_title('Success Rate Over Games')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Admission/Rejection distribution
            successful_games = [r for r in results if r.get('success', False)]
            if successful_games:
                admissions = [r['total_admitted'] for r in successful_games]
                rejections = [r['total_rejected'] for r in successful_games]
                
                axes[0, 1].hist(admissions, bins=20, alpha=0.7, label='Admissions')
                axes[0, 1].set_xlabel('Count')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Admission Distribution')
                axes[0, 1].legend()
                
                axes[1, 0].hist(rejections, bins=20, alpha=0.7, label='Rejections', color='orange')
                axes[1, 0].set_xlabel('Count')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Rejection Distribution')
                axes[1, 0].legend()
                
                # Constraint satisfaction
                constraint_data = []
                for r in successful_games:
                    for c in r.get('constraint_status', []):
                        constraint_data.append({
                            'attribute': c['attribute'],
                            'surplus': c['current'] - c['required']
                        })
                
                if constraint_data:
                    df = pd.DataFrame(constraint_data)
                    df.boxplot(column='surplus', by='attribute', ax=axes[1, 1])
                    axes[1, 1].set_xlabel('Attribute')
                    axes[1, 1].set_ylabel('Surplus/Deficit')
                    axes[1, 1].set_title('Constraint Achievement')
                    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()


def compare_models(
    models: List[Tuple[str, Path, Path]],  # (name, model_path, encoder_path)
    scenario: int = 1,
    games_per_model: int = 10,
    save_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Compare multiple transformer models."""
    
    comparison_results = []
    
    for name, model_path, encoder_path in models:
        print(f"\nEvaluating model: {name}")
        
        evaluator = TransformerEvaluator(
            model_path=model_path,
            encoder_path=encoder_path,
            scenario=scenario
        )
        
        results = evaluator.evaluate_multiple_games(
            num_games=games_per_model
        )
        
        analysis = evaluator.analyze_results(results)
        analysis['model'] = name
        
        comparison_results.append(analysis)
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"{name}_results.json", 'w') as f:
                json.dump({
                    'raw_results': results,
                    'analysis': analysis
                }, f, indent=2)
    
    # Create comparison dataframe
    df = pd.DataFrame(comparison_results)
    
    if save_dir:
        df.to_csv(save_dir / 'model_comparison.csv', index=False)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Success rates
        df.plot(x='model', y='success_rate', kind='bar', ax=axes[0])
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Model Success Rates')
        axes[0].set_ylim([0, 1])
        
        # Average admissions
        df.plot(x='model', y='avg_admitted', kind='bar', ax=axes[1])
        axes[1].set_ylabel('Average Admissions')
        axes[1].set_title('Average Admissions per Game')
        
        # Average time
        df.plot(x='model', y='avg_time', kind='bar', ax=axes[2])
        axes[2].set_ylabel('Time (seconds)')
        axes[2].set_title('Average Game Duration')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png', dpi=150)
    
    return df


def evaluate_against_baseline(
    transformer_model_path: Path,
    transformer_encoder_path: Path,
    baseline_results_path: Path,
    scenario: int = 1,
    num_games: int = 20
) -> Dict:
    """Compare transformer model against baseline results."""
    
    # Load baseline results
    with open(baseline_results_path) as f:
        baseline_data = json.load(f)
    
    # Run transformer evaluation
    evaluator = TransformerEvaluator(
        model_path=transformer_model_path,
        encoder_path=transformer_encoder_path,
        scenario=scenario
    )
    
    transformer_results = evaluator.evaluate_multiple_games(num_games=num_games)
    transformer_analysis = evaluator.analyze_results(transformer_results)
    
    # Extract baseline statistics
    baseline_analysis = {
        'success_rate': baseline_data.get('success_rate', 0),
        'avg_admitted': baseline_data.get('avg_admitted', 0),
        'avg_rejected': baseline_data.get('avg_rejected', 0)
    }
    
    # Compare
    comparison = {
        'transformer': transformer_analysis,
        'baseline': baseline_analysis,
        'improvement': {
            'success_rate': transformer_analysis['success_rate'] - baseline_analysis['success_rate'],
            'admitted_diff': transformer_analysis.get('avg_admitted', 0) - baseline_analysis.get('avg_admitted', 0)
        }
    }
    
    # Print comparison
    print("\n" + "="*50)
    print("Model Comparison Results")
    print("="*50)
    print(f"\nTransformer Success Rate: {transformer_analysis['success_rate']:.2%}")
    print(f"Baseline Success Rate: {baseline_analysis['success_rate']:.2%}")
    print(f"Improvement: {comparison['improvement']['success_rate']:.2%}")
    print("\n" + "="*50)
    
    return comparison