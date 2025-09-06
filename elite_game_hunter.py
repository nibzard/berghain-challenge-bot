#!/usr/bin/env python3
"""
ABOUTME: Elite Game Hunter - Optimized runner that only saves games meeting elite performance criteria
ABOUTME: Designed for continuous running on M4 Mac with memory efficiency and parallel execution
"""

import json
import time
import logging
import signal
import sys
import gc
import psutil
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import os

from berghain.core import GameResult
from berghain.runner.game_executor import GameExecutor
from berghain.runner.parallel_runner import GameTask
from berghain.config import ConfigManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elite_hunter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EliteGameConfig:
    """Configuration for the elite game hunter."""
    # Performance thresholds
    max_rejections_threshold: int = 850  # Games with fewer rejections are "elite"
    success_required: bool = True  # Only count successful games as elite
    
    # Strategies to test
    strategies: List[str] = None
    scenario_id: int = 1
    
    # Execution settings
    batch_size: int = 20  # Games per batch
    max_workers: int = 8  # Parallel threads
    max_memory_gb: float = 12.0  # Pause if memory usage exceeds this
    
    # Progress tracking
    stats_save_interval: int = 100  # Save stats every N games
    progress_display_interval: int = 10  # Show progress every N seconds
    
    # Output directories
    elite_games_dir: str = "elite_games"
    stats_file: str = "elite_hunter_stats.json"
    csv_file: str = "elite_games_summary.csv"
    
    def __post_init__(self):
        if self.strategies is None:
            # Best performing strategies based on our analysis
            self.strategies = [
                'dual', 'rbcr', 'rbcr2', 'ultimate3', 'apex', 
                'optimal', 'perfect', 'ultimate3h', 'ultimate2'
            ]


@dataclass
class EliteGameStats:
    """Statistics tracking for elite game hunting."""
    total_games: int = 0
    elite_games: int = 0
    games_per_strategy: Dict[str, int] = None
    elite_per_strategy: Dict[str, int] = None
    best_performance: Dict[str, int] = None  # Best (lowest) rejections per strategy
    start_time: datetime = None
    last_elite_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.games_per_strategy is None:
            self.games_per_strategy = defaultdict(int)
        if self.elite_per_strategy is None:
            self.elite_per_strategy = defaultdict(int)
        if self.best_performance is None:
            self.best_performance = defaultdict(lambda: float('inf'))
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def elite_rate(self) -> float:
        return self.elite_games / max(self.total_games, 1)
    
    @property
    def runtime_hours(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 3600
    
    @property
    def games_per_hour(self) -> float:
        return self.total_games / max(self.runtime_hours, 0.01)
    
    @property
    def elite_per_hour(self) -> float:
        return self.elite_games / max(self.runtime_hours, 0.01)


class EliteGameHunter:
    """Optimized game runner that only saves elite performances."""
    
    def __init__(self, config: EliteGameConfig):
        self.config = config
        self.stats = EliteGameStats()
        self.running = True
        self.executor = None
        self.stats_lock = threading.Lock()
        self.last_progress_display = 0
        
        # Setup config manager
        self.config_manager = ConfigManager()
        
        # Setup output directories
        Path(self.config.elite_games_dir).mkdir(exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load stats if they exist
        self._load_stats()
        
        logger.info(f"🚀 Elite Game Hunter initialized")
        logger.info(f"   Threshold: < {self.config.max_rejections_threshold} rejections")
        logger.info(f"   Strategies: {', '.join(self.config.strategies)}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Max workers: {self.config.max_workers}")
        logger.info(f"   Elite games dir: {self.config.elite_games_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=False)
    
    def _load_stats(self):
        """Load existing statistics if available."""
        if Path(self.config.stats_file).exists():
            try:
                with open(self.config.stats_file, 'r') as f:
                    data = json.load(f)
                
                self.stats.total_games = data.get('total_games', 0)
                self.stats.elite_games = data.get('elite_games', 0)
                self.stats.games_per_strategy = defaultdict(int, data.get('games_per_strategy', {}))
                self.stats.elite_per_strategy = defaultdict(int, data.get('elite_per_strategy', {}))
                self.stats.best_performance = defaultdict(lambda: float('inf'), data.get('best_performance', {}))
                
                if data.get('start_time'):
                    self.stats.start_time = datetime.fromisoformat(data['start_time'])
                if data.get('last_elite_time'):
                    self.stats.last_elite_time = datetime.fromisoformat(data['last_elite_time'])
                
                logger.info(f"📊 Loaded existing stats: {self.stats.total_games} games, {self.stats.elite_games} elite")
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}")
    
    def _save_stats(self):
        """Save current statistics."""
        try:
            data = {
                'total_games': self.stats.total_games,
                'elite_games': self.stats.elite_games,
                'games_per_strategy': dict(self.stats.games_per_strategy),
                'elite_per_strategy': dict(self.stats.elite_per_strategy),
                'best_performance': {k: v for k, v in self.stats.best_performance.items() if v != float('inf')},
                'start_time': self.stats.start_time.isoformat(),
                'last_elite_time': self.stats.last_elite_time.isoformat() if self.stats.last_elite_time else None,
                'elite_rate': self.stats.elite_rate,
                'runtime_hours': self.stats.runtime_hours,
                'games_per_hour': self.stats.games_per_hour,
                'elite_per_hour': self.stats.elite_per_hour
            }
            
            with open(self.config.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def _save_elite_game_csv(self, result: GameResult, strategy: str):
        """Save elite game to CSV summary."""
        csv_path = Path(self.config.csv_file)
        file_exists = csv_path.exists()
        
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write header if new file
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'strategy', 'game_id', 'rejections', 
                        'admitted', 'success', 'duration_seconds'
                    ])
                
                # Write elite game data
                writer.writerow([
                    datetime.now().isoformat(),
                    strategy,
                    result.game_state.game_id,
                    result.game_state.rejected_count,
                    result.game_state.admitted_count,
                    result.success,
                    0.0  # duration not available in GameResult
                ])
                
        except Exception as e:
            logger.error(f"Failed to save elite game to CSV: {e}")
    
    def _is_elite_game(self, result: GameResult) -> bool:
        """Check if a game result meets elite criteria."""
        if self.config.success_required and not result.success:
            return False
        
        return result.game_state.rejected_count < self.config.max_rejections_threshold
    
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        return psutil.virtual_memory().used / (1024 ** 3)
    
    def _should_pause_for_memory(self) -> bool:
        """Check if we should pause due to high memory usage."""
        return self._get_memory_usage_gb() > self.config.max_memory_gb
    
    def _run_single_game(self, strategy: str) -> Tuple[GameResult, bool]:
        """Run a single game and return result + elite status."""
        try:
            # Create game executor
            game_executor = GameExecutor()
            
            # Generate unique solver ID
            solver_id = f"{strategy}_{int(time.time() * 1000) % 100000:05d}"
            
            # Run the game using the correct method signature
            result = game_executor.execute_game(
                scenario_id=self.config.scenario_id,
                strategy_name=strategy,
                solver_id=solver_id,
                enable_high_score_check=True,
                mode="local"
            )
            
            # Check if elite and save accordingly
            is_elite = self._is_elite_game(result)
            
            if is_elite:
                # Save full game log for elite games
                self._save_elite_game_full(result, strategy)
                self._save_elite_game_csv(result, strategy)
                
                # Update stats immediately when elite game is found
                with self.stats_lock:
                    self.stats.elite_games += 1
                    self.stats.elite_per_strategy[strategy] += 1
                    self.stats.last_elite_time = datetime.now()
                
                logger.info(f"🏆 ELITE GAME: {strategy} - {result.game_state.rejected_count} rejections!")
            
            return result, is_elite
            
        except Exception as e:
            logger.error(f"Error running {strategy} game: {e}")
            # Return a failed result using the correct GameResult constructor
            from berghain.core.domain import GameState, GameStatus
            dummy_state = GameState()
            dummy_state.status = GameStatus.FAILED
            dummy_state.rejected_count = 9999
            dummy_result = GameResult(
                game_state=dummy_state,
                decisions=[],
                solver_id="error",
                strategy_params={}
            )
            return dummy_result, False
    
    def _save_elite_game_full(self, result: GameResult, strategy: str):
        """Save full game log for elite games."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elite_{strategy}_{result.game_state.rejected_count}rej_{timestamp}_{result.game_state.game_id}.json"
        filepath = Path(self.config.elite_games_dir) / filename
        
        try:
            # Create comprehensive log with safe serialization
            def safe_serialize_constraint(c):
                return {
                    'attribute': c.attribute,
                    'min_count': c.min_count
                }
            
            def safe_serialize_decision(d):
                return {
                    'person': {
                        'index': d.person.index,
                        'attributes': dict(d.person.attributes)
                    },
                    'accepted': d.accepted,
                    'reasoning': d.reasoning,
                    'timestamp': d.timestamp.isoformat() if hasattr(d, 'timestamp') and d.timestamp else None
                }
            
            log_data = {
                'game_id': result.game_state.game_id,
                'timestamp': timestamp,
                'scenario': self.config.scenario_id,
                'strategy': strategy,
                'success': result.success,
                'rejected_count': result.game_state.rejected_count,
                'admitted_count': result.game_state.admitted_count,
                'constraints': [safe_serialize_constraint(c) for c in result.game_state.constraints],
                'decisions': [safe_serialize_decision(d) for d in result.decisions],
                'elite_threshold': self.config.max_rejections_threshold,
                'hunter_version': '1.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            logger.info(f"💾 Saved elite game: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save elite game log: {e}")
    
    def _display_progress(self):
        """Display current progress statistics."""
        with self.stats_lock:
            elapsed = self.stats.runtime_hours
            
            print(f"\n{'='*80}")
            print(f"🎯 ELITE GAME HUNTER - Progress Report")
            print(f"{'='*80}")
            print(f"⏱️  Runtime: {elapsed:.1f} hours")
            print(f"🎮 Total Games: {self.stats.total_games:,}")
            print(f"🏆 Elite Games: {self.stats.elite_games:,} ({self.stats.elite_rate:.1%})")
            print(f"📊 Rate: {self.stats.games_per_hour:.1f} games/hr, {self.stats.elite_per_hour:.2f} elite/hr")
            print(f"💾 Memory: {self._get_memory_usage_gb():.1f}GB / {self.config.max_memory_gb}GB")
            
            if self.stats.last_elite_time:
                since_elite = (datetime.now() - self.stats.last_elite_time).total_seconds() / 60
                print(f"⏰ Last elite: {since_elite:.0f}m ago")
            
            print(f"\n📈 Performance by Strategy:")
            print(f"{'Strategy':<12} {'Games':<8} {'Elite':<8} {'Best':<8} {'Rate':<8}")
            print("-" * 50)
            
            for strategy in self.config.strategies:
                games = self.stats.games_per_strategy[strategy]
                elite = self.stats.elite_per_strategy[strategy]
                best = self.stats.best_performance[strategy]
                rate = elite / max(games, 1)
                
                best_str = f"{int(best)}" if best != float('inf') else "---"
                print(f"{strategy:<12} {games:<8} {elite:<8} {best_str:<8} {rate:<8.1%}")
            
            print(f"{'='*80}\n")
    
    def _update_stats(self, strategy: str, result: GameResult, is_elite: bool):
        """Update statistics for a completed game."""
        with self.stats_lock:
            self.stats.total_games += 1
            self.stats.games_per_strategy[strategy] += 1
            
            # Elite stats are updated immediately when elite game is found in _run_single_game
            # Update best performance
            if result.game_state.rejected_count < self.stats.best_performance[strategy]:
                self.stats.best_performance[strategy] = result.game_state.rejected_count
    
    def run_batch(self) -> int:
        """Run a batch of games and return number of elite games found."""
        elite_count = 0
        batch_start = time.time()
        
        # Create balanced task list (round-robin strategies)
        tasks = []
        strategy_cycle = iter(self.config.strategies * (self.config.batch_size // len(self.config.strategies) + 1))
        
        for _ in range(self.config.batch_size):
            tasks.append(next(strategy_cycle))
        
        logger.info(f"🚀 Starting batch of {len(tasks)} games...")
        
        # Execute batch with thread pool
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            self.executor = executor
            
            # Submit all tasks
            future_to_strategy = {
                executor.submit(self._run_single_game, strategy): strategy 
                for strategy in tasks
            }
            
            # Process completed games
            for future in as_completed(future_to_strategy):
                if not self.running:
                    break
                    
                strategy = future_to_strategy[future]
                
                try:
                    result, is_elite = future.result()
                    self._update_stats(strategy, result, is_elite)
                    
                    if is_elite:
                        elite_count += 1
                    
                except Exception as e:
                    logger.error(f"Game execution failed for {strategy}: {e}")
                    # Still count as a game attempt
                    with self.stats_lock:
                        self.stats.total_games += 1
                        self.stats.games_per_strategy[strategy] += 1
        
        self.executor = None
        batch_duration = time.time() - batch_start
        
        logger.info(f"✅ Batch complete: {elite_count}/{len(tasks)} elite games in {batch_duration:.1f}s")
        
        return elite_count
    
    def run_continuous(self):
        """Run the elite game hunter continuously."""
        logger.info("🎯 Starting continuous elite game hunting...")
        
        try:
            while self.running:
                # Check memory usage
                if self._should_pause_for_memory():
                    logger.warning(f"⚠️  High memory usage ({self._get_memory_usage_gb():.1f}GB), pausing...")
                    gc.collect()  # Force garbage collection
                    time.sleep(30)  # Wait for memory to stabilize
                    continue
                
                # Run batch
                elite_found = self.run_batch()
                
                # Save stats periodically
                if self.stats.total_games % self.config.stats_save_interval == 0:
                    self._save_stats()
                
                # Display progress
                current_time = time.time()
                if current_time - self.last_progress_display > self.config.progress_display_interval:
                    self._display_progress()
                    self.last_progress_display = current_time
                
                # Brief pause between batches
                if self.running:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.running = False
            self._save_stats()
            self._display_progress()
            logger.info("🏁 Elite Game Hunter stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Elite Game Hunter - Find games with exceptional performance")
    parser.add_argument('--max-rejections', type=int, default=850, help='Maximum rejections for elite status')
    parser.add_argument('--batch-size', type=int, default=20, help='Games per batch')
    parser.add_argument('--max-workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--strategies', nargs='*', help='Strategies to test')
    parser.add_argument('--scenario', type=int, default=1, help='Scenario ID')
    parser.add_argument('--max-memory', type=float, default=12.0, help='Max memory usage in GB')
    
    args = parser.parse_args()
    
    # Create configuration
    config = EliteGameConfig(
        max_rejections_threshold=args.max_rejections,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        scenario_id=args.scenario,
        max_memory_gb=args.max_memory
    )
    
    if args.strategies:
        config.strategies = args.strategies
    
    # Create and run hunter
    hunter = EliteGameHunter(config)
    hunter.run_continuous()


if __name__ == "__main__":
    main()