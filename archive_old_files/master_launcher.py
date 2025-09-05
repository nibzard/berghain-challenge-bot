# ABOUTME: Master launcher that coordinates parallel solvers with TUI streaming
# ABOUTME: The ultimate Berghain optimization system with real-time monitoring

import asyncio
import sys
import argparse
import signal
from pathlib import Path
from typing import List, Dict
import threading
import time

from stream_bridge import StreamBridge, TUIStreamAdapter
from parallel_runner import ParallelRunner, SolverConfig
from berghain_solver_ultimate import UltimateSolver

class MasterLauncher:
    def __init__(self):
        self.bridge = StreamBridge()
        self.runner = ParallelRunner(max_workers=6)
        self.running = False
        self.bridge_thread = None
        
    def start_stream_bridge(self):
        """Start the streaming bridge in background."""
        self.bridge_thread = self.bridge.start_background_server()
        time.sleep(2)  # Give server time to start
        print(f"🌐 Stream bridge running on ws://localhost:{self.bridge.port}")
        print(f"📁 Stream logs: {self.bridge.stream_file}")
        
    def create_streaming_solver_configs(self, scenarios: List[int], 
                                       num_variants: int = 8) -> List[SolverConfig]:
        """Create solver configs with streaming enabled."""
        configs = []
        callback = self.bridge.create_callback()
        
        # Enhanced base parameters based on analysis
        base_params_by_scenario = {
            1: {  # Scenario 1: Focus on young + well_dressed
                'ultra_rare_threshold': 0.15,
                'rare_accept_rate': 0.95,
                'common_reject_rate': 0.03,
                'phase1_multi_attr_only': True,
                'deficit_panic_threshold': 0.75,
                'early_game_threshold': 0.25,
                'mid_game_threshold': 0.65,
            },
            2: {  # Scenario 2: Has very rare attributes (creative 6.2%)
                'ultra_rare_threshold': 0.08,  # Lower threshold for rare detection
                'rare_accept_rate': 0.99,      # Nearly always accept rare
                'common_reject_rate': 0.01,    # Almost never accept common
                'phase1_multi_attr_only': False,  # Accept single rare attrs early
                'deficit_panic_threshold': 0.6, # Panic earlier
                'early_game_threshold': 0.4,   # Longer early phase
                'mid_game_threshold': 0.75,
            },
            3: {  # Scenario 3: Many constraints, some rare
                'ultra_rare_threshold': 0.12,
                'rare_accept_rate': 0.97,
                'common_reject_rate': 0.02,
                'phase1_multi_attr_only': False,
                'deficit_panic_threshold': 0.7,
                'early_game_threshold': 0.35,
                'mid_game_threshold': 0.7,
            }
        }
        
        for scenario in scenarios:
            base_params = base_params_by_scenario.get(scenario, base_params_by_scenario[1])
            
            # Generate variations
            param_variations = self.runner.generate_param_variations(base_params, num_variants)
            
            for i, params in enumerate(param_variations):
                config = SolverConfig(
                    solver_id=f"stream_s{scenario}_v{i:02d}",
                    strategy_params=params,
                    scenario=scenario,
                    priority=1 if i == 0 else 0  # Base params get priority
                )
                configs.append(config)
        
        return configs
    
    def run_streaming_solver(self, config: SolverConfig) -> Dict:
        """Run a single solver with streaming enabled."""
        callback = self.bridge.create_callback()
        
        # Notify game start
        callback({
            "type": "game_start",
            "solver_id": config.solver_id,
            "game_id": f"{config.solver_id}_game",
            "data": {
                "scenario": config.scenario,
                "strategy_params": config.strategy_params,
                "start_time": time.time()
            }
        })
        
        try:
            solver = UltimateSolver(
                strategy_params=config.strategy_params, 
                solver_id=config.solver_id
            )
            solver.set_stream_callback(callback)
            
            result = solver.play_game(config.scenario)
            
            # Notify game end
            callback({
                "type": "game_end",
                "solver_id": config.solver_id,
                "game_id": result['game_id'][:8],
                "data": {
                    "status": result['status'],
                    "rejected_count": result['rejected_count'],
                    "admitted_count": result['admitted_count'],
                    "success": result['status'] == 'completed'
                }
            })
            
            return result
            
        except Exception as e:
            callback({
                "type": "game_end",
                "solver_id": config.solver_id,
                "game_id": "error",
                "data": {
                    "status": "error",
                    "error": str(e),
                    "success": False
                }
            })
            raise
    
    def run_optimization_campaign(self, scenarios: List[int] = [1, 2, 3], 
                                 total_games: int = 24):
        """Run comprehensive optimization campaign with streaming."""
        print(f"🎯 BERGHAIN ULTIMATE OPTIMIZATION CAMPAIGN")
        print(f"📊 Scenarios: {scenarios}")
        print(f"🎮 Total games: {total_games}")
        print(f"⚡ Max parallel: {self.runner.max_workers}")
        print("="*60)
        
        # Start streaming
        self.start_stream_bridge()
        
        # Create streaming solver configs
        games_per_scenario = total_games // len(scenarios)
        configs = self.create_streaming_solver_configs(scenarios, games_per_scenario)
        
        # Run parallel optimization with streaming
        print(f"🚀 Launching {len(configs)} streaming solvers...")
        
        # Override the runner's single game method to use streaming
        original_run_single = self.runner.run_single_game
        self.runner.run_single_game = lambda config: self.run_streaming_solver(config)
        
        try:
            # Run the adaptive optimization
            self.runner.adaptive_run(scenarios, len(configs))
        finally:
            # Restore original method
            self.runner.run_single_game = original_run_single
    
    def quick_test(self, scenario: int = 1, num_games: int = 3):
        """Quick test with fewer games for development."""
        print(f"🧪 QUICK TEST - Scenario {scenario}, {num_games} games")
        self.start_stream_bridge()
        
        configs = self.create_streaming_solver_configs([scenario], num_games)
        
        for config in configs[:num_games]:
            print(f"🎮 Running {config.solver_id}...")
            try:
                result = self.run_streaming_solver(config)
                status_emoji = "🎉" if result['status'] == 'completed' else "❌"
                print(f"{status_emoji} {config.solver_id}: {result['rejected_count']} rejections")
            except Exception as e:
                print(f"💥 {config.solver_id}: Error - {e}")
        
        print("\n💡 Connect TUI with: python game_dashboard.py")
        print(f"📁 Check logs: {self.bridge.stream_file}")

def main():
    parser = argparse.ArgumentParser(description="Berghain Ultimate Solver")
    parser.add_argument("--mode", choices=["campaign", "quick"], default="quick",
                       help="Run full campaign or quick test")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=1,
                       help="Scenario for quick test")
    parser.add_argument("--games", type=int, default=3,
                       help="Number of games for quick test")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3],
                       help="Scenarios for campaign mode")
    parser.add_argument("--total-games", type=int, default=24,
                       help="Total games for campaign mode")
    
    args = parser.parse_args()
    
    launcher = MasterLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Stopping...")
        launcher.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.mode == "campaign":
            launcher.run_optimization_campaign(args.scenarios, args.total_games)
        else:
            launcher.quick_test(args.scenario, args.games)
            
        # Keep alive for streaming
        print("\n✅ Games completed. Stream bridge still running...")
        print("💡 Connect TUI: python game_dashboard.py")
        print("🛑 Press Ctrl+C to exit")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
    except Exception as e:
        print(f"💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()