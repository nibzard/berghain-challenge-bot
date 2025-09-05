# ABOUTME: Comprehensive launcher for the Berghain game monitoring and analysis system
# ABOUTME: Provides unified interface for dashboard, streaming, session management, and analytics

import sys
import subprocess
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
import argparse

class BerghainLauncher:
    """Main launcher for the Berghain game monitoring system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        
    def print_banner(self):
        """Print system banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    🏢 BERGHAIN GAME MONITORING SYSTEM                 ║
║                                                                       ║
║  Advanced TUI Dashboard • Real-time Streaming • Performance Analytics ║
║  Multi-game Sessions • Success Prediction • Strategy Extraction       ║
╚═══════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def show_main_menu(self):
        """Show main menu options"""
        print("\n🎯 Available Components:")
        print("━" * 50)
        print("1. 📊 Game Dashboard        - Multi-view TUI for monitoring games")  
        print("2. 🔍 Stream Monitor        - Real-time file watching and streaming")
        print("3. 🎮 Session Manager       - Run multiple games in parallel")
        print("4. 📈 Analytics Engine      - Performance analysis and insights")  
        print("5. 🚀 Quick Start           - Launch dashboard + stream monitor")
        print("6. 🏆 Full System           - Launch everything for comprehensive monitoring")
        print("7. ℹ️  System Status         - Check component status and requirements")
        print("8. ❓ Help                   - Show detailed usage information")
        print("0. 🚪 Exit")
        
    def launch_dashboard(self, args: List[str] = None):
        """Launch the game dashboard"""
        print("🚀 Launching Game Dashboard...")
        cmd = [sys.executable, "game_dashboard.py"]
        if args:
            cmd.extend(args)
        
        try:
            subprocess.run(cmd, cwd=self.base_dir)
        except KeyboardInterrupt:
            print("Dashboard stopped")
            
    def launch_stream_monitor(self, args: List[str] = None):
        """Launch the stream monitor"""
        print("🔍 Launching Stream Monitor...")
        cmd = [sys.executable, "stream_monitor.py"]
        if args:
            cmd.extend(args)
            
        try:
            subprocess.run(cmd, cwd=self.base_dir)
        except KeyboardInterrupt:
            print("Stream monitor stopped")
            
    def launch_session_manager(self, preset: str = "quick_test"):
        """Launch session manager with preset"""
        print(f"🎮 Launching Session Manager (preset: {preset})...")
        cmd = [sys.executable, "session_manager.py", "--preset", preset]
        
        try:
            subprocess.run(cmd, cwd=self.base_dir)
        except KeyboardInterrupt:
            print("Session manager stopped")
            
    def launch_analytics(self):
        """Launch analytics engine"""
        print("📈 Launching Analytics Engine...")
        cmd = [sys.executable, "analytics.py"]
        
        try:
            subprocess.run(cmd, cwd=self.base_dir)
        except KeyboardInterrupt:
            print("Analytics stopped")
            
    def quick_start(self):
        """Launch dashboard with stream monitor support"""
        print("🚀 Quick Start - Launching Dashboard with Streaming...")
        
        # Start stream monitor in background
        stream_process = None
        try:
            print("   Starting stream monitor...")
            stream_process = subprocess.Popen(
                [sys.executable, "stream_monitor.py"],
                cwd=self.base_dir
            )
            
            # Wait a moment for stream monitor to start
            time.sleep(2)
            
            print("   Launching dashboard...")
            self.launch_dashboard()
            
        except KeyboardInterrupt:
            print("\nStopping Quick Start...")
        finally:
            if stream_process:
                stream_process.terminate()
                try:
                    stream_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stream_process.kill()
                    
    def full_system(self):
        """Launch comprehensive monitoring system"""
        print("🏆 Full System Launch - All Components...")
        
        processes = []
        try:
            # Start stream monitor
            print("   1/3 Starting stream monitor...")
            stream_proc = subprocess.Popen(
                [sys.executable, "stream_monitor.py"],
                cwd=self.base_dir
            )
            processes.append(('stream', stream_proc))
            
            # Start session manager with bot comparison
            print("   2/3 Starting session manager...")
            session_proc = subprocess.Popen(
                [sys.executable, "session_manager.py", "--preset", "bot_comparison"],
                cwd=self.base_dir
            )
            processes.append(('session', session_proc))
            
            time.sleep(3)  # Let background processes start
            
            # Start dashboard (foreground)
            print("   3/3 Launching dashboard...")
            self.launch_dashboard()
            
        except KeyboardInterrupt:
            print("\nStopping Full System...")
        finally:
            # Clean up background processes
            for name, proc in processes:
                print(f"   Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    
    def check_system_status(self):
        """Check system status and requirements"""
        print("ℹ️  System Status Check")
        print("━" * 30)
        
        # Check Python version
        print(f"Python Version: {sys.version.split()[0]}")
        
        # Check required files
        required_files = [
            "game_dashboard.py",
            "stream_monitor.py", 
            "session_manager.py",
            "analytics.py",
            "data_collector.py",
            "requirements.txt"
        ]
        
        print("\nRequired Files:")
        for file in required_files:
            path = self.base_dir / file
            status = "✅" if path.exists() else "❌"
            print(f"  {status} {file}")
            
        # Check game logs directory
        logs_dir = self.base_dir / "game_logs"
        print(f"\nGame Logs Directory:")
        if logs_dir.exists():
            log_count = len(list(logs_dir.glob("*.json")))
            print(f"  ✅ game_logs/ ({log_count} log files)")
        else:
            print(f"  ⚠️  game_logs/ (will be created when needed)")
            
        # Check dependencies (basic check)
        print("\nDependencies Status:")
        dependencies = ["rich", "numpy", "scipy", "requests", "watchdog", "websockets"]
        
        for dep in dependencies:
            try:
                __import__(dep)
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep} - run: pip install {dep}")
                
        # Bot scripts check
        print("\nBot Scripts:")
        bot_scripts = [
            "berghain_bot.py",
            "berghain_bot_optimal.py", 
            "berghain_bot_simple.py"
        ]
        
        for bot in bot_scripts:
            path = self.base_dir / bot
            status = "✅" if path.exists() else "⚠️ "
            print(f"  {status} {bot}")
            
    def show_help(self):
        """Show detailed usage help"""
        help_text = """
🎯 BERGHAIN GAME MONITORING SYSTEM - HELP

OVERVIEW:
This system provides comprehensive monitoring and analysis for Berghain 
simulation games. It includes real-time dashboards, parallel game execution,
streaming updates, and performance analytics.

COMPONENTS:

1. 📊 GAME DASHBOARD (game_dashboard.py)
   - Multi-view TUI interface
   - Grid view: Monitor up to 9 games simultaneously
   - Detail view: Focus on single game with full stats  
   - Leaderboard: Rank games by performance
   - Analytics: Cross-game insights
   
   Usage: python game_dashboard.py
   
2. 🔍 STREAM MONITOR (stream_monitor.py) 
   - Real-time file watching
   - WebSocket streaming to dashboard
   - Partial JSON parsing for live updates
   
   Usage: python stream_monitor.py [--logs-dir game_logs] [--port 8765]
   
3. 🎮 SESSION MANAGER (session_manager.py)
   - Parallel game execution
   - Multiple bot coordination
   - Preset configurations
   - Crash recovery
   
   Usage: python session_manager.py --preset [quick_test|bot_comparison|marathon_s1]
   
4. 📈 ANALYTICS ENGINE (analytics.py)
   - Performance analysis
   - Pattern detection  
   - Success prediction
   - Strategy extraction
   
   Usage: python analytics.py [--logs-dir game_logs] [--output report.json]

WORKFLOWS:

🏃 QUICK MONITORING:
  1. Run your data collector: python data_collector.py
  2. Launch: python berghain_launcher.py
  3. Choose option 5 (Quick Start)
  
🏆 COMPREHENSIVE ANALYSIS:
  1. Launch: python berghain_launcher.py  
  2. Choose option 6 (Full System)
  3. Let it run multiple bots and analyze results
  
📊 POST-GAME ANALYSIS:
  1. After collecting game data
  2. Launch: python berghain_launcher.py
  3. Choose option 4 (Analytics Engine)

CONFIGURATION:
- Edit PRESET_SESSIONS in session_manager.py for custom game sessions
- Modify dashboard grid size in game_dashboard.py
- Adjust streaming ports in stream_monitor.py

FILE STRUCTURE:
- game_logs/          : JSON game log files
- streaming_logs/     : Real-time stream data
- session_reports/    : Session execution reports  
- analytics_reports/  : Performance analysis reports

For more details, see SPEC.md
        """
        print(help_text)
        
    def run_interactive_menu(self):
        """Run the interactive menu"""
        self.print_banner()
        
        while True:
            self.show_main_menu()
            
            try:
                choice = input("\n🎯 Enter your choice (0-8): ").strip()
                
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                elif choice == '1':
                    self.launch_dashboard()
                elif choice == '2':
                    self.launch_stream_monitor()
                elif choice == '3':
                    print("\nAvailable presets:")
                    print("  • quick_test - 3 games, 1 bot, scenario 1")
                    print("  • bot_comparison - All bots, all scenarios, 5 runs each")  
                    print("  • marathon_s1 - 100 games, optimal bot, scenario 1")
                    print("  • stress_test - 20 runs each, all combinations")
                    
                    preset = input("Choose preset (default: quick_test): ").strip()
                    if not preset:
                        preset = "quick_test"
                    self.launch_session_manager(preset)
                elif choice == '4':
                    self.launch_analytics()
                elif choice == '5':
                    self.quick_start()
                elif choice == '6':
                    self.full_system()
                elif choice == '7':
                    self.check_system_status()
                elif choice == '8':
                    self.show_help()
                else:
                    print("❌ Invalid choice. Please try again.")
                    
                if choice != '0':
                    input("\nPress Enter to return to menu...")
                    print("\n" * 50)  # Clear screen effect
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break

def main():
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(description='Berghain Game Monitoring System Launcher')
    parser.add_argument('component', nargs='?', choices=[
        'dashboard', 'stream', 'session', 'analytics', 'quick', 'full', 'status', 'help'
    ], help='Component to launch directly')
    
    parser.add_argument('--preset', default='quick_test', 
                       help='Preset for session manager')
    parser.add_argument('--interactive', action='store_true',
                       help='Force interactive mode')
    
    args = parser.parse_args()
    launcher = BerghainLauncher()
    
    # Direct component launch
    if args.component and not args.interactive:
        launcher.print_banner()
        
        if args.component == 'dashboard':
            launcher.launch_dashboard()
        elif args.component == 'stream':
            launcher.launch_stream_monitor()
        elif args.component == 'session':
            launcher.launch_session_manager(args.preset)
        elif args.component == 'analytics':
            launcher.launch_analytics()
        elif args.component == 'quick':
            launcher.quick_start()
        elif args.component == 'full':
            launcher.full_system()
        elif args.component == 'status':
            launcher.check_system_status()
        elif args.component == 'help':
            launcher.show_help()
            
    else:
        # Interactive mode
        launcher.run_interactive_menu()

if __name__ == "__main__":
    main()