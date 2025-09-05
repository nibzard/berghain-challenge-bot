# ABOUTME: Clean TUI dashboard for monitoring games using Rich
# ABOUTME: Single responsibility - display game information in a user-friendly interface

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.box import ROUNDED

from .file_watcher import GameLogWatcher


@dataclass
class GameDisplayInfo:
    """Information needed to display a game in the dashboard."""
    game_id: str
    solver_id: str
    scenario: int
    status: str
    success: bool
    admitted: int
    rejected: int
    progress: Dict[str, float]
    start_time: str
    duration: float
    constraints: List[Dict[str, Any]]


class TUIDashboard:
    """Clean TUI dashboard for monitoring game progress."""
    
    def __init__(self, logs_directory: str = "game_logs", refresh_rate: float = 1.0):
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.running = False
        
        # Data sources
        self.file_watcher = GameLogWatcher(logs_directory)
        self.active_games: Dict[str, GameDisplayInfo] = {}
        self.completed_games: List[GameDisplayInfo] = []
        self.max_completed_display = 5
        
        # Layout
        self.layout = self._create_layout()
        
        # Setup file watcher
        self.file_watcher.add_update_callback(self._handle_file_update)
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=6)
        )
        
        layout["main"].split_row(
            Layout(name="active", ratio=2),
            Layout(name="completed", ratio=1)
        )
        
        return layout
    
    def _handle_file_update(self, update_info: Dict[str, Any]):
        """Handle file system updates."""
        try:
            if update_info["type"] == "game_discovered":
                self._load_game_info(update_info)
            elif update_info["type"] == "game_updated":
                self._update_game_info(update_info)
            elif update_info["type"] == "live_update":
                self._handle_live_update(update_info)
            elif update_info["type"] == "game_completed":
                self._handle_game_completion(update_info)
        except Exception as e:
            # Don't crash dashboard on update errors
            pass
    
    def _load_game_info(self, update_info: Dict[str, Any]):
        """Load full game information from file."""
        try:
            import json
            from pathlib import Path
            
            filepath = Path(update_info["filepath"])
            if not filepath.exists():
                return
            
            with open(filepath, 'r') as f:
                game_data = json.loads(f.read())
            
            # Create display info
            game_info = GameDisplayInfo(
                game_id=game_data.get("game_id", "unknown")[:8],
                solver_id=game_data.get("solver_id", "unknown"),
                scenario=game_data.get("scenario_id", game_data.get("scenario", 0)),
                status=game_data.get("status", "unknown"),
                success=game_data.get("success", False),
                admitted=game_data.get("admitted_count", 0),
                rejected=game_data.get("rejected_count", 0),
                progress=self._extract_progress(game_data),
                start_time=game_data.get("start_time", ""),
                duration=game_data.get("duration_seconds", 0),
                constraints=game_data.get("constraints", [])
            )
            
            # Add to appropriate list
            if game_info.status == "running":
                self.active_games[game_info.game_id] = game_info
            else:
                # Move to completed games
                if game_info.game_id in self.active_games:
                    del self.active_games[game_info.game_id]
                
                # Keep only recent completed games
                self.completed_games.insert(0, game_info)
                self.completed_games = self.completed_games[:self.max_completed_display]
        
        except Exception:
            pass  # Ignore errors in file processing
    
    def _update_game_info(self, update_info: Dict[str, Any]):
        """Update existing game information."""
        game_id = update_info.get("game_id", "unknown")[:8]
        
        if game_id in self.active_games:
            game = self.active_games[game_id]
            game.status = update_info.get("status", game.status)
            game.admitted = update_info.get("admitted", game.admitted)
            game.rejected = update_info.get("rejected", game.rejected)
            
            # If game completed, move to completed list
            if game.status != "running":
                game.success = update_info.get("status") == "completed"
                self.completed_games.insert(0, game)
                self.completed_games = self.completed_games[:self.max_completed_display]
                del self.active_games[game_id]
    
    def _extract_progress(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract constraint progress from game data."""
        progress = {}
        
        # Try to get from constraint satisfaction summary
        constraint_summary = game_data.get("constraint_satisfaction", {})
        for attr, summary in constraint_summary.items():
            if isinstance(summary, dict) and "progress" in summary:
                progress[attr] = summary["progress"]
        
        # Fallback: calculate from admitted attributes and constraints
        if not progress:
            admitted_attrs = game_data.get("final_admitted_attributes", {})
            constraints = game_data.get("constraints", [])
            
            for constraint in constraints:
                attr = constraint["attribute"]
                min_count = constraint["min_count"]
                current = admitted_attrs.get(attr, 0)
                progress[attr] = current / min_count if min_count > 0 else 0
        
        return progress
    
    def _handle_live_update(self, update_info: Dict[str, Any]):
        """Handle live game updates."""
        solver_id = update_info["solver_id"]
        update_type = update_info["update_type"]
        data = update_info["data"]
        
        if update_type == "game_start":
            # Create new active game entry
            game_info = GameDisplayInfo(
                game_id=data.get("game_id", "unknown")[:8],
                solver_id=solver_id,
                scenario=data.get("scenario", 0),
                status="running",
                success=False,
                admitted=0,
                rejected=0,
                progress={},
                start_time=data.get("timestamp", ""),
                duration=0,
                constraints=data.get("constraints", [])
            )
            self.active_games[solver_id] = game_info
            
        elif update_type == "decision" and solver_id in self.active_games:
            # Update progress
            game = self.active_games[solver_id]
            game.admitted = data.get("admitted", 0)
            game.rejected = data.get("rejected", 0)
            game.progress = data.get("progress", {})
            
        elif update_type == "game_end" and solver_id in self.active_games:
            # Move to completed
            game = self.active_games[solver_id]
            game.status = data.get("status", "completed")
            game.success = data.get("success", False)
            game.admitted = data.get("admitted_count", 0)
            game.rejected = data.get("rejected_count", 0)
            game.duration = data.get("duration", 0)
            
            # Move to completed list
            self.completed_games.insert(0, game)
            if len(self.completed_games) > self.max_completed_display:
                self.completed_games = self.completed_games[:self.max_completed_display]
            
            # Remove from active
            del self.active_games[solver_id]
    
    def _handle_game_completion(self, update_info: Dict[str, Any]):
        """Handle game completion (live file deleted)."""
        # Extract solver_id from filename
        filepath = update_info["filepath"]
        filename = Path(filepath).name
        if filename.startswith("live_"):
            solver_id = filename[5:-5]  # Remove "live_" prefix and ".json" suffix
            
            # If game is still in active list, it completed without final update
            if solver_id in self.active_games:
                game = self.active_games[solver_id]
                game.status = "completed"
                
                # Move to completed
                self.completed_games.insert(0, game)
                if len(self.completed_games) > self.max_completed_display:
                    self.completed_games = self.completed_games[:self.max_completed_display]
                
                del self.active_games[solver_id]
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        current_time = datetime.now().strftime("%H:%M:%S")
        active_count = len(self.active_games)
        completed_count = len(self.completed_games)
        
        header_text = Text()
        header_text.append("🎯 Berghain Challenge Dashboard", style="bold white")
        header_text.append(f"  |  Time: {current_time}", style="dim")
        header_text.append(f"  |  Active: {active_count}", style="green")
        header_text.append(f"  |  Completed: {completed_count}", style="blue")
        
        return Panel(
            Align.center(header_text),
            box=ROUNDED,
            style="blue"
        )
    
    def _create_active_games_panel(self) -> Panel:
        """Create active games panel."""
        if not self.active_games:
            return Panel(
                Align.center(Text("No active games\n\nStart games with: python main.py run", 
                                style="dim")),
                title="🎮 Active Games",
                box=ROUNDED,
                style="yellow"
            )
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Solver", style="cyan", width=15)
        table.add_column("S", justify="center", width=2)  # Scenario
        table.add_column("Status", width=10)
        table.add_column("A/R", justify="right", width=8)  # Admitted/Rejected
        table.add_column("Progress", width=20)
        table.add_column("Time", justify="right", width=8)
        
        for game in self.active_games.values():
            # Status styling
            if game.status == "running":
                status_text = Text("▶ Running", style="yellow")
            else:
                status_text = Text(f"● {game.status.title()}", style="green" if game.success else "red")
            
            # Progress bar
            if game.constraints and game.progress:
                progress_text = ""
                for constraint in game.constraints:
                    attr = constraint["attribute"] 
                    prog = game.progress.get(attr, 0)
                    bar_length = 3
                    filled = int(prog * bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    progress_text += f"{attr[:3]}:{bar} "
            else:
                progress_text = "N/A"
            
            # Calculate elapsed time
            try:
                if game.start_time:
                    start = datetime.fromisoformat(game.start_time.replace('Z', '+00:00'))
                    elapsed = datetime.now() - start.replace(tzinfo=None)
                    time_str = f"{int(elapsed.total_seconds() // 60)}:{int(elapsed.total_seconds() % 60):02d}"
                else:
                    time_str = "N/A"
            except:
                time_str = "N/A"
            
            table.add_row(
                game.solver_id[:14],
                str(game.scenario),
                status_text,
                f"{game.admitted}/{game.rejected}",
                progress_text.strip(),
                time_str
            )
        
        return Panel(
            table,
            title="🎮 Active Games",
            box=ROUNDED,
            style="yellow"
        )
    
    def _create_completed_games_panel(self) -> Panel:
        """Create completed games panel."""
        if not self.completed_games:
            return Panel(
                Align.center(Text("No completed games", style="dim")),
                title="📊 Recent Results",
                box=ROUNDED,
                style="green"
            )
        
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Solver", style="cyan", width=12)
        table.add_column("S", justify="center", width=2)
        table.add_column("Result", width=8)
        table.add_column("Rejections", justify="right", width=10)
        table.add_column("Time", justify="right", width=6)
        
        for game in self.completed_games:
            result_text = Text("SUCCESS", style="green") if game.success else Text("FAILED", style="red")
            
            duration_str = f"{int(game.duration)}s" if game.duration > 0 else "N/A"
            
            table.add_row(
                game.solver_id[:11],
                str(game.scenario),
                result_text,
                f"{game.rejected:,}",
                duration_str
            )
        
        return Panel(
            table,
            title="📊 Recent Results",
            box=ROUNDED,
            style="green"
        )
    
    def _create_footer(self) -> Panel:
        """Create footer with instructions."""
        footer_text = Text()
        footer_text.append("Commands: ", style="bold")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" = Exit  |  ", style="dim")
        footer_text.append("python main.py run", style="bold green")
        footer_text.append(" = Start Games  |  ", style="dim")
        footer_text.append("Refresh: ", style="bold")
        footer_text.append(f"{self.refresh_rate}s", style="yellow")
        
        return Panel(
            Align.center(footer_text),
            box=ROUNDED,
            style="blue"
        )
    
    def _update_layout(self):
        """Update the layout with current data."""
        self.layout["header"].update(self._create_header())
        self.layout["active"].update(self._create_active_games_panel())
        self.layout["completed"].update(self._create_completed_games_panel())
        self.layout["footer"].update(self._create_footer())
    
    def run(self):
        """Run the TUI dashboard."""
        print("🚀 Starting Berghain Dashboard")
        print(f"📁 Monitoring: game_logs/")
        print("Press Ctrl+C to exit\n")
        
        self.running = True
        
        # Start file watcher
        self.file_watcher.start_watching()
        
        # Load existing games
        recent_games = self.file_watcher.get_recent_games(20)
        for game_summary in recent_games:
            self._load_game_info({
                "type": "game_discovered",
                "filepath": game_summary["filepath"],
                "game_id": game_summary["game_id"],
                "solver_id": game_summary["solver_id"]
            })
        
        try:
            with Live(self.layout, refresh_per_second=1/self.refresh_rate, screen=True):
                while self.running:
                    self._update_layout()
                    time.sleep(self.refresh_rate)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.file_watcher.stop_watching()
            print("\nDashboard stopped")
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False