# ABOUTME: Advanced multi-view TUI dashboard for monitoring parallel Berghain games
# ABOUTME: Supports grid view, detail view, leaderboard, analytics, and success highlighting

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import glob
import statistics

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn
from rich.tree import Tree
from rich.box import ROUNDED, DOUBLE
from rich import box

class ViewMode(Enum):
    GRID = "grid"
    DETAIL = "detail"
    LEADERBOARD = "leaderboard"
    ANALYTICS = "analytics"
    SPLIT = "split"

class GameStatus(Enum):
    RUNNING = ("▶", "yellow", "Running")
    COMPLETED = ("●", "green", "Completed") 
    FAILED = ("✗", "red", "Failed")
    PAUSED = ("⏸", "dim", "Paused")
    BEST = ("🏆", "purple", "BEST")

@dataclass
class GameSnapshot:
    game_id: str
    scenario: int
    bot_type: str
    status: GameStatus
    start_time: datetime
    end_time: Optional[datetime]
    constraints: List[Dict]
    admitted: int
    rejected: int
    efficiency: float
    constraint_fill_rates: Dict[str, float]
    file_path: str
    last_update: float
    
    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time
        
    @property
    def success_score(self) -> float:
        """Calculate overall success score for ranking"""
        if self.status != GameStatus.COMPLETED:
            return 0.0
            
        # Check if all constraints are met
        constraints_met = all(rate >= 1.0 for rate in self.constraint_fill_rates.values())
        if not constraints_met or self.admitted > 1000 or self.rejected > 20000:
            return 0.0
            
        # Calculate score: efficiency (40%) + constraint_bonus (40%) + time_bonus (20%)
        constraint_avg = statistics.mean(self.constraint_fill_rates.values()) 
        time_bonus = max(0, (3600 - self.duration.total_seconds()) / 3600)  # Bonus for speed
        
        return self.efficiency * 0.4 + constraint_avg * 0.4 + time_bonus * 0.2

@dataclass 
class SessionStats:
    total_games: int = 0
    successful_games: int = 0
    average_efficiency: float = 0.0
    average_rejections: float = 0.0
    best_score: float = 0.0
    best_game_id: Optional[str] = None

class GameDashboard:
    def __init__(self, logs_dir: str = "game_logs"):
        self.console = Console()
        self.logs_dir = Path(logs_dir)
        self.games: Dict[str, GameSnapshot] = {}
        self.view_mode = ViewMode.GRID
        self.selected_game = None
        self.session_stats = SessionStats()
        self.last_scan = 0
        self.focus_scenarios = [1, 2, 3]  # Which scenarios to monitor
        
        # Layout settings
        self.grid_cols = 3
        self.grid_rows = 2
        self.max_displayed_games = self.grid_cols * self.grid_rows
        
    def scan_games(self):
        """Scan for game files and update snapshots"""
        if not self.logs_dir.exists():
            return
            
        for json_file in self.logs_dir.glob("*.json"):
            file_mtime = os.path.getmtime(json_file)
            game_id = json_file.stem
            
            # Skip if file hasn't changed and we already have it
            if game_id in self.games and self.games[game_id].last_update >= file_mtime:
                continue
                
            try:
                snapshot = self._load_game_snapshot(json_file)
                if snapshot and snapshot.scenario in self.focus_scenarios:
                    self.games[snapshot.game_id] = snapshot
                    
            except Exception as e:
                self.console.print(f"[red]Error loading {json_file}: {e}[/red]")
                continue
                
        self._update_session_stats()
        self.last_scan = time.time()
        
    def _load_game_snapshot(self, filepath: Path) -> Optional[GameSnapshot]:
        """Load a game snapshot from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Determine bot type from filename or data
            bot_type = "unknown"
            if "optimal" in str(filepath):
                bot_type = "optimal"
            elif "simple" in str(filepath):
                bot_type = "simple"
            elif "dynamic" in str(filepath):
                bot_type = "dynamic"
                
            # Calculate constraint fill rates
            constraint_fill_rates = {}
            for constraint in data.get('constraints', []):
                attr = constraint['attribute']
                required = constraint.get('minCount', constraint.get('requiredCount', 0))
                
                # Count admitted people with this attribute
                admitted_with_attr = sum(1 for person in data.get('people', [])
                                       if person['decision'] and person['attributes'].get(attr, False))
                
                fill_rate = (admitted_with_attr / required) if required > 0 else 0
                constraint_fill_rates[attr] = fill_rate
                
            # Determine status
            status_str = data.get('final_status', '')
            if status_str == 'completed':
                # Check if this is the best performing game
                all_constraints_met = all(rate >= 1.0 for rate in constraint_fill_rates.values())
                if (all_constraints_met and 
                    data.get('final_admitted_count', 0) <= 1000 and 
                    data.get('final_rejected_count', 0) <= 20000):
                    status = GameStatus.BEST  # Will be refined in _update_session_stats
                else:
                    status = GameStatus.COMPLETED
            elif status_str == 'failed':
                status = GameStatus.FAILED
            else:
                status = GameStatus.RUNNING
                
            admitted = data.get('final_admitted_count', 0)
            rejected = data.get('final_rejected_count', 0)
            efficiency = (admitted / (admitted + rejected)) if (admitted + rejected) > 0 else 0
            
            return GameSnapshot(
                game_id=data['game_id'][:8],  # Short ID
                scenario=data['scenario'],
                bot_type=bot_type,
                status=status,
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
                constraints=data.get('constraints', []),
                admitted=admitted,
                rejected=rejected,
                efficiency=efficiency,
                constraint_fill_rates=constraint_fill_rates,
                file_path=str(filepath),
                last_update=os.path.getmtime(filepath)
            )
            
        except Exception as e:
            return None
            
    def _update_session_stats(self):
        """Update overall session statistics"""
        if not self.games:
            return
            
        completed_games = [g for g in self.games.values() if g.status in [GameStatus.COMPLETED, GameStatus.BEST]]
        successful_games = [g for g in completed_games if g.success_score > 0]
        
        self.session_stats = SessionStats(
            total_games=len(self.games),
            successful_games=len(successful_games),
            average_efficiency=statistics.mean([g.efficiency for g in self.games.values()]) if self.games else 0,
            average_rejections=statistics.mean([g.rejected for g in self.games.values()]) if self.games else 0,
            best_score=max([g.success_score for g in self.games.values()]) if self.games else 0,
            best_game_id=max(self.games.values(), key=lambda g: g.success_score).game_id if successful_games else None
        )
        
        # Update best game highlighting
        if self.session_stats.best_game_id:
            best_game = self.games[self.session_stats.best_game_id]
            if best_game.success_score > 0.8:  # High threshold for "BEST" status
                best_game.status = GameStatus.BEST
    
    def create_main_layout(self) -> Layout:
        """Create the main dashboard layout"""
        layout = Layout()
        
        if self.view_mode == ViewMode.GRID:
            layout.split(
                Layout(name="header", size=3),
                Layout(name="games_grid", size=20),
                Layout(name="sidebar", size=8),
                Layout(name="footer", size=3)
            )
        elif self.view_mode == ViewMode.DETAIL:
            layout.split(
                Layout(name="header", size=3),
                Layout(name="detail_main"),
                Layout(name="footer", size=3)
            )
        elif self.view_mode == ViewMode.LEADERBOARD:
            layout.split(
                Layout(name="header", size=3),
                Layout(name="leaderboard_main"),
                Layout(name="footer", size=3)
            )
        
        return layout
        
    def create_header(self) -> Panel:
        """Create dashboard header"""
        # Mode indicator
        mode_text = f"[bold]{self.view_mode.value.upper()} VIEW[/bold]"
        
        # Session stats
        stats_text = (f"[dim]Games: {self.session_stats.total_games} | "
                     f"Success: {self.session_stats.successful_games} | "
                     f"Avg Efficiency: {self.session_stats.average_efficiency:.1%}[/dim]")
        
        # Best game highlight
        best_text = ""
        if self.session_stats.best_game_id:
            best_text = f" | [purple]Best: {self.session_stats.best_game_id} (Score: {self.session_stats.best_score:.2f})[/purple]"
        
        header_content = f"{mode_text}    {stats_text}{best_text}"
        
        return Panel(
            header_content,
            title="[bold white]🏢 Berghain Game Dashboard[/bold white]",
            border_style="bold blue"
        )
        
    def create_games_grid(self) -> Panel:
        """Create the games grid view"""
        if not self.games:
            return Panel(
                "[dim]No games found. Start some games to see them here.\n\n"
                "Run: python data_collector.py[/dim]",
                title="Games Grid",
                border_style="dim"
            )
            
        # Get most recent games for display
        recent_games = sorted(
            self.games.values(), 
            key=lambda g: g.last_update, 
            reverse=True
        )[:self.max_displayed_games]
        
        # Create grid panels
        game_panels = []
        for game in recent_games:
            panel = self._create_game_card(game)
            game_panels.append(panel)
            
        # Fill empty slots if needed
        while len(game_panels) < self.max_displayed_games:
            game_panels.append(Panel("[dim]Empty slot[/dim]", border_style="dim"))
            
        # Arrange in grid
        rows = []
        for i in range(0, len(game_panels), self.grid_cols):
            row_panels = game_panels[i:i + self.grid_cols]
            rows.append(Columns(row_panels, equal=True))
            
        return Panel(
            "\n".join(str(row) for row in rows),
            title="Active Games",
            border_style="blue"
        )
        
    def _create_game_card(self, game: GameSnapshot) -> Panel:
        """Create a compact game card for grid view"""
        # Status indicator
        status_icon, status_color, status_text = game.status.value
        status_line = f"[{status_color}]{status_icon} {status_text}[/{status_color}]"
        
        # Constraint progress
        constraint_lines = []
        for attr, fill_rate in game.constraint_fill_rates.items():
            percentage = min(fill_rate * 100, 100)
            color = "green" if fill_rate >= 1.0 else "yellow" if fill_rate >= 0.5 else "red"
            constraint_lines.append(f"[{color}]{attr}: {percentage:.0f}%[/{color}]")
            
        # Key metrics
        metrics = [
            f"Capacity: {game.admitted}",
            f"Rejected: {game.rejected:,}",
            f"Efficiency: {game.efficiency:.1%}"
        ]
        
        # Duration
        duration_str = f"{int(game.duration.total_seconds() // 60)}m {int(game.duration.total_seconds() % 60)}s"
        
        card_content = [
            f"[bold]Game {game.game_id}[/bold] (S{game.scenario})",
            status_line,
            "",
            *constraint_lines,
            "",
            *metrics,
            f"[dim]{duration_str}[/dim]"
        ]
        
        # Special border for best game
        border_style = "purple" if game.status == GameStatus.BEST else "white"
        
        return Panel(
            "\n".join(card_content),
            border_style=border_style,
            padding=(0, 1)
        )
        
    def create_sidebar(self) -> Panel:
        """Create sidebar with quick stats and controls"""
        # Scenario breakdown
        scenario_stats = {}
        for game in self.games.values():
            scenario = game.scenario
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {"total": 0, "successful": 0}
            scenario_stats[scenario]["total"] += 1
            if game.success_score > 0:
                scenario_stats[scenario]["successful"] += 1
                
        scenario_lines = []
        for scenario in sorted(scenario_stats.keys()):
            stats = scenario_stats[scenario]
            success_rate = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            scenario_lines.append(f"S{scenario}: {stats['successful']}/{stats['total']} ({success_rate:.0f}%)")
        
        # Recent activity
        recent_activity = []
        recent_games = sorted(self.games.values(), key=lambda g: g.last_update, reverse=True)[:3]
        for game in recent_games:
            ago = int(time.time() - game.last_update)
            recent_activity.append(f"{game.game_id}: {ago}s ago")
            
        sidebar_content = [
            "[bold]Scenario Stats[/bold]",
            *scenario_lines,
            "",
            "[bold]Recent Activity[/bold]", 
            *recent_activity,
            "",
            "[dim]Controls:[/dim]",
            "[dim]g: Grid view[/dim]",
            "[dim]d: Detail view[/dim]", 
            "[dim]l: Leaderboard[/dim]",
            "[dim]q: Quit[/dim]"
        ]
        
        return Panel(
            "\n".join(sidebar_content),
            title="Dashboard Stats",
            border_style="yellow"
        )
        
    def create_detail_view(self) -> Panel:
        """Create detailed view of selected game"""
        if not self.selected_game or self.selected_game not in self.games:
            # Auto-select best or most recent game
            if self.session_stats.best_game_id:
                self.selected_game = self.session_stats.best_game_id
            elif self.games:
                self.selected_game = max(self.games.values(), key=lambda g: g.last_update).game_id
            else:
                return Panel("[dim]No game selected[/dim]", title="Detail View")
                
        game = self.games[self.selected_game]
        
        # Create detailed constraint progress bars
        constraint_progress = []
        for constraint in game.constraints:
            attr = constraint['attribute']
            required = constraint.get('minCount', constraint.get('requiredCount', 0))
            fill_rate = game.constraint_fill_rates.get(attr, 0)
            current = int(required * fill_rate)
            
            # Progress bar
            bar_width = 40
            filled = int(bar_width * min(fill_rate, 1.0))
            empty = bar_width - filled
            bar = "█" * filled + "░" * empty
            
            color = "green" if fill_rate >= 1.0 else "yellow" if fill_rate >= 0.5 else "red"
            
            constraint_progress.append(
                f"[bold]{attr.upper().replace('_', ' ')}[/bold]\n"
                f"[{color}][{bar}][/{color}] {current}/{required} ({fill_rate*100:.1f}%)"
            )
            
        detail_content = [
            f"[bold white]Game Details: {game.game_id}[/bold white]",
            f"Scenario {game.scenario} • {game.bot_type.title()} Bot • Started {game.duration} ago",
            "",
            "[bold]## Constraint Progress[/bold]",
            "",
            *constraint_progress,
            "",
            "[bold]## Performance Metrics[/bold]",
            "",
            f"Status: {game.status.value[2]}",
            f"Venue Capacity: {game.admitted}/1000 ({game.admitted/10:.1f}% full)",
            f"Total Rejected: {game.rejected:,}",
            f"Efficiency: {game.efficiency:.1%}",
            f"Success Score: {game.success_score:.3f}",
            "",
            f"[dim]File: {Path(game.file_path).name}[/dim]"
        ]
        
        return Panel(
            "\n".join(detail_content),
            title="Detail View",
            border_style="green" if game.status == GameStatus.BEST else "blue"
        )
        
    def create_leaderboard(self) -> Panel:
        """Create leaderboard view"""
        if not self.games:
            return Panel("[dim]No games to rank[/dim]", title="Leaderboard")
            
        # Sort games by success score
        ranked_games = sorted(
            [g for g in self.games.values() if g.success_score > 0],
            key=lambda g: g.success_score,
            reverse=True
        )[:10]  # Top 10
        
        table = Table(
            "Rank", "Game ID", "Scenario", "Bot", "Success Score", "Efficiency", "Rejections",
            box=box.ROUNDED,
            header_style="bold blue"
        )
        
        for i, game in enumerate(ranked_games, 1):
            rank_style = "bold purple" if i == 1 else "bold yellow" if i <= 3 else ""
            
            table.add_row(
                f"[{rank_style}]{i}[/{rank_style}]" if rank_style else str(i),
                game.game_id,
                f"S{game.scenario}",
                game.bot_type.title(),
                f"{game.success_score:.3f}",
                f"{game.efficiency:.1%}",
                f"{game.rejected:,}"
            )
            
        return Panel(
            table,
            title="🏆 Leaderboard - Top Performing Games",
            border_style="purple"
        )
        
    def create_footer(self) -> Panel:
        """Create footer with status and controls"""
        now = datetime.now().strftime("%H:%M:%S")
        games_count = len(self.games)
        
        footer_text = (f"[dim]Last scan: {now} • "
                      f"Monitoring: {games_count} games • "
                      f"Press 'h' for help[/dim]")
                      
        return Panel(footer_text, border_style="dim")
        
    def update_display(self, layout: Layout):
        """Update the dashboard display"""
        layout["header"].update(self.create_header())
        layout["footer"].update(self.create_footer())
        
        if self.view_mode == ViewMode.GRID:
            layout["games_grid"].update(self.create_games_grid())
            layout["sidebar"].update(self.create_sidebar())
        elif self.view_mode == ViewMode.DETAIL:
            layout["detail_main"].update(self.create_detail_view())
        elif self.view_mode == ViewMode.LEADERBOARD:
            layout["leaderboard_main"].update(self.create_leaderboard())
            
    def handle_keypress(self, key: str):
        """Handle keyboard input for view switching"""
        if key == 'g':
            self.view_mode = ViewMode.GRID
        elif key == 'd':
            self.view_mode = ViewMode.DETAIL
        elif key == 'l':
            self.view_mode = ViewMode.LEADERBOARD
        elif key == 'n' and self.games:
            # Cycle through games in detail view
            game_ids = list(self.games.keys())
            if self.selected_game in game_ids:
                current_idx = game_ids.index(self.selected_game)
                self.selected_game = game_ids[(current_idx + 1) % len(game_ids)]
        elif key == 'b' and self.session_stats.best_game_id:
            # Jump to best game
            self.selected_game = self.session_stats.best_game_id
            self.view_mode = ViewMode.DETAIL
            
    def run_dashboard(self, refresh_rate: float = 2.0):
        """Run the live dashboard"""
        layout = self.create_main_layout()
        
        try:
            with Live(layout, console=self.console, refresh_per_second=refresh_rate, screen=True):
                while True:
                    self.scan_games()
                    self.update_display(layout)
                    time.sleep(1.0 / refresh_rate)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Dashboard stopped[/yellow]")

def main():
    dashboard = GameDashboard()
    
    # Start in grid view monitoring all scenarios
    dashboard.view_mode = ViewMode.GRID
    dashboard.focus_scenarios = [1, 2, 3]
    
    dashboard.console.print("[bold green]🚀 Starting Berghain Game Dashboard[/bold green]")
    dashboard.console.print("[dim]Monitoring game_logs/ directory for active games...[/dim]")
    dashboard.console.print("[dim]Press Ctrl+C to exit[/dim]")
    time.sleep(1)
    
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()