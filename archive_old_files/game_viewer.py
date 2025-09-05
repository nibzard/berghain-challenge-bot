# ABOUTME: Rich-based TUI viewer for real-time display of game statistics from log files
# ABOUTME: Watches game_logs directory for new files and displays live stats without running games

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import glob

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align

@dataclass 
class GameStats:
    game_id: str
    scenario: int
    start_time: datetime
    end_time: Optional[datetime]
    constraints: List[Dict]
    final_status: str
    final_rejected_count: int
    final_admitted_count: int
    total_time: float
    people: List[Dict]
    
    @classmethod
    def from_log_file(cls, filepath: str) -> 'GameStats':
        """Load game stats from a JSON log file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return cls(
            game_id=data['game_id'],
            scenario=data['scenario'], 
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            constraints=data['constraints'],
            final_status=data['final_status'],
            final_rejected_count=data['final_rejected_count'],
            final_admitted_count=data['final_admitted_count'],
            total_time=data['total_time'],
            people=data['people']
        )

@dataclass
class SessionSummary:
    scenario: int
    games: List[GameStats]
    total_games: int
    successful_games: int
    avg_rejections: float
    avg_admitted: float
    success_rate: float

class GameViewer:
    def __init__(self, logs_dir: str = "game_logs"):
        self.console = Console()
        self.logs_dir = Path(logs_dir)
        self.current_game: Optional[GameStats] = None
        self.sessions: Dict[int, SessionSummary] = {}
        self.last_scan = 0
        
    def scan_for_games(self):
        """Scan for new game log files."""
        if not self.logs_dir.exists():
            return
            
        # Find all JSON files
        json_files = list(self.logs_dir.glob("*.json"))
        
        if not json_files:
            return
            
        # Get the most recent file
        latest_file = max(json_files, key=os.path.getctime)
        
        # Only update if this is a new file
        if os.path.getctime(latest_file) > self.last_scan:
            try:
                self.current_game = GameStats.from_log_file(latest_file)
                self.last_scan = os.path.getctime(latest_file)
                
                # Update session summaries
                self._update_sessions()
                
            except Exception as e:
                self.console.print(f"[red]Error loading {latest_file}: {e}[/red]")
    
    def _update_sessions(self):
        """Update session summaries from all log files."""
        if not self.logs_dir.exists():
            return
            
        # Group games by scenario
        games_by_scenario = {}
        
        for json_file in self.logs_dir.glob("*.json"):
            try:
                game = GameStats.from_log_file(json_file)
                scenario = game.scenario
                
                if scenario not in games_by_scenario:
                    games_by_scenario[scenario] = []
                    
                games_by_scenario[scenario].append(game)
                
            except Exception:
                continue  # Skip corrupted files
                
        # Create session summaries
        for scenario, games in games_by_scenario.items():
            successful = [g for g in games if g.final_status == "completed"]
            
            self.sessions[scenario] = SessionSummary(
                scenario=scenario,
                games=games,
                total_games=len(games),
                successful_games=len(successful),
                avg_rejections=sum(g.final_rejected_count for g in games) / len(games) if games else 0,
                avg_admitted=sum(g.final_admitted_count for g in games) / len(games) if games else 0,
                success_rate=(len(successful) / len(games) * 100) if games else 0
            )
    
    def create_layout(self) -> Layout:
        """Create the main TUI layout."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", size=20), 
            Layout(name="footer", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """Create the header panel with game title and status."""
        if self.current_game:
            scenario_names = {1: "Friday Night", 2: "Saturday Night", 3: "Sunday Night"}
            scenario_name = scenario_names.get(self.current_game.scenario, f"Scenario {self.current_game.scenario}")
            
            # Determine status and color
            if self.current_game.final_status == "completed":
                status_color = "green"
                status_text = "● Completed"
            elif self.current_game.final_status == "failed":
                status_color = "red"
                status_text = "● Failed"
            else:
                status_color = "yellow"
                status_text = "▶ Running"
            
            # Calculate elapsed time 
            if self.current_game.end_time:
                elapsed = self.current_game.end_time - self.current_game.start_time
            else:
                elapsed = datetime.now() - self.current_game.start_time
                
            elapsed_str = f"{int(elapsed.total_seconds() // 60)}m {int(elapsed.total_seconds() % 60)}s ago"
            
            header_text = f"[bold white]# Game Details: {self.current_game.game_id[:8]}[/bold white]\n"
            header_text += f"[dim]Scenario {self.current_game.scenario}: {scenario_name}[/dim]    "
            header_text += f"[dim]Status:[/dim] [{status_color}]{status_text}[/{status_color}]    "
            header_text += f"[dim]Started:[/dim] {elapsed_str}"
        else:
            header_text = "[bold white]# Berghain Game Stats Viewer[/bold white]\n[dim]Watching for game logs...[/dim]"
            
        return Panel(header_text, style="bold")
    
    def create_constraint_progress(self, game: GameStats) -> Panel:
        """Create constraint progress panel."""
        if not game.constraints:
            return Panel("[dim]No constraints found...[/dim]", title="## Constraint Progress")
            
        constraint_display = []
        
        for constraint in game.constraints:
            attr = constraint['attribute']
            required = constraint.get('minCount', constraint.get('requiredCount', 0))
            
            # Count admitted people with this attribute
            admitted_with_attr = sum(1 for person in game.people 
                                   if person['decision'] and person['attributes'].get(attr, False))
            
            # Calculate percentage
            percentage = (admitted_with_attr / required * 100) if required > 0 else 0
            percentage = min(percentage, 100)  # Cap at 100%
            
            # Create progress bar visual
            bar_width = 40
            filled = int(bar_width * percentage / 100)
            empty = bar_width - filled
            bar = "█" * filled + "░" * empty
            
            # Color coding
            if percentage >= 100:
                color = "green"
                arrow = "●"
            elif percentage >= 50:
                color = "yellow"
                arrow = "▶"
            else:
                color = "red"
                arrow = "▶"
                
            constraint_text = f"[bold]{attr.upper().replace('_', ' ')}[/bold]"
            progress_text = f"[{color}]{arrow} {admitted_with_attr}/{required} ({percentage:.1f}%)[/{color}]"
            bar_text = f"[{color}][{bar}][/{color}] {admitted_with_attr} of {required} required (out of {sum(1 for p in game.people if p['attributes'].get(attr, False))} total)"
            
            constraint_display.append(f"{constraint_text:<20} {progress_text}\n{bar_text}")
            
        content = "\n\n".join(constraint_display) if constraint_display else "[dim]Loading constraints...[/dim]"
        
        # Add live update indicator
        update_indicator = "[green]● Live[/green] • Updated 1s ago"
        
        return Panel(content + f"\n\n[dim]{update_indicator}[/dim]", title="## Constraint Progress")
    
    def create_venue_overview(self, game: GameStats) -> Panel:
        """Create venue overview panel."""
        if not game.people:
            return Panel("[dim]No venue data yet...[/dim]", title="## Venue Overview")
            
        admitted = game.final_admitted_count
        total_capacity = 1000  # Based on the game
        capacity_percentage = (admitted / total_capacity * 100) if total_capacity > 0 else 0
        
        # Venue capacity bar
        bar_width = 40
        filled = int(bar_width * capacity_percentage / 100)
        empty = bar_width - filled
        capacity_bar = "█" * filled + "░" * empty
        
        capacity_color = "blue"
        arrow = "▶"
        
        venue_content = f"[bold]VENUE CAPACITY[/bold]                    [{capacity_color}]{arrow} {admitted}/{total_capacity} ({capacity_percentage:.1f}%)[/{capacity_color}]\n"
        venue_content += f"[{capacity_color}][{capacity_bar}][/{capacity_color}] {admitted} of {total_capacity} required (out of {admitted} total)"
        
        return Panel(venue_content, title="## Venue Overview")
    
    def create_summary_stats(self, game: GameStats) -> Panel:
        """Create summary statistics table."""
        table = Table.grid(padding=1)
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold")
        table.add_column("Label2", style="dim") 
        table.add_column("Value2", style="bold")
        table.add_column("Label3", style="dim")
        table.add_column("Value3", style="bold")
        
        admitted = game.final_admitted_count
        rejected = game.final_rejected_count
        efficiency = (admitted / (admitted + rejected) * 100) if (admitted + rejected) > 0 else 0
        
        table.add_row(
            "Capacity", f"[blue]{admitted}/1000[/blue]",
            "Rejected", f"[red]{rejected}[/red]",
            "Efficiency", f"[yellow]{efficiency:.1f}%[/yellow]"
        )
        table.add_row(
            f"[dim]{(admitted/1000*100):.1f}% full[/dim]", "",
            f"[dim]{20000 - rejected} until limit[/dim]", "", 
            "[dim]Acceptance rate[/dim]", ""
        )
        
        return Panel(table, padding=(1, 2))
    
    def create_session_overview(self) -> Panel:
        """Create session overview panel showing multi-game stats."""
        if not self.sessions:
            return Panel("[dim]No game sessions found yet...[/dim]", title="Session Overview")
            
        content = []
        for scenario, session in self.sessions.items():
            scenario_names = {1: "Friday Night", 2: "Saturday Night", 3: "Sunday Night"}
            name = scenario_names.get(scenario, f"Scenario {scenario}")
            
            line = f"[bold]Scenario {scenario}[/bold] ({name})"
            line += f"\n  Games: {session.total_games} | Avg Rejections: {session.avg_rejections:.1f}"
            line += f"\n  Success Rate: {session.success_rate:.1f}% | Avg Admitted: {session.avg_admitted:.1f}"
            content.append(line)
            
        return Panel("\n\n".join(content), title="Multi-Game Session Stats")
    
    def update_display(self, layout: Layout):
        """Update all layout components."""
        layout["header"].update(self.create_header())
        
        if self.current_game:
            # Main panels
            constraint_panel = self.create_constraint_progress(self.current_game)
            venue_panel = self.create_venue_overview(self.current_game)
            summary_panel = self.create_summary_stats(self.current_game)
            
            # Combine main content
            main_content = Layout()
            main_content.split(
                Layout(constraint_panel, size=10),
                Layout(venue_panel, size=6),
                Layout(summary_panel, size=4)
            )
            
            layout["left"].update(main_content)
        else:
            layout["left"].update(Panel(f"[dim]No game logs found in {self.logs_dir}/\nRun data_collector.py in another terminal[/dim]", title="Waiting for Games"))
            
        layout["right"].update(self.create_session_overview())
        layout["footer"].update(Panel("[dim]Press Ctrl+C to exit • Watching for new game files...[/dim]", title="Controls"))
    
    def run_viewer(self):
        """Run the live viewer dashboard."""
        layout = self.create_layout()
        
        try:
            with Live(layout, console=self.console, refresh_per_second=2, screen=True):
                while True:
                    self.scan_for_games()
                    self.update_display(layout)
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Viewer stopped[/yellow]")

def main():
    viewer = GameViewer()
    viewer.run_viewer()

if __name__ == "__main__":
    main()