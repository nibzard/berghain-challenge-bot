# ABOUTME: Rich-based TUI interface for real-time Berghain game statistics and multi-game collection
# ABOUTME: Provides live updates, progress tracking, and aggregated statistics across multiple game sessions

import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TaskID
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.columns import Columns

from data_collector import DataCollector, GameLog

@dataclass
class GameSession:
    scenario: int
    games_completed: int
    games_target: int
    total_rejections: int
    total_admitted: int
    total_time: float
    success_rate: float
    current_game: Optional[GameLog] = None

class GameTUI:
    def __init__(self):
        self.console = Console()
        self.collector = DataCollector()
        self.sessions: Dict[int, GameSession] = {}
        self.current_session: Optional[GameSession] = None
        self.game_logs_dir = Path("game_logs")
        self.is_running = False
        
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
        if self.current_session and self.current_session.current_game:
            game = self.current_session.current_game
            scenario_names = {1: "Friday Night", 2: "Saturday Night", 3: "Sunday Night"}
            scenario_name = scenario_names.get(game.scenario, f"Scenario {game.scenario}")
            
            status_color = "yellow" if game.final_status == "" else "green" if game.final_status == "completed" else "red"
            status_text = "▶ Running" if game.final_status == "" else f"● {game.final_status.title()}"
            
            elapsed = datetime.now() - game.start_time if game.start_time else timedelta(0)
            elapsed_str = f"{int(elapsed.total_seconds() // 60)}m {int(elapsed.total_seconds() % 60)}s ago"
            
            header_text = f"[bold white]# Game Details: nibz[/bold white]\n"
            header_text += f"[dim]Scenario {game.scenario}: {scenario_name}[/dim]    "
            header_text += f"[dim]Status:[/dim] [{status_color}]{status_text}[/{status_color}]    "
            header_text += f"[dim]Started:[/dim] {elapsed_str}"
        else:
            header_text = "[bold white]# Berghain Game Stats Dashboard[/bold white]\n[dim]Ready to collect game data...[/dim]"
            
        return Panel(header_text, style="bold")
        
    def create_constraint_progress(self, game: GameLog) -> Panel:
        """Create constraint progress panel."""
        if not game.constraints:
            return Panel("[dim]No constraints loaded yet...[/dim]", title="## Constraint Progress")
            
        constraint_display = []
        
        for constraint in game.constraints:
            attr = constraint['attribute']
            required = constraint.get('requiredCount', 0)
            
            # Count how many people with this attribute we've admitted
            admitted_with_attr = sum(1 for person in game.people 
                                   if person.decision and person.attributes.get(attr, False))
            
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
            bar_text = f"[{color}][{bar}][/{color}] {admitted_with_attr} of {required} required (out of {len([p for p in game.people if p.attributes.get(attr, False)])} total)"
            
            constraint_display.append(f"{constraint_text:<20} {progress_text}\n{bar_text}")
            
        content = "\n\n".join(constraint_display) if constraint_display else "[dim]Loading constraints...[/dim]"
        
        # Add live update indicator
        update_indicator = "[green]● Live[/green] • Updated 1s ago"
        
        return Panel(content + f"\n\n[dim]{update_indicator}[/dim]", title="## Constraint Progress")
        
    def create_venue_overview(self, game: GameLog) -> Panel:
        """Create venue overview panel."""
        if not game.people:
            return Panel("[dim]No venue data yet...[/dim]", title="## Venue Overview")
            
        admitted = sum(1 for p in game.people if p.decision)
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
        
    def create_summary_stats(self, game: GameLog) -> Table:
        """Create summary statistics table."""
        table = Table.grid(padding=1)
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold")
        table.add_column("Label2", style="dim")
        table.add_column("Value2", style="bold")
        table.add_column("Label3", style="dim") 
        table.add_column("Value3", style="bold")
        
        admitted = sum(1 for p in game.people if p.decision)
        rejected = len(game.people) - admitted
        efficiency = (admitted / len(game.people) * 100) if game.people else 0
        
        table.add_row(
            "Capacity", f"[blue]{admitted}/1000[/blue]",
            "Rejected", f"[red]{rejected}[/red]", 
            "Efficiency", f"[yellow]{efficiency:.1f}%[/yellow]"
        )
        table.add_row(
            "[dim]17.8% full[/dim]", "",
            "[dim]18812 until limit[/dim]", "",
            "[dim]Acceptance rate[/dim]", ""
        )
        
        return Panel(table, padding=(1, 2))
        
    def create_session_overview(self) -> Panel:
        """Create session overview panel showing multi-game stats."""
        if not self.sessions:
            return Panel("[dim]No sessions started yet...[/dim]", title="Session Overview")
            
        content = []
        for scenario, session in self.sessions.items():
            scenario_names = {1: "Friday Night", 2: "Saturday Night", 3: "Sunday Night"}
            name = scenario_names.get(scenario, f"Scenario {scenario}")
            
            progress = f"{session.games_completed}/{session.games_target}"
            avg_rejections = session.total_rejections / max(session.games_completed, 1)
            
            line = f"[bold]Scenario {scenario}[/bold] ({name})"
            line += f"\n  Games: {progress} | Avg Rejections: {avg_rejections:.1f} | Success: {session.success_rate:.1f}%"
            content.append(line)
            
        return Panel("\n\n".join(content), title="Multi-Game Session Stats")
        
    def update_display(self, layout: Layout):
        """Update all layout components."""
        layout["header"].update(self.create_header())
        
        if self.current_session and self.current_session.current_game:
            game = self.current_session.current_game
            
            # Main panels
            constraint_panel = self.create_constraint_progress(game)
            venue_panel = self.create_venue_overview(game)
            summary_panel = self.create_summary_stats(game)
            
            # Combine main content
            main_content = Layout()
            main_content.split(
                Layout(constraint_panel, size=10),
                Layout(venue_panel, size=6), 
                Layout(summary_panel, size=4)
            )
            
            layout["left"].update(main_content)
        else:
            layout["left"].update(Panel("[dim]No active game...[/dim]", title="Game Display"))
            
        layout["right"].update(self.create_session_overview())
        layout["footer"].update(Panel("[dim]Press Ctrl+C to stop[/dim]", title="Controls"))
        
    def start_data_collection(self, scenario: int, num_games: int = 5):
        """Start collecting data for a scenario with live updates."""
        # Initialize session
        self.sessions[scenario] = GameSession(
            scenario=scenario,
            games_completed=0,
            games_target=num_games,
            total_rejections=0,
            total_admitted=0,
            total_time=0.0,
            success_rate=0.0
        )
        self.current_session = self.sessions[scenario]
        
        def collect_data():
            session = self.sessions[scenario]
            
            for game_num in range(num_games):
                try:
                    self.console.print(f"[dim]Starting game {game_num + 1}/{num_games}...[/dim]")
                    
                    # Start new game and track it immediately
                    game_log = GameLog(
                        game_id="",
                        scenario=scenario,
                        start_time=datetime.now(),
                        end_time=None,
                        constraints=[],
                        attribute_frequencies={},
                        attribute_correlations={},
                        people=[],
                        final_status="",
                        final_rejected_count=0,
                        final_admitted_count=0,
                        total_time=0.0
                    )
                    
                    # Set the current game immediately so UI can show it
                    session.current_game = game_log
                    
                    # Now run the actual game
                    completed_game = self.collector.play_fast_game(scenario)
                    
                    # Update the session with completed game
                    session.current_game = completed_game
                    session.games_completed += 1
                    session.total_rejections += completed_game.final_rejected_count
                    session.total_admitted += completed_game.final_admitted_count
                    session.total_time += completed_game.total_time
                    
                    # Calculate success rate
                    if completed_game.final_status == "completed":
                        session.success_rate = (session.success_rate * (session.games_completed - 1) + 100) / session.games_completed
                    else:
                        session.success_rate = session.success_rate * (session.games_completed - 1) / session.games_completed
                        
                    # Save game log
                    self.collector.save_game_log(completed_game)
                    
                    time.sleep(2)  # Brief pause between games
                    
                except Exception as e:
                    self.console.print(f"[red]Error in game {game_num + 1}: {e}[/red]")
                    continue
                    
        # Start collection in background thread
        collection_thread = threading.Thread(target=collect_data)
        collection_thread.daemon = True
        collection_thread.start()
        
        return collection_thread
        
    def run_live_dashboard(self, scenario: int = 1, num_games: int = 5):
        """Run the live dashboard."""
        layout = self.create_layout()
        self.is_running = True
        
        # Start data collection
        collection_thread = self.start_data_collection(scenario, num_games)
        
        # Give the collection thread a moment to initialize
        time.sleep(1)
        
        try:
            with Live(layout, console=self.console, refresh_per_second=2, screen=True):
                while self.is_running and collection_thread.is_alive():
                    self.update_display(layout)
                    time.sleep(0.5)
                    
                # Final update after collection is done
                self.update_display(layout)
                time.sleep(2)  # Show final state briefly
                
        except KeyboardInterrupt:
            self.is_running = False
            self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")
            
def main():
    tui = GameTUI()
    
    # Run dashboard for scenario 1 with 3 games
    tui.run_live_dashboard(scenario=1, num_games=3)
    
    print("\n🎉 Data collection complete! Check the 'game_logs' directory for detailed logs.")

if __name__ == "__main__":
    main()