# ABOUTME: Clean TUI dashboard for monitoring games using Rich
# ABOUTME: Single responsibility - display game information in a user-friendly interface

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.box import ROUNDED
from berghain.config import ConfigManager

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
        self.config_manager = ConfigManager()
        self.show_completed: bool = False  # Recent Results hidden by default
        self._layout_needs_rebuild: bool = False
        
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
        if self.show_completed:
            layout["main"].split_row(
                Layout(name="active", ratio=2),
                Layout(name="completed", ratio=1)
            )
        else:
            layout["main"].split_row(
                Layout(name="active", ratio=3)
            )
        
        return layout

    def _compute_required_eff_text(self, game: GameDisplayInfo) -> str:
        """Compute required efficiency text (e.g., '52.3%@807') to beat HS threshold.
        Uses scenario high score with buffer. Estimates minimal admits needed using
        current shortages and, when available, expected frequencies + correlation.
        """
        try:
            hs_raw = self.config_manager.get_high_score_threshold(int(game.scenario)) or 850
            buf = self.config_manager.get_buffer_percentage() or 1.0
            hs_threshold = int(hs_raw * buf)

            predicted_total_admitted = max(game.admitted or 0, 1)
            max_shortage = 0
            if game.constraints:
                # Bound 1: Current admits + max remaining shortage (best-case overlap)
                if game.progress:
                    for constraint in game.constraints:
                        attr = constraint.get("attribute")
                        min_count = int(constraint.get("min_count", 0))
                        raw_prog = float(game.progress.get(attr, 0.0))
                        current = int(round(raw_prog * min_count)) if min_count > 0 else 0
                        current = max(0, min(current, min_count))
                        shortage = max(0, min_count - current)
                        if shortage > max_shortage:
                            max_shortage = shortage
                    predicted_total_admitted = max(predicted_total_admitted, (game.admitted or 0) + max_shortage)

                # Bound 2: Scenario frequencies/correlation union estimate (2-constraint only)
                try:
                    scen = self.config_manager.get_scenario_config(int(game.scenario)) or {}
                    exp_freq = scen.get("expected_frequencies", {})
                    exp_corr = scen.get("expected_correlations", {})
                    attrs = [c.get("attribute") for c in game.constraints]
                    mins = {c.get("attribute"): int(c.get("min_count", 0)) for c in game.constraints}
                    if len(attrs) == 2 and all(a in exp_freq for a in attrs):
                        a, b = attrs[0], attrs[1]
                        p1 = float(exp_freq.get(a, 0.0))
                        p2 = float(exp_freq.get(b, 0.0))
                        import math
                        phi = float(exp_corr.get(a, {}).get(b, exp_corr.get(b, {}).get(a, 0.0)))
                        p11 = phi * math.sqrt(max(p1*(1-p1)*p2*(1-p2), 0.0)) + p1*p2
                        p11 = max(0.0, min(p11, min(p1, p2)))
                        union = max(1e-6, p1 + p2 - p11)
                        need_a = mins.get(a, 0) * union / max(p1, 1e-6)
                        need_b = mins.get(b, 0) * union / max(p2, 1e-6)
                        x_needed_union = math.ceil(max(need_a, need_b))
                        predicted_total_admitted = max(predicted_total_admitted, x_needed_union)
                except Exception:
                    pass

            req_eff = predicted_total_admitted / (predicted_total_admitted + hs_threshold) if predicted_total_admitted > 0 else 0.0
            return f"{req_eff*100:.1f}%@{hs_threshold}"
        except Exception:
            return "—"
    
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
        
        # Create game entry if it doesn't exist (handles mid-stream detection)
        if solver_id not in self.active_games:
            # Extract constraints from progress keys if available
            constraints = data.get("constraints", [])
            if not constraints and "progress" in data:
                # Create dummy constraints from progress keys
                constraints = [{"attribute": attr} for attr in data["progress"].keys()]
                # Try to infer scenario from constraint keys and fill min_count
                inferred_scenario = self._infer_scenario_from_progress(list(data["progress"].keys()))
                if inferred_scenario:
                    scen_cfg = self.config_manager.get_scenario_config(inferred_scenario) or {}
                    targets = {c["attribute"]: int(c.get("min_count", 0)) for c in scen_cfg.get("constraints", [])}
                    for c in constraints:
                        attr = c.get("attribute")
                        if attr in targets:
                            c["min_count"] = targets[attr]
            
            game_info = GameDisplayInfo(
                game_id=data.get("game_id", "unknown")[:8],
                solver_id=solver_id,
                scenario=data.get("scenario", self._infer_scenario_from_progress(list(data.get("progress", {}).keys())) or 0),
                status="running",
                success=False,
                admitted=data.get("admitted", 0),
                rejected=data.get("rejected", 0),
                progress=data.get("progress", {}),
                start_time=update_info.get("timestamp", data.get("timestamp", "")),
                duration=0,
                constraints=constraints
            )
            self.active_games[solver_id] = game_info
        
        if update_type == "game_start":
            # Update game info with game_start specific data
            game = self.active_games[solver_id]
            game.game_id = data.get("game_id", "unknown")[:8]
            game.scenario = data.get("scenario", 0)
            game.constraints = data.get("constraints", game.constraints)
            game.start_time = data.get("timestamp", game.start_time)
            
        elif update_type == "decision":
            # Update progress
            game = self.active_games[solver_id]
            game.admitted = data.get("admitted", 0)
            game.rejected = data.get("rejected", 0)
            game.progress = data.get("progress", {})
            # If we still lack targets, try to fill them now
            if game.constraints and (not any('min_count' in c for c in game.constraints)):
                inferred_scenario = self._infer_scenario_from_progress(list(game.progress.keys()))
                if inferred_scenario:
                    scen_cfg = self.config_manager.get_scenario_config(inferred_scenario) or {}
                    targets = {c["attribute"]: int(c.get("min_count", 0)) for c in scen_cfg.get("constraints", [])}
                    for c in game.constraints:
                        if c.get("attribute") in targets:
                            c["min_count"] = targets[c["attribute"]]
                    if not game.scenario:
                        game.scenario = inferred_scenario
            
        elif update_type == "game_end" and solver_id in self.active_games:
            # Move to completed
            game = self.active_games[solver_id]
            game.status = data.get("status", "completed")
            game.success = data.get("success", False)
            game.admitted = data.get("admitted_count", game.admitted)
            game.rejected = data.get("rejected_count", game.rejected)
            game.duration = data.get("duration", 0)
            # Preserve scenario information
            if data.get("scenario") is not None:
                game.scenario = data.get("scenario")
            
            # Move to completed list
            self.completed_games.insert(0, game)
            if len(self.completed_games) > self.max_completed_display:
                self.completed_games = self.completed_games[:self.max_completed_display]
            
            # Remove from active
            del self.active_games[solver_id]

    def _infer_scenario_from_progress(self, attrs: List[str]) -> Optional[int]:
        """Infer scenario by matching constraint attribute sets with known scenarios."""
        try:
            keys = set(a for a in attrs if a)
            if not keys:
                return None
            for scen_id in self.config_manager.list_available_scenarios():
                cfg = self.config_manager.get_scenario_config(scen_id) or {}
                scen_keys = {c.get("attribute") for c in cfg.get("constraints", [])}
                if keys == scen_keys:
                    return scen_id
            return None
        except Exception:
            return None
    
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
        try:
            buffer_pct = self.config_manager.get_buffer_percentage()
        except Exception:
            buffer_pct = 1.0
        
        header_text = Text()
        header_text.append("🎯 Berghain Challenge Dashboard", style="bold white")
        header_text.append(f"  |  Time: {current_time}", style="dim")
        header_text.append(f"  |  Active: {active_count}", style="green")
        header_text.append(f"  |  Completed: {completed_count}", style="blue")
        header_text.append(f"  |  HS buf: {buffer_pct*100:.0f}%", style="dim")
        
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
        table.add_column("Solver", style="cyan", width=16)
        table.add_column("S", justify="center", width=3)  # Scenario
        table.add_column("Status", width=11)
        table.add_column("A/R", justify="right", width=10)  # Admitted/Rejected
        table.add_column("Eff%", justify="right", width=6)  # Acceptance rate
        table.add_column("ReqEff", justify="right", width=11)  # Min eff to beat HS (with threshold)
        table.add_column("Constraint Progress", width=32)
        table.add_column("Time", justify="right", width=8)
        
        for game in self.active_games.values():
            # Status styling
            if game.status == "running":
                status_text = Text("▶ Running", style="yellow")
            else:
                status_text = Text(f"● {game.status.title()}", style="green" if game.success else "red")
            
            # Improved progress display
            if game.constraints and game.progress:
                progress_parts = []
                for constraint in game.constraints:
                    attr = constraint["attribute"] 
                    raw_prog = game.progress.get(attr, 0)
                    prog = min(raw_prog, 1.0)  # Cap at 100%
                    
                    # Create a 6-character progress bar
                    bar_length = 6
                    filled = int(prog * bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    
                    # Color the bar based on progress
                    if prog >= 0.95:
                        bar_color = "green"
                    elif prog >= 0.5:
                        bar_color = "yellow" 
                    else:
                        bar_color = "red"
                    
                    # Current/target if we know min_count
                    min_count = constraint.get("min_count")
                    counts_str = ""
                    if isinstance(min_count, (int, float)) and min_count:
                        # Derive current from progress ratio when available
                        try:
                            current = int(max(0, round(raw_prog * float(min_count))))
                        except Exception:
                            current = None
                        if current is not None:
                            # Cap display at target for readability
                            current_capped = min(current, int(min_count))
                            counts_str = f" {current_capped}/{int(min_count)}"
                    
                    # Attribute label (truncated) + bar + counts + percentage
                    attr_short = attr[:4] if len(attr) > 4 else attr
                    progress_parts.append(f"{attr_short}:{bar}{counts_str} {prog:.1%}")
                
                progress_text = " ".join(progress_parts)
            else:
                progress_text = "N/A"
            
            # Calculate elapsed time
            try:
                if game.start_time:
                    # Handle different timestamp formats
                    timestamp_str = game.start_time
                    if 'T' in timestamp_str:
                        # ISO format with or without timezone
                        start = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        start = start.replace(tzinfo=None)
                    else:
                        # Fallback for other formats
                        start = datetime.fromisoformat(timestamp_str)
                    
                    elapsed = datetime.now() - start
                    time_str = f"{int(elapsed.total_seconds() // 60)}:{int(elapsed.total_seconds() % 60):02d}"
                else:
                    time_str = "N/A"
            except Exception as e:
                time_str = "N/A"
            
            # Format admitted/rejected with better spacing
            ar_text = f"{game.admitted:,}/{game.rejected:,}"
            # Efficiency (acceptance rate)
            total_proc = (game.admitted or 0) + (game.rejected or 0)
            eff_text = f"{(game.admitted/total_proc*100):.1f}%" if total_proc > 0 else "—"

            # Required efficiency to finish under high score threshold
            req_eff_text = self._compute_required_eff_text(game)
            
            table.add_row(
                game.solver_id[:15],
                str(game.scenario) if game.scenario is not None and game.scenario > 0 else "?",
                status_text,
                ar_text,
                eff_text,
                req_eff_text,
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
        table.add_column("Rej", justify="right", width=6)
        table.add_column("Eff%", justify="right", width=6)
        table.add_column("ReqEff", justify="right", width=11)
        table.add_column("Time", justify="right", width=6)
        
        for game in self.completed_games:
            result_text = Text("SUCCESS", style="green") if game.success else Text("FAILED", style="red")
            
            duration_str = f"{int(game.duration)}s" if game.duration > 0 else "N/A"
            total_proc = (game.admitted or 0) + (game.rejected or 0)
            eff_text = f"{(game.admitted/total_proc*100):.1f}%" if total_proc > 0 else "—"
            req_eff_text = self._compute_required_eff_text(game)
            
            table.add_row(
                game.solver_id[:11],
                str(game.scenario),
                result_text,
                f"{game.rejected:,}",
                eff_text,
                req_eff_text,
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
        footer_text.append("R+Enter", style="bold yellow")
        footer_text.append(" = Toggle Results  |  ", style="dim")
        footer_text.append("Refresh: ", style="bold")
        footer_text.append(f"{self.refresh_rate}s", style="yellow")
        
        return Panel(
            Align.center(footer_text),
            box=ROUNDED,
            style="blue"
        )
    
    def _update_layout(self):
        """Update the layout with current data."""
        # Clean up stale games (active for more than 10 minutes)
        self._cleanup_stale_games()
        
        self.layout["header"].update(self._create_header())
        self.layout["active"].update(self._create_active_games_panel())
        if self.show_completed:
            try:
                self.layout["completed"].update(self._create_completed_games_panel())
            except Exception:
                pass
        self.layout["footer"].update(self._create_footer())
    
    def _cleanup_stale_games(self):
        """Remove active games that have been running too long (likely dead)."""
        current_time = datetime.now()
        stale_games = []
        
        for solver_id, game in self.active_games.items():
            try:
                if game.start_time:
                    # Try to parse different timestamp formats
                    start_time_str = game.start_time
                    if 'T' in start_time_str:
                        # ISO format
                        start = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                        start = start.replace(tzinfo=None)  # Remove timezone for comparison
                    else:
                        continue  # Skip if we can't parse
                    
                    elapsed = current_time - start
                    # If game has been "active" for more than 10 minutes, it's probably stale
                    if elapsed.total_seconds() > 600:  # 10 minutes
                        stale_games.append(solver_id)
            except Exception:
                # If we can't parse the time, assume it's stale
                stale_games.append(solver_id)
        
        # Move stale games to completed with unknown status
        for solver_id in stale_games:
            game = self.active_games[solver_id]
            game.status = "timeout"
            game.success = False
            self.completed_games.insert(0, game)
            del self.active_games[solver_id]
    
    def run(self):
        """Run the TUI dashboard."""
        print("🚀 Starting Berghain Dashboard")
        print(f"📁 Monitoring: game_logs/")
        print("Press Ctrl+C to exit\n")
        
        self.running = True
        
        # Start file watcher
        self.file_watcher.start_watching()
        
        # Start a simple input thread to toggle recent results (press 'r' + Enter)
        def _input_loop():
            while self.running:
                try:
                    cmd = sys.stdin.readline()
                    if not cmd:
                        break
                    cmd = cmd.strip().lower()
                    if cmd in ("r", "toggle", "results", "recent"):
                        self.show_completed = not self.show_completed
                        self._layout_needs_rebuild = True
                except Exception:
                    break
        input_thread = threading.Thread(target=_input_loop, daemon=True)
        input_thread.start()
        
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
            with Live(self.layout, refresh_per_second=1/self.refresh_rate, screen=True) as live:
                while self.running:
                    # Rebuild layout on toggle
                    if self._layout_needs_rebuild:
                        self.layout = self._create_layout()
                        # Populate new layout immediately
                        self._update_layout()
                        live.update(self.layout)
                        self._layout_needs_rebuild = False
                    else:
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
