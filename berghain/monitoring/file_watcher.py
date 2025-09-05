# ABOUTME: File watcher for monitoring game logs without WebSocket dependency
# ABOUTME: Clean implementation using watchdog for file system events

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


logger = logging.getLogger(__name__)


class GameLogHandler(FileSystemEventHandler):
    """Handler for game log file changes."""
    
    def __init__(self, update_callback: Callable):
        super().__init__()
        self.update_callback = update_callback
        self.tracked_files: Dict[str, Dict] = {}  # filepath -> file_info
    
    def on_created(self, event):
        """Handle new log file creation."""
        if event.is_directory or not event.src_path.endswith('.json'):
            return
        
        filepath = Path(event.src_path)
        if filepath.name.startswith('live_'):
            self._process_live_file(filepath, is_new=True)
        else:
            self._process_new_file(filepath)
    
    def on_modified(self, event):
        """Handle log file modification."""
        if event.is_directory or not event.src_path.endswith('.json'):
            return
        
        filepath = Path(event.src_path)
        if filepath.name.startswith('live_'):
            self._process_live_file(filepath, is_new=False)
        else:
            self._process_file_update(filepath)
    
    def on_deleted(self, event):
        """Handle file deletion (live files when games complete)."""
        if event.is_directory or not event.src_path.endswith('.json'):
            return
        
        filepath = Path(event.src_path)
        if filepath.name.startswith('live_'):
            # Live file deleted means game completed
            self.update_callback({
                "type": "game_completed",
                "filepath": str(filepath),
                "timestamp": datetime.now().isoformat()
            })
    
    def _process_new_file(self, filepath: Path):
        """Process a newly created game log file."""
        try:
            # Wait a moment for file to be written
            time.sleep(0.1)
            
            with open(filepath, 'r') as f:
                content = f.read()
                
            if content.strip():
                game_data = json.loads(content)
                
                # Extract game info
                game_info = {
                    "type": "game_discovered",
                    "filepath": str(filepath),
                    "game_id": game_data.get("game_id", "unknown"),
                    "solver_id": game_data.get("solver_id", "unknown"),
                    "scenario": game_data.get("scenario_id", game_data.get("scenario", 0)),
                    "status": game_data.get("status", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.update_callback(game_info)
                
                # Track this file
                self.tracked_files[str(filepath)] = {
                    "game_id": game_data.get("game_id", "unknown"),
                    "last_modified": filepath.stat().st_mtime,
                    "last_size": filepath.stat().st_size
                }
                
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.debug(f"Could not process new file {filepath}: {e}")
    
    def _process_file_update(self, filepath: Path):
        """Process updates to existing log files."""
        try:
            # Check if this is a file we care about
            if str(filepath) not in self.tracked_files:
                return
            
            file_info = self.tracked_files[str(filepath)]
            current_size = filepath.stat().st_size
            current_mtime = filepath.stat().st_mtime
            
            # Check if file actually changed
            if (current_size == file_info["last_size"] and 
                current_mtime == file_info["last_modified"]):
                return
            
            # Update tracking info
            file_info["last_size"] = current_size  
            file_info["last_modified"] = current_mtime
            
            # Try to read updated content
            with open(filepath, 'r') as f:
                game_data = json.loads(f.read())
            
            # Send update
            update_info = {
                "type": "game_updated",
                "filepath": str(filepath),
                "game_id": file_info["game_id"],
                "solver_id": game_data.get("solver_id", "unknown"),
                "status": game_data.get("status", "unknown"),
                "admitted": game_data.get("admitted_count", 0),
                "rejected": game_data.get("rejected_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            self.update_callback(update_info)
            
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.debug(f"Could not process file update {filepath}: {e}")
    
    def _process_live_file(self, filepath: Path, is_new: bool):
        """Process live game status file."""
        try:
            # Wait a moment for file to be written
            time.sleep(0.05)
            
            with open(filepath, 'r') as f:
                live_data = json.loads(f.read())
            
            # Extract game info from live data
            live_info = {
                "type": "live_update",
                "filepath": str(filepath),
                "solver_id": live_data.get("solver_id", "unknown"),
                "update_type": live_data.get("type", "unknown"),
                "data": live_data.get("data", {}),
                "timestamp": live_data.get("timestamp", datetime.now().isoformat()),
                "is_new": is_new
            }
            
            self.update_callback(live_info)
            
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.debug(f"Could not process live file {filepath}: {e}")


class GameLogWatcher:
    """Watches game log directory for real-time updates."""
    
    def __init__(self, logs_directory: str = "game_logs"):
        self.logs_directory = Path(logs_directory)
        self.observer: Optional[Observer] = None
        self.update_callbacks: List[Callable] = []
        
        # Ensure directory exists
        self.logs_directory.mkdir(exist_ok=True)
    
    def add_update_callback(self, callback: Callable):
        """Add callback to be called on file updates."""
        self.update_callbacks.append(callback)
    
    def _handle_file_update(self, update_info: Dict[str, Any]):
        """Handle file system updates and notify callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(update_info)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    def start_watching(self):
        """Start watching the logs directory."""
        if self.observer:
            logger.warning("Watcher already running")
            return
        
        handler = GameLogHandler(self._handle_file_update)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.logs_directory), recursive=False)
        
        self.observer.start()
        logger.info(f"Started watching {self.logs_directory}")
        
        # Process existing files
        self._process_existing_files(handler)
    
    def stop_watching(self):
        """Stop watching the logs directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped file watching")
    
    def _process_existing_files(self, handler: GameLogHandler):
        """Process files that already exist in the directory."""
        json_files = list(self.logs_directory.glob("*.json"))
        logger.info(f"Processing {len(json_files)} existing log files")
        
        for json_file in json_files:
            try:
                handler._process_new_file(json_file)
            except Exception as e:
                logger.debug(f"Could not process existing file {json_file}: {e}")
    
    def get_recent_games(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent game data from log files."""
        json_files = list(self.logs_directory.glob("*.json"))
        
        # Sort by modification time (newest first)
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        recent_games = []
        for json_file in json_files[:limit]:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.loads(f.read())
                
                # Extract summary info
                game_summary = {
                    "filepath": str(json_file),
                    "game_id": game_data.get("game_id", "unknown"),
                    "solver_id": game_data.get("solver_id", "unknown"),
                    "scenario": game_data.get("scenario_id", game_data.get("scenario", 0)),
                    "status": game_data.get("status", "unknown"),
                    "success": game_data.get("success", False),
                    "admitted": game_data.get("admitted_count", 0),
                    "rejected": game_data.get("rejected_count", 0),
                    "duration": game_data.get("duration_seconds", 0),
                    "modified_time": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()
                }
                
                recent_games.append(game_summary)
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.debug(f"Could not read game file {json_file}: {e}")
        
        return recent_games