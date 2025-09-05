# ABOUTME: Real-time streaming monitor for Berghain games without modifying data_collector
# ABOUTME: Watches files, parses partial JSON, and streams updates to dashboard via WebSocket

import time
import json
import os
import asyncio
import websockets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
import queue
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

@dataclass
class StreamUpdate:
    game_id: str
    timestamp: float
    update_type: str  # 'game_start', 'decision', 'constraint_update', 'game_end'
    data: Dict[str, Any]

class GameFileWatcher(FileSystemEventHandler):
    """Watches game log files for changes and extracts updates"""
    
    def __init__(self, update_queue: queue.Queue):
        super().__init__()
        self.update_queue = update_queue
        self.tracked_files: Dict[str, dict] = {}  # filepath -> file_info
        
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.json'):
            return
            
        filepath = Path(event.src_path)
        self._process_file_change(filepath)
        
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.json'):
            return
            
        filepath = Path(event.src_path)
        self._process_new_file(filepath)
        
    def _process_new_file(self, filepath: Path):
        """Process a newly created game file"""
        try:
            # Extract game info from filename
            # Format: scenario_X_YYYYMMDD_HHMMSS_gameID.json
            filename = filepath.stem
            parts = filename.split('_')
            
            if len(parts) >= 4 and parts[0] == 'scenario':
                scenario = int(parts[1])
                game_id = parts[-1] if len(parts) > 4 else 'unknown'
                
                update = StreamUpdate(
                    game_id=game_id,
                    timestamp=time.time(),
                    update_type='game_start',
                    data={
                        'scenario': scenario,
                        'filepath': str(filepath),
                        'start_time': datetime.now().isoformat()
                    }
                )
                
                self.update_queue.put(update)
                
                # Track this file
                self.tracked_files[str(filepath)] = {
                    'game_id': game_id,
                    'scenario': scenario,
                    'last_size': 0,
                    'last_people_count': 0
                }
                
        except Exception as e:
            print(f"Error processing new file {filepath}: {e}")
            
    def _process_file_change(self, filepath: Path):
        """Process changes to an existing game file"""
        filepath_str = str(filepath)
        
        if filepath_str not in self.tracked_files:
            return
            
        try:
            file_info = self.tracked_files[filepath_str]
            current_size = os.path.getsize(filepath)
            
            # Skip if file size hasn't changed much (avoid reading same data)
            if current_size - file_info['last_size'] < 100:
                return
                
            # Try to parse the current JSON (might be incomplete)
            game_data = self._parse_partial_json(filepath)
            if not game_data:
                return
                
            # Extract updates
            current_people_count = len(game_data.get('people', []))
            
            # Check for new decisions
            if current_people_count > file_info['last_people_count']:
                new_decisions = current_people_count - file_info['last_people_count']
                
                # Calculate current constraint progress
                constraints_progress = self._calculate_constraint_progress(game_data)
                
                update = StreamUpdate(
                    game_id=file_info['game_id'],
                    timestamp=time.time(),
                    update_type='constraint_update',
                    data={
                        'people_processed': current_people_count,
                        'new_decisions': new_decisions,
                        'constraints': constraints_progress,
                        'admitted': sum(1 for p in game_data.get('people', []) if p.get('decision', False)),
                        'rejected': sum(1 for p in game_data.get('people', []) if not p.get('decision', False))
                    }
                )
                
                self.update_queue.put(update)
                file_info['last_people_count'] = current_people_count
                
            # Check for game completion
            final_status = game_data.get('final_status')
            if final_status and final_status != '':
                update = StreamUpdate(
                    game_id=file_info['game_id'],
                    timestamp=time.time(),
                    update_type='game_end',
                    data={
                        'final_status': final_status,
                        'final_admitted_count': game_data.get('final_admitted_count', 0),
                        'final_rejected_count': game_data.get('final_rejected_count', 0),
                        'total_time': game_data.get('total_time', 0)
                    }
                )
                
                self.update_queue.put(update)
                
            file_info['last_size'] = current_size
            
        except Exception as e:
            print(f"Error processing file change {filepath}: {e}")
            
    def _parse_partial_json(self, filepath: Path) -> Optional[Dict]:
        """Attempt to parse potentially incomplete JSON"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Try to parse as-is
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
                
            # Try to repair incomplete JSON
            repaired_content = self._repair_json(content)
            if repaired_content:
                try:
                    return json.loads(repaired_content)
                except json.JSONDecodeError:
                    pass
                    
            return None
            
        except Exception:
            return None
            
    def _repair_json(self, content: str) -> Optional[str]:
        """Attempt to repair incomplete JSON"""
        # Remove trailing comma if exists
        content = content.rstrip()
        if content.endswith(','):
            content = content[:-1]
            
        # Try to close incomplete structures
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        if open_brackets > 0:
            content += ']' * open_brackets
            
        if open_braces > 0:
            content += '}' * open_braces
            
        return content
        
    def _calculate_constraint_progress(self, game_data: Dict) -> List[Dict]:
        """Calculate current constraint progress from game data"""
        constraints = game_data.get('constraints', [])
        people = game_data.get('people', [])
        
        progress = []
        for constraint in constraints:
            attr = constraint['attribute']
            required = constraint.get('minCount', constraint.get('requiredCount', 0))
            
            # Count admitted people with this attribute
            admitted_with_attr = sum(1 for person in people
                                   if person.get('decision', False) and 
                                   person.get('attributes', {}).get(attr, False))
            
            progress.append({
                'attribute': attr,
                'required': required,
                'current': admitted_with_attr,
                'fill_rate': (admitted_with_attr / required) if required > 0 else 0
            })
            
        return progress

class StreamServer:
    """WebSocket server for streaming updates to dashboard"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.update_queue = queue.Queue()
        
    async def register_client(self, websocket, path):
        """Register a new client connection"""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            
    async def broadcast_updates(self):
        """Broadcast updates to all connected clients"""
        while True:
            try:
                # Get updates from queue (non-blocking)
                updates = []
                while not self.update_queue.empty():
                    update = self.update_queue.get_nowait()
                    updates.append(asdict(update))
                    
                if updates and self.clients:
                    message = json.dumps({
                        'type': 'batch_update',
                        'updates': updates,
                        'timestamp': time.time()
                    })
                    
                    # Send to all clients
                    disconnected = []
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.append(client)
                            
                    # Clean up disconnected clients
                    for client in disconnected:
                        self.clients.discard(client)
                        
                await asyncio.sleep(0.1)  # 10Hz update rate
                
            except Exception as e:
                print(f"Error in broadcast_updates: {e}")
                await asyncio.sleep(1)
                
    async def start_server(self):
        """Start the WebSocket server"""
        # Start the WebSocket server
        server = await websockets.serve(
            self.register_client,
            self.host,
            self.port
        )
        
        print(f"Stream server started on ws://{self.host}:{self.port}")
        
        # Start broadcast task
        broadcast_task = asyncio.create_task(self.broadcast_updates())
        
        # Keep running
        await server.wait_closed()

class StreamMonitor:
    """Main streaming monitor coordinating file watching and WebSocket server"""
    
    def __init__(self, logs_dir: str = "game_logs", host: str = 'localhost', port: int = 8765):
        self.logs_dir = Path(logs_dir)
        self.update_queue = queue.Queue()
        self.server = StreamServer(host, port)
        self.server.update_queue = self.update_queue  # Share queue
        
        # File watcher
        self.watcher = GameFileWatcher(self.update_queue)
        self.observer = Observer()
        
    def start_monitoring(self):
        """Start monitoring files and streaming updates"""
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup file watcher
        self.observer.schedule(self.watcher, str(self.logs_dir), recursive=False)
        self.observer.start()
        
        print(f"File watcher started on {self.logs_dir}")
        
        # Process existing files to get current state
        self._process_existing_files()
        
        # Start WebSocket server in asyncio
        asyncio.run(self.server.start_server())
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.observer.stop()
        self.observer.join()
        
    def _process_existing_files(self):
        """Process existing files to get current game state"""
        for json_file in self.logs_dir.glob("*.json"):
            self.watcher._process_new_file(json_file)
            # Give it time to process
            time.sleep(0.1)

# Client-side utilities for dashboard integration
class StreamClient:
    """WebSocket client for receiving stream updates in dashboard"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.is_connected = False
        self.update_callback: Optional[Callable] = None
        
    async def connect(self, update_callback: Callable):
        """Connect to stream server"""
        self.update_callback = update_callback
        
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            print(f"Connected to stream server at {self.uri}")
            
            # Listen for updates
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    if self.update_callback:
                        self.update_callback(data)
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
                    
        except Exception as e:
            print(f"Stream client error: {e}")
            self.is_connected = False
            
    async def disconnect(self):
        """Disconnect from stream server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False

def main():
    """Run the stream monitor server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Berghain Game Stream Monitor')
    parser.add_argument('--logs-dir', default='game_logs', help='Directory to watch for game logs')
    parser.add_argument('--host', default='localhost', help='WebSocket server host')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port')
    
    args = parser.parse_args()
    
    monitor = StreamMonitor(args.logs_dir, args.host, args.port)
    
    try:
        print("🔍 Starting Berghain Game Stream Monitor")
        print(f"Watching: {args.logs_dir}")
        print(f"Server: ws://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping stream monitor...")
        monitor.stop_monitoring()
        
if __name__ == "__main__":
    main()