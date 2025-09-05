# ABOUTME: Bridge between solvers and TUI for real-time game streaming
# ABOUTME: Handles WebSocket communication and file-based streaming for live updates

import asyncio
import json
import time
import threading
import websockets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import queue
import logging

@dataclass
class StreamUpdate:
    type: str  # "game_start", "decision", "game_end", "progress"
    solver_id: str
    game_id: str
    timestamp: datetime
    data: Dict[str, Any]

class StreamBridge:
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: set = set()
        self.update_queue = queue.Queue()
        self.active_games: Dict[str, Dict] = {}  # game_id -> game_state
        self.stream_file = Path("game_logs") / "live_stream.jsonl"
        self.running = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StreamBridge")
        
        # Ensure logs directory exists
        Path("game_logs").mkdir(exist_ok=True)
        
    async def register_client(self, websocket, path):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send current active games to new client
        for game_id, game_state in self.active_games.items():
            await websocket.send(json.dumps({
                "type": "game_state",
                "game_id": game_id,
                "data": game_state
            }))
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            self.logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast_update(self, update: StreamUpdate):
        """Broadcast update to all connected clients."""
        if not self.clients:
            return
            
        message = {
            "type": update.type,
            "solver_id": update.solver_id,
            "game_id": update.game_id,
            "timestamp": update.timestamp.isoformat(),
            "data": update.data
        }
        
        # Broadcast to WebSocket clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    def log_to_file(self, update: StreamUpdate):
        """Log update to file for persistence and analysis."""
        log_entry = {
            "type": update.type,
            "solver_id": update.solver_id,
            "game_id": update.game_id,
            "timestamp": update.timestamp.isoformat(),
            "data": update.data
        }
        
        with open(self.stream_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def add_update(self, update_data: Dict[str, Any]) -> StreamUpdate:
        """Add an update to the stream."""
        update = StreamUpdate(
            type=update_data.get("type", "unknown"),
            solver_id=update_data.get("solver_id", "unknown"),
            game_id=update_data.get("game_id", "unknown"),
            timestamp=datetime.now(),
            data=update_data.get("data", {})
        )
        
        # Update active games state
        if update.type == "game_start":
            self.active_games[update.game_id] = {
                "solver_id": update.solver_id,
                "scenario": update.data.get("scenario"),
                "status": "running",
                "admitted": 0,
                "rejected": 0,
                "start_time": update.timestamp.isoformat(),
                "constraints": update.data.get("constraints", []),
                "progress": {}
            }
        elif update.type == "decision" and update.game_id in self.active_games:
            game = self.active_games[update.game_id]
            game["admitted"] = update.data.get("admitted", 0)
            game["rejected"] = update.data.get("rejected", 0)
            game["progress"] = update.data.get("progress", {})
            game["last_decision"] = update.data.get("decision")
            game["last_reasoning"] = update.data.get("reasoning", "")
        elif update.type == "game_end" and update.game_id in self.active_games:
            game = self.active_games[update.game_id]
            game["status"] = update.data.get("status", "completed")
            game["final_rejected"] = update.data.get("rejected_count", 0)
            game["end_time"] = update.timestamp.isoformat()
        
        # Queue for async processing
        self.update_queue.put(update)
        
        # Log to file
        self.log_to_file(update)
        
        return update
    
    async def process_updates(self):
        """Process queued updates and broadcast them."""
        while self.running:
            try:
                # Process all queued updates
                while not self.update_queue.empty():
                    update = self.update_queue.get_nowait()
                    await self.broadcast_update(update)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error processing updates: {e}")
                await asyncio.sleep(1)
    
    def create_callback(self) -> Callable:
        """Create a callback function for solvers to use."""
        def callback(update_data: Dict[str, Any]):
            self.add_update(update_data)
        return callback
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.running = True
        
        # Start the update processor
        update_task = asyncio.create_task(self.process_updates())
        
        # Start WebSocket server
        server = await websockets.serve(self.register_client, "localhost", self.port)
        self.logger.info(f"StreamBridge server started on ws://localhost:{self.port}")
        
        try:
            await server.wait_closed()
        finally:
            self.running = False
            update_task.cancel()
    
    def start_background_server(self):
        """Start server in background thread."""
        def run_server():
            asyncio.run(self.start_server())
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread

# TUI Integration adapter
class TUIStreamAdapter:
    def __init__(self, tui_app):
        self.tui_app = tui_app
        self.bridge = StreamBridge()
        self.callback = self.bridge.create_callback()
        
        # Start bridge in background
        self.bridge.start_background_server()
    
    def get_callback(self):
        """Get the callback for solvers to use."""
        return self.callback
    
    def get_active_games(self) -> Dict[str, Dict]:
        """Get currently active games."""
        return self.bridge.active_games.copy()
    
    def get_recent_updates(self, limit: int = 100) -> List[Dict]:
        """Get recent updates from the stream file."""
        updates = []
        if self.bridge.stream_file.exists():
            with open(self.bridge.stream_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        updates.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return updates

def main():
    """Test the stream bridge."""
    bridge = StreamBridge()
    
    # Test updates
    callback = bridge.create_callback()
    
    # Simulate game start
    callback({
        "type": "game_start",
        "solver_id": "test_solver",
        "game_id": "game_001",
        "data": {
            "scenario": 1,
            "constraints": [{"attribute": "young", "min_count": 600}]
        }
    })
    
    # Simulate decisions
    for i in range(5):
        callback({
            "type": "decision", 
            "solver_id": "test_solver",
            "game_id": "game_001",
            "data": {
                "person_index": i,
                "decision": i % 2 == 0,
                "reasoning": f"test_reason_{i}",
                "admitted": i // 2,
                "rejected": i - i // 2,
                "progress": {"young": i * 0.1}
            }
        })
    
    # Simulate game end
    callback({
        "type": "game_end",
        "solver_id": "test_solver", 
        "game_id": "game_001",
        "data": {
            "status": "completed",
            "rejected_count": 1000
        }
    })
    
    print("Stream bridge test completed. Check game_logs/live_stream.jsonl")

if __name__ == "__main__":
    main()