# ABOUTME: Clean WebSocket streaming server for real-time game updates
# ABOUTME: Single responsibility - handle WebSocket communication and broadcasting

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Set, Dict, Any, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class StreamServer:
    """Clean WebSocket server for streaming game updates."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.active_games: Dict[str, Dict[str, Any]] = {}
        self.server: Optional[websockets.WebSocketServer] = None
        
        # Stream log file
        self.stream_log_path = Path("game_logs") / "live_stream.jsonl"
        Path("game_logs").mkdir(exist_ok=True)
    
    async def register_client(self, websocket, path):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        
        # Send current active games to new client
        for game_id, game_data in self.active_games.items():
            await self._send_to_client(websocket, {
                "type": "game_state_snapshot",
                "game_id": game_id,
                "data": game_data
            })
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected: {client_addr}")
    
    async def _send_to_client(self, websocket, message: Dict[str, Any]):
        """Send message to a specific client."""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            pass  # Client disconnected, will be cleaned up
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        # Add timestamp
        message["timestamp"] = datetime.now().isoformat()
        
        # Log to file
        self._log_to_file(message)
        
        # Update active games state
        self._update_active_games(message)
        
        # Broadcast to all clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Clean up disconnected clients
        self.clients -= disconnected_clients
    
    def _log_to_file(self, message: Dict[str, Any]):
        """Log message to JSONL file for persistence."""
        try:
            with open(self.stream_log_path, 'a') as f:
                f.write(json.dumps(message) + '\n')
        except Exception as e:
            logger.error(f"Failed to log stream message: {e}")
    
    def _update_active_games(self, message: Dict[str, Any]):
        """Update internal active games state."""
        msg_type = message.get("type")
        game_id = message.get("data", {}).get("game_id")
        
        if not game_id:
            return
        
        if msg_type == "game_start":
            data = message["data"]
            self.active_games[game_id] = {
                "solver_id": message.get("solver_id", "unknown"),
                "scenario": data.get("scenario"),
                "strategy": data.get("strategy"),
                "status": "running",
                "admitted": 0,
                "rejected": 0,
                "start_time": message["timestamp"],
                "constraints": data.get("constraints", []),
                "progress": {}
            }
            
        elif msg_type == "decision" and game_id in self.active_games:
            data = message["data"]
            game = self.active_games[game_id]
            game["admitted"] = data.get("admitted", 0)
            game["rejected"] = data.get("rejected", 0)
            game["progress"] = data.get("progress", {})
            game["last_decision"] = data.get("decision")
            game["last_reasoning"] = data.get("reasoning", "")
            
        elif msg_type == "game_end" and game_id in self.active_games:
            data = message["data"]
            game = self.active_games[game_id]
            game["status"] = data.get("status", "completed")
            game["success"] = data.get("success", False)
            game["final_rejected"] = data.get("rejected_count", 0)
            game["final_admitted"] = data.get("admitted_count", 0)
            game["end_time"] = message["timestamp"]
            game["duration"] = data.get("duration", 0)
    
    def create_callback(self):
        """Create a callback function for solvers to use."""
        def callback(update_data: Dict[str, Any]):
            # Convert to async call
            asyncio.create_task(self.broadcast(update_data))
        return callback
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.register_client,
            self.host,
            self.port
        )
        logger.info(f"StreamServer started on ws://{self.host}:{self.port}")
        return self.server
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("StreamServer stopped")
    
    def get_active_games(self) -> Dict[str, Dict[str, Any]]:
        """Get current active games state."""
        return self.active_games.copy()
    
    def clear_completed_games(self):
        """Clear completed games from active state."""
        completed_games = [
            game_id for game_id, game_data in self.active_games.items()
            if game_data.get("status") != "running"
        ]
        
        for game_id in completed_games:
            del self.active_games[game_id]
        
        logger.info(f"Cleared {len(completed_games)} completed games from active state")