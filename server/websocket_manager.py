import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from server.server_defines import WebSocketManager

logger = logging.getLogger(__name__)

websocket_manager: WebSocketManager

class WebSocketManagerImpl(WebSocketManager):
    """Manages WebSocket connections and routing of notifications to clients."""

    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Register a new WebSocket connection."""
        await websocket.accept()
        self._connections[connection_id] = websocket
        logger.info("WebSocket connection established: %s", connection_id)
        # Immediately notify the client of the assigned connection id for request mapping
        try:
            while True:
                # The server is not expecting to receive messages in this one-way example.
                # However, a receive call is typically needed to keep the connection open
                # and detect disconnects. You can add a timeout or handle potential client messages.
                # For this example, we'll just wait indefinitely or until disconnect.
                try:
                    # This will raise WebSocketDisconnect if the client closes the connection
                    _ = await websocket.receive_text()
                except WebSocketDisconnect:
                    break  # Exit the loop on disconnect
                except asyncio.CancelledError:
                    break  # Handle potential cancellation during server shutdown
        finally:
            self._connections.pop(connection_id, None)

    def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection and clean mappings."""
        if connection_id in self._connections:
            del self._connections[connection_id]
        logger.info("WebSocket connection closed: %s", connection_id)

    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific WebSocket connection."""
        ws = self._connections.get(connection_id)
        if not ws:
            logger.warning("Connection %s not found", connection_id)
            return False
        try:
            await ws.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error("Failed to send message to %s: %s", connection_id, e)
            self.disconnect(connection_id)
            return False

    async def send_message_all(self, message: Dict[str, Any]) -> None:
        """Send a message to all WebSocket connections."""

        for connection_id in self._connections.keys():
            ok = await self.send_message(connection_id, message)
            if ok == False:
                logger.warning("Failed to send message to %s", connection_id)

    def stop(self) -> None:
        for connection_id in self._connections:
            del self._connections[connection_id]
        self._connections.clear()