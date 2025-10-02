import json
import logging
import uuid
from typing import Any, Dict

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from server.server_defines import WebSocketManager
from server.task_manager import TaskManager

logger = logging.getLogger(__name__)

websocket_manager: WebSocketManager

class WebSocketManagerImpl(WebSocketManager):
    """Manages WebSocket connections and routing of TaskManager notifications to clients."""

    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._request_to_connection: Dict[str, str] = {}
        self._task_mgr: TaskManager | None = None

    def set_task_manager(self, task_mgr: TaskManager) -> None:
        self._task_mgr = task_mgr
        logger.info("TaskManager set")

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Register a new WebSocket connection."""
        await websocket.accept()
        self._connections[connection_id] = websocket
        logger.info("WebSocket connection established: %s", connection_id)
        # Immediately notify the client of the assigned connection id for request mapping
        try:
            await websocket.send_text(json.dumps({
                "type": "ws_connected",
                "connection_id": connection_id,
                "success": True
            }))
        except Exception:
            logger.exception("Failed to send ws_connected for %s", connection_id)

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection and clean mappings."""
        if connection_id in self._connections:
            del self._connections[connection_id]
        to_remove = [rid for rid, cid in self._request_to_connection.items() if cid == connection_id]
        for rid in to_remove:
            del self._request_to_connection[rid]
        logger.info("WebSocket connection closed: %s", connection_id)

    def register_request(self, request_id: str, connection_id: str):
        """Associate a task/request with the originating connection."""
        self._request_to_connection[request_id] = connection_id

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

    async def send_response(self, request_id: str, message: Dict[str, Any]) -> bool:
        """Send a message addressed by request_id via its mapped connection."""
        connection_id = self._request_to_connection.get(request_id)
        if not connection_id:
            logger.warning("No connection mapping found for request %s", request_id)
            return False

        message["request_id"] = request_id
        ok = await self.send_message(connection_id, message)

        # Cleanup mapping for terminal task states
        msg_type = message.get("type")
        if msg_type in ("task_completed", "task_failed", "task_cancelled"):
            self._request_to_connection.pop(request_id, None)

        return ok

    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        """
        Handle messages from client over WS.
        Expected messages:
          - {"type":"register_request","request_id":"..."} to bind future task notifications to this connection.
        """
        msg_type = message.get("type")
        if msg_type == "register_request":
            req_id = message.get("request_id")
            if not req_id:
                await self.send_message(connection_id, {
                    "type": "error",
                    "success": False,
                    "error": "Missing request_id for register_request"
                })
                return
            self.register_request(req_id, connection_id)
            await self.send_message(connection_id, {
                "type": "request_registered",
                "request_id": req_id,
                "success": True
            })
            return

        # Unknown/unsupported message types
        await self.send_message(connection_id, {
            "type": "error",
            "success": False,
            "error": f"Unsupported message type: {msg_type or 'unknown'}"
        })

    @staticmethod
    async def websocket_endpoint(websocket: WebSocket, connection_id: str = None):
        """WebSocket endpoint handler."""
        if connection_id is None:
            connection_id = str(uuid.uuid4())

        await websocket_manager.connect(websocket, connection_id)

        try:
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    await websocket_manager.handle_message(connection_id, message)
                except json.JSONDecodeError as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "success": False,
                        "error": f"Invalid JSON: {str(e)}"
                    }))
        except WebSocketDisconnect:
            websocket_manager.disconnect(connection_id)
        except Exception as e:
            logger.exception(f"WebSocket error for connection {connection_id}: {e}")
            websocket_manager.disconnect(connection_id)

    @staticmethod
    async def websocket_notifier(request_id: str, response: Dict[str, Any]) -> None:
        await websocket_manager.send_response(request_id, response)


# def init_websocket_manager(task_mgr: TaskManager):
#     """Initialize the global WebSocket manager."""
#     global websocket_manager
#     websocket_manager = WebSocketManagerImpl()
#     websocket_manager.set_task_manager(task_mgr)
#     # Wire TaskManager -> WS notifier so TaskManager can push updates
#     try:
#         task_mgr.add_websocket_notifier("server-bus", WebSocketManagerImpl.websocket_notifier)
#     except Exception as e:
#         # If TaskManager doesn't support this yet, leave a log to implement.
#         logger.warning("TaskManager does not support set_websocket_notifier; task updates won't be pushed via WS: %s", str(e))
#     return websocket_manager