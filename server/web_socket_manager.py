import json
import logging
import uuid
from typing import Any, Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

from server.server_defines import GraphRagManager

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and message routing."""

    def __init__(self, graph_rag_manager: GraphRagManager):
        # Use WeakValueDictionary to automatically clean up disconnected connections
        self._connections: Dict[str, WebSocket] = {}
        self._request_to_connection: Dict[str, str] = {}
        self._graph_rag_manager = graph_rag_manager

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Register a new WebSocket connection."""
        await websocket.accept()
        self._connections[connection_id] = websocket
        logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self._connections:
            del self._connections[connection_id]
        # Clean up any pending request mappings
        requests_to_remove = [req_id for req_id, conn_id in self._request_to_connection.items() if
                              conn_id == connection_id]
        for req_id in requests_to_remove:
            del self._request_to_connection[req_id]
        logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific WebSocket connection."""
        if connection_id not in self._connections:
            logger.warning(f"Connection {connection_id} not found")
            return False

        try:
            websocket = self._connections[connection_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False

    async def send_response(self, request_id: str, response: Dict[str, Any]) -> bool:
        """Send a response back to the client that made the request."""
        connection_id = self._request_to_connection.get(request_id)
        if not connection_id:
            logger.warning(f"No connection found for request {request_id}")
            return False

        response["request_id"] = request_id
        success = await self.send_message(connection_id, response)

        # For completed/failed tasks, clean up the mapping
        msg_type = response.get("type", "")
        if msg_type in ["task_completed", "task_failed", "task_cancelled"]:
            if request_id in self._request_to_connection:
                del self._request_to_connection[request_id]

        return success

    def register_request(self, request_id: str, connection_id: str):
        """Map a request to its originating connection."""
        self._request_to_connection[request_id] = connection_id

    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Process incoming WebSocket messages and route to GraphRagManager."""
        try:
            msg_type = message.get("type")
            request_id = message.get("request_id", str(uuid.uuid4()))

            # Register this request with the connection
            self.register_request(request_id, connection_id)

            if msg_type == "query":
                query = message.get("query", "")
                response = self._graph_rag_manager.query(request_id, query)
                # Don't send response here - it will be sent by the task completion handler

            elif msg_type == "cancel":
                target_request_id = message.get("target_request_id", request_id)
                response = self._graph_rag_manager.cancel_query(target_request_id)
                await self.send_response(request_id, {
                    "type": "cancel_response",
                    "success": response.get("ok", False),
                    "error": response.get("error")
                })

            elif msg_type == "refresh":
                response = self._graph_rag_manager.refresh_documents(request_id)
                # Don't send response here - it will be sent by the task completion handler

            elif msg_type == "list_documents":
                try:
                    documents = self._graph_rag_manager.list_documents(request_id)
                    await self.send_response(request_id, {
                        "type": "list_documents_response",
                        "documents": documents,
                        "success": True
                    })
                except Exception as e:
                    await self.send_response(request_id, {
                        "type": "list_documents_response",
                        "success": False,
                        "error": str(e)
                    })

            else:
                await self.send_response(request_id, {
                    "type": "error",
                    "success": False,
                    "error": f"Unknown message type: {msg_type}"
                })

        except Exception as e:
            logger.exception(f"Error handling message from {connection_id}: {e}")
            await self.send_message(connection_id, {
                "type": "error",
                "success": False,
                "error": str(e),
                "request_id": message.get("request_id")
            })


# Global WebSocket manager instance
websocket_manager: WebSocketManager = None


def init_websocket_manager(graph_rag_manager: GraphRagManager):
    """Initialize the global WebSocket manager."""
    global websocket_manager
    websocket_manager = WebSocketManager(graph_rag_manager)

    # Set up the notification callback for the TaskManager
    async def websocket_notifier(request_id: str, response: Dict[str, Any]):
        await websocket_manager.send_response(request_id, response)

    graph_rag_manager.set_websocket_notifier(websocket_notifier)
    return websocket_manager


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
