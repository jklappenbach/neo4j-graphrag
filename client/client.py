"""
WebSocket client for the Graph RAG server.
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, Callable, List
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

logger = logging.getLogger(__name__)

class GraphRagClient:
    """WebSocket client for communicating with the Graph RAG server."""

    def __init__(self, server_url: str = "ws://localhost:8000/ws"):
        self.server_url = server_url
        self.connection_id = str(uuid.uuid4())
        self.websocket = None
        self._is_connected = False
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._task_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._message_handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        try:
            full_url = f"{self.server_url}/{self.connection_id}"
            self.websocket = await websockets.connect(full_url)
            self._is_connected = True

            # Start listening for messages in the background
            asyncio.create_task(self._listen())
            logger.info(f"Connected to server at {full_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self._is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket and self._is_connected:
            self._is_connected = False
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from server")

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to the server."""
        return self._is_connected and self.websocket is not None

    async def _listen(self):
        """Listen for incoming messages from the server."""
        try:
            while self._is_connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse server message: {e}")
                except (ConnectionClosedError, ConnectionClosedOK):
                    logger.info("Server connection closed")
                    break

        except Exception as e:
            logger.error(f"Error in message listener: {e}")
        finally:
            self._is_connected = False

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from the server."""
        request_id = message.get("request_id")
        msg_type = message.get("type")

        logger.debug(f"Received message: {msg_type} for request {request_id}")

        # Handle responses to direct requests (list_documents, etc.)
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(message)
            return

        # Handle task status updates
        if request_id and request_id in self._task_callbacks:
            callback = self._task_callbacks.get(request_id)
            if callback:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in task callback for {request_id}: {e}")

                # Remove callback if task is finished
                if msg_type in ["task_completed", "task_failed", "task_cancelled"]:
                    del self._task_callbacks[request_id]

        # Handle generic message type handlers
        if msg_type and msg_type in self._message_handlers:
            handlers = self._message_handlers[msg_type]
            for handler in handlers:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler for {msg_type}: {e}")

    def add_message_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Add a handler for a specific message type."""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)

    def remove_message_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Remove a handler for a specific message type."""
        if message_type in self._message_handlers:
            try:
                self._message_handlers[message_type].remove(handler)
            except ValueError:
                pass

    async def _send_request(self, message: Dict[str, Any], wait_for_response: bool = True, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Send a request and optionally wait for response."""
        if not self.is_connected:
            raise RuntimeError("Not connected to server")

        request_id = message.get("request_id", str(uuid.uuid4()))
        message["request_id"] = request_id

        future = None
        if wait_for_response:
            future = asyncio.Future()
            self._pending_requests[request_id] = future

        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message['type']} with request_id: {request_id}")

            if wait_for_response and future:
                try:
                    return await asyncio.wait_for(future, timeout=timeout)
                except asyncio.TimeoutError:
                    self._pending_requests.pop(request_id, None)
                    raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")

        except Exception as e:
            if wait_for_response and request_id in self._pending_requests:
                self._pending_requests.pop(request_id, None)
            raise e

        return {"request_id": request_id, "success": True}

    async def query(self, query_text: str, on_status_update: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        """
        Send a query to the server.

        Args:
            query_text: The query string to send
            on_status_update: Optional callback for receiving task status updates

        Returns:
            request_id: The ID of the submitted query request
        """
        request_id = str(uuid.uuid4())

        # Set up status update callback if provided
        if on_status_update:
            self._task_callbacks[request_id] = on_status_update

        await self._send_request({
            "type": "query",
            "query": query_text,
            "request_id": request_id
        }, wait_for_response=False)

        return request_id

    async def cancel_query(self, request_id: str) -> Dict[str, Any]:
        """Cancel a running query."""
        return await self._send_request({
            "type": "cancel",
            "target_request_id": request_id
        })

    async def refresh_documents(self, on_status_update: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        """
        Refresh all documents in the server.

        Args:
            on_status_update: Optional callback for receiving task status updates

        Returns:
            request_id: The ID of the submitted refresh request
        """
        request_id = str(uuid.uuid4())

        # Set up status update callback if provided
        if on_status_update:
            self._task_callbacks[request_id] = on_status_update

        await self._send_request({
            "type": "refresh",
            "request_id": request_id
        }, wait_for_response=False)

        return request_id

    async def list_documents(self) -> List[str]:
        """List all documents on the server."""
        response = await self._send_request({
            "type": "list_documents"
        })

        if response.get("success"):
            return response.get("documents", [])
        else:
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Failed to list documents: {error}")

# Utility functions for common operations
async def simple_query(query_text: str, server_url: str = "ws://localhost:8000/ws", timeout: float = 60.0) -> Dict[str, Any]:
    """
    Perform a simple query with automatic connection handling.

    Args:
        query_text: The query to send
        server_url: WebSocket server URL
        timeout: Maximum time to wait for query completion

    Returns:
        Dict containing the query result or error information
    """
    client = GraphRagClient(server_url)
    result = {"success": False, "error": None, "result": None}

    # Event to wait for query completion
    query_complete = asyncio.Event()

    def on_status_update(message: Dict[str, Any]):
        nonlocal result
        msg_type = message.get("type")

        if msg_type == "task_completed":
            result["success"] = True
            result["result"] = message.get("result")
            query_complete.set()
        elif msg_type == "task_failed":
            result["success"] = False
            result["error"] = message.get("error", "Query failed")
            query_complete.set()
        elif msg_type == "task_started":
            logger.info("Query processing started...")

    try:
        if not await client.connect():
            result["error"] = "Failed to connect to server"
            return result

        request_id = await client.query(query_text, on_status_update)
        logger.info(f"Query submitted with ID: {request_id}")

        # Wait for query completion
        try:
            await asyncio.wait_for(query_complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            result["error"] = f"Query timed out after {timeout} seconds"
            try:
                await client.cancel_query(request_id)
            except:
                pass  # Ignore cancel errors

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Query error: {e}")
    finally:
        await client.disconnect()

    return result

# Example usage
async def main():
    """Example usage of the WebSocket client."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    client = GraphRagClient()

    try:
        if not await client.connect():
            print("Failed to connect to server")
            return

        # Define status update handler
        def query_handler(message: Dict[str, Any]):
            msg_type = message.get("type")
            if msg_type == "task_started":
                print("🔄 Query processing started...")
            elif msg_type == "task_completed":
                print("✅ Query completed successfully!")
                result = message.get("result", {})
                print(f"Result: {result}")
            elif msg_type == "task_failed":
                error = message.get("error", "Unknown error")
                print(f"❌ Query failed: {error}")

        # Send a query
        print("Submitting query...")
        request_id = await client.query("What is this codebase about?", query_handler)
        print(f"Query submitted with ID: {request_id}")

        # List documents
        print("\nListing documents...")
        try:
            docs = await client.list_documents()
            print(f"Found {len(docs)} documents:")
            for doc in docs[:5]:  # Show first 5
                print(f"  - {doc}")
            if len(docs) > 5:
                print(f"  ... and {len(docs) - 5} more")
        except Exception as e:
            print(f"Failed to list documents: {e}")

        # Keep connection alive for a bit to see async responses
        print("\nWaiting for responses...")
        await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())