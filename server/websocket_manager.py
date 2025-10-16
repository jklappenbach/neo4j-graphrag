import asyncio
import json
import logging
import contextlib
from typing import Dict, Any

import anyio
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManagerImpl:
    def __init__(self) -> None:
        self._connections: Dict[str, WebSocket] = {}
        self._request_to_connection: Dict[str, str] = {}
        # Track per-connection receive tasks (anyio tasks) so we can cancel on stop()
        self._receive_tasks: Dict[str, anyio.abc.TaskGroup] = {}
        self._stopping = False

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept, register, send hello, start background receive task (portable across asyncio/trio), then return."""
        await websocket.accept()
        self._connections[connection_id] = websocket
        logger.info("WebSocket connection established: %s", connection_id)

        try:
            await websocket.send_text(json.dumps({
                "type": "ws_connected",
                "connection_id": connection_id
            }))
        except Exception as e:
            logger.warning("Failed sending ws_connected for %s: %s", connection_id, e)
            await self._safe_disconnect(connection_id)
            return

        # Start background receive loop using anyio TaskGroup; store the group for cancellation in stop()
        tg = anyio.create_task_group()
        await tg.__aenter__()
        try:
            tg.start_soon(self._receive_loop, connection_id)
            self._receive_tasks[connection_id] = tg
        except Exception:
            # If starting fails, make sure to close the group and cleanup
            with contextlib.suppress(Exception):
                await tg.__aexit__(None, None, None)
            await self._safe_disconnect(connection_id)
            raise

    async def _receive_loop(self, connection_id: str) -> None:
        """Background task: receive, route/process, cleanup on disconnect."""
        ws = self._connections.get(connection_id)
        if not ws:
            return

        try:
            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Receive failed for %s: %s", connection_id, e)
                    break

                try:
                    msg = json.loads(raw)
                except Exception:
                    await self._send_safe(ws, {"type": "error", "success": False, "error": "Invalid JSON"})
                    continue

                await self.handle_message(connection_id, msg)
        finally:
            self.disconnect(connection_id)

    def disconnect(self, connection_id: str) -> None:
        """Remove connection and mappings; cancel receive task if running."""
        to_remove = [rid for rid, cid in self._request_to_connection.items() if cid == connection_id]
        for rid in to_remove:
            self._request_to_connection.pop(rid, None)

        # Cancel receive task group
        tg = self._receive_tasks.pop(connection_id, None)
        if tg is not None:
            # Schedule async exit; if called from sync context, let the event loop run it later
            async def _close_group():
                with contextlib.suppress(Exception):
                    await tg.__aexit__(None, None, None)
            try:
                # Try to run within current loop if exists; else spawn a new loop briefly
                anyio.from_thread.run(_close_group)
            except RuntimeError:
                # Not in a thread managed by anyio; try best-effort with asyncio
                try:
                    asyncio.get_running_loop().create_task(_close_group())
                except RuntimeError:
                    # As last resort, run a temporary loop
                    asyncio.run(_close_group())

        self._connections.pop(connection_id, None)
        logger.info("WebSocket connection closed: %s", connection_id)

    async def _safe_disconnect(self, connection_id: str) -> None:
        ws = self._connections.get(connection_id)
        if ws:
            with contextlib.suppress(Exception):
                await ws.close()
        self.disconnect(connection_id)

    async def _send_safe(self, ws: WebSocket, payload: Dict[str, Any]) -> bool:
        try:
            await ws.send_text(json.dumps(payload))
            return True
        except Exception:
            return False

    def register_request(self, request_id: str, connection_id: str) -> None:
        self._request_to_connection[request_id] = connection_id

    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        ws = self._connections.get(connection_id)
        if not ws:
            return

        mtype = message.get("type")
        if mtype == "register_request":
            rid = message.get("request_id")
            if not rid:
                await self._send_safe(ws, {"type": "error", "success": False, "error": "Missing request_id"})
                return
            self.register_request(rid, connection_id)
            await self._send_safe(ws, {"type": "request_registered", "request_id": rid, "success": True})
            return

        await self._send_safe(ws, {"type": "error", "success": False, "error": f"Unsupported message type: {mtype}"})

    async def send_message(self, connection_id: str, payload: Dict[str, Any]) -> bool:
        ws = self._connections.get(connection_id)
        if not ws:
            return False
        try:
            await ws.send_text(json.dumps(payload))
            return True
        except Exception:
            self.disconnect(connection_id)
            return False

    async def send_message_all(self, payload: Dict[str, Any]) -> None:
        for cid in list(self._connections.keys()):
            await self.send_message(cid, payload)

    async def send_response(self, request_id: str, payload: Dict[str, Any]) -> bool:
        cid = self._request_to_connection.get(request_id)
        if not cid:
            return False
        msg = dict(payload)
        msg.setdefault("request_id", request_id)
        ok = await self.send_message(cid, msg)
        if ok and msg.get("type") in {"task_completed", "task_failed", "task_cancelled"}:
            self._request_to_connection.pop(request_id, None)
        return ok

    def stop(self) -> None:
        """Shutdown: cancel receive tasks and close sockets."""
        self._stopping = True

        # Cancel/exit task groups
        for cid, tg in list(self._receive_tasks.items()):
            async def _close_group(group: anyio.abc.TaskGroup):
                with contextlib.suppress(Exception):
                    await group.__aexit__(None, None, None)
            try:
                anyio.from_thread.run(_close_group, tg)
            except RuntimeError:
                try:
                    asyncio.get_running_loop().create_task(_close_group(tg))
                except RuntimeError:
                    asyncio.run(_close_group(tg))
        self._receive_tasks.clear()

        # Remove all connections bookkeeping
        for cid in list(self._connections.keys()):
            self.disconnect(cid)