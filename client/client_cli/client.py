# ... existing code ...
import json
import uuid
from typing import Any, Dict, List, Optional
import requests
import httpx
import websockets
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
import websocket

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000/ws"

@dataclass
class WsEvent:
    type: str
    payload: Dict[str, Any]

class AsyncClient:
    """
    HTTP ack + WebSocket notifications client.

    - call_* methods return {"request_id": "..."} immediately.
    - Events for that request_id arrive via WebSocket.
    """
    def __init__(self, base_url: str):
        self._session_id = uuid.uuid4()
        self.base_url = base_url.rstrip("/")
        self._ws_url = self.base_url.replace("http", "ws") + "/ws"
        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._should_run = False
        self._on_event: Optional[Callable[[WsEvent], None]] = None
        self._pending_register: "set[str]" = set()
        self._lock = threading.Lock()

    def start_ws(self, on_event: Optional[Callable[[WsEvent], None]] = None):
        if websocket is None:
            raise RuntimeError("websocket-client package is required for WS support")
        self._on_event = on_event
        if self._ws and self._ws_thread and self._ws_thread.is_alive():
            return
        self._should_run = True

        def _run():
            while self._should_run:
                try:
                    ws = websocket.WebSocketApp(
                        self._ws_url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_close=self._on_close,
                        on_error=self._on_error,
                    )
                    self._ws = ws
                    ws.run_forever()
                except Exception:
                    time.sleep(2)  # backoff
        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()

    def stop_ws(self):
        self._should_run = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._ws_thread:
            self._ws_thread.join(timeout=2)

    # ---- WS callbacks ----
    def _on_open(self, ws):
        # Register any pending request_ids queued before the socket connected
        with self._lock:
            pending = list(self._pending_register)
        for rid in pending:
            self._send_register(rid)

    def _on_message(self, ws, message: str):
        try:
            msg = json.loads(message)
        except Exception:
            return
        evt = WsEvent(type=msg.get("type") or "", payload=msg)
        if self._on_event:
            self._on_event(evt)

    def _on_close(self, ws, *args):
        pass

    def _on_error(self, ws, err):
        pass

    def _send_register(self, request_id: str):
        if not self._ws:
            return
        try:
            self._ws.send(json.dumps({"type": "register_request", "request_id": request_id}))
        except Exception:
            # if send fails, queue it to attempt on next connect
            with self._lock:
                self._pending_register.add(request_id)

    def _register_session(self, resp: requests.Response) -> Dict[str, Any]:
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        request_id = data.get("request_id") or data.get("requestId")
        if request_id:
            # attempt immediate register; if socket not ready, queue it
            if self._ws:
                self._send_register(request_id)
            else:
                with self._lock:
                    self._pending_register.add(request_id)
        return {"request_id": request_id}

    # ---- HTTP methods that return ACKs ----
    def create_project(self, name: str, source_roots: list[str], args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"name": name, "source_roots": source_roots, "args": args or {}}
        r = requests.post(f"{self.base_url}/api/projects", json=payload, timeout=60)
        return self._register_session(r)

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.patch(f"{self.base_url}/api/projects/{project_id}", json=updates, timeout=60)
        return self._register_session(r)

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        r = requests.delete(f"{self.base_url}/api/projects/{project_id}", timeout=60)
        return self._register_session(r)

    def list_documents(self, project_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/projects/{project_id}/documents", timeout=60)
        return self._register_session(r)

    def refresh_documents(self, project_id: str) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/api/projects/{project_id}/documents:refresh", timeout=60)
        return self._register_session(r)

    def query(self, project_id: str, query_str: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"query": query_str, "args": args or {}}
        r = requests.post(f"{self.base_url}/api/projects/{project_id}/query", json=payload, timeout=300)
        return self._register_session(r)

    def cancel_query(self, request_id: str) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/api/tasks/{request_id}:cancel", timeout=30)
        return self._register_session(r)

def example_usage():
    """
    Demonstrates basic usage:
      - start WS and event handler
      - create a project (ACK only)
      - run a query (ACK only)
      - handle async task_* updates via WS
    """
    events: List[dict] = []

    def on_event(evt):
        # evt.type is like 'task_enqueued'|'task_started'|'task_completed'|'task_failed'
        # evt.payload contains full message, including 'request_id'
        print(f"[WS EVENT] {evt.type}: {evt.payload}")
        events.append(evt.payload)

    client = AsyncClient(base_url="http://localhost:8000")
    client.start_ws(on_event=on_event)

    # Create a project (returns ACK)
    ack1 = client.create_project(
        name="My Project",
        source_roots=["/path/to/src"],
        args={"embedder_model_name": "default"}
    )
    print("Create project ACK:", ack1)

    # Run a query (returns ACK)
    ack2 = client.query(
        project_id="your-project-id",
        query_str="How is the code organized?",
        args={"temperature": 0.2}
    )
    print("Query ACK:", ack2)

    # Let events arrive for a bit (in a real app, integrate with your event loop/UI)
    time.sleep(5)

    # Optionally, cancel a query using its request_id
    if ack2.get("request_id"):
        cancel_ack = client.cancel_query(ack2["request_id"])
        print("Cancel query ACK:", cancel_ack)

    # Shutdown WS when done
    client.stop_ws()

    return {
        "create_project_request_id": ack1.get("request_id"),
        "query_request_id": ack2.get("request_id"),
        "received_events_count": len(events),
    }
if __name__ == "__main__":
    example_usage()