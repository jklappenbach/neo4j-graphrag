# ... existing code ...
import json
from typing import Any, Dict, List, Optional

import httpx
import websockets
import asyncio

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000/ws"

class GraphRagClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, ws_url: str = DEFAULT_WS_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ----------------------
    # Projects CRUD
    # ----------------------
    def create_project(self, name: str, source_roots: List[str], args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"name": name, "source_roots": source_roots, "args": args or {}}
        r = self._client.post("/api/projects", json=payload)
        r.raise_for_status()
        return r.json()

    def list_projects(self) -> List[Dict[str, Any]]:
        r = self._client.get("/api/projects")
        r.raise_for_status()
        return r.json()

    def get_project(self, project_id: str) -> Dict[str, Any]:
        r = self._client.get(f"/api/projects/{project_id}")
        r.raise_for_status()
        return r.json()

    def update_project(self, project_id: str, name: Optional[str] = None,
                       source_roots: Optional[List[str]] = None,
                       args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if source_roots is not None:
            payload["source_roots"] = source_roots
        if args is not None:
            payload["args"] = args
        r = self._client.patch(f"/api/projects/{project_id}", json=payload)
        r.raise_for_status()
        return r.json()

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        r = self._client.delete(f"/api/projects/{project_id}")
        r.raise_for_status()
        return r.json()

    # ----------------------
    # Sync API
    # ----------------------
    def sync_project(self, project_id: str, force: bool = False) -> Dict[str, Any]:
        r = self._client.post(f"/api/projects/{project_id}/sync", json={"force": force})
        r.raise_for_status()
        return r.json()

    # ----------------------
    # Query API
    # ----------------------
    def start_query(self, project_id: str, query: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Starts a query. Expects server to enqueue and return { ok, request_id, ... }.
        """
        payload = {"query": query, "args": args or {}}
        r = self._client.post(f"/api/projects/{project_id}/query", json=payload)
        r.raise_for_status()
        return r.json()

    def cancel_query(self, request_id: str) -> Dict[str, Any]:
        """
        Cancels a query by request_id.
        """
        r = self._client.post(f"/api/queries/{request_id}/cancel")
        r.raise_for_status()
        return r.json()

    def get_query_result(self, request_id: str) -> Dict[str, Any]:
        """
        Polls for a query result by request_id, if your server exposes such endpoint.
        """
        r = self._client.get(f"/api/queries/{request_id}")
        r.raise_for_status()
        return r.json()

    # ----------------------
    # Tasks API
    # ----------------------
    def list_scheduled_operations(self) -> Dict[str, Any]:
        r = self._client.get("/api/tasks/scheduled")
        r.raise_for_status()
        return r.json()

    # ----------------------
    # WebSocket
    # ----------------------
    async def listen_ws(self, on_message, path: Optional[str] = None):
        """
        Connect to WebSocket and receive messages.
        on_message: async or sync callable that receives parsed message (dict or str)
        path: optional suffix path to append to ws_url, e.g. "/notifications"
        """
        url = self.ws_url if not path else self.ws_url.rstrip("/") + path
        async with websockets.connect(url) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    data = msg
                if asyncio.iscoroutinefunction(on_message):
                    await on_message(data)
                else:
                    on_message(data)

# Convenience run example
async def _print_ws(msg):
    print("WS:", msg)

def example_usage():
    client = GraphRagClient()
    # REST calls
    projects = client.list_projects()
    print("Projects:", projects)
    # WebSocket
    asyncio.run(client.listen_ws(_print_ws))

if __name__ == "__main__":
    example_usage()