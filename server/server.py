import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import neo4j
from fastapi import Body, FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi import Request
from pathlib import Path
from neo4j import GraphDatabase

from client.client_defines import ProjectCreate, ProjectUpdate, SyncRequest
from server.graph_rag_manager import GraphRagManagerImpl
from server.server_defines import Project, TaskManager, GraphRagManager
from server.task_manager import TaskManagerImpl
from server.websocket_manager import WebSocketManagerImpl

# Main logging configuration
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

# Module logger
logger = logging.getLogger(__name__)

db_credentials: Dict[str, Any] = {
    'neo4j_url': os.environ.get('NEO4J_URL', 'bolt://localhost:7687'),
    'username': os.environ.get('NEO4j_USERNAME', 'neo4j'),
    'password': os.environ.get('NEO4J_PASSWORD', 'your_neo4j_password'),
    'database': os.environ.get('NEO4J_DB_NAME', 'graph-rag')
}

db_driver: neo4j.Driver
task_manager: TaskManager
graph_rag_manager: GraphRagManager
TEMPLATES_DIR = Path(__file__).parent / "templates"
DASHBOARD_FILE = TEMPLATES_DIR / "dashboard.html"

app = FastAPI(title="Graph RAG Server", version="0.1.0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_driver, task_manager, graph_rag_manager
    # Startup
    logger.info("Starting up FastAPI application")
    uri = db_credentials.get('neo4j_url')
    user = db_credentials.get('username')
    pwd = db_credentials.get('password')

    db_driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with db_driver.session(database=db_credentials.get('database')) as session:
            session.run("RETURN 1")
        logger.info("Connected to Neo4j at %s", uri)
    except Exception as e:
        logger.exception("Failed to connect to Neo4j: %s", e)
        # Ensure driver is closed on failure
        try:
            db_driver.close()
        except Exception:
            pass
        raise

    # Initialize managers
    task_manager = TaskManagerImpl(db_driver, WebSocketManagerImpl.websocket_notifier)
    graph_rag_manager = GraphRagManagerImpl(db_driver, task_manager)
    websocket_manager = WebSocketManagerImpl()
    logger.info("Managers initialized")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application")
    try:
        if db_driver:
            db_driver.close()
            logger.info("Neo4j driver closed")
    except Exception as e:
        logger.exception("Error closing Neo4j driver: %s", e)

@app.get("/", response_class=HTMLResponse)
async def dashboard(_: Request):
    try:
        html = DASHBOARD_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        html = "<html><body><h3>Dashboard not found</h3><p>Expected at: {}</p></body></html>".format(DASHBOARD_FILE)
    return HTMLResponse(content=html)

@app.websocket("/ws/{connection_id}")
async def websocket_route(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for client connections."""
    await WebSocketManagerImpl.websocket_endpoint(websocket, connection_id)

@app.websocket("/ws")
async def websocket_route_auto(websocket: WebSocket):
    """WebSocket endpoint with auto-generated connection ID."""
    await WebSocketManagerImpl.websocket_endpoint(websocket)

@app.post("/query")
async def create_query(project_id: str, query: str = Body(..., embed=True),
                       args: Dict[str, str] = Body(..., embed=True)) -> Dict[str, Any]:
    """Legacy REST endpoint for queries. Consider using WebSocket instead."""
    request_id = str(uuid.uuid4())
    return graph_rag_manager.query(project_id, request_id, query, args)

@app.post("/api/projects", response_model=Dict[str, Any])
def create_project(p: ProjectCreate):
    try:
        proj = Project(p.name, p.source_roots, p.args)
        graph_rag_manager.create_project(proj)
        return {"ok": True, "project_id": proj.project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects", response_model=List[Dict[str, Any]])
def list_projects():
    try:
        items = graph_rag_manager.list_projects()
        return [ {"project_id": pr.project_id, "name": pr.name, "source_roots": pr.source_roots} for pr in items ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}", response_model=Dict[str, Any])
def get_project(project_id: str):
    pr = graph_rag_manager.get_project(project_id)
    if not pr:
        raise HTTPException(status_code=404, detail="Not found")
    return {"project_id": pr.project_id, "name": pr.name, "source_roots": pr.source_roots}

@app.patch("/api/projects/{project_id}", response_model=Dict[str, Any])
def update_project(project_id: str, upd: ProjectUpdate):
    try:
        res = graph_rag_manager.update_project(project_id, name=upd.name, source_roots=upd.source_roots, args=upd.args)
        return {"ok": True, "project": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}", response_model=Dict[str, Any])
def delete_project(project_id: str):
    try:
        res = graph_rag_manager.delete_project(project_id)
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/sync", response_model=Dict[str, Any])
def sync_project(project_id: str, req: SyncRequest):
    try:
        res = graph_rag_manager.sync_project(project_id, force_all=req.force)
        return {"ok": True, "project_id": project_id, "sync": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/scheduled", response_model=Dict[str, Any])
def list_scheduled():
    try:
        ops = task_manager.list_active_tasks()
        return {"ok": True, "operations": ops}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Optional: run with `python -m server.server`
if __name__ == "__main__":
    try:
        import uvicorn  # type: ignore

        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    except ModuleNotFoundError:
        # Allow running without uvicorn installed, but inform the user how to start.
        print(
            "Uvicorn is not installed. Install with `pip install uvicorn[standard]` and run: "
            "python server.py"
        )

##############################################################
# Python
import asyncio
import json
import uuid
from typing import Dict, Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

# In-memory connection storage
connections: Dict[str, WebSocket] = {}
request_to_connection: Dict[str, str] = {}

@app.get("/")
async def index():
    return HTMLResponse("<html><body><h3>WS server running</h3></body></html>")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    await websocket.accept()
    connections[connection_id] = websocket
    try:
        # Inform client of connection_id
        await websocket.send_text(json.dumps({
            "type": "ws_connected",
            "connection_id": connection_id,
            "success": True
        }))

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            mtype = msg.get("type")
            if mtype == "register_request":
                req_id = msg.get("request_id")
                if not req_id:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "missing request_id"
                    }))
                    continue
                request_to_connection[req_id] = connection_id
                await websocket.send_text(json.dumps({
                    "type": "request_registered",
                    "request_id": req_id,
                    "success": True
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": f"unsupported message type: {mtype}"
                }))
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup on disconnect
        connections.pop(connection_id, None)
        for rid, cid in list(request_to_connection.items()):
            if cid == connection_id:
                request_to_connection.pop(rid, None)

async def notify(request_id: str, payload: Dict[str, Any]) -> None:
    """Send a message addressed by request_id to its mapped connection."""
    connection_id = request_to_connection.get(request_id)
    if not connection_id:
        return
    ws = connections.get(connection_id)
    if not ws:
        return
    payload = dict(payload)
    payload["request_id"] = request_id
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        # If sending fails, drop the connection mapping
        connections.pop(connection_id, None)
        for rid, cid in list(request_to_connection.items()):
            if cid == connection_id:
                request_to_connection.pop(rid, None)

# Demo task flow: enqueue -> started -> completed
@app.post("/demo/start")
async def demo_start():
    request_id = str(uuid.uuid4())

    async def run_task():
        await notify(request_id, {"type": "task_enqueued", "success": True})
        await asyncio.sleep(0.5)
        await notify(request_id, {"type": "task_started", "success": True})
        await asyncio.sleep(1.0)
        await notify(request_id, {
            "type": "task_completed",
            "success": True,
            "result": {"answer": 42}
        })

    # Fire-and-forget
    asyncio.create_task(run_task())
    # Return immediate ACK
    return {"request_id": request_id, "ok": True}