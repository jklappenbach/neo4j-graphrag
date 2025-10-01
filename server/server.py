import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List
from fastapi import Body, FastAPI, WebSocket

import neo4j
from fastapi import Body, FastAPI, WebSocket, HTTPException
from neo4j import GraphDatabase

from client.client_defines import ProjectCreate, ProjectUpdate, SyncRequest
from server.graph_rag_manager import GraphRagManagerImpl
from server.server_defines import Project
from server.task_manager import TaskManagerImpl
from server.web_socket_manager import WebSocketManager

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

db_driver = GraphDatabase.driver(db_credentials.get('neo4j_url'),
            auth=(db_credentials.get('username'), db_credentials.get('password')))

task_manager = TaskManagerImpl(db_driver, WebSocketManager.websocket_notifier)
graph_rag_manager = GraphRagManagerImpl(db_driver, task_manager)

app = FastAPI(title="Graph RAG Server", version="0.1.0")

@app.websocket("/ws/{connection_id}")
async def websocket_route(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for client connections."""
    await WebSocketManager.websocket_endpoint(websocket, connection_id)

@app.websocket("/ws")
async def websocket_route_auto(websocket: WebSocket):
    """WebSocket endpoint with auto-generated connection ID."""
    await WebSocketManager.websocket_endpoint(websocket)

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
        res = graph_rag_manager.sync_project(project_id, force=req.force)
        return {"ok": True, "project_id": project_id, "sync": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/scheduled", response_model=Dict[str, Any])
def list_scheduled():
    try:
        ops = graph_rag_manager.task_mgr.list_active_tasks()
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
