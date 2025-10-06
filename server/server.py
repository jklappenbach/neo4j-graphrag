import contextlib
import logging
import os
import platform
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import neo4j
from fastapi import Body, FastAPI, WebSocket, HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse
from flask import abort
from neo4j import GraphDatabase

from client.client_defines import ProjectCreate, SyncRequest
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
websocket_manager: WebSocketManagerImpl
graph_rag_manager: GraphRagManager
TEMPLATES_DIR = Path(__file__).parent / "templates"
DASHBOARD_FILE = TEMPLATES_DIR / "dashboard.html"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_driver, task_manager, graph_rag_manager, websocket_manager
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
    websocket_manager = WebSocketManagerImpl()
    task_manager = TaskManagerImpl(db_driver, websocket_manager)
    graph_rag_manager = GraphRagManagerImpl(db_driver, task_manager)
    logger.info("Managers initialized")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application")
    try:
        if db_driver:
            db_driver.close()
            logger.info("Neo4j driver closed")
        task_manager.stop()
        websocket_manager.stop()
        graph_rag_manager.stop()
    except Exception as e:
        logger.exception("Error closing Neo4j driver: %s", e)

app = FastAPI(title="Graph RAG Server", version="0.1.0", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def dashboard(_: Request):
    try:
        html = DASHBOARD_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        html = "<html><body><h3>Dashboard not found</h3><p>Expected at: {}</p></body></html>".format(DASHBOARD_FILE)
    return HTMLResponse(content=html)

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    await websocket_manager.connect(websocket, connection_id)

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
        global graph_rag_manager
        graph_rag_manager.create_project(proj)
        return {"ok": True, "project_id": proj.project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects", response_model=List[Dict[str, Any]])
def list_projects():
    try:
        request_id = str(uuid.uuid4())
        global graph_rag_manager
        items = graph_rag_manager.list_projects(request_id)
        return [ {"project_id": pr.project_id, "name": pr.name, "source_roots": pr.source_roots} for pr in items ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}", response_model=Dict[str, Any])
def get_project(project_id: str):
    request_id = str(uuid.uuid4())
    global graph_rag_manager
    pr = graph_rag_manager.get_project(request_id, project_id)
    if not pr:
        raise HTTPException(status_code=404, detail="Not found")
    return {"project_id": pr.project_id, "name": pr.name, "source_roots": pr.source_roots}

@app.patch("/api/projects/{project_id}", response_model=Dict[str, Any])
def update_project(project_id: str, args: Dict[str, Any]):
    request_id = str(uuid.uuid4())
    try:
        global graph_rag_manager
        res = graph_rag_manager.update_project(request_id, project_id, args=args)
        return {"ok": True, "project": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}", response_model=Dict[str, Any])
def delete_project(project_id: str):
    request_id = str(uuid.uuid4())
    try:
        global graph_rag_manager
        res = graph_rag_manager.delete_project(request_id, project_id)
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/sync", response_model=Dict[str, Any])
def sync_project(project_id: str, req: SyncRequest):
    request_id = str(uuid.uuid4())
    try:
        global graph_rag_manager
        res = graph_rag_manager.handle_sync_project(request_id, project_id, force_all=req.force)
        return {"ok": True, "project_id": project_id, "sync": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/scheduled", response_model=Dict[str, Any])
def list_scheduled():
    request_id = str(uuid.uuid4())
    try:
        global task_manager
        ops = task_manager.list_active_tasks()
        return {"ok": True, "operations": ops}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Directory Picker routes and methods
def _normalize_path(p: str) -> str:
    """Resolve user and env, keep within filesystem, no write ops"""
    p = os.path.expanduser(os.path.expandvars(p or ""))
    if platform.system() == "Windows":
        # On Windows, empty or invalid path -> current drive root
        if not p:
            p = os.path.splitdrive(os.getcwd())[0] + "\\"
    else:
        if not p:
            p = "/"
    return os.path.abspath(p)

def _is_hidden(name: str) -> bool:
    """Check if filename should be hidden from listing"""
    return name.startswith(".") or name in ("System Volume Information", "$Recycle.Bin")

def _list_dir(path: str, include_files: bool, page: int, page_size: int):
    """List directory contents with pagination"""
    try:
        entries: List[Dict[str, str]] = []
        with os.scandir(path) as it:
            for e in it:
                # Skip broken symlinks and unreadable entries safely
                try:
                    is_dir = e.is_dir(follow_symlinks=False)
                    is_file = e.is_file(follow_symlinks=False)
                except OSError:
                    continue

                if not include_files and not is_dir:
                    continue

                # Skip hidden files/folders
                if _is_hidden(e.name):
                    continue

                # Collect minimal info
                entries.append({
                    "name": e.name,
                    "path": os.path.abspath(os.path.join(path, e.name)),
                    "type": "dir" if is_dir else ("file" if is_file else "other"),
                })
    except FileNotFoundError:
        abort(404, description="Path not found")
    except PermissionError:
        abort(403, description="Permission denied")

    # Sort: dirs first then files, case-insensitive by name
    entries.sort(key=lambda x: (0 if x["type"] == "dir" else 1, x["name"].lower()))

    # Pagination
    total = len(entries)
    start = max(0, (page - 1) * page_size)
    end = min(total, start + page_size)

    # Determine parent path
    abs_path = os.path.abspath(path)
    parent_path = os.path.abspath(os.path.join(abs_path, os.pardir))
    # Only include parent if we're not at root
    has_parent = abs_path != parent_path

    return {
        "path": abs_path,
        "entries": entries[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "parent": parent_path if has_parent else None,
        "separator": os.sep,
    }

def _list_roots() -> List[str]:
    """Get list of filesystem roots (drives on Windows, / on Unix)"""
    roots: List[str] = []
    sysname = platform.system()

    try:
        if sysname == "Windows":
            import string
            # Detect existing drive letters safely
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                # Use os.path.isdir to avoid odd errors on phantom drives
                with contextlib.suppress(Exception):
                    if os.path.isdir(drive):
                        roots.append(drive)
            # Fallback to current drive if none detected
            if not roots:
                drive, _ = os.path.splitdrive(os.getcwd())
                if drive:
                    roots.append(drive + "\\")
        else:
            # Always include root
            roots.append(os.sep)
            # Optionally include common mount points; guard all FS ops
            for candidate in ("/Volumes", "/mnt", "/media", "~"):
                if os.path.isdir(candidate):
                    with contextlib.suppress(Exception):
                        for name in os.listdir(candidate):
                            full = os.path.join(candidate, name)
                            if os.path.isdir(full):
                                roots.append(full)
    except Exception:
        # As a safety net, ensure at least one root is returned per platform
        if sysname == "Windows":
            drive, _ = os.path.splitdrive(os.getcwd())
            roots = [drive + os.sep] if drive else ["C:\\"]
        else:
            roots = ["/"]

    # Deduplicate, normalize, and sort robustly
    unique: List[str] = []
    seen = set()
    for r in roots:
        try:
            ar = os.path.abspath(r)
        except Exception:
            continue
        key = ar.lower() if sysname == "Windows" else ar
        if key not in seen:
            seen.add(key)
            unique.append(ar)

    # Avoid ValueError from .lower on non-str by ensuring all are str and sort safely
    try:
        unique.sort(key=lambda x: x.lower() if isinstance(x, str) else "")
    except Exception:
        unique.sort()

    return unique


@app.get("/api/fs/roots")
def api_fs_roots():
    """Get filesystem roots"""
    return {"roots": _list_roots()}

@app.get("/api/fs/list/")
def api_fs_list(path: str, include_files: bool, page: int = 1, page_size: int = 20):
    """List directory contents"""
    norm = _normalize_path(path)
    return _list_dir(norm, include_files, page, page_size)

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
