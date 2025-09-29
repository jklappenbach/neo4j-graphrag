import logging
import uuid
from datetime import datetime
from typing import Any, Dict
from fastapi import Body, FastAPI, WebSocket

# Change relative imports to absolute imports
from server.graph_rag_manager import GraphRagManager
from web_socket_manager import init_websocket_manager, websocket_endpoint

# Main logging configuration
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

# Module logger
logger = logging.getLogger(__name__)

# Initialize GraphRagManager
graph_rag_manager = GraphRagManager("./")

# Initialize WebSocket manager
ws_manager = init_websocket_manager(graph_rag_manager)

app = FastAPI(title="Graph RAG Server", version="0.1.0")

@app.websocket("/ws/{connection_id}")
async def websocket_route(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for client connections.

    Returns a JSON with status and current server time.
    """
    request_id = uuid.uuid4()
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.websocket("/ws")
async def websocket_route_auto(websocket: WebSocket):
    """WebSocket endpoint with auto-generated connection ID."""
    await websocket_endpoint(websocket)

@app.post("/query")
async def query(query: str = Body(..., embed=True)) -> Dict[str, Any]:
    """Legacy REST endpoint for queries. Consider using WebSocket instead."""
    request_id = str(uuid.uuid4())
    return graph_rag_manager.query(request_id, query)

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
