# Graph RAG - Code-Aware Retrieval-Augmented Generation

A graph-based Retrieval-Augmented Generation (RAG) system that provides intelligent question-answering capabilities for codebases. This system combines semantic search with graph-aware retrieval to understand code relationships, imports, function calls, and dependencies.

## What It Does

Graph RAG analyzes your codebase and creates a knowledge graph that captures:

- **Code Structure**: Functions, classes, and modules
- **Relationships**: Import dependencies, function calls, and code relationships  
- **Semantic Understanding**: Vector embeddings for intelligent content retrieval
- **Real-time Updates**: Automatically monitors file changes and updates the knowledge graph

When you ask questions about your code, the system:

1. Uses semantic search to find relevant code chunks
2. Leverages graph relationships to discover connected code
3. Generates contextual answers using a local LLM (Ollama)
4. Provides real-time status updates via WebSocket connections

## Supported File Types

- Python (.py)
- JavaScript (.js)
- CSS (.css)
- HTML (.html)
- JSON (.json)

## Prerequisites

### 1. Neo4j Database

Install and run Neo4j:

```bash
# Using Docker (recommended)
docker run
--name neo4j
-p7474:7474 -p7687:7687
-d
-v HOME/neo4j/data:/data \ -vHOME/neo4j/logs:/logs
-v HOME/neo4j/import:/var/lib/neo4j/import \ -vHOME/neo4j/plugins:/plugins
--env NEO4J_AUTH=neo4j/your_neo4j_password
neo4j:latest
``` 

Or download from [Neo4j Download Center](https://neo4j.com/download/)

**Important**: Update the password in `server/graph_rag_manager.py`

### 2. Ollama (Local LLM)

Install Ollama and required models:
```bash
Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
Pull required models
ollama pull llama3.1 # For text generation ollama pull nomic-embed-text # For embeddings``` 
```
### 3. Python Dependencies

Install Python dependencies:


```bash
# Create virtual environment
python -m venv .venv source .venv/bin/activate # On Windows: .venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt``` 
````
## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd graph-rag
   ```

2. **Start Neo4j** (see prerequisites above)

3. **Start Ollama** and pull models:
   ```bash
   ollama serve
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

4. **Configure the document root**: Update the path in `server/server.py`:
   ```python
   graph_rag_manager = GraphRagManager("./your-codebase-path")
   ```

5. **Start the server**:
   ```bash
   python -m server.server
   ```

The server will:
- Connect to Neo4j and Ollama
- Start ingesting files from the document root
- Begin monitoring for file changes
- Start the WebSocket server on `ws://localhost:8000/ws`

## Usage

### Simple Query (Recommended)
```python 
import asyncio from client.client import simple_query
async def main(): result = await simple_query("What does this code do?") if result["success"]: print(result["result"]) else: print(f"Error: {result['error']}")
asyncio.run(main())
``` 

### Advanced Usage with Status Callbacks
```python
import asyncio from client.client import GraphRagClient
async def main(): client = GraphRagClient() await client.connect()
def on_query_update(message):
    if message.get("type") == "task_started":
        print("🔄 Processing your query...")
    elif message.get("type") == "task_completed":
        print("✅ Query completed!")
        result = message.get("result", {})
        print(f"Answer: {result}")
    elif message.get("type") == "task_failed":
        print(f"❌ Query failed: {message.get('error')}")

request_id = await client.query("Explain this codebase", on_query_update)
print(f"Submitted query with ID: {request_id}")

# Keep connection alive to receive updates
await asyncio.sleep(30)
await client.disconnect()
asyncio.run(main())
``` 

### Other Operations

```python
# List all indexed documents
docs = await client.list_documents() print(f"Found {len(docs)} documents")
# Refresh the document index
def on_refresh_update(message): if message.get("type") == "task_completed": print("✅ Document refresh completed!")
refresh_id = await client.refresh_documents(on_refresh_update)
# Cancel a running query
await client.cancel_query(request_id)
``` 

## WebSocket API

The server exposes a WebSocket interface at `ws://localhost:8000/ws` with the following message types:

### Client → Server

- **Query**: `{"type": "query", "query": "your question", "request_id": "uuid"}`
- **Cancel**: `{"type": "cancel", "target_request_id": "uuid"}`
- **Refresh**: `{"type": "refresh", "request_id": "uuid"}`
- **List Documents**: `{"type": "list_documents", "request_id": "uuid"}`

### Server → Client

- **Task Started**: `{"type": "task_started", "request_id": "uuid", "task_info": {...}}`
- **Task Completed**: `{"type": "task_completed", "request_id": "uuid", "result": {...}}`
- **Task Failed**: `{"type": "task_failed", "request_id": "uuid", "error": "message"}`

## Architecture

- **Server** (`server/`): FastAPI WebSocket server with task management
- **Client** (`client/`): WebSocket client with async support
- **Graph RAG Manager**: Core logic for document ingestion and querying
- **Task Manager**: Handles async task execution with status tracking
- **Neo4j**: Graph database for code relationships and vector storage
- **Ollama**: Local LLM for embeddings and text generation

## File Watching

The system automatically monitors your codebase for changes and updates the knowledge graph in real-time. Supported operations:

- **File Added**: Automatically ingests new files
- **File Modified**: Re-processes and updates the graph
- **File Deleted**: Removes from the knowledge graph
- **Directory Changes**: Recursively handles directory operations

## Logging

The system provides detailed logging for debugging and monitoring:

```
bash
Enable debug logging
export PYTHONPATH=. python -c " import logging logging.basicConfig(level=logging.DEBUG)
Your code here
``` 

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**: Check that Neo4j is running and credentials are correct
2. **Ollama Models Not Found**: Run `ollama pull llama3.1` and `ollama pull nomic-embed-text`
3. **WebSocket Connection Failed**: Ensure the server is running on port 8000
4. **No Documents Found**: Check that the document root path contains supported file types

### Health Check

Test server connectivity:

```bash
curl http://localhost:8000/health
```
Neo4j Browser
Access the Neo4j browser at http://localhost:7474 to inspect the knowledge graph directly.

License
[MIT License] 

This README provides comprehensive documentation covering:

1. **Clear project description** - What the system does and its capabilities
2. **Prerequisites** - Neo4j, Ollama, and Python dependencies with installation instructions
3. **Step-by-step setup** - Complete installation and configuration guide
4. **Usage examples** - Both simple and advanced usage patterns as requested
5. **WebSocket API documentation** - Complete message format reference
6. **Architecture overview** - High-level system components
7. **Troubleshooting section** - Common issues and solutions
8. **Health checks** - How to verify the system is working

The README is structured to get users up and running quickly while providing enough detail for advanced usage scenarios.

# neo4j-graphrag
A client-server solution for GraphRAG using neo4j, specifically tailored for local LLM coding assistance.
