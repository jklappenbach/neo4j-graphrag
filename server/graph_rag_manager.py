import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import neo4j
from haystack import tracing
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline import Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack.tracing.logging_tracer import LoggingTracer
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from neo4j import GraphDatabase
from neo4j_haystack import Neo4jDocumentStore
from watchdog.observers import Observer

from server.code_aware_splitter import CodeAwareSplitter
from server.code_relationship_extractor import CodeRelationshipExtractor
from server.graph_document_expander import GraphAwareRetriever
from server.project_manager import ProjectManagerImpl
from server.server_defines import GraphRagManager, Project, TaskManager
from server.task_manager import RefreshTask, QueryTask, ListDocumentsTask

logging.getLogger("haystack").setLevel(logging.DEBUG)

# Module logger
logger = logging.getLogger(__name__)

# Enable content tracing for component inputs/outputs
tracing.tracer.is_content_tracing_enabled = True

# Activate the LoggingTracer
tracing.enable_tracing(LoggingTracer())

@dataclass
class _ChunkRecord:
    id: str
    path: str
    index: int
    text: str
    ext: str

class _ProjectEntry:
    project: Project
    observers: List[Observer]
    db_driver: neo4j.Driver
    document_store: Neo4jDocumentStore
    document_embedder: OllamaDocumentEmbedder
    text_converter: TextFileToDocument
    rel_extractor: CodeRelationshipExtractor
    retriever: GraphAwareRetriever
    doc_splitter: CodeAwareSplitter
    doc_writer: DocumentWriter
    ingestion_pipeline: Pipeline

    def __init__(self, project: Project) -> None:
        url = os.environ.get('NEO4J_URL', 'bolt://localhost:7687')
        username = os.environ.get('NEO4j_USERNAME', 'neo4j')
        password = os.environ.get('NEO4J_PASSWORD', 'your_neo4j_password')

        self.project = project
        # Initialize Document Store and Embedder [1.2.7, 1.5.3]
        self.document_store = Neo4jDocumentStore(
            url=url,
            username=username,
            password=password,  # Replace with your password
            database=self.project.name,
            embedding_dim=768  # Ensure this matches your Ollama model's dimension
        )

        self.document_embedder = OllamaDocumentEmbedder(
            model=self.project.embedder_model_name,
            url="http://localhost:11434"
        )

        # Define the ingestion pipeline
        self.text_converter = TextFileToDocument()
        self.rel_extractor = CodeRelationshipExtractor()
        self.doc_splitter = CodeAwareSplitter()
        self.doc_writer = DocumentWriter(document_store=self.document_store,
                                        policy=DuplicatePolicy.OVERWRITE)

        # Create the pipeline
        self.ingestion_pipeline = Pipeline()
        self.ingestion_pipeline.add_component("converter", self.text_converter)
        self.ingestion_pipeline.add_component("splitter", self.doc_splitter)
        self.ingestion_pipeline.add_component("extractor", self.rel_extractor)
        self.ingestion_pipeline.add_component("embedder", self.document_embedder)
        self.ingestion_pipeline.add_component("writer", self.doc_writer)

        # Link the components
        self.ingestion_pipeline.connect("converter.documents", "splitter.documents")
        self.ingestion_pipeline.connect("splitter.documents", "extractor.documents")
        self.ingestion_pipeline.connect("extractor.documents", "embedder.documents")
        self.ingestion_pipeline.connect("embedder.documents", "writer.documents")

class GraphRagManagerImpl(GraphRagManager):
    """
    Graph-based RAG manager using Haystack with code-aware ingestion.
    """

    def __init__(self, db_driver: neo4j.Driver, task_mgr: TaskManager) -> None:
        self._neo4j_url = os.environ.get('NEO4J_URL', 'bolt://localhost:7687')
        self._username = os.environ.get('NEO4j_USERNAME', 'neo4j')
        self._password = os.environ.get('NEO4J_PASSWORD', 'your_neo4j_password')
        database = os.environ.get('NEO4J_DB_NAME', 'graph-rag')

        self._db_driver = db_driver
        self._task_mgr = task_mgr
        self._project_mgr = ProjectManagerImpl(self._db_driver)
        self._project_entries: Dict[str, _ProjectEntry] = {}

    # ---------------------
    # Hash utilities and metadata
    # ---------------------
    def _compute_file_hash(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_tracked_files_with_hashes(self, project_id: str) -> Dict[str, str]:
        project_entry = self._project_entries.get(project_id)
        if not project_entry:
            raise ValueError(f"Project {project_id} not found")
        tracked: Dict[str, str] = {}
        with project_entry.db_driver.session(database=project_entry.project.name) as session:
            res = session.run("""
                MATCH (d:Document)
                WHERE d.file_path IS NOT NULL
                RETURN d.file_path AS file_path, d.content_hash AS content_hash
            """)
            for r in res:
                tracked[r["file_path"]] = r.get("content_hash") or ""
        return tracked

    def _store_file_metadata(self, project_id: str, file_path: Path, content_hash: str) -> None:
        project_entry = self._project_entries.get(project_id)
        if not project_entry:
            raise ValueError(f"Project {project_id} not found")
        stat = file_path.stat()
        with project_entry.db_driver.session(database=project_entry.project.name) as session:
            session.run("""
                MERGE (d:Document {file_path: $file_path})
                SET d.content_hash = $content_hash,
                    d.last_synced_at = timestamp(),
                    d.last_modified_at = $last_modified_at,
                    d.file_size = $file_size
            """, file_path=file_path.as_posix(),
                 content_hash=content_hash,
                 last_modified_at=int(stat.st_mtime * 1000),
                 file_size=stat.st_size)
    def set_websocket_notifier(self, websocket_notifier) -> None:
        self.websocket_notifier = websocket_notifier



    # ---------------------
    # Synchronization
    # ---------------------
    def sync_project(self, project_id: str, force: bool = False) -> Dict[str, Any]:
        logger.info("Starting sync project_id=%s force=%s", project_id, force)
        project_entry = self._project_entries.get(project_id)
        if not project_entry:
            raise ValueError(f"Project {project_id} not found")

        if force:
            self.handle_refresh_documents(project_id)
            return {"mode": "full", "ok": True}

        changes = {"added": [], "modified": [], "deleted": [], "unchanged": 0}
        tracked = self._get_tracked_files_with_hashes(project_id)

        current: Dict[str, str] = {}
        for root in project_entry.project.source_roots:
            root_path = Path(root).expanduser().resolve()
            if not root_path.exists():
                continue
            for fp in self._iter_code_files(root_path):
                try:
                    current[fp.as_posix()] = self._compute_file_hash(fp)
                except Exception as e:
                    logger.exception("Hash failed for %s, cause: %s", fp, str(e))

        for path_str, h in current.items():
            if path_str not in tracked:
                changes["added"].append(path_str)
            elif tracked[path_str] != h:
                changes["modified"].append(path_str)
            else:
                changes["unchanged"] += 1

        for path_str in tracked.keys():
            if path_str not in current:
                changes["deleted"].append(path_str)

        # Apply
        for d in changes["deleted"]:
            try:
                self.handle_delete_path(project_id, Path(d))
            except Exception:
                logger.exception("Delete during sync failed for %s", d)
        for a in changes["added"]:
            try:
                self.handle_add_path(project_id, Path(a))
            except Exception:
                logger.exception("Add during sync failed for %s", a)
        for m in changes["modified"]:
            try:
                self.handle_update_path(project_id, Path(m))
            except Exception:
                logger.exception("Update during sync failed for %s", m)

        return {"mode": "incremental", **{k: (len(v) if isinstance(v, list) else v) for k, v in changes.items()}}

    # ---------------------
    # Projects API
    # ---------------------
    def create_project(self, project: Project) -> None:
        try:
            if self._project_entries.get(project.project_id) is not None:
                raise Exception('Project already exists')
            project_entry = _ProjectEntry(project)
            self._project_entries[project.project_id] = project_entry
            project_entry.db_driver = GraphDatabase.driver(self._neo4j_url, auth=(self._username, self._password))
            self._project_mgr.create_project(project)
            self._create_project_observers(project_entry)
            for path_str in project.source_roots:
                path = Path(path_str)
                self.handle_add_path(project.project_id, path)
        except Exception as e:
            logger.exception("Failed to create project %s: %s", project.project_id, str(e))

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        try:
            entry = self._project_entries.get(project_id)
            if entry:
                for obs in entry.observers or []:
                    try:
                        obs.stop()
                        obs.join(timeout=1)
                    except Exception:
                        pass
                self._project_entries.pop(project_id, None)
            return self._project_mgr.delete_project(project_id)
        except Exception as e:
            logger.exception("Failed to delete project %s: %s", project_id, str(e))
            raise

    def update_project(self, project_id: str, name: str | None = None, source_roots: List[str] | None = None, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            updated = self._project_mgr.update_project(project_id, name=name, source_roots=source_roots, args=args)
            # Reload entry in memory
            proj = self._project_mgr.get_project(project_id)
            if proj:
                entry = self._project_entries.get(project_id)
                if entry is None:
                    entry = _ProjectEntry(proj)
                    entry.db_driver = GraphDatabase.driver(self._neo4j_url, auth=(self._username, self._password))
                    self._project_entries[project_id] = entry
                else:
                    entry.project = proj
                # Recreate observers for new roots
                for obs in entry.observers or []:
                    try:
                        obs.stop(); obs.join(timeout=1)
                    except Exception as e:
                        logger.warning(f"Failed to stop observer for project {project_id}: {e}")
                entry.observers = []
                # Sync after update
                self.sync_project(project_id, force=False)
            return updated
        except Exception as e:
            logger.exception("Failed to update project %s: %s", project_id, str(e))
            raise

    def list_projects(self) -> List[Project]:
        return self._project_mgr.list_projects()

    def get_project(self, project_id: str) -> Project | None:
        return self._project_mgr.get_project(project_id)

    def _create_project_observers(self, project_entry: _ProjectEntry) -> None:
        if len(project_entry.observers) > 0:
            for observer in project_entry.observers:
                observer.stop()
                observer.join()

        for path_str in project_entry.project.source_roots:
            path = Path(path_str)
            observer = Observer()
            observer.schedule(self, path.as_posix(), recursive=True)
            observer.daemon = True
            observer.start()
            project_entry.observers.append(observer)
            logger.info("Filesystem watcher started on %s", path.as_posix())

    def _load_and_activate_all_projects(self) -> None:
        for project in self._project_mgr.list_projects():
            try:
                project_entry = _ProjectEntry(project)
                self._project_entries[project.project_id] = project_entry
                project_entry.db_driver = GraphDatabase.driver(self._neo4j_url, auth=(self._username, self._password))
                self._project_entries[project.project_id] = project_entry
                # Perform incremental sync at startup
                self.sync_project(project.project_id, force=False)
            except Exception as e:
                logger.exception("Startup sync failed for %s, cause: %s", project.project_id, str(e))

    # ---------------------
    # Query API
    # ---------------------
    def query(self, request_id: str, project_id: str, query_str: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._task_mgr.add_task(
                QueryTask(project_id, request_id, query_str, args, self.handle_query, self._task_mgr))
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to query %s, request_id: %s", query_str, request_id)
            return {"ok": False, "error": str(e)}

    def handle_query(self, request_id: str, project_id: str, query_str: str, args: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Retrieve graph-aware relevant chunks from Neo4j via Haystack and answer with a local model.
        The retrieval leverages relationships (IMPORTS, CALLS, DEFINES) using a Cypher-aware retriever.
        """
        logger.info("Entering GraphRagManager.handle_query request_id=%s query=%s", request_id, query_str)
        project_entry = self._project_entries.get(project_id)
        if not project_entry:
            raise Exception('Invalid project ID provided.  Project does not exist.')

        try:
            # Embed the query
            q_embedder = OllamaTextEmbedder(
                model=project_entry.project.embedder_model_name,
                url="http://localhost:11434"
            )

            # Use custom graph-aware retriever with the project-specific driver
            retriever = GraphAwareRetriever(
                document_store=project_entry.document_store,
                driver=project_entry.db_driver,
                top_k=15,
                relationship_weight=0.3
            )

            # Local generator
            generator = OllamaGenerator(model=project_entry.project.llm_model_name, url="http://localhost:11434")

            # Build pipeline: embed -> retrieve
            pipeline = Pipeline()
            pipeline.add_component("q_embedder", q_embedder)
            pipeline.add_component("retriever", retriever)
            pipeline.connect("q_embedder.embedding", "retriever.query_embedding")

            # Correct v2-style run: feed data into the q_embedder
            ret = pipeline.run(data={"q_embedder": {"text": query_str}})

            # Safely extract documents
            retr_out = ret.get("retriever") or {}
            docs = retr_out.get("documents") or []

            if not docs:
                logger.info("No documents retrieved for request_id=%s", request_id)
                return {
                    "ok": True,
                    "request_id": request_id,
                    "result": {
                        "answer": "No relevant documents were found for your query.",
                        "sources": [],
                        "context": ""
                    }
                }

            context = "\n\n".join(getattr(d, "content", "") for d in docs if getattr(d, "content", ""))

            prompt = f"""You are a code-aware assistant. Use the graph-related context to answer.
                Context: {context}

                Question: {query_str}
                Answer succinctly:"""

            # Call generator directly instead of rerunning the whole pipeline
            gen_res = generator.run(prompt=prompt)
            # OllamaGenerator typically returns {"replies": [text, ...]}
            replies = gen_res.get("replies") or []
            answer = replies[0] if replies else ""

            sources = []
            for d in docs:
                fp = None
                # Try common places where file path may reside
                if hasattr(d, "meta") and isinstance(d.meta, dict):
                    fp = d.meta.get("file_path") or d.meta.get("name") or d.meta.get("source")
                if not fp and hasattr(d, "id"):
                    fp = str(d.id)
                if fp:
                    sources.append(fp)

            return {
                "ok": True,
                "request_id": request_id,
                "result": {
                    "answer": answer,
                    "sources": sources,
                    "context": context
                }
            }
        except Exception as e:
            logger.exception("Query operation failed request_id=%s: %s", request_id, str(e))
            raise e

    def cancel_query(self, request_id: str):
        try:
            self._task_mgr.cancel_task(request_id)
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to cancel request_id: %s", request_id)
            return {"ok": False, "error": str(e)}

    # ---------------------
    # Documents API
    # ---------------------

    def list_documents(self, request_id: str, project_id: str) -> Dict[str, Any]:
        logger.info("Entering GraphRagManager.list_documents, request_id=%s", request_id)
        self._task_mgr.add_task(ListDocumentsTask(request_id, project_id, self.handle_list_documents, self._task_mgr))
        return {"ok": True, "request_id": request_id}

    def refresh_documents(self, request_id: str, project_id: str) -> Dict[str, Any]:
        logger.info("Entering GraphRagManager.clear_documents, request_id: %s", request_id)
        try:
            self._task_mgr.add_task(RefreshTask(request_id, project_id, self.handle_refresh_documents, self._task_mgr))
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to refresh documents, request_id: %s", request_id)
            return {"ok": False, "error": str(e)}

    def handle_add_path(self, project_id: str, path: Path) -> None:
        logger.info("Entering GraphRagManager.ingest_file file_path=%s", path)
        project_entry = self._project_entries.get(project_id)

        try:
            if not path.exists():
                return
            code_files: List[Path] = []
            if path.is_dir():
                code_files.extend(list(self._iter_code_files(path)))
            else:
                code_files.append(path)

            if len(code_files) == 0:
                return

            project_entry.ingestion_pipeline.run({"converter": {"sources": code_files}})
            self._create_doc_relationships()
            for fp in code_files:
                ch = self._compute_file_hash(fp)
                self._store_file_metadata(project_id, fp, ch)
            logger.info("Ingested %d files", len(code_files))
        except Exception as e:
            logger.exception("Ingestion failed for %s", path)
            raise e

    def handle_update_path(self, project_id: str, path: Path) -> None:
        try:
            self.handle_delete_path(project_id, path)
            self.handle_add_path(project_id, path)
        except Exception as e:
            logger.exception("Update failed for %s", path)
            raise e

    def handle_delete_path(self, project_id: str, path: Path) -> None:
        """
        If the path is a directory, delete every document under it from the store,
        along with all the relationships.  For each subdirectory, recursively call
        this method.
        """
        logger.info("Entering GraphRagManager.delete_path path: %s", path)
        project_entry = self._project_entries.get(project_id)

        try:
            target = Path(path).expanduser().resolve()
        except Exception as e:
            logger.exception("Invalid path provided to delete_path: %s", path, e)
            raise e

        if not target.exists():
            logger.warning("Path does not exist, nothing to delete: %s", target.as_posix())
            raise FileNotFoundError()

        # If directory: traverse and delete all supported files under it
        if target.is_dir():
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS or p.is_dir():
                    self.handle_delete_path(project_id, p)

        # If file: remove corresponding nodes/embeddings/relationships
        if target.is_file():
            file_path_str = target.as_posix()
            # Remove relationships before removing the node
            try:
                with project_entry.db_driver.session() as s:
                    # Delete Chunk nodes contained by the Document, then the Document itself.
                    s.run(
                        """
                        MATCH (d:Document {file_path: $path})
                        OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                        DETACH DELETE c, d
                        """,
                        {"path": file_path_str},
                    )
            except Exception as e:
                logger.exception("Failed deleting document graph for %s", file_path_str)
                raise e

            # Then remove the node itself
            try:
                project_entry.document_store.delete_documents([file_path_str])
            except Exception as e:
                logger.debug("Document store deletion attempt skipped/failed for %s", file_path_str)
                raise e

    def handle_list_documents(self, project_id: str) -> Dict[str, Any]:
        try:
            entry = self._project_entries.get(project_id)
            if not entry:
                raise ValueError(f"Project {project_id} not found")

            # Gather filesystem files from all source roots using _iter_code_files as in sync_project
            fs_files: List[str] = []
            for root in entry.project.source_roots:
                root_path = Path(root).expanduser().resolve()
                if not root_path.exists():
                    continue
                for fp in self._iter_code_files(root_path):
                    fs_files.append(fp.as_posix())

            # Gather document store files (by file_path if available, else by id/name/source)
            ds_files: List[str] = []
            try:
                with entry.db_driver.session(database=entry.project.name) as session:
                    res = session.run("""
                        MATCH (d:Document)
                        RETURN coalesce(d.file_path, d.name, d.source, d.id) AS path
                    """)
                    for r in res:
                        v = r.get("path")
                        if v:
                            ds_files.append(str(v))
            except Exception as e:
                logger.exception("Failed to fetch document store files for project %s, cause: %s", project_id, e)
                raise e
            return {
                "fileSystem": fs_files,
                "documentStore": ds_files
            }
        except Exception as e:
            logger.exception("Failed listing documents", e)
            raise e

    def handle_refresh_documents(self, project_id: str) -> None:
        logger.info("Refreshing all documents for project %s", project_id)
        entry = self._project_entries.get(project_id)
        if not entry:
            raise ValueError(f"Project {project_id} not found")
        for root in entry.project.source_roots:
            rp = Path(root).expanduser().resolve()
            if rp.exists():
                self.handle_delete_path(project_id, rp)
        for root in entry.project.source_roots:
            rp = Path(root).expanduser().resolve()
            if rp.exists():
                self.handle_add_path(project_id, rp)

    # ---------------------
    # Helpers
    # ---------------------
    def _iter_code_files(self, root: Path) -> Iterable[Path]:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS:
                yield p

    def _create_doc_relationships(self):
        with self._db_driver.session() as session:
            # Create IMPORTS relationships
            session.run("""
                MATCH (d:Document)
                WHERE d.imports IS NOT NULL
                UNWIND d.imports AS imported_module
                MATCH (d2:Document)
                WHERE d2.file_path CONTAINS(imported_module)
                MERGE (d)-[:IMPORTS]->(d2)
            """)

            # Create CALLS relationships
            session.run("""
                MATCH (d:Document)
                WHERE d.calls IS NOT NULL
                UNWIND d.calls AS called_function
                MATCH (d2:Document)
                WHERE d2.content CONTAINS 'def ' + called_function
                MERGE (d)-[:CALLS]->(d2)
            """)

    @property
    def task_mgr(self):
        return self._task_mgr
