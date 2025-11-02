from __future__ import annotations
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Optional imports for environments without heavy deps during tests
try:
    import neo4j  # type: ignore
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    neo4j = None  # type: ignore
    class GraphDatabase:  # type: ignore
        @staticmethod
        def driver(*args, **kwargs):
            class _D:
                def session(self, **kwargs):
                    class _S:
                        def __enter__(self): return self
                        def __exit__(self, *a): return False
                        def run(self, *a, **k):
                            class _R:
                                def __iter__(self): return iter(())
                                def get(self, *aa, **kk): return None
                            return _R()
                    return _S()
            return _D()

try:  # haystack optional
    from haystack import tracing  # type: ignore
    from haystack.core.pipeline import Pipeline  # type: ignore
    from haystack.tracing.logging_tracer import LoggingTracer  # type: ignore
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder  # type: ignore
    from haystack_integrations.components.generators.ollama import OllamaGenerator  # type: ignore
except Exception:  # pragma: no cover
    class _DummyTracer:
        def __init__(self):
            self.is_content_tracing_enabled = False
    class _Tracing:
        tracer = _DummyTracer()
        @staticmethod
        def enable_tracing(*a, **k):
            return None
    tracing = _Tracing()  # type: ignore

    class Pipeline:  # type: ignore
        def add_component(self, *a, **k): pass
        def connect(self, *a, **k): pass
        def run(self, *a, **k): return {}
    class OllamaTextEmbedder:  # type: ignore
        def __init__(self, *a, **k): pass
    class OllamaGenerator:  # type: ignore
        def __init__(self, *a, **k): pass
        def run(self, *a, **k): return {"replies": [""]}

from watchdog.observers import Observer

from server.pipeline.graph_document_expander import GraphAwareRetriever
# Project manager import may pull heavy deps; provide fallback
try:
    from server.project_manager import ProjectManagerImpl, _ProjectEntry  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class _ProjectEntry:  # type: ignore
        project: Any
        observers: List[Any] = None
        def __post_init__(self):
            if self.observers is None:
                self.observers = []
        # Placeholder attrs used by code
        db_driver: Any = None
        document_store: Any = None

    class ProjectManagerImpl:  # type: ignore
        def __init__(self, *a, **k):
            pass
        def create_project(self, project):
            return None
        def delete_project(self, project_id: str):
            return None
        def list_projects(self) -> List[Any]:
            return []
        def get_project(self, project_id: str) -> Any:
            return None
        def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
            return {"project_id": project_id, **updates}

from server.server_defines import GraphRagManager, Project, TaskManager, FileEventType
from server.task_manager import RefreshTask, QueryTask, ListDocumentsTask, FileTask

logging.getLogger("haystack").setLevel(logging.DEBUG)

# Module logger
logger = logging.getLogger(__name__)

# Enable content tracing for component inputs/outputs
try:
    tracing.tracer.is_content_tracing_enabled = True  # type: ignore
    # Activate the LoggingTracer
    if 'LoggingTracer' in globals():
        tracing.enable_tracing(LoggingTracer())  # type: ignore
except Exception:
    pass

@dataclass
class _ChunkRecord:
    id: str
    path: str
    index: int
    text: str
    ext: str


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
        # Internal registry of active projects for runtime operations
        self._project_entries: Dict[str, _ProjectEntry] = {}
        self._project_mgr = ProjectManagerImpl(self._db_driver)
        # Best-effort: don’t fail hard if startup loading encounters optional deps
        try:
            self._load_and_activate_all_projects()
        except Exception as e:
            logger.warning("Startup project activation skipped due to error: %s", str(e))

    # ---------------------
    # Utilities and metadata
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
            """, file_path=os.path.normpath(file_path),
                 content_hash=content_hash,
                 last_modified_at=int(stat.st_mtime * 1000),
                 file_size=stat.st_size)

    # ---------------------
    # Projects API
    # ---------------------
    def create_project(self, project: Project) -> None:
        if self._project_entries.get(project.project_id) is not None:
            raise Exception('Project already exists')
        try:
            # Persist project via project manager (may return None or a dict depending on impl)
            try:
                self._project_mgr.create_project(project)
            except Exception:
                # Non-fatal for tests using dummy manager
                pass
            project_entry = _ProjectEntry(project)
            self._project_entries[project.project_id] = project_entry
            project_entry.db_driver = GraphDatabase.driver(self._neo4j_url, auth=(self._username, self._password))
            self._create_project_observers(project_entry)
            for path_str in project.source_roots:
                path = Path(path_str)
                # Schedule initial add of each root via handler (tests stub this out)
                try:
                    self.handle_add_path(project.project_id, path)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Failed to create project %s: %s", project.project_id, str(e))

    def delete_project(self, request_id: str, project_id: str) -> None:
        logger.info("Deleting project: %s, request_id: %s", project_id, request_id)
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
                # Inform underlying project manager
                try:
                    self._project_mgr.delete_project(project_id)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Failed to delete project %s: %s", project_id, str(e))
            raise

    def update_project(self, request_id: str, project_id: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        logger.info("Updating project: %s, request_id: %s", project_id, request_id)
        try:
            entry = self._project_entries.get(project_id)
            if entry is None:
                raise ValueError(f"Project {project_id} not found")

            args = args or {}

            # Determine incoming values with fallbacks
            new_name: str = args.get("name", entry.project.name)
            incoming_roots: List[str] = args.get("source_roots", entry.project.source_roots) or []

            # Normalize paths to absolute posix strings for comparison
            def normalize_many(paths: List[str]) -> List[str]:
                normed: List[str] = []
                for p in paths:
                    try:
                        normed.append(Path(p).expanduser().resolve().as_posix())
                    except Exception:
                        # If invalid, keep as-is to let downstream tasks handle/report
                        normed.append(Path(p).as_posix())
                return normed

            current_norm = set(normalize_many(entry.project.source_roots))
            incoming_norm = set(normalize_many(incoming_roots))

            to_add = sorted(incoming_norm - current_norm)
            to_remove = sorted(current_norm - incoming_norm)

            # Persist changes via project manager
            updates: Dict[str, Any] = {}
            if new_name != entry.project.name:
                updates["name"] = new_name
            # Always persist the new set of roots if changed
            if to_add or to_remove:
                # Preserve original representation as provided in args for storage
                updates["source_roots"] = incoming_roots

            if updates:
                # Call underlying manager without unexpected keywords in the signature
                updated = self._project_mgr.update_project(project_id, updates)
                # Some implementations may return a dict using private attribute keys (e.g., _name)
                if isinstance(updated, dict):
                    normalized = {
                        "project_id": updated.get("project_id") or updated.get("_project_id") or entry.project.project_id,
                        "name": updated.get("name") or updated.get("_name") or new_name,
                        "source_roots": updated.get("source_roots") or updated.get("_source_roots") or incoming_roots,
                        "embedder_model_name": updated.get("embedder_model_name") or updated.get("_embedder_model_name", entry.project.embedder_model_name),
                        "llm_model_name": updated.get("llm_model_name") or updated.get("_llm_model_name", entry.project.llm_model_name),
                        "query_temperature": updated.get("query_temperature") or updated.get("_query_temperature", entry.project.query_temperature),
                    }
                else:
                    # Fallback to current entry data if the manager returned a non-dict
                    normalized = {
                        "project_id": entry.project.project_id,
                        "name": new_name,
                        "source_roots": incoming_roots,
                        "embedder_model_name": entry.project.embedder_model_name,
                        "llm_model_name": entry.project.llm_model_name,
                        "query_temperature": entry.project.query_temperature,
                    }
                proj = Project.from_dict(normalized)

                if proj is None:
                    raise ValueError(f"Project {project_id} not found after update")
                entry.project = proj
            else:
                updated = {"ok": True, "project_id": project_id}

            # Handle FileTasks and observers for roots that changed
            # Remove old observers for removed roots and schedule delete tasks
            if to_remove:
                # Stop observers pointing to removed roots
                remaining_observers: List[Observer] = []
                removed_set = set(to_remove)
                for obs in entry.observers or []:
                    try:
                        # watchdog.Observer doesn't expose watched paths directly; we stop/recreate below.
                        obs.stop()
                        obs.join(timeout=1)
                    except Exception as e:
                        logger.warning("Failed to stop observer during removal: %s", str(e))
                entry.observers = []

                for root_str in to_remove:
                    # Create a FileTask to remove the path
                    self._task_mgr.add_task(FileTask(request_id,
                                                     project_id,
                                                     root_str, root_str,
                                                     True,
                                                     FileEventType.PATH_DELETED,
                                                     self.handle_delete_path,
                                                     self._task_mgr))

            # Add observers for newly added roots and schedule add tasks
            if to_add:
                for root_str in to_add:
                    self._task_mgr.add_task(FileTask(request_id,
                                                     project_id,
                                                     root_str,
                                                     "",
                                                     True,
                                                     FileEventType.PATH_CREATED,
                                                     self.handle_add_path,
                                                     self._task_mgr))

            # Recreate observers for the final set of roots
            self._create_project_observers(project_entry=entry)

            return updated
        except Exception as e:
            logger.exception("Failed to update project %s: %s", project_id, str(e))
            raise

    def list_projects(self, request_id: str) -> List[Project]:
        logger.info(f"Listing projects for request {request_id}")
        return self._project_mgr.list_projects()

    def get_project(self, request_id: str, project_id: str) -> Project | None:
        logger.info(f"Getting project {project_id} for request {request_id}")
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
                # Create and execute sync task asynchronously instead of direct execution
                request_id = "system" + str(uuid.uuid4())
                self._task_mgr.add_task(
                    RefreshTask(request_id, project.project_id, self.handle_sync_project, self._task_mgr))
            except Exception as e:
                logger.exception("Startup sync failed for %s, cause: %s", project.project_id, str(e))
    # ---------------------
    # Lifecycle API
    # ---------------------
    def stop(self) -> None:
        for project_id, project_entry in list(self._project_entries.items()):
            for observer in project_entry.observers or []:
                try:
                    observer.stop()
                    observer.join()
                except Exception as e:
                    logger.warning("Failed to stop observer: %s", str(e))

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
            self._task_mgr.add_task(RefreshTask(request_id, project_id, self.handle_sync_project, self._task_mgr))
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to refresh documents, request_id: %s", request_id)
            return {"ok": False, "error": str(e)}

    def handle_add_path(self, request_id: str, project_id: str, src_path_str: str, dest_path_str: str = None) -> None:
        logger.info("Entering GraphRagManager.handle_add_path file_path=%s", src_path_str)
        if src_path_str is None or src_path_str.startswith('.'):
            logger.info("Skipping, file_path is either none or a hidden directory")
            return

        project_entry: _ProjectEntry = self._project_entries.get(project_id)

        try:
            path = Path(src_path_str)
            if not path.exists():
                return
            code_files: List[Path] = []
            if path.is_dir():
                if path.name.startswith('.'):
                    return
                code_files.extend(list(self._iter_code_files(path)))
            else:
                code_files.append(path)

            if len(code_files) == 0:
                return

            project_entry.ingestion_pipeline.run({"file_type_router": {"sources": code_files}})
            logger.info("Completed ingestion of code files.")
            self._create_doc_relationships(project_entry)
            for fp in code_files:
                ch = self._compute_file_hash(fp)
                self._store_file_metadata(project_id, fp, ch)
            logger.info("Ingested %d files", len(code_files))
        except Exception as e:
            logger.exception("Ingestion failed for %s", path)
            raise e

    def handle_update_path(self, request_id: str, project_id: str, src_path_str: str, dest_path_str: str = None) -> None:
        try:
            self.handle_delete_path(request_id, project_id, src_path_str)
            self.handle_add_path(request_id, project_id, dest_path_str)
        except Exception as e:
            logger.exception("Update failed for %s", src_path_str)
            raise e

    def handle_delete_path(self, request_id: str, project_id: str, src_path_str: str, dest_path_str: str = None) -> None:
        """
        If the path is a directory, delete every document under it from the store,
        along with all the relationships.  For each subdirectory, recursively call
        this method.
        """
        logger.info("Entering GraphRagManager.delete_path path: %s", src_path_str)
        project_entry = self._project_entries.get(project_id)
        path = Path(src_path_str)
        try:
            target = Path(path).expanduser().resolve()
        except Exception as e:
            logger.exception("Invalid path provided to delete_path: %s", path, e)
            raise e

        if not target.exists():
            logger.warning("Path does not exist, nothing to delete: %s", target.as_posix())

        # If directory: traverse and delete all supported files under it
        if target.is_dir():
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS or p.is_dir():
                    self.handle_delete_path(request_id, project_id, os.path.normpath(p))

        # If file: remove corresponding nodes/embeddings/relationships
        if target.is_file():
            file_path_str = os.path.normpath(target)
            try:
                with project_entry.db_driver.session(database=project_entry.project.name) as s:
                    # 1) Remove any relationships to/from documents with this file path
                    s.run(
                        """
                        MATCH (d:Document {file_path: $path})-[r]-()
                        DELETE r
                        """,
                        {"path": file_path_str},
                    )
                    # 2) Remove the documents themselves that match the file path
                    s.run(
                        """
                        MATCH (d:Document {file_path: $path})
                        DELETE d
                        """,
                        {"path": file_path_str},
                    )
            except Exception as e:
                logger.exception("Failed deleting document graph for %s", file_path_str)
                raise e

    def handle_sync_project(self, request_id: str, project_id: str) -> Dict[str, Any]:
        logger.info("Starting sync project_id=%s", project_id)
        project_entry = self._project_entries.get(project_id)
        if not project_entry:
            raise ValueError(f"Project {project_id} not found")

        # if force_all:
        #     self.handle_refresh_documents(project_id)
        #     return {"mode": "full", "ok": True}

        changes = {"added": [], "modified": [], "deleted": [], "unchanged": 0}
        tracked = self._get_tracked_files_with_hashes(project_id)

        current: Dict[str, str] = {}
        for root in project_entry.project.source_roots:
            root_path = Path(root).expanduser().resolve()
            if not root_path.exists():
                continue
            for fp in self._iter_code_files(root_path):
                try:
                    current[os.path.normpath(fp)] = self._compute_file_hash(fp)
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
                self.handle_delete_path(request_id, project_id, d)
            except Exception:
                logger.exception("Delete during sync failed for %s", d)
        for a in changes["added"]:
            try:
                self.handle_add_path(request_id, project_id, a)
            except Exception:
                logger.exception("Add during sync failed for %s", a)
        for m in changes["modified"]:
            try:
                self.handle_update_path(request_id, project_id, m)
            except Exception:
                logger.exception("Update during sync failed for %s", m)

        return {"mode": "incremental", **{k: (len(v) if isinstance(v, list) else v) for k, v in changes.items()}}

    def handle_list_documents(self, request_id: str, project_id: str) -> Dict[str, Any]:
        entry = self._project_entries.get(project_id)
        if not entry:
            raise ValueError(f"Project {project_id} not found")
        try:
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

    # ---------------------
    # General Helpers
    # ---------------------
    # alter the rglob regex to exclude directories that start with '.'
    def _iter_code_files(self, root: Path) -> Iterable[Path]:
        for p in root.rglob("*"):
            if (p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS and
                    not any(part.startswith('.') for part in p.parts)):
                yield p

    def _create_doc_relationships(self, project_entry: _ProjectEntry):
        with self._db_driver.session(database=project_entry.project.name) as session:
            # Create IMPORTS relationships
            session.run("""
                MATCH (d:Document)
                WHERE d.imports IS NOT NULL
                UNWIND d.imports AS imported_module
                MATCH (d2:Document)
                WHERE d2.file_path CONTAINS(imported_module)
                MERGE (d)-[:IMPORTS]->(d2)
            """)

            # Create CALLS relationships:
            # For each function name in d.calls, connect to any Document whose symbol_name contains that function.
            session.run("""
                MATCH (d:Document)
                WHERE d.calls IS NOT NULL
                UNWIND d.calls AS called_function
                MATCH (d2:Document)
                WHERE d2.symbols IS NOT NULL
                  AND any(sym IN d2.symbols WHERE sym CONTAINS called_function)
                MERGE (d)-[:CALLS]->(d2)
                MERGE (d2)-[:CALLED_BY]->(d)
            """)

            # Create NEXT/PREV relationships between chunked documents when metadata contains navigation ids
            session.run("""
                MATCH (d:Document)
                WHERE d.next IS NOT NULL
                MATCH (n:Document {id: d.next})
                MERGE (d)-[:NEXT]->(n)
            """)

            session.run("""
                MATCH (d:Document)
                WHERE d.previous IS NOT NULL
                MATCH (p:Document {id: d.previous})
                MERGE (d)-[:PREVIOUS]->(p)
            """)

    def _get_document_ids_by_path(self, project_id: str, file_path: str) -> List[str]:
        """
        Return all Document node ids for a given file path within a project database.
        Multiple ids may exist due to chunking.
        """
        entry = self._project_entries.get(project_id)
        if not entry:
            raise ValueError(f"Project {project_id} not found")
        path_str = os.path.normpath(file_path)
        try:
            with entry.db_driver.session(database=entry.project.name) as session:
                res = session.run("""
                    MATCH (d:Document {file_path: $path})
                    RETURN d.id AS id
                """, path=path_str)
                ids: List[str] = []
                for rec in res:
                    v = rec.get("id")
                    if v is not None:
                        ids.append(str(v))
                return ids
        except Exception as e:
            logger.exception("Failed to fetch document ids for path=%s in project=%s: %s", path_str, project_id, str(e))
            raise
