import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, LiteralString

import neo4j
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument, CSVToDocument, DOCXToDocument, HTMLToDocument, \
    JSONConverter, PyPDFToDocument, MarkdownToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from neo4j import Driver, GraphDatabase
from neo4j_haystack import Neo4jDocumentStore
from watchdog.observers import Observer
from server.pipeline.code_relationship_extractor import CodeRelationshipExtractor
from server.pipeline.css_splitter import CssSplitter
from server.pipeline.graph_document_expander import GraphAwareRetriever
from server.pipeline.html_splitter import HtmlSplitter
from server.pipeline.javascript_splitter import JavascriptSplitter
from server.pipeline.python_splitter import PythonSplitter
from server.server_defines import ProjectManager, Project, FileEventType
from server.task_manager import FileTask

logger = logging.getLogger(__name__)

class _ProjectEntry:
    project: Project
    observers: List[Observer] = []
    db_driver: neo4j.Driver
    document_store: Neo4jDocumentStore
    document_embedder: OllamaDocumentEmbedder
    text_converter: TextFileToDocument
    csv_converter: CSVToDocument
    docx_converter: DOCXToDocument
    html_converter: HTMLToDocument
    json_converter: JSONConverter
    pdf_converter: PyPDFToDocument
    md_converter: MarkdownToDocument
    text_splitter: DocumentSplitter
    css_splitter: CssSplitter
    html_splitter: HtmlSplitter
    md_splitter: DocumentSplitter
    js_splitter: JavascriptSplitter
    python_splitter: PythonSplitter

    rel_extractor: CodeRelationshipExtractor
    retriever: GraphAwareRetriever
    doc_writer: DocumentWriter
    ingestion_pipeline: Pipeline

    def __init__(self, project: Project) -> None:
        try:
            url = os.environ.get('NEO4J_URL', 'bolt://localhost:7687')
            username = os.environ.get('NEO4j_USERNAME', 'neo4j')
            password = os.environ.get('NEO4J_PASSWORD', 'your_neo4j_password')
            self.default_embedder = os.environ.get('GRAPH_RAG_DEFAULT_EMBEDDER',
                                                   'hf.co/nomic-ai/nomic-embed-code-GGUF:Q4_K_M')
            self.embedder_name = self.default_embedder if project.embedder_model_name == 'default' else project.embedder_model_name
            self.default_llm = os.environ.get('GRAPH_RAG_DEFAULT_LLM',
                                              'hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q5_K_M')
            self.llm_name = self.default_llm if project.llm_model_name == 'default' else project.llm_model_name

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
                model=self.embedder_name,
                url="http://localhost:11434"
            )

            # Define the ingestion pipeline
            self.file_type_router = FileTypeRouter(mime_types=["text/plain", "text/css", "text/html",
                                                               "text/javascript", "text/x-python", "application/pdf",
                                                               "text/markdown", "application/json"])
            self.text_converter = TextFileToDocument(store_full_path=True)
            self.python_converter = TextFileToDocument(store_full_path=True)
            self.js_converter = TextFileToDocument(store_full_path=True)
            self.css_converter = TextFileToDocument(store_full_path=True)
            self.html_converter = HTMLToDocument(store_full_path=True)
            self.md_converter = MarkdownToDocument(store_full_path=True)
            self.pdf_converter = PyPDFToDocument(store_full_path=True)
            self.css_splitter = CssSplitter()
            self.js_splitter = JavascriptSplitter()
            self.md_splitter = DocumentSplitter()
            self.text_splitter = DocumentSplitter()
            self.css_splitter = CssSplitter()
            self.html_splitter = HtmlSplitter()
            self.js_splitter = JavascriptSplitter()
            self.pdf_splitter = DocumentSplitter()
            self.python_splitter = PythonSplitter()
            self.rel_extractor = CodeRelationshipExtractor()
            self.document_joiner = DocumentJoiner()

            self.doc_writer = DocumentWriter(document_store=self.document_store,
                                            policy=DuplicatePolicy.OVERWRITE)

            # Create the pipeline
            self.ingestion_pipeline = Pipeline()
            self.ingestion_pipeline.add_component("file_type_router", self.file_type_router)
            self.ingestion_pipeline.add_component("text_converter",  self.text_converter)
            self.ingestion_pipeline.add_component("python_converter", self.python_converter)
            self.ingestion_pipeline.add_component("js_converter", self.js_converter)
            self.ingestion_pipeline.add_component("md_converter", self.md_converter)
            self.ingestion_pipeline.add_component("css_converter", self.css_converter)
            self.ingestion_pipeline.add_component("pdf_converter", self.pdf_converter)
            self.ingestion_pipeline.add_component("html_converter", self.html_converter)
            self.ingestion_pipeline.add_component("text_splitter", self.text_splitter)
            self.ingestion_pipeline.add_component("md_splitter", self.md_splitter)
            self.ingestion_pipeline.add_component("pdf_splitter", self.pdf_splitter)
            self.ingestion_pipeline.add_component("css_splitter", self.css_splitter)
            self.ingestion_pipeline.add_component("html_splitter", self.html_splitter)
            self.ingestion_pipeline.add_component("js_splitter", self.js_splitter)
            self.ingestion_pipeline.add_component("python_splitter", self.python_splitter)
            self.ingestion_pipeline.add_component("extractor", self.rel_extractor)
            self.ingestion_pipeline.add_component("joiner", self.document_joiner)
            self.ingestion_pipeline.add_component("embedder", self.document_embedder)
            self.ingestion_pipeline.add_component("writer", self.doc_writer)

            # Link the components
            self.ingestion_pipeline.connect("file_type_router.text/plain", "text_converter.sources")
            self.ingestion_pipeline.connect("file_type_router.text/markdown", "md_converter.sources")
            self.ingestion_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
            self.ingestion_pipeline.connect("file_type_router.text/x-python", "python_converter.sources")
            self.ingestion_pipeline.connect("file_type_router.text/css", "css_converter.sources")
            self.ingestion_pipeline.connect("file_type_router.text/javascript", "js_converter.sources")
            self.ingestion_pipeline.connect("file_type_router.text/html", "html_converter.sources")

            self.ingestion_pipeline.connect("text_converter.documents", "text_splitter.documents")
            self.ingestion_pipeline.connect("md_converter.documents", "md_splitter.documents")
            self.ingestion_pipeline.connect("pdf_converter.documents", "pdf_splitter.documents")
            self.ingestion_pipeline.connect("python_converter.documents", "python_splitter.documents")
            self.ingestion_pipeline.connect("css_converter.documents", "css_splitter.documents")
            self.ingestion_pipeline.connect("html_converter.documents", "html_splitter.documents")
            self.ingestion_pipeline.connect("js_converter.documents", "js_splitter.documents")
            self.ingestion_pipeline.connect("python_splitter.documents", "extractor.documents")
            self.ingestion_pipeline.connect("extractor.documents", "joiner.documents")
            self.ingestion_pipeline.connect("text_splitter.documents", "joiner.documents")
            self.ingestion_pipeline.connect("css_splitter.documents", "joiner.documents")
            self.ingestion_pipeline.connect("html_splitter.documents", "joiner.documents")
            self.ingestion_pipeline.connect("js_splitter.documents", "joiner.documents")
            self.ingestion_pipeline.connect("joiner.documents", "embedder.documents")
            self.ingestion_pipeline.connect("embedder.documents", "writer.documents")
        except Exception as e:
            logger.exception("Error initializing project pipelines %s", str(e))

class ProjectManagerImpl(ProjectManager):
    """
    Neo4j-backed implementation.

    - Uses the system database 'system' to manage databases (create/drop).
    - Stores project metadata in a central catalog database 'projects_catalog'
      in a node labeled Project with properties:
        id (string, uuid), name (string), source_roots (list<string>), args (map<string,string>)
    - The per-project database is created/dropped with the project's name.
    """

    CATALOG_DB = "graph-rag"

    _project_entries: Dict[str, _ProjectEntry]

    def __init__(self, driver: Driver):
        self.driver = driver
        # Ensure catalog DB and indexes exist
        self._ensure_catalog()
        self._project_entries = {}

    # Getter property for _project_entries
    @property
    def project_entries(self) -> Dict[str, _ProjectEntry]:
        return self._project_entries

    def _ensure_catalog(self) -> None:
        logger.info("Ensuring catalog database and schema")
        try:
            # Ensure catalog database exists
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run(
                    "CREATE DATABASE $db IF NOT EXISTS",
                    db=self.CATALOG_DB)

            # Ensure unique constraint on Project.id
            with self.driver.session(database=self.CATALOG_DB) as cat_sess:
                cat_sess.run(
                    "CREATE CONSTRAINT project_id_unique IF NOT EXISTS "
                    "FOR (p:Project) REQUIRE p.id IS UNIQUE")
        except Exception as e:
            logger.exception("Failed to ensure catalog: %s", e)
            raise

    def create_project(self, project: Project) -> Project:
        logger.info("create_project called: name=%s", project.name)
        try:
            # 1) Create per-project database by name
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run(
                    "CREATE DATABASE $db IF NOT EXISTS",
                    db=project.name,
                )

            # After DB creation, ensure required fulltext indexes exist in the project's database
            try:
                with self.driver.session(database=project.name) as proj_sess:
                    index_statements = [
                        # Full-text over content
                        """
                        CREATE FULLTEXT INDEX document_content_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.content]
                        """,
                        # Full-text over calls (array of strings)
                        """
                        CREATE FULLTEXT INDEX document_calls_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.calls]
                        """,
                        # Full-text over file_path
                        """
                        CREATE FULLTEXT INDEX document_file_path_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.file_path]
                        """,
                        # Full-text over symbol_name (on Chunk nodes)
                        """
                        CREATE FULLTEXT INDEX document_symbol_name_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.symbol_name]
                        """,
                        # Full-text over symbol_scope (on Chunk nodes)
                        """
                        CREATE FULLTEXT INDEX document_symbol_scope_fti IF NOT EXISTS
                        FOR (d:Document) ON EACH [d.symbol_scope]
                        """,
                        # Create/ensure indexes for Document navigation fields
                        """
                        CREATE INDEX document_next_idx IF NOT EXISTS 
                        FOR (d:Document) ON (d.next)
                        """,
                        # Create/ensure indexes for Document navigation fields
                        """
                        CREATE INDEX document_previous_idx IF NOT EXISTS
                        FOR (d:Document) ON (d.previous)
                        """
                    ]
                    for stmt in index_statements:
                        try:
                            proj_sess.run(stmt)
                            logger.debug("Ensured index: %s", stmt.strip().splitlines()[0])
                        except Exception as e:
                            logger.warning("Failed creating index in project DB %s: %s", project.name, e)
            except Exception as e:
                logger.exception("Error ensuring full-text indexes for project %s: %s", project.name, e)
                # Non-fatal; continue so project is at least created

            # 2) Insert metadata into catalog
            with self.driver.session(database=self.CATALOG_DB) as cat_sess:
                cat_sess.run(
                    """
                    CREATE (p:Project {
                        project_id: $project_id,
                        name: $name,
                        source_roots: $source_roots
                    })
                    """,
                    project_id=project.project_id,
                    name=project.name,
                    source_roots=list(project.source_roots or [])
                )

            return {
                "project_id": project.project_id,
                "name": project.name,
                "source_roots": project.source_roots
            }
        except Exception as e:
            logger.exception("Error in create_project: %s", e)
            raise

    def get_project(self, project_id: str) -> Optional[Project]:
        logger.info("get_project called: id=%s", project_id)
        try:
            with self.driver.session(database=self.CATALOG_DB) as sess:
                rec = sess.run(
                    "MATCH (p:Project {project_id: $id}) RETURN p LIMIT 1", id=project_id
                ).single()
                if not rec:
                    return None
                p = rec["p"]
                return Project.from_dict({
                    "project_id": p["project_id"],
                    "name": p["name"],
                    "source_roots": list(p.get("source_roots", [])),
                    "args": dict(p.get("args", {})),
                    "database": p["name"],
                })
        except Exception as e:
            logger.exception("Error in get_project: %s", e)
            raise

    def list_projects(self) -> List[Project]:
        logger.info("list_projects called")
        try:
            with self.driver.session(database=self.CATALOG_DB) as sess:
                res = sess.run("MATCH (p:Project) RETURN p ORDER BY p.name ASC")
                items: List[Project] = []
                for rec in res:
                    p = rec["p"]
                    items.append(Project.from_dict({
                            "project_id": p["project_id"],
                            "name": p["name"],
                            "source_roots": list(p.get("source_roots", []))
                    }))
                return items
        except Exception as e:
            logger.exception("Error in list_projects: %s", e)
            raise

    def update_project(self, request_id: str, project_id: str, args: Dict[str, Any] = None) -> Project:
        logger.info("Updating project: %s, request_id: %s", project_id, request_id)
        try:
            entry = self._project_entries.get(project_id)
            if entry is None:
                raise ValueError(f"Project {project_id} not found")

            args = args or {}
            incoming_name: str = args.get("name", entry.project.name)
            incoming_roots_raw: List[str] = args.get("source_roots", entry.project.source_roots) or []

            # Normalize paths for reliable comparison, keep original incoming for persistence
            def norm_many(paths: List[str]) -> List[str]:
                out: List[str] = []
                for p in paths:
                    try:
                        out.append(Path(p).expanduser().resolve().as_posix())
                    except Exception:
                        out.append(Path(p).as_posix())
                return out

            current_norm = set(norm_many(entry.project.source_roots))
            incoming_norm = set(norm_many(incoming_roots_raw))

            to_add = sorted(incoming_norm - current_norm)
            to_remove = sorted(current_norm - incoming_norm)

            # Persist changes (single write to underlying project manager)
            updates_to_persist: Dict[str, Any] = {}
            name_changed = incoming_name != entry.project.name
            roots_changed = bool(to_add or to_remove)

            if name_changed:
                updates_to_persist["name"] = incoming_name
            if roots_changed:
                updates_to_persist["source_roots"] = incoming_roots_raw

            if updates_to_persist:
                updated_dict = self._project_mgr.update_project(project_id, updates_to_persist)
                proj = Project.from_dict(updated_dict)
                if proj is None:
                    raise ValueError(f"Project {project_id} not found after update")
                entry.project = proj
            else:
                # Nothing changed; return early
                return {"ok": True, "project_id": project_id}

            # If name changed, switch database-bound resources (document store, db driver-backed items, pipelines)
            if name_changed:
                # Close existing observers; they'll be recreated below if needed
                for obs in entry.observers or []:
                    with contextlib.suppress(Exception):
                        obs.stop(); obs.join(timeout=1)
                entry.observers = []

                # Recreate project entry resources bound to DB name
                # Keep the same embedder/llm selections; only DB name changes
                # Rebuild a fresh _ProjectEntry with the new Project and rewire DB driver
                new_entry = _ProjectEntry(entry.project)
                new_entry.db_driver = GraphDatabase.driver(self._neo4j_url, auth=(self._username, self._password))
                # Replace the stored entry
                self._project_entries[project_id] = new_entry
                entry = new_entry  # continue using entry

            # Schedule removals first (clean up)
            if to_remove:
                for obs in entry.observers or []:
                    with contextlib.suppress(Exception):
                        obs.stop(); obs.join(timeout=1)
                entry.observers = []
                for root_norm in to_remove:
                    # Use original arg value when possible; fall back to normalized
                    root_for_task = root_norm
                    self._task_mgr.add_task(FileTask(request_id,
                                                     project_id,
                                                     root_for_task,
                                                     root_for_task,
                                                     True,
                                                     FileEventType.PATH_DELETED,
                                                     self.handle_delete_path,
                                                     self._task_mgr))

            # Schedule additions
            if to_add:
                for root_norm in to_add:
                    root_for_task = root_norm
                    self._task_mgr.add_task(FileTask(request_id,
                                                     project_id,
                                                     root_for_task,
                                                     "",
                                                     True,
                                                     FileEventType.PATH_CREATED,
                                                     self.handle_add_path,
                                                     self._task_mgr))

            # Only recreate observers if roots changed
            if roots_changed:
                self._create_project_observers(project_entry=entry)

            return {"ok": True, "project_id": project_id, "name_changed": name_changed, "roots_changed": roots_changed}
        except Exception as e:
            logger.exception("Failed to update project %s: %s", project_id, str(e))
            raise

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        logger.info("delete_project called: id=%s", project_id)
        try:
            # Find project to know its database name
            proj = self.get_project(project_id)
            if proj is None:
                return {"deleted": False, "reason": "not_found"}

            db_name = proj.name

            # Delete metadata
            with self.driver.session(database=self.CATALOG_DB) as sess:
                sess.run("MATCH (p:Project {project_id: $project_id}) DETACH DELETE p", project_id=project_id)

            # Drop database
            with self.driver.session(database="system") as sys_sess:
                sys_sess.run("DROP DATABASE $db IF EXISTS", db=db_name)

            return {"deleted": True, "project_id": project_id, "database": db_name}
        except Exception as e:
            logger.exception("Error in delete_project: %s", e)
            raise

    def stop(self) -> None:
        pass
