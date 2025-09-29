
import logging
from dataclasses import dataclass
from pathlib import Path
from haystack_integrations.document_stores.neo4j import Neo4jDocumentStore  # type: ignore
from haystack.components.embedders import OllamaTextEmbedder  # type: ignore
from haystack.components.retrievers import CygGraphRetriever  # type: ignore
from haystack.components.generators import OllamaGenerator  # type: ignore  # type: ignore
from haystack import Pipeline  # type: ignore
from haystack import tracing
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder  # type: ignore
from haystack.components.writers import DocumentWriter
from haystack.core.component import Component  # type: ignore
from haystack.dataclasses import Document  # type: ignore
from haystack.document_stores import InMemoryDocumentStore  # type: ignore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tracing.logging_tracer import LoggingTracer
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from neo4j import GraphDatabase
from typing import Any, Dict, Iterable, List, Optional, Tuple
from neo4j_haystack import Neo4jDocumentStore

from server.code_aware_splitter import CodeAwareSplitter
from server.code_change_handler import CodeChangeHandler
from server.code_relationship_extractor import CodeRelationshipExtractor
from watchdog.observers import Observer

from server.task_manager import TaskManagerImpl, QueryTask, RefreshTask
from server.graph_document_expander import GraphAwareRetriever

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

class GraphRagManager:
    """
    Graph-based RAG manager using Haystack with code-aware ingestion.
    """

    SUPPORTED_EXTS = {".html", ".js", ".css", ".py", ".json"}

    def __init__(self, doc_root: str) -> None:
        url = "bolt://localhost:7687"
        username = "neo4j"
        password = "your_neo4j_password"
        database = "neo4j"

        self._doc_root = Path(doc_root).expanduser().resolve()
        self._driver = GraphDatabase.driver(url, auth=(username, password))
        self._event_executor = TaskManagerImpl(self._driver)

        # Initialize Document Store and Embedder [1.2.7, 1.5.3]
        self._document_store = Neo4jDocumentStore(
            url=url,
            username=username,
            password=password,  # Replace with your password
            database=database,
            embedding_dim=768  # Ensure this matches your Ollama model's dimension
        )

        self.document_embedder = OllamaDocumentEmbedder(
            model=self.embedder_model_name,
            url="http://localhost:11434"
        )

        # Define the ingestion pipeline
        text_converter = TextFileToDocument()
        rel_extractor = CodeRelationshipExtractor()
        doc_splitter = CodeAwareSplitter()
        doc_writer = DocumentWriter(document_store=self._document_store, policy=DuplicatePolicy.OVERWRITE)

        # Create the pipeline
        self.ingestion_pipeline = Pipeline()
        self.ingestion_pipeline.add_component("converter", text_converter)
        self.ingestion_pipeline.add_component("splitter", doc_splitter)
        self.ingestion_pipeline.add_component("extractor", rel_extractor)
        self.ingestion_pipeline.add_component("embedder", self.document_embedder)
        self.ingestion_pipeline.add_component("writer", doc_writer)

        # Link the components
        self.ingestion_pipeline.connect("converter.documents", "splitter.documents")
        self.ingestion_pipeline.connect("splitter.documents", "extractor.documents")
        self.ingestion_pipeline.connect("extractor.documents", "embedder.documents")
        self.ingestion_pipeline.connect("embedder.documents", "writer.documents")

        # Auto-ingest all supported files under doc_root
        try:
            self._add_path(self._doc_root)
        except Exception:
            logger.exception("Auto-ingest failed during initialization")

        self._observer = Observer()
        self.code_change_handler = CodeChangeHandler(self)
        self._observer.schedule(self.code_change_handler, self._doc_root.as_posix(), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info("Filesystem watcher started on %s", self._doc_root.as_posix())

    # ---------------------
    # Basic API
    # ---------------------

    def list_documents(self, request_id: str) -> List[str]:
        logger.info("Entering GraphRagManager.list_documents, request_id=%s", request_id)
        return [str(path) for path in self._iter_code_files()]

    def refresh_documents(self, request_id: str) -> Dict[str, Any]:
        logger.info("Entering GraphRagManager.clear_documents, request_id: %s", request_id)
        try:
            self._event_executor.add_task(RefreshTask(request_id, self._refresh_documents()))
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to refresh documents, request_id: %s", request_id)
            return {"ok": False, "error": str(e)}

    def query(self, request_id: str, query: str) -> Dict[str, Any]:
        try:
            self._event_executor.add_task(QueryTask(request_id, query))
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to query %s, request_id: %s", query, request_id)
            return {"ok": False, "error": str(e)}

    def cancel_query(self, request_id: str):
        try:
            self._event_executor.cancel_task(request_id)
            return {"ok": True, "request_id": request_id}
        except Exception as e:
            logger.exception("Failed to cancel request_id: %s", request_id)
            return {"ok": False, "error": str(e)}

    def _add_path(self, path: Path) -> None:
        logger.info("Entering GraphRagManager.ingest_file file_path=%s", path)
        try:
            code_files = []

            if not path.exists() or not path.is_file():
                return

            if path.is_dir():
                code_files.append(list(self._iter_code_files(path)))
            else:
                code_files.append(path)

            self.ingestion_pipeline.run({"converter": {"sources": code_files}})
            self._create_code_relationships()
            print("Nodes created successfully.")
        except Exception as e:
            logger.exception("Ingestion failed for file %s", path)
            raise e
        return

    def _update_path(self, path: Path) -> None:
        try:
            self._delete_path(path)
            self._add_path(path)
        except Exception as e:
            logger.exception("Update failed for file %s", path)
            raise e
        return

    def _delete_path(self, path: Path) -> None:
        """
        If the path is a directory, delete every document under it from the store,
        along with all the relationships.  For each subdirectory, recursively call
        this method.
        """
        logger.info("Entering GraphRagManager.delete_path path: %s", path)
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
                    self.delete_path(p)

        # If file: remove corresponding nodes/embeddings/relationships
        if target.is_file():
            file_path_str = target.as_posix()
            # Remove relationships before removing the node
            try:
                with self._driver.session() as s:
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
                self._document_store.delete_documents([file_path_str])
            except Exception as e:
                logger.debug("Document store deletion attempt skipped/failed for %s", file_path_str)
                raise e

    def _list_documents(self) -> List[str]:
        try:
            return  [str(path) for path in self._iter_code_files()]
        except Exception as e:
            logger.exception("Failed listing documents", e)
            raise e

    def _refresh_documents(self) -> None:
        logger.info("Entering GraphRagManager._refresh_documents")
        self._delete_path(self._doc_root)
        self._add_path(self._doc_root)

    def _query(self, query: str) -> Dict[str, Any]:
        """
        Retrieve graph-aware relevant chunks from Neo4j via Haystack and answer with a local model.
        The retrieval leverages relationships (IMPORTS, CALLS, DEFINES) using a Cypher-aware retriever.
        """
        logger.info("Entering GraphRagManager.start_query query=%s", query)

        try:
            # Embed the query
            q_embedder = OllamaTextEmbedder(model="nomic-embed-text", url="http://localhost:11434")

            # Use custom graph-aware retriever
            retriever = GraphAwareRetriever(
                document_store=self._document_store,
                driver=self._driver,
                top_k=15,
                relationship_weight=0.3
            )

            # Local generator
            generator = OllamaGenerator(model="llama3.1", url="http://localhost:11434")

            # Build pipeline: embed -> retrieve -> generate
            pipeline = Pipeline()
            pipeline.add_component("q_embedder", q_embedder)
            pipeline.add_component("retriever", retriever)
            pipeline.add_component("llm", generator)
            pipeline.connect("q_embedder.embedding", "retriever.query_embedding")

            ret = pipeline.run(data={"q_embedder": {"text": query}})
            docs = ret.get("retriever", {}).get("documents", [])

            # Rest of the method remains the same...
            context = "\n\n".join(d.content for d in docs)

            prompt = f"""You are a code-aware assistant. Use the graph-related context to answer.
    Context:
    {context}

    Question: {query}
    Answer succinctly:"""

            return pipeline.run(data={"llm": {"prompt": prompt}}, include_outputs_from=["llm"])
        except Exception as e:
            logger.exception("Query operation failed: %s", str(e))
            raise e

    # ---------------------
    # Helpers
    # ---------------------
    def _iter_code_files(self, root: Path = None) -> Iterable[Path]:
        target = root
        if target is None:
            target = self._doc_root
        for p in target.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS:
                yield p

    def _create_code_relationships(self):
        with self._driver.session() as session:
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
