import logging
from typing import Any, List, Optional, Dict

from haystack import Document
from haystack.core.component import component
from neo4j_haystack import Neo4jDocumentStore, Neo4jEmbeddingRetriever

# Module logger
logger = logging.getLogger(__name__)

@component
class GraphAwareRetriever:
    def __init__(
        self,
        document_store: Neo4jDocumentStore,
        driver: Any,
        top_k: int = 10,
        relationship_weight: float = 0.3,
        max_hops: int = 2
    ):
        self.document_store = document_store
        self.driver = driver
        self.top_k = top_k
        self.relationship_weight = relationship_weight
        self.max_hops = max_hops

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        # Step 1: Get initial embedding-based matches
        embedding_retriever = Neo4jEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k * 2  # Get more candidates
        )
        initial_result = embedding_retriever.run(
            query_embedding=query_embedding,
            filters=filters
        )
        initial_docs = initial_result["documents"]
        
        # Step 2: Expand through graph relationships
        expanded_docs = self._expand_via_relationships(initial_docs)
        
        # Step 3: Re-rank and return top_k
        final_docs = self._rerank_documents(initial_docs + expanded_docs, query_embedding)
        
        return {"documents": final_docs[:self.top_k]}

    def _expand_via_relationships(self, docs: List[Document]) -> List[Document]:
        """Traverse IMPORTS, CALLS, DEFINES relationships to find related documents"""
        related_docs = []
        
        with self.driver.session() as session:
            for doc in docs:
                # Get documents related through various relationships
                result = session.run("""
                    MATCH (d:Document) WHERE d.id = $doc_id
                    MATCH (d)-[r:imports|calls|defines|next|prev*1..2]-(related:Document)
                    WHERE related.id <> d.id
                    RETURN DISTINCT related.id as related_id, 
                           related.content as content,
                           related.file_path as file_path,
                           type(r) as relationship_type
                    LIMIT 10
                """, {"doc_id": doc.id})
                
                for record in result:
                    # Create document from Neo4j result
                    related_doc = Document(
                        id=record["related_id"],
                        content=record["content"],
                        meta={
                            "file_path": record["file_path"],
                            "relationship_type": record["relationship_type"],
                            "source": "graph_expansion"
                        }
                    )
                    related_docs.append(related_doc)
        
        return related_docs

    def _rerank_documents(self, docs: List[Document], query_embedding: List[float]) -> List[Document]:
        """Re-rank documents considering both embedding similarity and graph relationships"""
        # This is a simplified reranking - you could implement more sophisticated scoring
        original_docs = [d for d in docs if d.meta.get("source") != "graph_expansion"]
        related_docs = [d for d in docs if d.meta.get("source") == "graph_expansion"]
        
        # Boost related documents slightly but keep original order mostly intact
        return original_docs + related_docs
