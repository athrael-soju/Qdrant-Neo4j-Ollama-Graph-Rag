"""
Retrieval module for GraphRAG.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np

from graphrag.connectors.neo4j_connection import get_connection as get_neo4j_connection
from graphrag.connectors.qdrant_connection import get_connection as get_qdrant_connection
from graphrag.core.ingest import embed_text

# Setup logging
logger = logging.getLogger(__name__)

class VectorRetriever:
    """Vector-based retrieval using Qdrant."""
    
    def __init__(self):
        """Initialize the retriever."""
        self.qdrant = get_qdrant_connection()
        
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve chunks based on vector similarity.
        
        Args:
            query: Query string
            top_k: Number of top results to retrieve
            
        Returns:
            List of (chunk_id, text, score) tuples
        """
        # Generate query embedding
        query_embedding = embed_text(query)
        
        # Search in Qdrant
        results = self.qdrant.search("chunks", query_embedding, top_k)
        
        # Format results
        chunks = []
        for result in results:
            chunk_id = result.payload.get("chunk_id", "unknown")
            text = result.payload.get("text", "")
            score = result.score
            chunks.append((chunk_id, text, score))
            
        return chunks

class GraphRetriever:
    """Graph-based retrieval using Neo4j."""
    
    def __init__(self):
        """Initialize the retriever."""
        self.neo4j = get_neo4j_connection()
        
    def retrieve_chunks_by_terms(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve chunks based on term overlap.
        
        Args:
            query: Query string
            top_k: Number of top results to retrieve
            
        Returns:
            List of (chunk_id, text, score) tuples
        """
        # Extract terms from query
        query_terms = set(query.lower().split())
        
        # Query Neo4j for chunks containing these terms
        cypher = """
        MATCH (t:Term)-[:APPEARS_IN]->(c:Chunk)
        WHERE t.text IN $query_terms
        WITH c, COUNT(DISTINCT t) AS term_overlap
        ORDER BY term_overlap DESC
        LIMIT $top_k
        RETURN c.chunk_id AS chunk_id, c.text AS text, term_overlap AS score
        """
        
        results = self.neo4j.run_query(cypher, {"query_terms": list(query_terms), "top_k": top_k})
        
        # Format results
        chunks = []
        for record in results:
            chunk_id = record["chunk_id"]
            text = record["text"]
            score = record["score"] / len(query_terms)  # Normalize score
            chunks.append((chunk_id, text, score))
            
        return chunks
    
    def retrieve_triplets(self, query: str, top_k: int = 5) -> List[Tuple[str, str, str, str, float]]:
        """
        Retrieve knowledge graph triplets relevant to the query.
        
        Args:
            query: Query string
            top_k: Number of top results to retrieve
            
        Returns:
            List of (subject, relation, object, chunk_id, score) tuples
        """
        # Extract terms from query
        query_terms = query.lower().split()
        
        # Query Neo4j for triplets related to query terms
        cypher = """
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE any(term IN $query_terms WHERE toLower(s.text) CONTAINS term OR toLower(o.text) CONTAINS term)
        WITH s, r, o, r.chunk_id AS chunk_id,
             size([term IN $query_terms WHERE toLower(s.text) CONTAINS term OR toLower(o.text) CONTAINS term]) AS relevance
        ORDER BY relevance DESC
        LIMIT $top_k
        RETURN s.text AS subject, r.type AS relation, o.text AS object, chunk_id, relevance AS score
        """
        
        results = self.neo4j.run_query(cypher, {"query_terms": query_terms, "top_k": top_k})
        
        # Format results
        triplets = []
        for record in results:
            subject = record["subject"]
            relation = record["relation"]
            obj = record["object"]
            chunk_id = record["chunk_id"]
            score = record["score"] / len(query_terms)  # Normalize score
            triplets.append((subject, relation, obj, chunk_id, score))
            
        return triplets

def hybrid_retrieve(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Retrieve chunks using hybrid vector and graph approach.
    
    Args:
        query: Query string
        top_k: Number of top results to retrieve
        
    Returns:
        List of (chunk_id, text, score) tuples
    """
    logger.info(f"Hybrid retrieval for query: '{query}'")
    
    # Get results from both retrievers
    vector_retriever = VectorRetriever()
    graph_retriever = GraphRetriever()
    
    vector_results = vector_retriever.retrieve_chunks(query, top_k)
    graph_results = graph_retriever.retrieve_chunks_by_terms(query, top_k)
    
    # Combine results
    combined_results = {}
    
    # Add vector results
    for chunk_id, text, score in vector_results:
        combined_results[chunk_id] = {
            "text": text,
            "vector_score": score,
            "graph_score": 0.0
        }
    
    # Add or update with graph results
    for chunk_id, text, score in graph_results:
        if chunk_id in combined_results:
            combined_results[chunk_id]["graph_score"] = score
        else:
            combined_results[chunk_id] = {
                "text": text,
                "vector_score": 0.0,
                "graph_score": score
            }
    
    # Calculate combined score (simple average)
    results = []
    for chunk_id, data in combined_results.items():
        combined_score = (data["vector_score"] + data["graph_score"]) / 2
        results.append((chunk_id, data["text"], combined_score))
    
    # Sort by combined score
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Return top_k results
    return results[:top_k]

def hybrid_retrieve_with_triplets(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve chunks and knowledge graph triplets.
    
    Args:
        query: Query string
        top_k: Number of top results to retrieve
        
    Returns:
        Dictionary with chunks and triplets
    """
    logger.info(f"Hybrid retrieval with triplets for query: '{query}'")
    
    # Get chunks
    chunks = hybrid_retrieve(query, top_k)
    
    # Get triplets
    graph_retriever = GraphRetriever()
    triplets = graph_retriever.retrieve_triplets(query, top_k)
    
    return {
        "chunks": chunks,
        "triplets": triplets
    } 