"""
Retrieval utilities for GraphRAG
"""

import re
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from graphrag.neo4j_connection import get_connection as get_neo4j_connection
from graphrag.qdrant_connection import get_connection as get_qdrant_connection

# Initialize logger
logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = 'intfloat/e5-base-v2'

class Retriever:
    """Base class for retrieval in GraphRAG"""
    
    def __init__(self, neo4j_conn=None, qdrant_conn=None):
        """Initialize Retriever
        
        Args:
            neo4j_conn: Neo4j connection instance
            qdrant_conn: Qdrant connection instance
        """
        self.neo4j = neo4j_conn or get_neo4j_connection()
        self.qdrant = qdrant_conn or get_qdrant_connection()
        logger.info("Initialized Retriever")

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            List[Tuple[str, str, float]]: List of (chunk_id, chunk_text, score) tuples
        """
        raise NotImplementedError("Subclasses must implement retrieve_chunks")

    def _fetch_chunk_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Fetch text for a list of chunk IDs
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Dict[str, str]: Dictionary mapping chunk_id to chunk_text
        """
        if not chunk_ids:
            return {}
            
        result = self.neo4j.run_query(
            """
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            RETURN c.id AS id, c.text AS text
            """,
            {"chunk_ids": chunk_ids}
        )
        
        return {r["id"]: r["text"] for r in result}


class VectorRetriever(Retriever):
    """Vector-based retrieval for GraphRAG using Qdrant vector database"""
    
    def __init__(self, neo4j_conn=None, qdrant_conn=None, embedding_model=DEFAULT_EMBEDDING_MODEL):
        """Initialize VectorRetriever
        
        Args:
            neo4j_conn: Neo4j connection instance
            qdrant_conn: Qdrant connection instance
            embedding_model: Name or path of the embedding model
        """
        super().__init__(neo4j_conn, qdrant_conn)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("Embedding model loaded successfully")
        
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query
        
        Args:
            query: Query string
            
        Returns:
            np.ndarray: Query embedding
        """
        # Add logging to debug query embedding
        logger.info(f"Embedding query: {query}")
        
        try:
            # E5 models expect "query: " prefix
            # Check if model name contains 'e5'
            model_name = str(self.embedding_model).lower()
            if 'e5' in model_name:
                query = f"query: {query}"
                logger.info(f"Using E5 prefix: {query}")
            
            # Encode the query and normalize
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            return query_embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            # Return a zero vector as fallback
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return np.zeros(dim)
    
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve chunks using vector search with Qdrant
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            List[Tuple[str, str, float]]: List of (chunk_id, chunk_text, score) tuples
        """
        # Embed query
        logger.info(f"Vector retrieval for query: {query}")
        query_embedding = self.embed_query(query)
        
        # Search Qdrant
        try:
            logger.info(f"Searching Qdrant collection 'tokens' with limit {top_k}")
            search_results = self.qdrant.search(
                collection_name="tokens",
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            logger.info(f"Qdrant returned {len(search_results)} results")
            
            # Format results
            results = []
            for hit in search_results:
                # Get original chunk ID from metadata
                if hit.payload and 'original_id' in hit.payload:
                    chunk_id = hit.payload['original_id']
                else:
                    chunk_id = hit.id  # Fall back to UUID if original ID not found
                    
                logger.info(f"Processing hit with ID: {chunk_id}, score: {hit.score}")
                
                # If text is in metadata, use it directly, otherwise fetch from Neo4j
                if hit.payload and "text" in hit.payload:
                    chunk_text = hit.payload["text"]
                    # If text was truncated in metadata, fetch full text from Neo4j
                    if len(chunk_text) >= 990:  # Likely truncated
                        logger.info(f"Text truncated in metadata, fetching from Neo4j")
                        chunk_text = self._fetch_chunk_text(chunk_id)
                else:
                    logger.info(f"No text in metadata, fetching from Neo4j")
                    chunk_text = self._fetch_chunk_text(chunk_id)
                    
                results.append((chunk_id, chunk_text, hit.score))
            
            logger.info(f"Returning {len(results)} results from vector search")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector retrieval: {str(e)}")
            return []
    
    def _fetch_chunk_text(self, chunk_id: str) -> str:
        """Fetch text for a single chunk ID from Neo4j
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            str: Chunk text
        """
        query = "MATCH (c:Chunk {id: $chunk_id}) RETURN c.text as text"
        results = self.neo4j.run_query(query, {"chunk_id": chunk_id})
        
        if results and results[0].get("text"):
            return results[0]["text"]
        else:
            logger.warning(f"No text found for chunk {chunk_id}")
            return ""


class GraphRetriever(Retriever):
    """Graph-based retrieval for GraphRAG"""
    
    def __init__(self, neo4j_conn=None):
        """Initialize GraphRetriever
        
        Args:
            neo4j_conn: Neo4j connection instance
        """
        super().__init__(neo4j_conn)
        
    def term_search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for chunks containing query terms
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            List[Tuple[str, str, float]]: List of (chunk_id, chunk_text, score) tuples
        """
        logger.info(f"Term-based search for query: {query}")
        
        # Use Neo4j's fulltext index to search for terms
        result = self.neo4j.run_query(
            """
            CALL db.index.fulltext.queryNodes('TermIndex', $query) YIELD node, score
            MATCH (c:Chunk)-[:HAS_TERM]->(node)
            WITH c, SUM(score) AS relevance
            ORDER BY relevance DESC
            LIMIT $k
            RETURN c.id AS id, relevance AS score
            """, 
            {"query": query, "k": top_k}
        )
        
        # Get chunk IDs and scores
        chunk_results = [(r["id"], r["score"]) for r in result]
        
        # Fetch chunk texts
        chunk_ids = [cid for cid, _ in chunk_results]
        chunk_texts = self._fetch_chunk_texts(chunk_ids)
        
        # Create result tuples: (chunk_id, chunk_text, score)
        results = [(cid, chunk_texts.get(cid, ""), score) for cid, score in chunk_results]
        
        logger.info(f"Retrieved {len(results)} chunks via term search")
        return results
        
    def entity_search(self, entity_name: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for chunks mentioning a specific entity
        
        Args:
            entity_name: Entity name to search for
            top_k: Number of top chunks to retrieve
            
        Returns:
            List[Tuple[str, str, float]]: List of (chunk_id, chunk_text, score) tuples
        """
        logger.info(f"Entity search for: {entity_name}")
        
        # Find chunks mentioning the entity
        result = self.neo4j.run_query(
            """
            MATCH (e:Entity {name: $entity_name})<-[:MENTIONS_ENTITY]-(c:Chunk)
            RETURN DISTINCT c.id AS id
            LIMIT $k
            """,
            {"entity_name": entity_name, "k": top_k}
        )
        
        # Get chunk IDs
        chunk_ids = [r["id"] for r in result]
        
        # Fetch chunk texts
        chunk_texts = self._fetch_chunk_texts(chunk_ids)
        
        # Create result tuples with a default score of 1.0
        results = [(cid, chunk_texts.get(cid, ""), 1.0) for cid in chunk_ids]
        
        logger.info(f"Retrieved {len(results)} chunks mentioning entity: {entity_name}")
        return results
        
    def relationship_search(self, entity_name: str, relation_keyword: str = "") -> List[Tuple[str, str, str, str, str]]:
        """Search for relationships involving an entity
        
        Args:
            entity_name: Subject entity name
            relation_keyword: Optional keyword to filter relation types
            
        Returns:
            List[Tuple[str, str, str, str, str]]: List of (subject, relation, object, chunk_id, chunk_text) tuples
        """
        logger.info(f"Relationship search for entity: {entity_name}, relation: {relation_keyword}")
        
        # Prepare relation pattern for filtering
        rel_pattern = f".*{relation_keyword}.*" if relation_keyword else None
        
        # Query Neo4j for relationships
        if rel_pattern:
            result = self.neo4j.run_query(
                """
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE s.name = $name AND type(r) =~ $rel
                RETURN s.name AS subject, type(r) AS relation, o.name AS object, r.source AS chunk_id
                """,
                {"name": entity_name, "rel": rel_pattern}
            )
        else:
            result = self.neo4j.run_query(
                """
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE s.name = $name
                RETURN s.name AS subject, type(r) AS relation, o.name AS object, r.source AS chunk_id
                """,
                {"name": entity_name}
            )
            
        # Get unique chunk IDs
        chunk_ids = list({r["chunk_id"] for r in result if r["chunk_id"]})
        
        # Fetch chunk texts
        chunk_texts = self._fetch_chunk_texts(chunk_ids)
        
        # Create result tuples
        results = []
        for row in result:
            subj = row["subject"]
            rel = row["relation"]
            obj = row["object"]
            chunk_id = row["chunk_id"]
            chunk_text = chunk_texts.get(chunk_id, "") if chunk_id else ""
            
            results.append((subj, rel, obj, chunk_id, chunk_text))
            
        logger.info(f"Found {len(results)} relationships for entity: {entity_name}")
        return results
        
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve relevant chunks using graph-based methods
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            List[Tuple[str, str, float]]: List of (chunk_id, chunk_text, score) tuples
        """
        logger.info(f"Graph-based retrieval for query: {query}")
        
        # First, try term-based search
        term_results = self.term_search(query, top_k)
        
        # Try to identify entities in the query (simple heuristic for capitalized phrases)
        entity_candidates = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
        
        entity_results = []
        for entity in entity_candidates:
            # Check if entity exists in our graph
            exists = self.neo4j.run_query(
                "MATCH (e:Entity {name: $name}) RETURN count(e) > 0 AS exists",
                {"name": entity}
            )
            
            if exists and exists[0]["exists"]:
                # Entity exists, get chunks mentioning it
                entity_chunks = self.entity_search(entity, top_k)
                entity_results.extend(entity_chunks)
                
        # Combine results, removing duplicates and keeping highest scores
        combined = {}
        
        # Add term results
        for chunk_id, text, score in term_results:
            combined[chunk_id] = (text, score)
            
        # Add entity results, potentially updating scores
        for chunk_id, text, score in entity_results:
            if chunk_id in combined:
                # If chunk already exists, use higher score
                _, existing_score = combined[chunk_id]
                combined[chunk_id] = (text, max(existing_score, score))
            else:
                combined[chunk_id] = (text, score)
                
        # Convert back to list of tuples
        results = [(cid, text, score) for cid, (text, score) in combined.items()]
        
        # Sort by score and limit to top_k
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:top_k]
        
        logger.info(f"Retrieved {len(results)} chunks via graph-based methods")
        return results


class HybridRetriever(Retriever):
    """Hybrid retrieval combining vector and graph-based approaches"""
    
    def __init__(self, neo4j_conn=None, qdrant_conn=None, embedding_model=DEFAULT_EMBEDDING_MODEL):
        """Initialize HybridRetriever
        
        Args:
            neo4j_conn: Neo4j connection instance
            qdrant_conn: Qdrant connection instance
            embedding_model: Name or path of the embedding model
        """
        super().__init__(neo4j_conn, qdrant_conn)
        self.vector_retriever = VectorRetriever(neo4j_conn, qdrant_conn, embedding_model)
        self.graph_retriever = GraphRetriever(neo4j_conn)
        logger.info("Initialized HybridRetriever")
        
    def retrieve_chunks(self, query: str, top_k: int = 5, 
                     vector_weight: float = 0.5) -> List[Tuple[str, str, float]]:
        """Retrieve chunks using hybrid approach
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            vector_weight: Weight for vector scores (0.0-1.0)
            
        Returns:
            List[Tuple[str, str, float]]: List of (chunk_id, chunk_text, score) tuples
        """
        # Get results from both retrievers
        vector_results = self.vector_retriever.retrieve_chunks(query, top_k=top_k*2)
        graph_results = self.graph_retriever.retrieve_chunks(query, top_k=top_k*2)
        
        # Normalize scores separately
        def normalize_scores(results):
            if not results:
                return []
            
            # Extract scores
            scores = [score for _, _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            
            # Avoid division by zero
            if max_score == min_score:
                normalized_results = [(cid, text, 1.0) for cid, text, _ in results]
            else:
                # Normalize to [0,1]
                normalized_results = [
                    (cid, text, (score - min_score) / (max_score - min_score))
                    for cid, text, score in results
                ]
                
            return normalized_results
        
        vector_results = normalize_scores(vector_results)
        graph_results = normalize_scores(graph_results)
        
        # Combine results with weighted scores
        result_dict = {}
        
        for chunk_id, text, score in vector_results:
            result_dict[chunk_id] = {
                "text": text,
                "vector_score": score,
                "graph_score": 0.0
            }
            
        for chunk_id, text, score in graph_results:
            if chunk_id in result_dict:
                result_dict[chunk_id]["graph_score"] = score
            else:
                result_dict[chunk_id] = {
                    "text": text,
                    "vector_score": 0.0,
                    "graph_score": score
                }
                
        # Calculate combined scores
        combined_results = []
        for chunk_id, data in result_dict.items():
            combined_score = (
                vector_weight * data["vector_score"] + 
                (1 - vector_weight) * data["graph_score"]
            )
            combined_results.append((chunk_id, data["text"], combined_score))
            
        # Sort by combined score and take top_k
        combined_results.sort(key=lambda x: x[2], reverse=True)
        return combined_results[:top_k]

    def retrieve_with_triplets(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Enhanced retrieval that includes triplets from relevant chunks
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dict[str, Any]: Dictionary with chunks and triplets keys
        """
        # Get relevant chunks
        chunk_results = self.retrieve_chunks(query, top_k)
        chunk_ids = [cid for cid, _, _ in chunk_results]
        
        # Extract entities mentioned in the query
        entity_candidates = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
        
        # Find direct facts about entities in the query
        triplet_results = []
        for entity in entity_candidates:
            relation_keyword = ""  # Get all relations
            triplets = self.graph_retriever.relationship_search(entity, relation_keyword)
            triplet_results.extend(triplets)
            
        return {
            "chunks": chunk_results,
            "triplets": triplet_results
        }

# Standalone function implementations

def vector_search(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """Perform vector search for a query"""
    retriever = VectorRetriever()
    return retriever.retrieve_chunks(query, top_k)

def term_search(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """Perform term-based search for a query"""
    retriever = GraphRetriever()
    return retriever.term_search(query, top_k)

def relationship_search(entity_name: str, relation_keyword: str = "") -> List[Tuple[str, str, str, str, str]]:
    """Search for relationships involving an entity"""
    retriever = GraphRetriever()
    return retriever.relationship_search(entity_name, relation_keyword)

def hybrid_retrieve(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """Retrieve chunks using hybrid approach"""
    retriever = HybridRetriever()
    return retriever.retrieve_chunks(query, top_k)

def hybrid_retrieve_with_triplets(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve chunks and related triplets using hybrid approach"""
    retriever = HybridRetriever()
    return retriever.retrieve_with_triplets(query, top_k)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Demo with example query
    example_query = "What type of business is Hugging Face?"
    
    print("\nHybrid retrieval example:")
    results = hybrid_retrieve(example_query, top_k=3)
    print(f"Retrieved {len(results)} chunks:")
    for cid, text, score in results:
        print(f"Chunk {cid} (score: {score:.3f}):")
        print(f"  {text[:100]}..." if len(text) > 100 else text)
        print() 