"""
Qdrant vector database connection utilities for GraphRAG
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import os
import uuid
import hashlib

logger = logging.getLogger(__name__)

class QdrantConnection:
    """Manages connection to Qdrant for vector search in GraphRAG"""
    
    def __init__(self, url="http://localhost:6333", api_key=None):
        """
        Initialize Qdrant client
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication (if needed)
        """
        self.url = url
        self.api_key = api_key
        
        # Create client
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)
        
        logger.info(f"Initialized Qdrant connection to {url}")
    
    def test_connection(self):
        """
        Test the Qdrant connection
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # Get collection info to test connection
            collections = self.client.get_collections()
            logger.info(f"Qdrant connection test result: {collections}")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection failed: {str(e)}")
            return False
    
    def create_collection(self, collection_name="tokens", vector_size=768):
        """
        Create a collection for storing vector embeddings
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of embedding vectors
            
        Returns:
            bool: True if successful
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False
    
    def _string_to_uuid(self, string_id):
        """
        Convert string ID to a deterministic UUID
        
        Args:
            string_id: String ID
            
        Returns:
            uuid.UUID: UUID generated from string
        """
        # Create a deterministic UUID v5 using the string ID and a namespace
        namespace = uuid.UUID('bf8def8c-49bf-4e0d-93d5-1c1d1c6b6956')  # Fixed namespace for our app
        return uuid.uuid5(namespace, string_id)
    
    def upsert_vectors(self, collection_name, vectors, ids, metadata=None):
        """
        Insert or update vectors in collection
        
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            ids: List of IDs corresponding to vectors
            metadata: List of metadata dicts for each vector
            
        Returns:
            bool: True if successful
        """
        try:
            if metadata is None:
                metadata = [{} for _ in ids]
            
            points = []
            for i, (vec, id_str, meta) in enumerate(zip(vectors, ids, metadata)):
                # Convert string ID to UUID for Qdrant
                uuid_id = self._string_to_uuid(id_str)
                
                # Store original ID in metadata
                meta['original_id'] = id_str
                
                points.append(
                    models.PointStruct(
                        id=str(uuid_id),
                        vector=vec,
                        payload=meta
                    )
                )
            
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(vectors)} vectors to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            return False
    
    def search(self, collection_name, query_vector, limit=10, filter_condition=None):
        """
        Search for similar vectors
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_condition: Optional filter condition
            
        Returns:
            list: Search results with scores and payloads
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_condition
            )
            
            logger.info(f"Found {len(results)} results in collection '{collection_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def setup_collections(self, collection_name="tokens", vector_size=768, only_if_not_exists=True):
        """
        Set up necessary collections for GraphRAG
        
        Args:
            collection_name: Name of the collection to create
            vector_size: Size of the embedding vectors
            only_if_not_exists: Only create the collection if it doesn't exist
            
        Returns:
            bool: True if successful
        """
        try:
            # Check if collection exists
            if only_if_not_exists:
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name in collection_names:
                    logger.info(f"Collection '{collection_name}' already exists")
                    return True
            
            # Create token collection with the specified vector size
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up collections: {str(e)}")
            return False

    def clear_collection(self, collection_name="tokens"):
        """
        Clear a collection in Qdrant by deleting all points
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            bool: True if successful
        """
        try:
            # Check if collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False
                
            # Delete all points
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            
            # Recreate the collection
            self.setup_collections(collection_name=collection_name)
            logger.info(f"Recreated empty collection '{collection_name}'")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing collection '{collection_name}': {str(e)}")
            raise


# Default connection instance
default_connection = None

def get_connection(url="http://localhost:6333", api_key=None):
    """
    Get or create the default Qdrant connection
    
    Args:
        url: Qdrant server URL
        api_key: API key for authentication (if needed)
        
    Returns:
        QdrantConnection: A connection to Qdrant
    """
    global default_connection
    if default_connection is None:
        # Check for environment variables
        env_url = os.environ.get("QDRANT_URL", url)
        env_api_key = os.environ.get("QDRANT_API_KEY", api_key)
        
        default_connection = QdrantConnection(env_url, env_api_key)
    return default_connection


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Test connection
    conn = get_connection()
    if conn.test_connection():
        print("Qdrant connection successful!")
        conn.setup_collections()
    else:
        print("Failed to connect to Qdrant") 