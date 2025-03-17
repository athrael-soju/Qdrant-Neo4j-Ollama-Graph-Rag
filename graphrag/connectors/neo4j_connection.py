"""
Neo4j database connection utilities for GraphRAG
"""

from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Manages connection to Neo4j database for GraphRAG"""
    
    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "testpassword")):
        """
        Initialize Neo4j driver
        
        Args:
            uri: Neo4j connection URI
            auth: Authentication tuple (username, password)
        """
        self.uri = uri
        self.auth = auth
        self.driver = GraphDatabase.driver(uri, auth=auth)
        logger.info(f"Initialized Neo4j connection to {uri}")
        
    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()
            
    def test_connection(self):
        """
        Test the Neo4j connection
        
        Returns:
            bool: True if connection is successful
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Neo4j OK' AS status")
                status = result.single()["status"]
                logger.info(f"Neo4j connection test result: {status}")
                return status == "Neo4j OK"
        except Exception as e:
            logger.error(f"Neo4j connection failed: {str(e)}")
            return False
            
    def run_query(self, query, parameters=None):
        """
        Run a Cypher query
        
        Args:
            query: Cypher query string
            parameters: Dictionary of query parameters
            
        Returns:
            list: Query results
        """
        if parameters is None:
            parameters = {}
            
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return list(result.data())
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def setup_indexes(self):
        """
        Create necessary indexes for GraphRAG, including vector indexes.
        Uses new fulltext syntax (Neo4j 5.x) and a vector index if supported.
        """
        # Standard indexes that work on all Neo4j versions
        standard_indexes = [
            # Fulltext index for Term(text)
            """
            CREATE FULLTEXT INDEX TermIndex IF NOT EXISTS
            FOR (t:Term)
            ON EACH [t.text]
            """,
            # Fulltext index for Chunk(text)
            """
            CREATE FULLTEXT INDEX ChunkIndex IF NOT EXISTS
            FOR (c:Chunk)
            ON EACH [c.text]
            """,
            # Property indexes
            "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX term_text_idx IF NOT EXISTS FOR (t:Term) ON (t.text)",
        ]
        
        # Create standard indexes
        for query in standard_indexes:
            try:
                self.run_query(query)
                logger.info(f"Successfully executed index query: {query}")
            except Exception as e:
                logger.warning(f"Error creating index with query {query}: {str(e)}")

        # Try to check if vector indexes are supported
        try:
            version_info = self.run_query("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
            neo4j_version = None
            neo4j_edition = None
            
            for item in version_info:
                if item.get('name') == 'Neo4j Kernel':
                    neo4j_version = item.get('versions', [''])[0]
                    neo4j_edition = item.get('edition', '')
            
            logger.info(f"Detected Neo4j version {neo4j_version}, edition {neo4j_edition}")
            
            # Vector indexes require Neo4j 5.11+ Enterprise Edition
            supports_vector = False
            if neo4j_edition == 'enterprise' and neo4j_version:
                major, minor = map(int, neo4j_version.split('.')[:2])
                if major > 5 or (major == 5 and minor >= 11):
                    supports_vector = True
            
            if supports_vector:
                vector_index = """
                CREATE VECTOR INDEX vector_index_token IF NOT EXISTS
                FOR (t:Token)
                ON (t.embeddings)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
                try:
                    self.run_query(vector_index)
                    logger.info(f"Successfully created vector index")
                except Exception as e:
                    logger.warning(f"Error creating vector index: {str(e)}")
            else:
                logger.warning("Vector indexes not supported in this Neo4j version/edition. Skipping vector index creation.")
                logger.info("Vector indexes require Neo4j 5.11+ Enterprise Edition with vector plugin installed.")
        except Exception as e:
            logger.warning(f"Could not determine Neo4j version/vector support: {str(e)}")
            logger.info("Skipping vector index creation")


# Default connection instance
default_connection = None

def get_connection(uri="bolt://localhost:7687", auth=("neo4j", "testpassword")):
    """
    Get or create the default Neo4j connection
    
    Args:
        uri: Neo4j connection URI
        auth: Authentication tuple (username, password)
        
    Returns:
        Neo4jConnection: A connection to Neo4j
    """
    global default_connection
    if default_connection is None:
        default_connection = Neo4jConnection(uri, auth)
    return default_connection


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Test connection
    conn = get_connection()
    if conn.test_connection():
        print("Neo4j connection successful!")
    else:
        print("Failed to connect to Neo4j")
