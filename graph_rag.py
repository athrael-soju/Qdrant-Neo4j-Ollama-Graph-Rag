from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from collections import defaultdict
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os
import concurrent.futures

# Import the processor factory
from processors.processor_factory import get_processor

# Get the selected processor components
processor = get_processor()
llm_parser = processor["llm_parser"]
embeddings = processor["embeddings"]
embeddings_batch = processor["embeddings_batch"]
graphrag_query = processor["graphrag_query"]
GraphComponents = processor["GraphComponents"]
single = processor["Single"]
MODEL_PROVIDER = processor["MODEL_PROVIDER"]
VECTOR_DIMENSION = processor["VECTOR_DIMENSION"]
LLM_MODEL = processor["LLM_MODEL"]
EMBEDDING_MODEL = processor["EMBEDDING_MODEL"]

def process_data_chunk(chunk_text):
    """Process a chunk of text to extract nodes and relationships."""
    prompt = f"Extract nodes and relationships from the following text:\n{chunk_text}"
    parsed_response = llm_parser(prompt).graph
    
    chunk_nodes = {}
    chunk_relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node
        relationship = entry.relationship

        # Add nodes to the dictionary with a unique ID
        if node not in chunk_nodes:
            chunk_nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in chunk_nodes:
            chunk_nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            chunk_relationships.append({
                "source": chunk_nodes[node],
                "target": chunk_nodes[target_node],
                "type": relationship
            })
            
    return chunk_nodes, chunk_relationships

def extract_graph_components_parallel(raw_data, chunk_size=5000, max_workers=4):
    """
    Extract graph components in parallel using multiple workers.
    
    Args:
        raw_data: The raw text data to process
        chunk_size: Size of each text chunk in characters
        max_workers: Maximum number of parallel workers
    """
    # Split the text into chunks of roughly equal size
    chunks = []
    words = raw_data.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
            
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    if len(chunks) == 1:
        # If there's only one chunk, no need for parallel processing
        return extract_graph_components(raw_data)
    
    # Process chunks in parallel
    all_nodes = {}
    all_relationships = []
    
    print(f"Processing {len(chunks)} chunks in parallel with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_data_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in concurrent.futures.as_completed(futures):
            chunk_index = futures[future]
            try:
                chunk_nodes, chunk_relationships = future.result()
                
                # Merge the chunk results
                node_id_mapping = {}  # Map chunk node IDs to global node IDs
                
                # Add new nodes to the global set
                for node_name, chunk_node_id in chunk_nodes.items():
                    if node_name not in all_nodes:
                        # New node - add to global nodes with the chunk's UUID
                        all_nodes[node_name] = chunk_node_id
                    else:
                        # Node already exists globally - map chunk node ID to global node ID
                        node_id_mapping[chunk_node_id] = all_nodes[node_name]
                
                # Adjust relationship IDs to use global node IDs
                for rel in chunk_relationships:
                    source_id = rel["source"]
                    target_id = rel["target"]
                    
                    # Use mapped IDs if they exist
                    if source_id in node_id_mapping:
                        source_id = node_id_mapping[source_id]
                    if target_id in node_id_mapping:
                        target_id = node_id_mapping[target_id]
                    
                    # Add to global relationships with updated IDs
                    all_relationships.append({
                        "source": source_id,
                        "target": target_id,
                        "type": rel["type"]
                    })
                    
                print(f"Processed chunk {chunk_index+1}/{len(chunks)}")
                
            except Exception as e:
                print(f"Error processing chunk {chunk_index}: {str(e)}")
    
    return all_nodes, all_relationships

def extract_graph_components(raw_data):
    """Extract graph components from text data"""
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed_response = llm_parser(prompt).graph  # Assuming this returns a list of dictionaries

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node  # Get target node if available
        relationship = entry.relationship  # Get relationship if available

        # Add nodes to the dictionary with a unique ID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship
            })

    return nodes, relationships

def ingest_to_neo4j(neo4j_driver, nodes, relationships, batch_size=100):
    """
    Ingest nodes and relationships into Neo4j using batch operations.
    """
    with neo4j_driver.session() as session:
        # Create nodes in Neo4j in batches
        node_items = list(nodes.items())
        for i in range(0, len(node_items), batch_size):
            batch = node_items[i:i+batch_size]
            # Create batch query
            query = "UNWIND $nodes as node CREATE (n:Entity {id: node.id, name: node.name})"
            session.run(
                query,
                nodes=[{"id": node_id, "name": name} for name, node_id in batch]
            )
            print(f"Created nodes batch {i//batch_size + 1}/{(len(node_items) + batch_size - 1)//batch_size}")

        # Create relationships in Neo4j in batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            # Create batch query
            query = """
            UNWIND $rels as rel
            MATCH (a:Entity {id: rel.source})
            MATCH (b:Entity {id: rel.target})
            CREATE (a)-[r:RELATIONSHIP {type: rel.type}]->(b)
            """
            session.run(
                query,
                rels=batch
            )
            print(f"Created relationships batch {i//batch_size + 1}/{(len(relationships) + batch_size - 1)//batch_size}")

    return nodes

def create_collection(client, collection_name, vector_dimension):
    """Create a Qdrant collection if it doesn't exist"""
    # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")

def ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping):
    """
    Ingest data to Qdrant with optimized batching.
    """
    # Split the text into meaningful chunks (paragraphs)
    paragraphs = [p for p in raw_data.split("\n") if p.strip()]
    
    print(f"Generating embeddings for {len(paragraphs)} paragraphs...")
    embeddings_result = embeddings_batch(paragraphs)
    
    # Prepare batch points
    points = []
    node_ids = list(node_id_mapping.values())
    
    # Use min to handle cases where we have more paragraphs than node IDs or vice versa
    for i in range(min(len(embeddings_result), len(node_ids))):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embeddings_result[i],
            "payload": {"id": node_ids[i], "text": paragraphs[i]}
        })
    
    # Use batch upsert for better performance
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size} to Qdrant")

def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    """Search for relevant nodes and relationships"""
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.search(query_vector=embeddings(query), top_k=10)
    
    return results

def fetch_related_graph(neo4j_driver, entity_ids):
    """Fetch the related graph from Neo4j for the given entity IDs"""
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_driver.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"]
            })
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"]
                })
    return subgraph

def format_graph_context(subgraph):
    """Format the subgraph as a context for RAG"""
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}

def graphRAG_run(graph_context, user_query, stream=True):
    """
    Run RAG with the graph context
    
    Args:
        graph_context: The graph context with nodes and edges
        user_query: The user's question
        stream: Whether to stream the response (default True)
        
    Returns:
        If stream=True: Iterator yielding chunks of the response
        If stream=False: Complete response message
    """
    return graphrag_query(graph_context, user_query, stream=stream)

def clear_data(neo4j_driver, qdrant_client, collection_name):
    """
    Clear all data from Neo4j and the specified Qdrant collection.
    """
    # Clear Neo4j data
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    # Clear Qdrant collection - recreate it
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
        
        # Recreate the empty collection
        vector_dimension = VECTOR_DIMENSION  # Using dimension from environment variables
        create_collection(qdrant_client, collection_name, vector_dimension)
        
    except Exception as e:
        print(f"Error clearing Qdrant collection: {str(e)}")
    
    return True

def initialize_clients():
    """Initialize Neo4j and Qdrant clients from environment variables"""
    load_dotenv('.env')
    
    # Get credentials from environment variables
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    collection_name = os.getenv("COLLECTION_NAME", "graphRAGstoreds")    
    
    # Initialize clients
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    qdrant_client = QdrantClient(
        host=qdrant_host,
        port=int(qdrant_port) if qdrant_port else None
    )
    
    return neo4j_driver, qdrant_client, collection_name 