from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from collections import defaultdict
from functools import lru_cache
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os
import concurrent.futures
import time
import requests
import json
import traceback

# Import our processor factory
from processors.processor_factory import ProcessorFactory

# Define data models
class single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[single]

# Get processor type from environment variables
def get_processor_type():
    """Get processor type from environment variables"""
    return os.getenv("PROCESSOR_TYPE", "ollama").lower()

# Get processor configuration from environment variables
def get_processor_config():
    """Get processor configuration from environment variables"""
    return {
        "model": os.getenv("PROCESSOR_MODEL", "qwen2.5:3b"),
        "host": os.getenv("OLLAMA_HOST", "localhost"),
        "port": int(os.getenv("OLLAMA_PORT", "11434")),
    }

# Process data chunk using selected processor
def process_data_chunk(chunk_text, processor_type=None, processor_config=None):
    """
    Process a chunk of text to extract nodes and relationships.
    
    Args:
        chunk_text: Text to process
        processor_type: Type of processor to use (default from environment)
        processor_config: Configuration for the processor (default from environment)
        
    Returns:
        Tuple of (nodes, relationships)
    """
    if processor_type is None:
        processor_type = get_processor_type()
        
    if processor_config is None:
        processor_config = get_processor_config()
    
    # Create processor
    processor = ProcessorFactory.create_processor(processor_type, processor_config)
    
    # Process the chunk
    return processor.process_chunk(chunk_text)

def extract_graph_components_parallel(raw_data, chunk_size=5000, max_workers=8, processor_type=None, processor_config=None):
    """
    Extract graph components in parallel using multiple workers.
    
    Args:
        raw_data: Raw text data to process
        chunk_size: Size of chunks to process
        max_workers: Maximum number of worker threads
        processor_type: Type of processor to use
        processor_config: Configuration for the processor
        
    Returns:
        Tuple of (nodes, relationships)
    """
    # Process text in chunks to handle large documents
    chunks = [raw_data[i:i+chunk_size] for i in range(0, len(raw_data), chunk_size)]
    all_nodes = {}
    all_relationships = []
    
    # Set processor type and config from environment if not provided
    if processor_type is None:
        processor_type = get_processor_type()
        
    if processor_config is None:
        processor_config = get_processor_config()
    
    print(f"Using processor: {processor_type} with config: {processor_config}")
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        futures = [executor.submit(process_data_chunk, chunk, processor_type, processor_config) for chunk in chunks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_nodes, chunk_relationships = future.result()
                
                # Merge results
                for node_name, node_id in chunk_nodes.items():
                    if node_name not in all_nodes:
                        all_nodes[node_name] = node_id
                
                all_relationships.extend(chunk_relationships)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    return all_nodes, all_relationships

def extract_graph_components(raw_data, processor_type=None, processor_config=None):
    """
    Extract graph components from raw text.
    
    Args:
        raw_data: Raw text data to process
        processor_type: Type of processor to use
        processor_config: Configuration for the processor
        
    Returns:
        Tuple of (nodes, relationships)
    """
    # Set processor type and config from environment if not provided
    if processor_type is None:
        processor_type = get_processor_type()
        
    if processor_config is None:
        processor_config = get_processor_config()
    
    print(f"Using processor: {processor_type} with config: {processor_config}")
    
    # Create processor
    processor = ProcessorFactory.create_processor(processor_type, processor_config)
    
    # Process the data
    return processor.extract_components(raw_data)

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

def openai_embeddings(text):
    """Generate embeddings for a single text string using OpenAI"""
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding

def local_embeddings(text):
    """Generate embeddings using a local model via Ollama API"""
    # Default to localhost Ollama server
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))
    
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("embedding", [])
        else:
            print(f"Error getting local embeddings: {response.status_code}")
            # Fallback to a dummy embedding if needed
            return [0.0] * vector_dimension
    except Exception as e:
        print(f"Exception in local embeddings: {str(e)}")
        return [0.0] * vector_dimension

def get_embeddings(text, use_local=False):
    """Get embeddings using either OpenAI or local model based on configuration"""
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true" or use_local
    
    if use_local_embeddings:
        return local_embeddings(text)
    else:
        return openai_embeddings(text)

def openai_embeddings_batch(texts, batch_size=20):
    """
    Get embeddings for a list of texts in batches to reduce API calls.
    """
    client = OpenAI()
    vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))
    all_embeddings = []
    
    # Process in batches to avoid rate limits and improve performance
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            # Return empty embeddings for this batch
            all_embeddings.extend([[0.0] * vector_dimension] * len(batch))
    
    return all_embeddings

def local_embeddings_batch(texts, batch_size=20):
    """
    Get embeddings for a list of texts in batches using local Ollama model.
    """
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch:
            try:
                response = requests.post(
                    url,
                    json={"model": model, "prompt": text}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_embeddings.append(result.get("embedding", [0.0] * vector_dimension))
                else:
                    print(f"Error getting local embeddings: {response.status_code}")
                    batch_embeddings.append([0.0] * vector_dimension)
            except Exception as e:
                print(f"Exception in local embeddings: {str(e)}")
                batch_embeddings.append([0.0] * vector_dimension)
        
        all_embeddings.extend(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    return all_embeddings

def get_embeddings_batch(texts, use_local=False):
    """Get batch embeddings using either OpenAI or local model based on configuration"""
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true" or use_local
    
    if use_local_embeddings:
        return local_embeddings_batch(texts)
    else:
        return openai_embeddings_batch(texts)

def ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping):
    """
    Ingest data to Qdrant with optimized batching.
    """
    # Split the text into meaningful chunks (paragraphs)
    paragraphs = [p for p in raw_data.split("\n") if p.strip()]
    
    # Get the local embeddings setting
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    
    print(f"Generating embeddings for {len(paragraphs)} paragraphs...")
    print(f"Using {'local' if use_local_embeddings else 'OpenAI'} embeddings...")
    embeddings = get_embeddings_batch(paragraphs, use_local=use_local_embeddings)
    
    # Prepare batch points
    points = []
    node_ids = list(node_id_mapping.values())
    
    # Use min to handle cases where we have more paragraphs than node IDs or vice versa
    for i in range(min(len(embeddings), len(node_ids))):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embeddings[i],
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

def retriever_search(neo4j_driver, qdrant_client, collection_name, query, use_local=False):
    """Search for relevant nodes and relationships"""
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    # Use appropriate embeddings based on configuration
    query_vector = get_embeddings(query, use_local)
    
    try:
        # Try to get more results to ensure we find something
        results = retriever.search(query_vector=query_vector, top_k=15)
        
        # Convert results to list if needed - handle different return types
        if hasattr(results, '__iter__') and not isinstance(results, list):
            # It's an iterable but not a list (like RetrieverResult)
            results_list = list(results)
            print(f"Found {len(results_list)} results in vector search")
        elif isinstance(results, list):
            results_list = results
            print(f"Found {len(results_list)} results in vector search")
        else:
            # Single result or unknown type
            print("Vector search returned results of type:", type(results))
            if results:
                results_list = [results]
            else:
                results_list = []
        
        if not results_list:
            # If no results found, try direct entity search
            print("No results found. Attempting direct entity search...")
            lower_query = query.lower()
            # Check if important entities like "Tesla" are in query and try direct search
            key_entities = ["tesla", "musk", "spacex", "ai", "electric", "car", "vehicle", "battery", "solar"]
            search_terms = []
            
            for entity in key_entities:
                if entity in lower_query:
                    search_terms.append(entity)
            
            if search_terms:
                # Try direct entity search in Neo4j
                entity_search_term = search_terms[0]  # Take first match
                print(f"Trying direct entity search with: {entity_search_term}")
                
                with neo4j_driver.session() as session:
                    # Check which properties exist on nodes
                    property_check = session.run(
                        """
                        MATCH (n) WHERE n.name IS NOT NULL
                        RETURN keys(n) as properties LIMIT 1
                        """
                    )
                    properties = []
                    for record in property_check:
                        if 'properties' in record:
                            properties = record['properties']
                    
                    print(f"Node properties available: {properties}")
                    
                    # Build query based on available properties
                    query_conditions = []
                    if 'name' in properties:
                        query_conditions.append("toLower(n.name) CONTAINS $search_term")
                    if 'description' in properties:
                        query_conditions.append("toLower(n.description) CONTAINS $search_term")
                    if 'text' in properties:
                        query_conditions.append("toLower(n.text) CONTAINS $search_term")
                    if 'content' in properties:
                        query_conditions.append("toLower(n.content) CONTAINS $search_term")
                    
                    if not query_conditions:
                        # Fallback to just searching by name
                        query_conditions = ["toLower(n.name) CONTAINS $search_term"]
                    
                    query_string = f"""
                    MATCH (n)
                    WHERE {" OR ".join(query_conditions)}
                    RETURN n.id as id, n.name as name
                    LIMIT 10
                    """
                    
                    result = session.run(query_string, search_term=entity_search_term)
                    direct_results = [(record["id"], record) for record in result]
                    
                    if direct_results:
                        print(f"Found {len(direct_results)} results through direct entity search")
                        return direct_results
        
        return results_list
    except Exception as e:
        print(f"Error in retriever search: {str(e)}")
        traceback.print_exc()  # Print full stack trace
        return []

def debug_retriever_search(neo4j_driver, qdrant_client, collection_name, query_text):
    """Debug function to diagnose search issues"""
    print(f"\n--- Debugging Search for: '{query_text}' ---")
    
    # Check vector dimensions in environment and Qdrant
    vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))
    print(f"Configured vector dimension: {vector_dimension}")
    
    try:
        # Check if collection exists and get its configuration
        collection_info = qdrant_client.get_collection(collection_name)
        actual_vector_dimension = collection_info.config.params.vectors.size
        print(f"Collection '{collection_name}' exists. Vector dimension: {actual_vector_dimension}")
        
        if vector_dimension != actual_vector_dimension:
            print(f"WARNING: Vector dimension mismatch between environment ({vector_dimension}) and Qdrant ({actual_vector_dimension})")
    except Exception as e:
        print(f"Error checking collection: {str(e)}")
    
    # Check Neo4j connectivity
    try:
        with neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            for record in result:
                node_count = record["count"]
                print(f"Neo4j database contains {node_count} nodes")
                
            # Try a sample query for Tesla
            result = session.run(
                """
                MATCH (n) 
                WHERE toLower(n.name) CONTAINS 'tesla' OR toLower(n.description) CONTAINS 'tesla'
                RETURN count(n) as count
                """
            )
            for record in result:
                tesla_count = record["count"]
                print(f"Neo4j database contains {tesla_count} nodes that mention 'tesla'")
                
            if tesla_count > 0:
                # Get sample nodes
                result = session.run(
                    """
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS 'tesla' OR toLower(n.description) CONTAINS 'tesla'
                    RETURN n.name as name, n.description as description
                    LIMIT 3
                    """
                )
                print("\nSample Tesla-related nodes:")
                for i, record in enumerate(result):
                    print(f"{i+1}. Name: {record['name']}")
                    desc = record['description']
                    if desc and len(desc) > 100:
                        desc = desc[:100] + "..."
                    print(f"   Description: {desc}")
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")
    
    # Test vector search directly
    try:
        use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        query_vector = get_embeddings(query_text, use_local)
        print(f"\nGenerated query vector with length: {len(query_vector)}")
        
        # Try direct search in Qdrant
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )
        
        print(f"Direct Qdrant search returned {len(search_result)} results")
        
        if search_result:
            print("\nTop search results:")
            for i, result in enumerate(search_result):
                print(f"{i+1}. ID: {result.id}, Score: {result.score}")
                if hasattr(result.payload, 'item'):
                    print(f"   Content: {result.payload.get('content', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
    
    print("--- End Debug Info ---\n")

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

def local_llm_generation(prompt, model="llama3"):
    """Generate text using a local model via Ollama API"""
    import requests
    import json
    
    # Default to localhost Ollama server
    url = os.getenv("OLLAMA_URL_CHAT", "http://localhost:11434/api/chat")
    model_name = os.getenv("OLLAMA_LLM_MODEL", model)
    json_mode = os.getenv("OLLAMA_JSON_MODE", "true").lower() == "true"
    
    try:
        # Create request payload
        payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Provide the answer for the following question:"},
                    {"role": "user", "content": prompt}
            ],
            "stream": False  # Ensure we don't get streaming responses
        }
        
        # Add options if configured
        if json_mode:
            if "options" not in payload:
                payload["options"] = {}
            payload["options"]["temperature"] = 0.1
        
        response = requests.post(
            url,
            json=payload
        )
        
        if response.status_code == 200:
            try:
            result = response.json()
            return result.get("message", {}).get("content", "No response from model")
            except json.JSONDecodeError as e:
                # Handle streaming responses (multiple JSON objects)
                content_parts = []
                raw_text = response.text
                
                # Try to parse as multiple JSON objects (one per line)
                for line in raw_text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        json_obj = json.loads(line)
                        # Extract content from each message
                        if "message" in json_obj and "content" in json_obj["message"]:
                            content_parts.append(json_obj["message"]["content"])
                    except json.JSONDecodeError:
                        pass
                
                # If we successfully parsed any content, return it
                if content_parts:
                    return "".join(content_parts)
                
                # Otherwise, return an error with truncated raw output
                print(f"JSON parsing error: {str(e)}")
                if len(raw_text) > 500:
                    return f"Error parsing model response. Raw output (truncated): {raw_text[:500]}..."
                return f"Error parsing model response. Raw output: {raw_text}"
        else:
            print(f"Error getting response from local LLM: {response.status_code}")
            print(f"Response: {response.text}")
            return f"Error: Status code {response.status_code}"
    except Exception as e:
        print(f"Exception calling local LLM: {str(e)}")
        # Try fallback to completion API if chat fails
        try:
            completion_url = os.getenv("OLLAMA_URL_COMPLETION", "http://localhost:11434/api/generate")
            completion_response = requests.post(
                completion_url,
                json={"model": model_name, "prompt": prompt, "stream": False}
            )
            if completion_response.status_code == 200:
                result = completion_response.json()
                return result.get("response", f"Error: {str(e)}")
            else:
        return f"Error: {str(e)}"
        except Exception as fallback_error:
            return f"Error: {str(e)}. Fallback also failed: {str(fallback_error)}"

def graphRAG_run(graph_context, user_query):
    """Run RAG with the graph context"""
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    # Check if using local LLM
    use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    
    if use_local_llm:
        # Use local LLM via Ollama
        return local_llm_generation(prompt)
    else:
        # Use OpenAI
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Provide the answer for the following question:"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message
        
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

def recreate_qdrant_collection(qdrant_client, collection_name):
    """
    Recreate the Qdrant collection with the correct vector dimensions.
    This is useful when you need to reset or update dimensions without clearing Neo4j data.
    
    Args:
        qdrant_client: Qdrant client
        collection_name: Name of the collection to recreate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete the collection if it exists
        try:
            qdrant_client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            print(f"Note: Collection may not exist yet or could not be deleted: {e}")
        
        # Create the collection with the correct dimensions
        vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))  # Get from environment
        create_collection(qdrant_client, collection_name, vector_dimension)
        return True
    except Exception as e:
        print(f"Error recreating Qdrant collection: {str(e)}")
        return False

def clear_data(neo4j_driver, qdrant_client, collection_name):
    """
    Clear all data from Neo4j and the specified Qdrant collection.
    """
    # Clear Neo4j data
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    # Recreate the Qdrant collection
    success = recreate_qdrant_collection(qdrant_client, collection_name)
    
    return success

def initialize_clients():
    """Initialize Neo4j and Qdrant clients from environment variables"""
    load_dotenv('.env')
    
    # Get credentials from environment variables
    qdrant_key = os.getenv("QDRANT_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USER")  # Match the variable name used in main.py
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    # Initialize clients
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key
    )
    
    return neo4j_driver, qdrant_client 