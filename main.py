#!/usr/bin/env python3
"""
GraphRAG Interactive Console

This script provides an interactive console interface for the GraphRAG system.
"""

import time
import os
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from graph_rag import (
    extract_graph_components,
    extract_graph_components_parallel,
    ingest_to_neo4j,
    create_collection,
    ingest_to_qdrant,
    retriever_search,
    fetch_related_graph,
    format_graph_context,
    graphRAG_run,
    clear_data
)

def initialize_clients():
    """
    Initialize Neo4j and Qdrant clients.
    
    Returns:
        Tuple of (neo4j_driver, qdrant_client)
    """
    # Get Neo4j configuration
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Get Qdrant configuration
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Initialize Neo4j driver
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    return neo4j_driver, qdrant_client

# Function to handle a complete question-answering session with follow-ups
def ask_question_session(neo4j_driver, qdrant_client, collection_name):
    """
    Start an interactive question answering session that allows follow-up questions.
    
    Args:
        neo4j_driver: Neo4j driver
        qdrant_client: Qdrant client
        collection_name: Collection name for vector search
    """
    # Get model information
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    llm_model = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    use_local_llm = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    
    print("\nAsk a Question")
    print("-" * 30)
    print(f"Using: Embeddings: {embedding_model}, LLM: {llm_model}")
    print("(Type 'exit', 'quit', or 'menu' to return to the main menu)")
    print("(Type 'new: your question' to search for new graph context)")
    print("(Type 'debug: your question' to run diagnostic search tests)")
    
    # Keep track of conversation context
    conversation_history = []
    graph_context = None
    
    while True:
        query = input("\nEnter your question: ")
        
        # Check for exit commands
        if query.strip().lower() in ['exit', 'quit', 'menu']:
            print("Returning to main menu...")
            return
            
        if not query.strip():
            print("No question entered. Please try again.")
            continue
        
        # Check for debug command
        if query.lower().startswith("debug:"):
            debug_query = query[6:].strip()  # Remove the "debug:" prefix
            from graph_rag import debug_retriever_search
            debug_retriever_search(neo4j_driver, qdrant_client, collection_name, debug_query)
            continue
            
        start_time = time.time()
        
        # For first question or if explicitly requesting a new search
        if not graph_context or query.lower().startswith("new:"):
            if query.lower().startswith("new:"):
                query = query[4:].strip()  # Remove the "new:" prefix
                print(f"Searching for new context with: '{query}'")
                
            print("Retrieving relevant graph context...")
            # Always use local embeddings (set in environment variables)
            use_local_embeddings = True 
            retriever_results = retriever_search(neo4j_driver, qdrant_client, collection_name, query, use_local=use_local_embeddings)
            
            # Convert retriever_results to a subgraph and format it for RAG
            entity_ids = [result[0] for result in retriever_results]
            subgraph = fetch_related_graph(neo4j_driver, entity_ids)
            graph_context = format_graph_context(subgraph)
            
            if len(graph_context['nodes']) == 0:
                print("No relevant graph nodes found. The answer will be based on the model's knowledge only.")
                # Offer to run debug
                if "tesla" in query.lower() or "electric" in query.lower() or "musk" in query.lower():
                    run_debug = input("Would you like to run search diagnostics? (y/n): ").lower()
                    if run_debug == 'y':
                        from graph_rag import debug_retriever_search
                        debug_retriever_search(neo4j_driver, qdrant_client, collection_name, query)
            else:
                print(f"Found {len(graph_context['nodes'])} nodes and {len(graph_context['edges'])} edges")
            
        # Add query to conversation history
        conversation_history.append(query)
        
        # If we have a conversation history, include it in the final query
        if len(conversation_history) > 1:
            print("Using conversation history for context...")
            conversation_context = "\n".join([f"Q: {q}" for q in conversation_history[:-1]])
            full_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
        else:
            full_query = query
        
        print("Running GraphRAG...")
        answer = graphRAG_run(graph_context, full_query)
        end_time = time.time()
        query_time = end_time - start_time
        
        print("\nGraph Context:")
        print(f"Nodes: {len(graph_context['nodes'])}, Edges: {len(graph_context['edges'])}")
        
        print("\nAnswer:")
        print(answer)
        print(f"Query processing time: {query_time:.2f} seconds")
        
        # Show tips after the first question
        if len(conversation_history) == 1:
            print("\nTips:")
            print("- Ask follow-up questions to explore the topic further")
            print("- Type 'new: your question' to search for new graph context")
            print("- Type 'exit' to return to the main menu")

def ingest_data(neo4j_driver, qdrant_client, collection_name, batch_size=10, chunk_size=1000,
                 parallel=False, num_workers=4):
    """
    Process to ingest data into both Neo4j and Qdrant
    
    Args:
        neo4j_driver: Neo4j driver
        qdrant_client: Qdrant client
        collection_name: Collection name for vector search
        batch_size: Size of batches for processing
        chunk_size: Size of text chunks
        parallel: Whether to use parallel processing
        num_workers: Number of workers for parallel processing
    """
    print("\nIngest Data")
    print("-" * 30)

    # Get file path from user
    file_path = input("Enter the path to your data file (default: sample_data.txt): ").strip()
    if not file_path:
        file_path = "sample_data.txt"
        
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    try:
        print(f"Reading data from {file_path}...")
        start_time = time.time()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = f.read()
        
        print(f"Data read successfully. Size: {len(raw_data)} characters")
        print("Extracting graph components...")
        
        # Get processor configuration from environment variables
        processor_type = os.getenv("PROCESSOR_TYPE", "ollama").lower()
        processor_config = {
            "model": os.getenv("PROCESSOR_MODEL", "qwen2.5:3b"),
            "host": os.getenv("OLLAMA_HOST", "localhost"),
            "port": int(os.getenv("OLLAMA_PORT", "11434"))
        }
        # Remove None values
        processor_config = {k: v for k, v in processor_config.items() if v is not None}
        
        if parallel:
            print(f"Using parallel processing with {num_workers} workers...")
            nodes, relationships = extract_graph_components_parallel(
                raw_data, 
                chunk_size=chunk_size, 
                max_workers=num_workers,
                processor_type=processor_type,
                processor_config=processor_config
            )
        else:
            nodes, relationships = extract_graph_components(
                raw_data,
                processor_type=processor_type,
                processor_config=processor_config
            )
        
        print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
        
        print("Ingesting to Neo4j...")
        ingest_to_neo4j(neo4j_driver, nodes, relationships, batch_size=batch_size)
        print("Neo4j ingestion complete")
        
        print("Ingesting to Qdrant...")
        ingest_to_qdrant(qdrant_client, collection_name, raw_data, nodes)
        print("Qdrant ingestion complete")
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nTotal processing time: {processing_time:.2f} seconds")
        
    except FileNotFoundError:
        print(f"Error: {file_path} file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    print("GraphRAG Interactive Console")
    print("=" * 30)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    neo4j_driver, qdrant_client = initialize_clients()
    print("Clients initialized")
    
    # Get settings from environment variables
    collection_name = os.getenv("COLLECTION_NAME", "graphRAGstoreds")
    vector_dimension = int(os.getenv("VECTOR_DIMENSION", "768"))
    
    # Get optimization parameters from environment variables
    parallel_processing = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    max_workers = int(os.getenv("MAX_WORKERS", "4"))
    batch_size = int(os.getenv("BATCH_SIZE", "100"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "5000"))
    
    # Set default processor parameters - default to Ollama/open source
    processor_type = os.getenv("PROCESSOR_TYPE", "ollama").lower()
    processor_config = {
        "model": os.getenv("PROCESSOR_MODEL", "qwen2.5:3b"),
        "host": os.getenv("OLLAMA_HOST", "localhost"),
        "port": int(os.getenv("OLLAMA_PORT", "11434"))
    }
    # Remove None values
    processor_config = {k: v for k, v in processor_config.items() if v is not None}
    
    # Set default to use local LLM and embeddings if not specified
    if os.getenv("USE_LOCAL_EMBEDDINGS") is None:
        os.environ["USE_LOCAL_EMBEDDINGS"] = "true"
    if os.getenv("USE_LOCAL_LLM") is None:
        os.environ["USE_LOCAL_LLM"] = "true"
    
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    use_local_llm = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    
    # Ensure collection exists
    create_collection(qdrant_client, collection_name, vector_dimension)
    
    # Display current configuration
    print("\nCurrent Configuration:")
    print("-" * 30)
    print(f"Collection: {collection_name} | Vector Dim: {vector_dimension} | Batch: {batch_size} | Chunk: {chunk_size}")
    print(f"Embeddings: {'Local (' + os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text') + ')' if use_local_embeddings else 'OpenAI'} | LLM: {'Local (' + os.getenv('OLLAMA_LLM_MODEL', 'qwen2.5:3b') + ')' if use_local_llm else 'OpenAI'}")
    print(f"Processor: {processor_type.capitalize()} | Parallel: {'Yes (' + str(max_workers) + ' workers)' if parallel_processing else 'No'}")
    print("-" * 30)
    
    # Main menu loop
    while True:
        print("\n==================================================")
        print("GraphRAG Menu")
        print("==================================================")
        print("1. Ingest data")
        print("2. Clear data")
        print("3. Ask a question")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            # Ingest data
            ingest_data(
                neo4j_driver, 
                qdrant_client, 
                collection_name, 
                batch_size, 
                chunk_size, 
                parallel_processing, 
                max_workers
            )
            
        elif choice == "2":
            # Clear data
            print("\nClear Data")
            print("-" * 30)
            confirm = input("Are you sure you want to clear all data? (y/n): ").lower()
            if confirm == 'y':
                clear_data(neo4j_driver, qdrant_client, collection_name)
                print("All data cleared")
            else:
                print("Operation cancelled")
                
        elif choice == "3":
            # Ask a question
            ask_question_session(neo4j_driver, qdrant_client, collection_name)
            
        elif choice.lower() in ["4", "exit", "quit"]:
            print("\nExiting GraphRAG Console...")
            neo4j_driver.close()
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")