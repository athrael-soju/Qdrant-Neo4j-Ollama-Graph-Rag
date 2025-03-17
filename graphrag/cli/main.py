"""
Main entry point for GraphRAG system
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Optional, Union

from graphrag.connectors.neo4j_connection import get_connection as get_neo4j_connection
from graphrag.connectors.qdrant_connection import get_connection as get_qdrant_connection
from graphrag.core.ingest import process_document
from graphrag.core.nlp_graph import process_chunk
from graphrag.core.triplets import process_chunk as process_chunk_triplets
from graphrag.core.retrieval import hybrid_retrieve, hybrid_retrieve_with_triplets
from graphrag.utils.common import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize Neo4j database with necessary indexes and Qdrant collections"""
    logger.info("Setting up Neo4j database...")
    neo4j = get_neo4j_connection()
    neo4j.setup_indexes()
    
    logger.info("Setting up Qdrant vector database...")
    qdrant = get_qdrant_connection()
    qdrant.setup_collections()
    
    logger.info("Database setup complete.")
    
def reset_database():
    """Reset Neo4j and Qdrant databases by removing all data"""
    logger.info("Resetting Neo4j database...")
    neo4j = get_neo4j_connection()
    
    # Delete all nodes and relationships from Neo4j
    try:
        neo4j.run_query("MATCH (n) DETACH DELETE n")
        logger.info("Successfully deleted all nodes and relationships from Neo4j")
    except Exception as e:
        logger.error(f"Error resetting Neo4j database: {str(e)}")
        raise
    
    logger.info("Resetting Qdrant vector database...")
    qdrant = get_qdrant_connection()
    
    # Delete all points from the collection
    try:
        qdrant.clear_collection("tokens")
        logger.info("Successfully cleared the tokens collection in Qdrant")
    except Exception as e:
        logger.error(f"Error clearing Qdrant collection: {str(e)}")
        raise
    
    logger.info("Database reset complete.")
    
def process_document_full(doc_id: str, text: str = None, pdf_path: str = None, 
                       max_tokens: int = 200) -> Dict[str, Any]:
    """Process a document with all GraphRAG components
    
    Args:
        doc_id: Document identifier
        text: Document text (if provided directly)
        pdf_path: Path to PDF file (if loading from PDF)
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Dict[str, Any]: Processing results
    """
    logger.info(f"Processing document {doc_id}...")
    
    # Step 1: Ingest document, chunk, embed, and store in Neo4j
    chunks, embeddings = process_document(doc_id, text, pdf_path, max_tokens)
    
    # Step 2: Extract n-grams and build term-based graph
    term_results = {}
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk{idx}"
        unigrams, bigrams, trigrams = process_chunk(chunk_id, chunk_text)
        term_results[chunk_id] = {
            "unigrams": len(unigrams),
            "bigrams": len(bigrams),
            "trigrams": len(trigrams)
        }
        
    # Step 3: Extract and store triplets for knowledge graph
    triplet_results = {}
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk{idx}"
        triplets = process_chunk_triplets(chunk_id, chunk_text)
        triplet_results[chunk_id] = len(triplets)
        
    logger.info(f"Document {doc_id} processed successfully.")
    
    return {
        "document_id": doc_id,
        "chunks_count": len(chunks),
        "term_results": term_results,
        "triplet_counts": triplet_results
    }
    
def process_files(files: List[str], is_pdf: bool = False) -> Dict[str, Any]:
    """Process multiple files
    
    Args:
        files: List of file paths
        is_pdf: Whether files are PDFs
        
    Returns:
        Dict[str, Any]: Processing results by document ID
    """
    results = {}
    
    for file_path in files:
        doc_id = os.path.basename(file_path).split('.')[0]  # Use filename without extension as ID
        
        try:
            if is_pdf:
                result = process_document_full(doc_id, pdf_path=file_path)
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    result = process_document_full(doc_id, text=text)
                except UnicodeDecodeError:
                    try:
                        # Try with different encoding if UTF-8 fails
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                        result = process_document_full(doc_id, text=text)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path} with latin-1 encoding: {str(e)}")
                        continue
                except IOError as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    continue
                    
            results[doc_id] = result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            continue
        
    return results
    
def query_graphrag(query: str, top_k: int = 5, include_triplets: bool = True) -> Dict[str, Any]:
    """Query the GraphRAG system
    
    Args:
        query: Query string
        top_k: Number of top results to retrieve
        include_triplets: Whether to include triplets in results
        
    Returns:
        Dict[str, Any]: Query results
    """
    logger.info(f"Querying GraphRAG: '{query}'")
    
    try:
        if include_triplets:
            results = hybrid_retrieve_with_triplets(query, top_k)
        else:
            chunks = hybrid_retrieve(query, top_k)
            results = {"chunks": chunks, "triplets": []}
            
        return results
    except Exception as e:
        error_msg = str(e)
        if "ProcedureNotFound" in error_msg and "vector" in error_msg:
            logger.warning("Neo4j vector search is not supported. Using only Qdrant for vector search.")
            # Try again with a different approach that doesn't use Neo4j vector search
            from graphrag.core.retrieval import VectorRetriever
            retriever = VectorRetriever()
            chunks = retriever.retrieve_chunks(query, top_k)
            return {"chunks": chunks, "triplets": []}

def print_query_results(results: Dict[str, Any]):
    """Print query results in a readable format
    
    Args:
        results: Query results from query_graphrag
    """
    print("\n" + "="*80)
    print("QUERY RESULTS:")
    print("="*80)
    
    # Print chunks
    chunks = results.get("chunks", [])
    print(f"\nRetrieved {len(chunks)} relevant chunks:")
    for i, (chunk_id, text, score) in enumerate(chunks, 1):
        print(f"\n{i}. Chunk {chunk_id} (score: {score:.3f}):")
        print("-" * 40)
        print(text)
        
    # Print triplets if available
    triplets = results.get("triplets", [])
    if triplets:
        print("\n" + "="*80)
        print(f"Found {len(triplets)} relevant knowledge graph triplets:")
        print("="*80)
        
        for i, (subj, rel, obj, chunk_id, _) in enumerate(triplets, 1):
            print(f"{i}. {subj} --[{rel}]--> {obj}  (source: {chunk_id})")
            
    print("\n" + "="*80)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GraphRAG: Graph-based Retrieval Augmented Generation")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup database command
    setup_parser = subparsers.add_parser("setup", help="Setup Neo4j database")
    
    # Reset database command
    reset_parser = subparsers.add_parser("reset", help="Reset Neo4j and Qdrant databases by removing all data")
    
    # Process files command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("files", nargs="+", help="Files to process")
    process_parser.add_argument("--pdf", action="store_true", help="Files are PDFs")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the GraphRAG system")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of top results")
    query_parser.add_argument("--no-triplets", action="store_true", help="Don't include triplets in results")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive query session")
    
    return parser.parse_args()
    
def run_interactive_session():
    """Run an interactive query session"""
    print("\nGraphRAG Interactive Query Session")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'help' for available commands")
    
    while True:
        try:
            user_input = input("\nGraphRAG> ").strip()
            
            if user_input.lower() in ("exit", "quit"):
                break
                
            if user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  help       - Show this help message")
                print("  exit, quit - Exit the session")
                print("  Any other input will be treated as a query to the GraphRAG system")
                continue
                
            if not user_input:
                continue
                
            # Process query
            results = query_graphrag(user_input)
            print_query_results(results)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")
            
    print("\nExiting GraphRAG interactive session.")

def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        if args.command == "setup":
            setup_database()
            print("Neo4j database setup complete.")
            
        elif args.command == "reset":
            reset_database()
            print("Neo4j and Qdrant databases reset complete.")
            
        elif args.command == "process":
            results = process_files(args.files, args.pdf)
            print(f"Processed {len(results)} documents successfully.")
            
        elif args.command == "query":
            results = query_graphrag(args.query, args.top_k, not args.no_triplets)
            print_query_results(results)
            
        elif args.command == "interactive":
            run_interactive_session()
            
        else:
            print("Please specify a command. Use --help for available commands.")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 