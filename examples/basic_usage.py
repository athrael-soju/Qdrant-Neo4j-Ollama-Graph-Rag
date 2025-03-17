#!/usr/bin/env python
"""
Basic usage example for GraphRAG.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import graphrag
sys.path.insert(0, str(Path(__file__).parent.parent))

import graphrag
from graphrag.core.ingest import process_document
from graphrag.core.retrieval import hybrid_retrieve_with_triplets

def main():
    """Run the basic usage example."""
    # Print version
    print(f"GraphRAG version: {graphrag.__version__}")
    
    # Process a sample document (this is a placeholder since we don't have actual DB connections)
    print("This is a basic example of using GraphRAG.")
    print("In a real application, you would:")
    print("1. Connect to Neo4j and Qdrant databases")
    print("2. Process documents to extract text and generate embeddings")
    print("3. Store the embeddings in Qdrant and create a knowledge graph in Neo4j")
    print("4. Query the system using hybrid retrieval")
    
    # Example query
    query = "What is Hugging Face?"
    print(f"\nExample query: {query}")
    print("In a real application with connected databases, this would return relevant chunks and knowledge graph triplets.")
    
if __name__ == "__main__":
    main() 