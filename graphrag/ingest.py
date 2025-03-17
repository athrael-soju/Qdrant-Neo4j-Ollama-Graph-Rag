"""
Document ingestion and processing utilities for GraphRAG
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import nltk

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from sentence_transformers import SentenceTransformer
from graphrag.neo4j_connection import get_connection as get_neo4j_connection
from graphrag.qdrant_connection import get_connection as get_qdrant_connection

# Initialize logger
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

# Default embedding model
DEFAULT_EMBEDDING_MODEL = 'intfloat/e5-base-v2'

class DocumentIngestor:
    """Handles document loading, processing, and storage in Neo4j and Qdrant"""
    
    def __init__(self, neo4j_conn=None, qdrant_conn=None, embedding_model=DEFAULT_EMBEDDING_MODEL):
        """Initialize DocumentIngestor
        
        Args:
            neo4j_conn: Neo4j connection instance
            qdrant_conn: Qdrant connection instance
            embedding_model: Name or path of the embedding model
        """
        self.neo4j = neo4j_conn or get_neo4j_connection()
        self.qdrant = qdrant_conn or get_qdrant_connection()
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("Embedding model loaded successfully")
        
    def load_pdf(self, path: str) -> str:
        """Load text from a PDF file
        
        Args:
            path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required to load PDF files. Install with 'pip install pymupdf'")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")
            
        logger.info(f"Loading PDF from {path}")
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            logger.info(f"Successfully extracted text from PDF: {path}")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {path}: {str(e)}")
            raise
            
    def chunk_text(self, text: str, max_tokens: int = 200) -> List[str]:
        """Split text into chunks
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            logger.warning("Received empty text for chunking")
            return []
            
        logger.info(f"Chunking text ({len(text)} chars) with max {max_tokens} tokens per chunk")
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            if current_length + len(tokens) > max_tokens and current_chunk:
                # Start new chunk if current one would exceed max_tokens
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sent)
            current_length += len(tokens)
            
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        logger.info(f"Created {len(chunks)} chunks from input text")
        return chunks
        
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            np.ndarray: Matrix of embeddings (chunks × embedding_dim)
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return np.array([])
            
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            # Add "passage: " prefix to each chunk as required by E5 model
            prefixed_chunks = [f"passage: {ch}" for ch in chunks]
            embeddings = self.embedding_model.encode(prefixed_chunks, 
                                               normalize_embeddings=True,
                                               show_progress_bar=len(chunks) > 10)
            logger.info(f"Successfully generated embeddings of shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
            
    def store_chunks_in_neo4j(self, doc_id: str, chunks: List[str], 
                          embeddings: np.ndarray) -> None:
        """Store document chunks and metadata in Neo4j
        
        Args:
            doc_id: Document identifier
            chunks: List of text chunks
            embeddings: Chunk embeddings
        """
        # Create document node if it doesn't exist
        doc_query = """
        MERGE (d:Document {id: $doc_id})
        RETURN d
        """
        self.neo4j.run_query(doc_query, {"doc_id": doc_id})
        
        # Create chunk nodes and connect to document
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk{i}"
            chunk_query = """
            MATCH (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text, 
                c.index = $index
            MERGE (d)-[:CONTAINS]->(c)
            RETURN c
            """
            params = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "index": i
            }
            self.neo4j.run_query(chunk_query, params)
            
        logger.info(f"Stored {len(chunks)} chunks for document {doc_id} in Neo4j")
        
    def store_embeddings_in_qdrant(self, doc_id: str, chunks: List[str], 
                                embeddings: np.ndarray) -> None:
        """Store document chunk embeddings in Qdrant
        
        Args:
            doc_id: Document identifier
            chunks: List of text chunks
            embeddings: Chunk embeddings
        """
        # Prepare IDs and metadata for Qdrant
        ids = []
        metadata = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{i}"
            ids.append(chunk_id)
            metadata.append({
                "doc_id": doc_id, 
                "chunk_index": i,
                "text": chunk_text[:1000]  # Store truncated text in metadata for quick access
            })
        
        # Store in Qdrant
        result = self.qdrant.upsert_vectors(
            collection_name="tokens", 
            vectors=embeddings.tolist(), 
            ids=ids, 
            metadata=metadata
        )
        
        if result:
            logger.info(f"Stored {len(chunks)} embeddings for document {doc_id} in Qdrant")
        else:
            logger.error(f"Failed to store embeddings for document {doc_id} in Qdrant")
            
    def process_document(self, doc_id: str, text: str = None, pdf_path: str = None, 
                     max_tokens: int = 200) -> Tuple[List[str], np.ndarray]:
        """Process a document: load, chunk, embed, and store
        
        Args:
            doc_id: Document identifier
            text: Document text (if provided directly)
            pdf_path: Path to PDF file (if loading from PDF)
            max_tokens: Maximum tokens per chunk
            
        Returns:
            Tuple[List[str], np.ndarray]: Chunks and embeddings
        """
        # Load document
        if text is None and pdf_path is not None:
            text = self.load_pdf(pdf_path)
        elif text is None:
            raise ValueError("Either text or pdf_path must be provided")
            
        # Chunk text
        chunks = self.chunk_text(text, max_tokens)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embed_chunks(chunks)
        logger.info(f"Generated embeddings of shape {embeddings.shape}")
        
        # Store in Neo4j
        self.store_chunks_in_neo4j(doc_id, chunks, embeddings)
        
        # Store in Qdrant
        self.store_embeddings_in_qdrant(doc_id, chunks, embeddings)
        
        return chunks, embeddings

# Convenience functions

def load_pdf(path: str) -> str:
    """Load text from a PDF file (standalone function)"""
    ingestor = DocumentIngestor()
    return ingestor.load_pdf(path)

def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    """Chunk text into segments (standalone function)"""
    ingestor = DocumentIngestor()
    return ingestor.chunk_text(text, max_tokens)

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Embed text chunks (standalone function)"""
    ingestor = DocumentIngestor()
    return ingestor.embed_chunks(chunks)

def store_chunks_in_neo4j(doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
    """Store document chunks in Neo4j (standalone function)"""
    ingestor = DocumentIngestor()
    ingestor.store_chunks_in_neo4j(doc_id, chunks, embeddings)

def store_embeddings_in_qdrant(doc_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
    """Store document chunk embeddings in Qdrant (standalone function)"""
    ingestor = DocumentIngestor()
    ingestor.store_embeddings_in_qdrant(doc_id, chunks, embeddings)

def process_document(doc_id: str, text: str = None, pdf_path: str = None, 
                 max_tokens: int = 200) -> Tuple[List[str], np.ndarray]:
    """Process a document (standalone function)"""
    ingestor = DocumentIngestor()
    return ingestor.process_document(doc_id, text, pdf_path, max_tokens)
    
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Demo with example text
    example_text = """
    Hugging Face, Inc. is an American company that develops tools for building applications using machine learning.
    It was founded in 2016 and its headquarters is in New York City. The company is known for its libraries in natural
    language processing (NLP) and its platform that allows users to share machine learning models and datasets.
    """
    
    print("Processing example document...")
    chunks, embeddings = process_document("example_doc", text=example_text)
    print(f"Created {len(chunks)} chunks with embeddings of shape {embeddings.shape}")
    print("First chunk:", chunks[0]) 