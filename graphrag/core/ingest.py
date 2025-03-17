"""
Document ingestion module for GraphRAG.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

def split_text_into_chunks(text: str, max_tokens: int = 200) -> List[str]:
    """
    Split text into chunks of approximately max_tokens.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk (approximate)
        
    Returns:
        List of text chunks
    """
    # Simple splitting by sentences and then combining until max_tokens
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Approximate token count by word count
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > max_tokens and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
        
    return chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    try:
        import fitz  # PyMuPDF
        
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except ImportError:
        logger.error("PyMuPDF (fitz) is required for PDF support. Install with: pip install pymupdf")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def embed_text(text: str) -> np.ndarray:
    """
    Generate embeddings for text.
    
    Args:
        text: Text to embed
        
    Returns:
        Text embedding
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model (first time will download the model)
        model = SentenceTransformer('intfloat/e5-base')
        
        # Generate embedding
        embedding = model.encode(text)
        return embedding
    except ImportError:
        logger.error("SentenceTransformer is required for embedding. Install with: pip install sentence-transformers")
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def process_document(doc_id: str, text: str = None, pdf_path: str = None, 
                      max_tokens: int = 200) -> Tuple[List[str], List[np.ndarray]]:
    """
    Process a document by ingesting, chunking, and embedding.
    
    Args:
        doc_id: Document identifier
        text: Document text (if provided directly)
        pdf_path: Path to PDF file (if loading from PDF)
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Tuple of (chunks, embeddings)
    """
    logger.info(f"Processing document: {doc_id}")
    
    # Get text from either direct input or PDF
    if text is None and pdf_path is None:
        raise ValueError("Either text or pdf_path must be provided")
    
    if text is None:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    logger.info("Splitting text into chunks")
    chunks = split_text_into_chunks(text, max_tokens)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Generate embeddings for each chunk
    logger.info("Generating embeddings")
    embeddings = []
    for chunk in chunks:
        embedding = embed_text(chunk)
        embeddings.append(embedding)
    
    logger.info(f"Document processing complete: {doc_id}")
    return chunks, embeddings 