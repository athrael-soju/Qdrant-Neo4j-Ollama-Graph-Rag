import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get processor type from environment
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1536"))

# Main processor selection function
def get_processor():
    """
    Returns the appropriate processor module based on environment settings.
    
    The selection is based on the MODEL_PROVIDER environment variable.
    
    Returns:
        module: The selected processor module with all necessary functions.
    """
    
    if MODEL_PROVIDER == "ollama":
        from processors.ollama_processor import (
            ollama_llm_parser as llm_parser,
            ollama_embeddings as embeddings,
            ollama_embeddings_batch as embeddings_batch,
            graphrag_query,
            GraphComponents,
            Single,
            VECTOR_DIMENSION,
            OLLAMA_LLM_MODEL as LLM_MODEL,
            OLLAMA_EMBEDDING_MODEL as EMBEDDING_MODEL
        )
        
        print(f"Using Ollama processor with model: {LLM_MODEL}")
        print(f"Using Ollama embeddings with model: {EMBEDDING_MODEL}")
        
    else:  # Default to OpenAI
        from processors.openai_processor import (
            openai_llm_parser as llm_parser,
            openai_embeddings as embeddings,
            openai_embeddings_batch as embeddings_batch,
            graphrag_query,
            GraphComponents,
            Single,
            VECTOR_DIMENSION,
            OPENAI_LLM_MODEL as LLM_MODEL,
            OPENAI_EMBEDDING_MODEL as EMBEDDING_MODEL
        )
        
        print(f"Using OpenAI processor with model: {LLM_MODEL}")
        print(f"Using OpenAI embeddings with model: {EMBEDDING_MODEL}")
    
    return {
        "llm_parser": llm_parser,
        "embeddings": embeddings,
        "embeddings_batch": embeddings_batch,
        "graphrag_query": graphrag_query,
        "GraphComponents": GraphComponents,
        "Single": Single,
        "VECTOR_DIMENSION": VECTOR_DIMENSION,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL
    } 