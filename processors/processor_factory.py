import os
from dotenv import load_dotenv

load_dotenv()

# Remove any surrounding quotes and lower the provider name
MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").strip("'").lower()

if MODEL_PROVIDER == "openai":
    VECTOR_DIMENSION = int(os.getenv("OPENAI_VECTOR_DIMENSION", "1536"))
else:
    VECTOR_DIMENSION = int(os.getenv("OLLAMA_VECTOR_DIMENSION", "768"))

def get_processor() -> dict:
    """
    Returns the appropriate processor module based on environment settings.
    
    Supported providers:
      - "ollama": Uses Ollama configuration.
      - "openai": Uses OpenAI configuration.
    
    Returns:
        A dictionary containing processor components.
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
            OLLAMA_INFERENCE_MODEL as LLM_MODEL,
            OLLAMA_EMBEDDING_MODEL as EMBEDDING_MODEL
        )
    else:  # Default to OpenAI
        from processors.openai_processor import (
            openai_llm_parser as llm_parser,
            openai_embeddings as embeddings,
            openai_embeddings_batch as embeddings_batch,
            graphrag_query,
            GraphComponents,
            Single,
            VECTOR_DIMENSION,
            OPENAI_INFERENCE_MODEL as LLM_MODEL,
            OPENAI_EMBEDDING_MODEL as EMBEDDING_MODEL
        )

    return {
        "llm_parser": llm_parser,
        "embeddings": embeddings,
        "embeddings_batch": embeddings_batch,
        "graphrag_query": graphrag_query,
        "GraphComponents": GraphComponents,
        "Single": Single,
        "VECTOR_DIMENSION": VECTOR_DIMENSION,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "MODEL_PROVIDER": MODEL_PROVIDER
    }
