import os
from dotenv import load_dotenv

def reload_config():
    """
    Reloads environment variables and returns updated configuration.
    """
    load_dotenv(override=True)
    
    model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").strip("'").lower()
    
    if model_provider == "openai":
        vector_dimension = int(os.getenv("OPENAI_VECTOR_DIMENSION", "1536"))
    else:
        vector_dimension = int(os.getenv("OLLAMA_VECTOR_DIMENSION", "768"))
        
    return model_provider, vector_dimension

# Initial load of environment variables
load_dotenv()

MODEL_PROVIDER, VECTOR_DIMENSION = reload_config()

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

    processor = {
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

    # Override only the extraction process with the spaCy extractor if enabled.
    # This ensures that only the graph extraction is handled by spaCy while query and embeddings remain with the default provider.
    if os.getenv("USE_SPACY_EXTRACTOR", "false").lower() == "true":
        from processors.spacy_processor import spacy_llm_parser, GraphComponents as SpacyGraphComponents, Single as SpacySingle
        processor["llm_parser"] = spacy_llm_parser
        processor["GraphComponents"] = SpacyGraphComponents
        processor["Single"] = SpacySingle
        # Do NOT override graphrag_query or other components since spaCy is only used for extraction.
    return processor
