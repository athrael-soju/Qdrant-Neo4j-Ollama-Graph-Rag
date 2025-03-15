import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine model provider and extractor
MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").strip("'").lower()
EXTRACTOR = os.getenv("DEFAULT_EXTRACTOR", "").strip("'").lower()

# Define vector dimension based on model provider
if MODEL_PROVIDER == "openai":
    VECTOR_DIMENSION = int(os.getenv("OPENAI_VECTOR_DIMENSION", "1536"))
else:
    VECTOR_DIMENSION = int(os.getenv("OLLAMA_VECTOR_DIMENSION", "768"))


def get_processor() -> dict:
    """
    Returns the appropriate processor module based on environment settings.
    """
    if MODEL_PROVIDER == "ollama":
        from processors.ollama_processor import (
            ollama_embeddings as embeddings,
            ollama_embeddings_batch as embeddings_batch,
            graphrag_query,
            GraphComponents,
            Single,
            VECTOR_DIMENSION,
            OLLAMA_INFERENCE_MODEL as LLM_MODEL,
            OLLAMA_EMBEDDING_MODEL as EMBEDDING_MODEL,
        )
    else:
        from processors.openai_processor import (
            openai_embeddings as embeddings,
            openai_embeddings_batch as embeddings_batch,
            graphrag_query,
            GraphComponents,
            Single,
            VECTOR_DIMENSION,
            OPENAI_INFERENCE_MODEL as LLM_MODEL,
            OPENAI_EMBEDDING_MODEL as EMBEDDING_MODEL,
        )

    processor = {
        "embeddings": embeddings,
        "embeddings_batch": embeddings_batch,
        "graphrag_query": graphrag_query,
        "GraphComponents": GraphComponents,
        "Single": Single,
        "VECTOR_DIMENSION": VECTOR_DIMENSION,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "MODEL_PROVIDER": MODEL_PROVIDER,
        "EXTRACTOR": EXTRACTOR,
    }

    if EXTRACTOR == "spacy":
        from processors.spacy_processor import (
            spacy_llm_parser,
            GraphComponents as SpacyGraphComponents,
            Single as SpacySingle,
        )

        processor["llm_parser"] = spacy_llm_parser
        processor["GraphComponents"] = SpacyGraphComponents
        processor["Single"] = SpacySingle
    elif EXTRACTOR == "stanza":
        from processors.stanza_processor import (
            stanza_llm_parser,
            GraphComponents as StanzaGraphComponents,
            Single as StanzaSingle,
        )

        processor["llm_parser"] = stanza_llm_parser
        processor["GraphComponents"] = StanzaGraphComponents
        processor["Single"] = StanzaSingle
    else:
        if MODEL_PROVIDER == "ollama":
            from processors.ollama_processor import ollama_llm_parser as llm_parser
        else:
            from processors.openai_processor import openai_llm_parser as llm_parser
        processor["llm_parser"] = llm_parser

    return processor
