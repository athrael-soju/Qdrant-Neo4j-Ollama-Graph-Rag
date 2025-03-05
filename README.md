# GraphRAG

A knowledge graph-based Retrieval-Augmented Generation (RAG) system that allows you to ingest text data, extract entities and relationships, and query the resulting knowledge graph.

## Features

- **Provider Agnostic**: Works with both OpenAI and Ollama LLM providers
- **Knowledge Graph Extraction**: Automatically extracts entities and relationships from text
- **Vector Search**: Uses Qdrant for semantic similarity search
- **Graph Database**: Uses Neo4j to store and query relationship data
- **Interactive Console**: Simple console interface for ingesting data and asking questions
- **Streaming Responses**: Support for streaming responses for a better user experience

## Architecture

GraphRAG combines the power of:

1. **LLMs** (Large Language Models) for entity extraction and question answering
2. **Vector Database** (Qdrant) for semantic search 
3. **Graph Database** (Neo4j) for storing entity relationships

This creates a powerful retrieval system that can answer questions based on both semantic similarity and relationships between entities.

## Prerequisites

- Python 3.8+
- Docker (for running Neo4j and Qdrant)
- OpenAI API key (if using OpenAI) or Ollama running locally (if using Ollama)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/graph-rag.git
cd graph-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Neo4j and Qdrant with Docker:
```bash
docker-compose up -d
```

5. Copy the environment variables template and configure it:
```bash
cp .env.example .env
```

6. Edit the `.env` file with your configuration settings:
   - Set `MODEL_PROVIDER` to either `openai` or `ollama`
   - Configure your vector dimensions based on the model provider
   - Set the appropriate embedding and LLM models
   - Add your API keys or connection details
   - Configure performance parameters as needed

7. If using Ollama, pull the required models:
```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Pull the LLM for inference
ollama pull qwen2.5:3b
```

## Usage

### Running the Interactive Console

Start the interactive console:

```bash
python main.py
```

### Workflow

1. **Ingest Data**: Choose option 1 to ingest text data
2. **Ask Questions**: Choose option 3 to ask questions about the ingested data
3. **Configure Settings**: Option 4 allows you to configure various settings
4. **Clear Data**: Option 2 clears all data from Neo4j and Qdrant

### Example

```
GraphRAG Interactive Console
Loading environment variables and initializing clients...
Using model provider: ollama
Using LLM model: qwen2.5:3b
Using embedding model: nomic-embed-text
Vector dimension: 768
NEO4J_URI: bolt://localhost:7687
NEO4J_USERNAME: neo4j
NEO4J_PASSWORD: *****
QDRANT_HOST: localhost
COLLECTION_NAME: graphRAGstoreds
Clients initialized

==================================================
GraphRAG Console - Choose an option:
1. Ingest data
2. Clear all data
3. Ask a question
4. Configure settings
5. Exit
==================================================
Current LLM provider: OLLAMA
==================================================
Enter your choice (1-5): 3

Ask a Question
Enter your question: Who is Dave?
```

## Configuration Options

The system can be configured through environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_PROVIDER | LLM provider (`openai` or `ollama`) | - |
| NEO4J_URI | URI for Neo4j connection | bolt://localhost:7687 |
| NEO4J_USERNAME | Neo4j username | neo4j |
| NEO4J_PASSWORD | Neo4j password | - |
| QDRANT_HOST | Qdrant host | localhost |
| QDRANT_PORT | Qdrant port | 6333 |
| COLLECTION_NAME | Qdrant collection name | - |
| VECTOR_DIMENSION | Dimension of embedding vectors | 1536 (OpenAI) or 768 (Ollama) |
| EMBEDDING_MODEL | Model for embeddings | text-embedding-3-small (OpenAI) or nomic-embed-text (Ollama) |
| LLM_MODEL | Model for inference | gpt-4o-mini (OpenAI) or qwen2.5:3b (Ollama) |
| OPENAI_API_KEY | OpenAI API key (if using OpenAI) | - |
| OLLAMA_HOST | Ollama host (if using Ollama) | localhost |
| OLLAMA_PORT | Ollama port (if using Ollama) | 11434 |
| PARALLEL_PROCESSING | Enable parallel processing | true |
| MAX_WORKERS | Number of parallel workers | 4 |
| BATCH_SIZE | Batch size for database operations | 100 |
| CHUNK_SIZE | Size of text chunks for processing | 5000 |

## Extending the System

### Adding a New Model Provider

To add a new model provider:

1. Create a new file in the `processors` directory (e.g., `my_provider_processor.py`)
2. Implement the required functions following the pattern in existing processors
3. Update `processor_factory.py` to include your new provider

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses the [neo4j-graphrag](https://github.com/langchain-ai/neo4j-graphrag) library for handling graph-based retrieval. 