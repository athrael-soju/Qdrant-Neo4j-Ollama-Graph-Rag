![image](https://github.com/user-attachments/assets/900cdf8f-5dca-4105-983b-8534632d78fd)
# GraphRAG

A knowledge graph-based Retrieval-Augmented Generation (RAG) system that allows you to ingest text data, extract entities and relationships, and query the resulting knowledge graph.

## Features

- **Provider Agnostic**: Works with both OpenAI and Ollama LLM providers
- **Knowledge Graph Extraction**: Automatically extracts entities and relationships from text
- **Multiple Extraction Methods**: Choose between LLM-based or spaCy-based entity extraction
- **Vector Search**: Uses Qdrant for semantic similarity search
- **Graph Database**: Uses Neo4j to store and query relationship data
- **Interactive Console**: Simple console interface for ingesting data and asking questions
- **Streaming Responses**: Support for streaming responses for a better user experience
- **Parallel Processing**: Optimized for handling large documents with parallel processing

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
- spaCy (if using spaCy-based entity extraction)

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

4. If using the spaCy extractor, download the language model:
```bash
python -m spacy download en_core_web_sm
```

5. Start Neo4j and Qdrant with Docker:
```bash
docker-compose up -d
```

6. Copy the environment variables template and configure it:
```bash
cp .env.example .env
```

7. Edit the `.env` file with your configuration settings:
   - Set `DEFAULT_MODEL_PROVIDER` to either `openai` or `ollama`
   - Set `USE_SPACY_EXTRACTOR` to `true` if you want to use spaCy instead of LLM for entity extraction
   - Configure your vector dimensions based on the model provider
   - Set the appropriate embedding and LLM models
   - Add your API keys or connection details
   - Configure performance parameters as needed

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
Enter your question: Tell me about Carol

Starting retriever search...
Extracting entity IDs...
Fetching related graph...
Formatting graph context...
Running GraphRAG...

Answer: Based on the information provided in the knowledge graph, here's an overview of Carol:

Carol has several roles and responsibilities:
1. She led the expansion of the New York office under her leadership.
2. She implemented new processes in the New York office.
3. She leads the East Coast team for TechCorp.
4. Her expertise is crucial for the Alpha project managed from New York office.

These roles were all held during or after she transferred to the New York office last year, and Carol was mentored by Alice (data scientist at TechCorp's Seattle office) who also worked with Dave on the Alpha project.

Carol appears to be an important figure within TechCorp, especially in managing the New York office and its team. She is involved with several projects and teams across the organization.
Query processing time: 54.46 seconds (Answer generation: 54.25 seconds)
```

## Configuration Options

The system can be configured through environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| DEFAULT_MODEL_PROVIDER | LLM provider (`openai` or `ollama`) | `ollama` |
| USE_SPACY_EXTRACTOR | Whether to use spaCy for entity extraction instead of LLM | `false` |
| NEO4J_URI | URI for Neo4j connection | `bolt://localhost:7687` |
| NEO4J_USERNAME | Neo4j username | `neo4j` |
| NEO4J_PASSWORD | Neo4j password | `morpheus4j` |
| QDRANT_HOST | Qdrant host | `localhost` |
| QDRANT_PORT | Qdrant port | `6333` |
| COLLECTION_NAME | Qdrant collection name | `graphRAGstoreds` |
| OPENAI_API_KEY | OpenAI API key (if using OpenAI) | - |
| OPENAI_INFERENCE_MODEL | Model for inference with OpenAI | `gpt-4o-mini` |
| OPENAI_EMBEDDING_MODEL | Model for embeddings with OpenAI | `text-embedding-3-small` |
| OPENAI_VECTOR_DIMENSION | Dimension of embedding vectors for OpenAI | `1536` |
| OLLAMA_HOST | Ollama host (if using Ollama) | `localhost` |
| OLLAMA_PORT | Ollama port (if using Ollama) | `11434` |
| OLLAMA_INFERENCE_MODEL | Model for inference with Ollama | `qwen2.5:3b` |
| OLLAMA_EMBEDDING_MODEL | Model for embeddings with Ollama | `nomic-embed-text` |
| OLLAMA_VECTOR_DIMENSION | Dimension of embedding vectors for Ollama | `768` |
| PARALLEL_PROCESSING | Enable parallel processing | `true` |
| MAX_WORKERS | Number of parallel workers | `8` |
| BATCH_SIZE | Batch size for database operations | `100` |
| CHUNK_SIZE | Size of text chunks for processing | `5000` |
| USE_STREAMING | Enable streaming responses from LLM | `true` |

## Entity Extraction Options

### LLM-based Extraction (Default)
- Uses the configured LLM (either OpenAI or Ollama) to extract entities and their relationships
- More accurate for complex texts and can infer implicit relationships
- Slower and requires API calls or local LLM service
- Configured by setting `USE_SPACY_EXTRACTOR=false`

### spaCy-based Extraction
- Uses spaCy's natural language processing capabilities to extract entities and relationships
- Faster than LLM-based extraction and works offline
- May be less accurate for complex or domain-specific relationships
- Configured by setting `USE_SPACY_EXTRACTOR=true`
- Requires downloading the spaCy model: `python -m spacy download en_core_web_sm`

## Extending the System

### Adding a New Model Provider

To add a new model provider:

1. Create a new file in the `processors` directory (e.g., `my_provider_processor.py`)
2. Implement the required functions following the pattern in existing processors
3. Update `processor_factory.py` to include your new provider

### Adding a New Extraction Method

To add a new extraction method:

1. Create a new file in the `processors` directory (e.g., `custom_extractor.py`)
2. Implement the required functions, particularly the `*_llm_parser` function
3. Update `processor_factory.py` to include your new extractor

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses the [neo4j-graphrag](https://github.com/langchain-ai/neo4j-graphrag) library for handling graph-based retrieval. 
