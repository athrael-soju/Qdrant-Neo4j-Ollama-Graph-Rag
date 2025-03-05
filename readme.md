# GraphRAG v2

A simple Graph RAG implementation with modular processor support for OpenAI and Ollama LLMs.

## Features

- Extract graph components (nodes and relationships) from text
- Modular processor architecture:
  - OpenAI processor for high-quality extraction
  - Ollama processor for local, private extraction
- Interactive console interface
- Full RAG implementation with Neo4j and Qdrant
- Configurable via environment variables or command-line arguments
- Parallel processing for large documents

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/graph-rag-v2.git
cd graph-rag-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file):
```
# General settings
PROCESSOR_TYPE=ollama  # or "openai"
PROCESSOR_MODEL=qwen2.5:3b  # or any other model

# OpenAI settings (required if using OpenAI processor)
OPENAI_API_KEY=your_openai_api_key

# Ollama settings (required if using Ollama processor)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# Neo4j settings (required for full GraphRAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Qdrant settings (required for full GraphRAG)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

4. Start the Ollama service (if using Ollama):
```bash
ollama run qwen2.5:3b
```

## Interactive Console

Run the interactive console to work with GraphRAG:

```bash
python main.py
```

The console provides the following options:

1. **Ingest data**: Process a text file and ingest the extracted graph components into Neo4j and Qdrant
2. **Clear all data**: Delete all data from Neo4j and Qdrant
3. **Ask a question**: Query the graph with natural language
4. **Configure optimization settings**: Adjust parameters like parallelization, chunk size, etc.
5. **Configure processor settings**: Choose between OpenAI and Ollama processors and set their parameters
6. **Exit**: Close the application

### Processor Configuration

In the interactive console, you can switch between processors:

1. **OpenAI processor**: Uses OpenAI models for high-quality extraction (requires API key)
2. **Ollama processor**: Uses Ollama for local extraction (requires Ollama service)

You can configure the model and other settings for each processor.

## Project Structure

- `graph_rag.py`: Core functionality for graph extraction and RAG operations
- `main.py`: Interactive console for working with the GraphRAG system
- `processors/`: Package containing processor implementations
  - `base_processor.py`: Abstract base class for processors
  - `openai_processor.py`: Processor using OpenAI APIs
  - `ollama_processor.py`: Processor using local Ollama models
  - `processor_factory.py`: Factory for creating processor instances

## Using the GraphRAG System Programmatically

You can also use GraphRAG programmatically:

```python
from graph_rag import extract_graph_components, ingest_to_neo4j

# Extract graph components from text
nodes, relationships = extract_graph_components(text)

# Ingest into Neo4j
with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
    ingest_to_neo4j(driver, nodes, relationships)
```

## Configuration

### Processor Configuration

You can configure processors in the following ways:

1. Environment variables (in `.env` file):
```
PROCESSOR_TYPE=ollama
PROCESSOR_MODEL=qwen2.5:3b
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

2. Programmatically:
```python
from graph_rag import extract_graph_components

# Configure processor parameters
processor_type = "ollama"
processor_config = {
    "model": "qwen2.5:3b",
    "host": "localhost",
    "port": 11434
}

# Process text with specified processor
nodes, relationships = extract_graph_components(
    text,
    processor_type=processor_type,
    processor_config=processor_config
)
```

## Advanced Usage

### Processing Large Documents

For large documents, you can use parallel processing:

```python
from graph_rag import extract_graph_components_parallel

# Process large text with parallel workers
nodes, relationships = extract_graph_components_parallel(
    large_text,
    chunk_size=5000,  # Size of text chunks
    max_workers=4,    # Number of parallel workers
    processor_type="ollama",
    processor_config={"model": "qwen2.5:3b"}
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
