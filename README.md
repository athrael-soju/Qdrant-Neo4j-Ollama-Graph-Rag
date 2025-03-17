# GraphRAG: Graph-based Retrieval Augmented Generation

GraphRAG is a Python library that implements a Graph-based Retrieval Augmented Generation system. It combines vector embeddings with knowledge graphs to improve retrieval for RAG systems.

## Project Structure

The project follows a standard Python package structure:

```
graphrag/
├── core/           # Core functionality
│   ├── ingest.py   # Document ingestion
│   ├── retrieval.py # Retrieval mechanisms
│   ├── nlp_graph.py # Graph construction
│   └── triplets.py  # Triplet extraction
├── connectors/     # Database connectors
│   ├── neo4j_connection.py
│   └── qdrant_connection.py
├── models/         # ML models and embeddings
├── utils/          # Utility functions
│   ├── common.py
│   └── config.py
├── cli/            # Command-line interface
│   └── main.py
├── data/           # Data handling
└── tests/          # Tests
    ├── unit/
    └── integration/
```

## Quick Start

```bash
# Create and activate virtual environment
py -m venv venv    
venv/Scripts/activate

# Install the package
pip install -e .
python -m pip install --upgrade pip

# Set up the Neo4j database with required indexes
graphrag setup

# Process a document
graphrag process path/to/document.txt

# Query the system
graphrag query "What type of business is Hugging Face?"

# Start interactive session
graphrag interactive
```

## Configuration

By default, GraphRAG connects to Neo4j at `bolt://localhost:7687` with the credentials `neo4j/testpassword`. You can modify these settings by editing the configuration file in `graphrag/utils/config.py`.

## Architecture

GraphRAG consists of several key components:

1. **Document Ingestion** - Loads text/PDF files and splits them into manageable chunks
2. **Vector Indexing** - Embeds text chunks using a transformer model (default: E5-base)
3. **Term Graph Construction** - Extracts tokens, bigrams, and trigrams, linking them to chunks
4. **Triplet Extraction** - Uses a T5 model to identify subject-relation-object triplets
5. **Hybrid Retrieval** - Combines vector similarity with graph traversal for better context
6. **CLI Interface** - Provides command-line tools for processing and querying

## Advanced Usage

### Using Spark NLP for Large-Scale Processing (Optional)

For processing large document collections, GraphRAG can use Spark NLP for distributed NLP processing. Install the additional dependencies:

```bash
pip install pyspark spark-nlp
```

The system will automatically detect and use Spark NLP if available.

### Using GPU for Embedding and Triplet Extraction

To accelerate embedding and triplet extraction, ensure you have PyTorch installed with CUDA support and the required GPU drivers.

## Examples

See the `examples/` directory for usage examples.

### Process a document about a tech company:

```bash
echo "Hugging Face, Inc. is an American company that develops tools for building applications using machine learning. It was founded in 2016 and its headquarters is in New York City." > huggingface.txt

graphrag process huggingface.txt
```

### Query the system:

```bash
graphrag query "Where is Hugging Face headquartered?"
```

Output might include:
- Retrieved chunks containing the headquarters information
- A knowledge graph triplet: (Hugging Face, Inc.) -[HEADQUARTERS_IN]-> (New York City)

## Documentation

For more detailed documentation, see the `docs/` directory.

## Limitations

- The triplet extraction model works best on simple, factual statements
- Large documents may require significant processing time and memory
- Complex queries might need refinement for optimal results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use GraphRAG in your research, please cite:

```
@software{graphrag2023,
  author = {GraphRAG Team},
  title = {GraphRAG: Graph-based Retrieval Augmented Generation},
  year = {2023},
  url = {https://github.com/yourusername/graphrag}
}
``` 