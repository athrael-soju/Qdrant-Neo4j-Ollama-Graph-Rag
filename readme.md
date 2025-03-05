# GraphRAG with Neo4j, Qdrant, and LLMs

## Overview

This project demonstrates how to build a Graph Retrieval-Augmented Generation (RAG) pipeline that extracts graph relationships from raw text using language models, stores and queries these relationships in a Neo4j graph database, and enhances the process with Qdrant's vector search capabilities. By integrating these technologies, users can extract structured insights from unstructured text and perform complex graph queries to generate context-aware natural language responses.

The system is built on three main components:

- **Vector Search & RAG**: Uses Qdrant to index text embeddings for semantic search and combines these results with graph data to generate informed responses via LLMs.
- **Graph Extraction**: Leverages language models like OpenAI's GPT or Ollama's Qwen to parse text and extract entities (nodes) and relationships (edges) in a structured JSON format.
- **Graph Storage & Querying**: Utilizes Neo4j to ingest, store, and query the extracted graph components, enabling advanced relationship and subgraph queries.

---

## Setup

Before running any code, ensure you have the necessary API keys and database credentials. You will need:

- **Qdrant**: API key and URL for your Qdrant instance (or local setup using Docker).
- **Neo4j**: Connection URI, username, and password.
- **Language Model**: Either an OpenAI API key or a local Ollama instance running.

### Prerequisites

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/graphrag.git
   cd graphrag
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a file named `.env` in the root of the repository and add your credentials. Refer to the .env.sample for guidance:

   ```env
   # Connection Settings
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=password

   # OpenAI Settings
   OPENAI_API_KEY=your_openai_api_key
   LLM_PROVIDER=openai  # or 'ollama' for local models
   LLM_MODEL=gpt-4o-mini
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small

   # Ollama Settings (if using local models)
   OLLAMA_HOST=localhost
   OLLAMA_PORT=11434
   OLLAMA_LLM_MODEL=qwen2.5:3b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text

   # Collection Settings
   COLLECTION_NAME=graphRAGstoreds
   VECTOR_DIMENSION=1536

   # Performance Settings
   PARALLEL_PROCESSING=true
   MAX_WORKERS=4
   BATCH_SIZE=100
   CHUNK_SIZE=5000
   USE_STREAMING=true
   ```

5. **Docker Setup (Optional):**

   For local development, you can use the provided Docker Compose file to spin up Neo4j, Qdrant, and Ollama:

   ```bash
   docker-compose up -d
   ```

---

## Usage

Once your environment is set up, you can run the pipeline by executing the main script:

```bash
python main.py
```

The interactive console allows you to:

1. **Ingest Data**: Extract graph components from text and store them in Neo4j and Qdrant.
2. **Clear Data**: Remove all data from Neo4j and Qdrant.
3. **Ask Questions**: Query the knowledge graph using natural language.
4. **Configure Settings**: Adjust performance parameters.
5. **Switch LLM Provider**: Toggle between OpenAI and Ollama for language model processing.

---

## LLM Processors

The system supports two LLM processors:

### OpenAI Processor
- Uses OpenAI's models (default: gpt-4o-mini) for graph extraction and RAG
- Requires an OpenAI API key
- Generally provides higher quality extraction but incurs API costs

### Ollama Processor
- Uses local models via Ollama (default: qwen2.5:3b)
- Runs entirely on your local machine - no API costs
- Requires the Ollama server running (included in Docker setup)
- Suitable for development and testing

You can switch between processors at runtime through the console interface.

---

## Components

### Graph Extraction

- **Functionality:**  
  Uses a custom prompt with OpenAI's GPT model to extract entities and their relationships from unstructured text.
- **Output:**  
  A structured JSON object containing:
  - **`graph`**: An array of relationship objects, each with:
    - `node`: The source entity.
    - `target_node`: The related entity.
    - `relationship`: The type of relationship.

### Neo4j Integration

- **Functionality:**  
  Ingests the extracted nodes and relationships into a Neo4j database, enabling robust graph queries.
- **Usage:**  
  Creates nodes labeled `Entity` and relationships between them based on the extracted data.

### Qdrant Vector Search

- **Functionality:**  
  Computes text embeddings for segments of the raw text using OpenAI and stores these vectors in Qdrant.
- **Usage:**  
  Facilitates semantic search by matching query embeddings with stored vectors.

### Retrieval-Augmented Generation (RAG)

- **Functionality:**  
  Integrates graph context (from Neo4j) with vector search results (from Qdrant) to enrich the prompt for natural language generation.
- **Usage:**  
  Uses OpenAI's GPT to generate detailed, context-aware answers to user queries.

---

## Project Structure

```
.
├── .env.sample           # Environment variables configuration (change this to .env)
├── requirements.txt     # Python dependencies
├── graphrag.py       # Main Python script containing the pipeline code
└── README.md            # Project documentation
```

- **`graphrag.py`**:  
  Contains the complete pipeline implementation, including graph extraction, ingestion into Neo4j, vector indexing in Qdrant, and the retrieval-augmented generation process.

- **`requirements.txt`**:  
  Lists all necessary Python packages (e.g., `neo4j-graphrag[qdrant]`, `python-dotenv`, `pydantic`, `openai`).

---

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request describing your changes.

---
