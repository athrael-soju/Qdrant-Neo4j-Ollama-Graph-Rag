# GraphRAG with Neo4j, Qdrant, and OpenAI

## Overview

This project demonstrates how to build a Graph Retrieval-Augmented Generation (RAG) pipeline that extracts graph relationships from raw text using either OpenAI's GPT models or spaCy's NLP capabilities, stores and queries these relationships in a Neo4j graph database, and enhances the process with Qdrant's vector search capabilities. By integrating these technologies, users can extract structured insights from unstructured text and perform complex graph queries to generate context-aware natural language responses.

The system is built on three main components:

- **Vector Search & RAG**: Uses Qdrant to index text embeddings for semantic search and combines these results with graph data to generate informed responses via OpenAI's GPT.
- **Graph Extraction**: You can choose between:
  - **OpenAI GPT**: Leverages GPT models to parse text and extract entities and relationships in a structured JSON format (requires API key and may incur costs).
  - **spaCy**: Uses local NLP processing for entity recognition and relationship extraction (faster, free, but potentially less accurate).
- **Graph Storage & Querying**: Utilizes Neo4j to ingest, store, and query the extracted graph components, enabling advanced relationship and subgraph queries.

---

## Setup

Before running any code, ensure you have the necessary API keys and database credentials. You will need:

- **Qdrant**: API key and URL for your Qdrant instance.
- **Neo4j**: Connection URI, username, and password.
- **OpenAI** (Optional if using spaCy parser): API key for OpenAI services.

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

   Create a file named `.env` in the root of the repository and add your credentials. Refer to the .env.sample for guidance.:

   ```env
   # Qdrant configuration
   QDRANT_KEY=your_qdrant_api_key
   QDRANT_URL=your_qdrant_instance_url

   # Neo4j configuration
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password

   # OpenAI configuration
   OPENAI_API_KEY=your_openai_api_key
   ```

### Installing spaCy

If you plan to use the spaCy parser, install the required language model:

```bash
# After installing requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

Once your environment is set up, you can run the pipeline by executing the main script:

```bash
python graphrag.py
```

The script performs the following steps:

1. **Environment Initialization:**  
   Loads API keys and database credentials from `.env`.

2. **Graph Extraction:**  
   Uses OpenAI's GPT model to extract graph components (nodes and relationships) from raw text input.

3. **Data Ingestion:**

   - **Neo4j Ingestion:** Inserts the extracted nodes and relationships into a Neo4j graph database.
   - **Qdrant Ingestion:** Computes text embeddings for segments of the raw data and uploads them to a Qdrant collection.

4. **Retrieval & Graph Querying:**  
   Performs a vector search in Qdrant to identify relevant sections of the text, then queries Neo4j to fetch related graph context.

5. **Retrieval-Augmented Generation (RAG):**  
   Combines the graph context with the vector search results to generate a detailed answer to a user query via OpenAI's GPT.

Upon running the main application:

```
python main.py
```

You will be presented with a console-based interactive menu with several options:

1. **Ingest Data:**  
   Processes raw text data from the sample file, extracts graph components, and ingests them into Neo4j and Qdrant.

2. **Clear All Data:**  
   Wipes all data from both Neo4j and Qdrant.

3. **Ask a Question:**  
   Enter a natural language query to search the knowledge graph and receive an AI-generated response.

4. **Configure Optimization Settings:**  
   Adjust parameters for parallel processing, workers, batch sizes, and chunking.

5. **Choose Parser:**  
   Select which parser to use for extracting entities and relationships:
   - **OpenAI LLM**: More accurate but requires an API key and incurs costs.
   - **spaCy**: Faster, free local processing but may be less accurate.

6. **Exit:**  
   Close the application.

### Parser Comparison

**OpenAI LLM Parser:**
- Pros: Higher accuracy, better relationship extraction, understands context
- Cons: Requires API key, incurs costs, slower processing time

**spaCy Parser:**
- Pros: Free, faster processing, works offline
- Cons: May miss complex relationships, less context-aware

Choose the parser based on your needs: use spaCy for quick testing or when budget is a concern, and OpenAI for production-quality relationship extraction.

Console logs will provide detailed information on each step, including extraction progress, ingestion status, retrieval results, and the final generated response.

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
