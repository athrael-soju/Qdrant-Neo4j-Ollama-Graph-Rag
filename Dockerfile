FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements separately for better caching
COPY setup.py README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY graphrag ./graphrag/

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download transformer models (to cache them in the image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"

# Set Neo4j connection environment variables (can be overridden at runtime)
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=testpassword

# Create a sample text file
RUN echo "Hugging Face, Inc. is an American company that develops tools for building applications using machine learning. It was founded in 2016 and its headquarters is in New York City." > /app/huggingface.txt

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Override Neo4j connection from environment variables\n\
sed -i "s|uri = \"bolt://localhost:7687\"|uri = \"${NEO4J_URI}\"|g" /app/graphrag/neo4j_connection.py\n\
sed -i "s|auth=(\"neo4j\", \"testpassword\")|auth=(\"${NEO4J_USER}\", \"${NEO4J_PASSWORD}\")|g" /app/graphrag/neo4j_connection.py\n\
\n\
# Run the specified command\n\
exec "$@"\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["graphrag", "interactive"] 