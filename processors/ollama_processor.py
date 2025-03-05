from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Iterator
import os
import json
import requests
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))

# Export variables for processor_factory.py
LLM_MODEL = OLLAMA_LLM_MODEL
EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL

# Define data models (same as in OpenAI processor)
class Single(BaseModel):
    """
    Represents a single relationship between two nodes in a knowledge graph.
    """
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    """
    Represents a collection of relationships in a knowledge graph.
    """
    graph: list[Single]

@lru_cache(maxsize=128)
def cached_ollama_call(prompt: str, model: Optional[str] = None) -> str:
    """
    Cached version of Ollama API call to avoid redundant calls.
    
    Args:
        prompt: The text prompt to send to the model
        model: The model name to use, defaults to OLLAMA_LLM_MODEL if None
        
    Returns:
        The model's response as a string
    """
    if model is None:
        model = OLLAMA_LLM_MODEL

    # Prepare the request payload
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": 
                   
                """ You are a precise graph relationship extractor. Extract all 
                    relationships from the text and format them as a JSON object 
                    with this exact structure:
                    {
                        "graph": [
                            {"node": "Person/Entity", 
                             "target_node": "Related Entity", 
                             "relationship": "Type of Relationship"},
                            ...more relationships...
                        ]
                    }
                    Include ALL relationships mentioned in the text, including 
                    implicit ones. Be thorough and precise. """
                    
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # Call the Ollama API using the chat completions endpoint
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the message content from the chat completion result
        if 'message' in result and 'content' in result['message']:
            return result['message']['content']
        else:
            print(f"Warning: Unexpected API response format: {result}")
            return """{"graph": []}"""
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_BASE_URL}")
        return """{"graph": []}"""
    except Exception as e:
        print(f"Error in Ollama API call: {str(e)}")
        return """{"graph": []}"""

def ollama_llm_parser(prompt: str) -> GraphComponents:
    """
    Parse text into graph components using Ollama.
    
    Args:
        prompt: The text to extract relationships from
        
    Returns:
        GraphComponents object containing the extracted relationships
    """
    # Use the cached version
    content = cached_ollama_call(prompt)

    try:
        # Try to parse the JSON
        return GraphComponents.model_validate_json(content)
    except Exception as e:
        print(f"Error parsing JSON: {str(e)}")
        return GraphComponents(graph=[])

def ollama_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a single text string using Ollama.
    
    Args:
        text: The text to generate embeddings for
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        payload = {
            "model": OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [0] * VECTOR_DIMENSION)
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0] * VECTOR_DIMENSION

def ollama_embeddings_batch(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """
    Get embeddings for a list of texts in batches to reduce API calls.
    
    Args:
        texts: List of text strings to generate embeddings for
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embedding vectors (each as a list of floats)
    """
    all_embeddings = []
    
    # Process in batches to improve performance
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        
        try:
            for text in batch:
                embedding = ollama_embeddings(text)
                batch_embeddings.append(embedding)
                
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            # Add empty embeddings as placeholders for failed items
            all_embeddings.extend([[0] * VECTOR_DIMENSION] * len(batch))
    
    return all_embeddings

def graphrag_query(graph_context: Dict[str, List[str]], user_query: str, 
                  model: Optional[str] = None, stream: bool = True) -> Union[str, Iterator[str]]:
    """
    Run RAG with the graph context using Ollama.
    
    Args:
        graph_context: The graph context with nodes and edges
        user_query: The user's question
        model: The Ollama model to use (defaults to env variable)
        stream: Whether to stream the response (default True)
        
    Returns:
        If stream=True: Iterator yielding chunks of the response
        If stream=False: Complete response message
    """
    if model is None:
        model = OLLAMA_LLM_MODEL
        
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    
    # Create a simple system prompt for answering queries with graph context
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    try:
        # Create the appropriate payload based on streaming or not
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ],
            "stream": stream,
        }

        if stream:
            # For streaming, we need to set stream=True in the requests call
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True)
        else:
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            
        response.raise_for_status()
        
        if stream:
            # Simple generator for streaming responses
            def response_generator():
                for chunk in response.iter_lines():
                    if chunk:
                        try:
                            json_chunk = json.loads(chunk.decode('utf-8'))
                            if 'message' in json_chunk and 'content' in json_chunk['message']:
                                content = json_chunk['message']['content']
                                # Only yield content if it's not empty
                                if content.strip():
                                    yield content
                        except json.JSONDecodeError:
                            # If not valid JSON, just yield the raw text
                            raw_text = chunk.decode('utf-8')
                            if raw_text.strip():
                                yield raw_text
            
            # Return the generator directly, don't consume it yet
            return response_generator()
        else:
            # Parse the complete response
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                return result['message']['content']
            else:
                return f"Error: Unexpected API response format: {result}"
                
    except Exception as e:
        error_msg = f"Error querying LLM: {str(e)}"
        if stream:
            # Create a generator that yields the error message
            def error_generator():
                yield error_msg
            return error_generator()
        else:
            return error_msg 