import requests
import json
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Iterator
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OLLAMA_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
JSON_MODE = os.getenv("JSON_MODE", "true").lower() == "true"

# Base URL for Ollama API
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Define data models (same as OpenAI processor)
class Single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[Single]

@lru_cache(maxsize=128)
def cached_ollama_call(prompt, model=None):
    """Cached version of Ollama API call to avoid redundant calls"""
    if model is None:
        model = OLLAMA_LLM_MODEL
        
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # System prompt for graph extraction
    system_prompt = """You are a precise graph relationship extractor. Extract all 
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
        implicit ones. Be thorough and precise."""
    
    # Request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "format": "json" if JSON_MODE else None,
        "stream": False
    }
    
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    # Make the API call
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract content from response
        content = result.get("response", "{}")
        
        # Ensure it's valid JSON
        try:
            if not content.strip().startswith('{'):
                # Try to extract JSON from the text if it's not properly formatted
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > 0:
                    content = content[start_idx:end_idx]
            
            # Parse as JSON to validate
            json.loads(content)
            return content
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Invalid JSON response: {e}")
            # Return a minimal valid JSON as fallback
            return '{"graph": []}'
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {str(e)}")
        return '{"graph": []}'

def ollama_llm_parser(prompt):
    """Parse text into graph components using Ollama"""
    # Use the cached version
    content = cached_ollama_call(prompt)
    return GraphComponents.model_validate_json(content)

def ollama_embeddings(text):
    """Generate embeddings for a single text string using Ollama"""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    
    payload = {
        "model": OLLAMA_EMBEDDING_MODEL,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract embedding from response
        embedding = result.get("embedding", [0] * VECTOR_DIMENSION)
        
        # Ensure correct dimension
        if len(embedding) != VECTOR_DIMENSION:
            print(f"Warning: Expected dimension {VECTOR_DIMENSION}, got {len(embedding)}")
            # Pad or truncate to match expected dimension
            if len(embedding) < VECTOR_DIMENSION:
                embedding.extend([0] * (VECTOR_DIMENSION - len(embedding)))
            else:
                embedding = embedding[:VECTOR_DIMENSION]
                
        return embedding
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from Ollama: {str(e)}")
        return [0] * VECTOR_DIMENSION

def ollama_embeddings_batch(texts, batch_size=20):
    """
    Get embeddings for a list of texts in batches
    """
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch:
            try:
                embedding = ollama_embeddings(text)
                batch_embeddings.append(embedding)
                # Add a small delay to prevent overwhelming the API
                time.sleep(0.1)
            except Exception as e:
                print(f"Error embedding text: {str(e)}")
                batch_embeddings.append([0] * VECTOR_DIMENSION)
        
        all_embeddings.extend(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    return all_embeddings

def graphrag_query(graph_context, user_query, model=None, stream=True):
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
    
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    system_prompt = "Provide the answer for the following question:"
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": stream
    }
    
    try:
        if stream:
            # Handle streaming response
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                # Create a generator to mimic OpenAI's streaming format
                def response_generator():
                    for line in response.iter_lines():
                        if line:
                            try:
                                json_response = json.loads(line.decode('utf-8'))
                                # Create a structure similar to OpenAI's streaming format
                                if 'response' in json_response:
                                    # Create a class to mimic OpenAI's format
                                    class OllamaStreamResponse:
                                        class Choice:
                                            class Delta:
                                                def __init__(self, content):
                                                    self.content = content
                                            
                                            def __init__(self, content):
                                                self.delta = self.Delta(content)
                                                
                                        def __init__(self, content):
                                            self.choices = [self.Choice(content)]
                                            
                                    yield OllamaStreamResponse(json_response['response'])
                            except json.JSONDecodeError:
                                continue
                
                return response_generator()
        else:
            # Handle non-streaming response
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Create a class to mimic OpenAI's response format
            class OllamaResponse:
                class Message:
                    def __init__(self, content):
                        self.content = content
                
                def __init__(self, content):
                    self.content = content
                    
            return OllamaResponse(result.get('response', ''))
            
    except Exception as e:
        error_msg = f"Error querying Ollama LLM: {str(e)}"
        if stream:
            # Create a generator that yields the error message
            def error_generator():
                class ErrorResponse:
                    class Choice:
                        class Delta:
                            def __init__(self, content):
                                self.content = content
                        
                        def __init__(self, content):
                            self.delta = self.Delta(content)
                            
                    def __init__(self, content):
                        self.choices = [self.Choice(content)]
                
                yield ErrorResponse(error_msg)
            
            return error_generator()
        else:
            return error_msg 