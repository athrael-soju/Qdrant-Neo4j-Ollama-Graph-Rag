import os
import json
import requests
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Iterator
from dotenv import load_dotenv

load_dotenv()

# Use the new environment variable names for Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_INFERENCE_MODEL = os.getenv("OLLAMA_INFERENCE_MODEL", "qwen2.5:3b")
# Use OLLAMA_VECTOR_DIMENSION for this provider
VECTOR_DIMENSION = int(os.getenv("OLLAMA_VECTOR_DIMENSION", "768"))

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
    """
    if model is None:
        model = OLLAMA_INFERENCE_MODEL

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise graph relationship extractor. Extract all relationships from the text "
                    "and format them as a JSON object with this exact structure:\n"
                    '{ "graph": [ {"node": "Person/Entity", "target_node": "Related Entity", "relationship": "Type of Relationship"}, ... ] }\n'
                    "Include ALL relationships mentioned in the text, including implicit ones. Be thorough and precise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
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
    """
    content = cached_ollama_call(prompt)
    try:
        return GraphComponents.model_validate_json(content)
    except Exception as e:
        print(f"Error parsing JSON: {str(e)}")
        return GraphComponents(graph=[])

def ollama_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a single text string using Ollama.
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
    Get embeddings for a list of texts in batches.
    """
    all_embeddings = []
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
            all_embeddings.extend([[0] * VECTOR_DIMENSION] * len(batch))
    return all_embeddings

def graphrag_query(graph_context: Dict[str, List[str]], user_query: str,
                   model: Optional[str] = None, stream: bool = True) -> Union[str, Iterator[str]]:
    """
    Run RAG with the graph context using Ollama.
    """
    if model is None:
        model = OLLAMA_INFERENCE_MODEL

    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = (
        f"You are an intelligent assistant with access to the following knowledge graph:\n\n"
        f"Nodes: {nodes_str}\n\n"
        f"Edges: {edges_str}\n\n"
        f"Using this graph, answer the following question:\n\n"
        f'User Query: "{user_query}"'
    )

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ],
            "stream": stream,
        }
        if stream:
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True)
        else:
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()

        if stream:
            def response_generator():
                for chunk in response.iter_lines():
                    if chunk:
                        try:
                            json_chunk = json.loads(chunk.decode('utf-8'))
                            if 'message' in json_chunk and 'content' in json_chunk['message']:
                                content = json_chunk['message']['content']
                                if content.strip():
                                    yield content
                        except json.JSONDecodeError:
                            raw_text = chunk.decode('utf-8')
                            if raw_text.strip():
                                yield raw_text
            return response_generator()
        else:
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                return result['message']['content']
            else:
                return f"Error: Unexpected API response format: {result}"
    except Exception as e:
        error_msg = f"Error querying LLM: {str(e)}"
        if stream:
            def error_generator():
                yield error_msg
            return error_generator()
        else:
            return error_msg
