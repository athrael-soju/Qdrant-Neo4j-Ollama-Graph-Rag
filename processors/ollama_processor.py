import requests
import json
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Iterator
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OLLAMA_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_URL_CHAT = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"
OLLAMA_URL_EMBEDDINGS = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
OLLAMA_URL_GENERATE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
OLLAMA_JSON_MODE = os.getenv("JSON_MODE", "true").lower() == "true"

# Define RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that provides accurate information based on the context provided.

Context information:
{context}

User question: {question}

Please provide a helpful, accurate answer based only on the information in the context. If the context doesn't contain the answer, say you don't know.
"""

# Define data models (same as OpenAI processor for compatibility)
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
    
    headers = {"Content-Type": "application/json"}
    
    # Construct system prompt for graph extraction
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

    # Create payload for chat endpoint
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "format": "json" if OLLAMA_JSON_MODE else None,
    }
    
    # Remove None values from payload
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.post(OLLAMA_URL_CHAT, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()
        
        # Get content from response
        if "message" in response_data and "content" in response_data["message"]:
            content = response_data["message"]["content"]
            
            # Try to extract JSON from the content
            # First, check if content is already valid JSON
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                # Not valid JSON, try to extract JSON part
                try:
                    # Look for JSON-like structure between curly braces
                    import re
                    json_pattern = r'({[\s\S]*})'
                    match = re.search(json_pattern, content)
                    if match:
                        potential_json = match.group(1)
                        # Validate it's proper JSON
                        json.loads(potential_json)
                        return potential_json
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Failed to extract JSON from content: {str(e)}")
                    print(f"Raw content: {content[:100]}...")
                    
            # If we get here, we couldn't extract valid JSON
            # Return a fallback empty JSON structure
            print(f"Could not extract valid JSON from Ollama response, returning empty graph")
            return '{"graph": []}'
        else:
            print(f"Unexpected response structure from Ollama: {response_data}")
            return '{"graph": []}'
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        # Return empty JSON structure on error
        return '{"graph": []}'

def ollama_llm_parser(prompt):
    """Parse text into graph components using Ollama"""
    # Use the cached version
    content = cached_ollama_call(prompt)
    
    # Try to parse the content as JSON
    try:
        # First try to parse as standard JSON
        try:
            graph_components = GraphComponents.model_validate_json(content)
            if graph_components and hasattr(graph_components, 'graph') and len(graph_components.graph) > 0:
                print(f"Successfully extracted {len(graph_components.graph)} relationships")
                return graph_components
        except Exception as e:
            print(f"Error parsing standard JSON: {str(e)}")
        
        # If that fails, try to extract nested JSON structure
        try:
            parsed_json = json.loads(content)
            
            # Check for various possible formats
            if isinstance(parsed_json, dict):
                # Direct format: {"graph": [...]}
                if "graph" in parsed_json:
                    return GraphComponents(graph=[
                        Single(
                            node=item.get("node", "Unknown"), 
                            target_node=item.get("target_node", "Unknown"), 
                            relationship=item.get("relationship", "Unknown")
                        ) 
                        for item in parsed_json["graph"]
                    ])
                
                # Possible nested formats
                for key in parsed_json:
                    if isinstance(parsed_json[key], dict) and "graph" in parsed_json[key]:
                        nested_graph = parsed_json[key]["graph"]
                        return GraphComponents(graph=[
                            Single(
                                node=item.get("node", "Unknown"), 
                                target_node=item.get("target_node", "Unknown"), 
                                relationship=item.get("relationship", "Unknown")
                            ) 
                            for item in nested_graph
                        ])
            
            # If no valid structure found, log and return empty
            print("No valid graph structure found in JSON")
            
        except Exception as e:
            print(f"Error extracting nested JSON: {str(e)}")
        
        # Last resort: try to manually extract relationships from text
        try:
            print("Attempting manual relationship extraction from text")
            # Basic regex to find relationship patterns
            import re
            relationships = []
            
            # Look for patterns like "Entity -> Relationship -> Target"
            pattern = r"[\"']?([^\"']+)[\"']?\s*(?:->|:|\|)\s*[\"']?([^\"']+)[\"']?\s*(?:->|:|\|)\s*[\"']?([^\"']+)[\"']?"
            matches = re.findall(pattern, content)
            
            # Add any found relationships
            for match in matches:
                if len(match) >= 3:
                    relationships.append(
                        Single(node=match[0].strip(), target_node=match[2].strip(), relationship=match[1].strip())
                    )
            
            if relationships:
                print(f"Manually extracted {len(relationships)} potential relationships")
                return GraphComponents(graph=relationships)
            
        except Exception as e:
            print(f"Error in manual extraction: {str(e)}")
        
        # If all else fails, return empty graph
        print("All parsing attempts failed, returning empty graph")
        return GraphComponents(graph=[])
        
    except Exception as e:
        print(f"Unhandled error parsing Ollama response: {str(e)}")
        # Return empty graph on error
        return GraphComponents(graph=[])

def ollama_embeddings(text):
    """Generate embeddings for a single text string"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_EMBEDDING_MODEL,
        "prompt": text
    }
    
    # Maximum retries
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(OLLAMA_URL_EMBEDDINGS, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            content = response.json()
            
            if "embedding" in content and content["embedding"]:
                # Check if the embedding is the expected dimension
                if len(content["embedding"]) != VECTOR_DIMENSION:
                    print(f"Warning: Embedding dimension mismatch. Expected {VECTOR_DIMENSION}, got {len(content['embedding'])}.")
                    # Pad or truncate to match expected dimension
                    if len(content["embedding"]) < VECTOR_DIMENSION:
                        # Pad with zeros
                        return content["embedding"] + [0] * (VECTOR_DIMENSION - len(content["embedding"]))
                    else:
                        # Truncate
                        return content["embedding"][:VECTOR_DIMENSION]
                return content["embedding"]
            else:
                print(f"Error: No embedding found in response: {content}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying ({retry_count}/{max_retries})...")
                    time.sleep(1)  # Add a short delay before retrying
                    
        except requests.exceptions.Timeout:
            print(f"Timeout error when getting embeddings. Retrying ({retry_count+1}/{max_retries})...")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)  # Slightly longer delay for timeout errors
                
        except Exception as e:
            print(f"Error getting embeddings from Ollama: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying ({retry_count}/{max_retries})...")
                time.sleep(1)
    
    # If we've exhausted retries, return a zeros vector
    print(f"Failed to get embeddings after {max_retries} attempts, returning zeros vector")
    return [0] * VECTOR_DIMENSION

def ollama_embeddings_batch(texts, batch_size=20):
    """
    Get embeddings for a list of texts in batches.
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
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                # Add zeros vector on error
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
    
    headers = {"Content-Type": "application/json"}
    
    # Construct the payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Provide the answer for the following question:"},
            {"role": "user", "content": prompt}
        ],
        "stream": stream
    }
    
    try:
        if stream:
            # For streaming mode, create a generator that yields response chunks
            def stream_generator():
                response = requests.post(
                    OLLAMA_URL_CHAT, 
                    headers=headers, 
                    data=json.dumps(payload),
                    stream=True
                )
                response.raise_for_status()
                
                # Parse the streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'message' in chunk and 'content' in chunk['message']:
                                # Create a structure similar to OpenAI's streaming format
                                yield type('OllamaStreamChunk', (), {
                                    'choices': [
                                        type('Choice', (), {
                                            'delta': type('Delta', (), {
                                                'content': chunk['message']['content']
                                            })
                                        })
                                    ]
                                })
                        except json.JSONDecodeError:
                            continue
            
            return stream_generator()
        else:
            # For non-streaming mode, return the complete response
            payload["stream"] = False
            response = requests.post(OLLAMA_URL_CHAT, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            content = response.json()
            
            # Create a structure similar to OpenAI's response format
            return type('OllamaResponse', (), {
                'content': content['message']['content']
            })
    
    except Exception as e:
        error_msg = f"Error querying Ollama LLM: {str(e)}"
        if stream:
            # Create a generator that yields the error message
            def error_generator():
                yield type('OllamaErrorChunk', (), {
                    'choices': [
                        type('Choice', (), {
                            'delta': type('Delta', (), {
                                'content': error_msg
                            })
                        })
                    ]
                })
            return error_generator()
        else:
            return error_msg 

def generate_rag_response(prompt, context, prompt_template=RAG_PROMPT_TEMPLATE):
    """Generate a RAG response using the Ollama API"""
    if not context.strip():
        context = "No relevant context found. Please respond based on your general knowledge."
    
    final_prompt = prompt_template.format(context=context, question=prompt)
    
    # Prepare the request payload
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": final_prompt,
        "stream": False
    }
    
    # Maximum retries
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = cached_ollama_call(OLLAMA_URL_GENERATE, headers, payload)
            
            if "response" in response:
                return {
                    "answer": response["response"],
                    "sources": []  # Ollama doesn't provide source attribution
                }
            else:
                print(f"Error: No 'response' key in Ollama response: {response}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying RAG response generation ({retry_count}/{max_retries})...")
                    time.sleep(1)
                    
        except Exception as e:
            print(f"Error generating RAG response from Ollama: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying RAG response generation ({retry_count}/{max_retries})...")
                time.sleep(2)
    
    # Fallback response if all retries fail
    return {
        "answer": "I'm sorry, I couldn't generate a proper response at this time. There was an error communicating with the language model. Please try again or check the system logs for more information.",
        "sources": []
    } 