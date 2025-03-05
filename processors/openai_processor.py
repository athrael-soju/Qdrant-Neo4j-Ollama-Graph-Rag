from openai import OpenAI
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Iterator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") 
VECTOR_DIMENSION = int(os.getenv("OPENAI_VECTOR_DIMENSION", "1536"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Define data models
class Single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[Single]

@lru_cache(maxsize=128)
def cached_openai_call(prompt, model=None):
    """Cached version of OpenAI API call to avoid redundant calls"""
    if model is None:
        model = OPENAI_LLM_MODEL
        
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
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
    )
    
    return completion.choices[0].message.content

def openai_llm_parser(prompt):
    """Parse text into graph components using OpenAI"""
    # Use the cached version
    content = cached_openai_call(prompt)
    return GraphComponents.model_validate_json(content)

def openai_embeddings(text):
    """Generate embeddings for a single text string"""
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL
    )
    
    return response.data[0].embedding

def openai_embeddings_batch(texts, batch_size=20):
    """
    Get embeddings for a list of texts in batches to reduce API calls.
    """
    client = OpenAI()
    all_embeddings = []
    
    # Process in batches to avoid rate limits and improve performance
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=OPENAI_EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            # Add empty embeddings as placeholders for failed items
            all_embeddings.extend([[0] * VECTOR_DIMENSION] * len(batch))
    
    return all_embeddings

def graphrag_query(graph_context, user_query, model=None, stream=True):
    """
    Run RAG with the graph context using OpenAI.
    
    Args:
        graph_context: The graph context with nodes and edges
        user_query: The user's question
        model: The OpenAI model to use (defaults to env variable)
        stream: Whether to stream the response (default True)
        
    Returns:
        If stream=True: Iterator yielding chunks of the response
        If stream=False: Complete response message
    """
    if model is None:
        model = OPENAI_LLM_MODEL
        
    client = OpenAI()
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ],
            stream=stream
        )
        
        if stream:
            return response  # Return the streaming response generator
        else:
            return response.choices[0].message
    
    except Exception as e:
        error_msg = f"Error querying LLM: {str(e)}"
        if stream:
            # Create a generator that yields the error message
            def error_generator():
                yield error_msg
            return error_generator()
        else:
            return error_msg 