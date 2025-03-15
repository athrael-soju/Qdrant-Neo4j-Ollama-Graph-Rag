from openai import OpenAI
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Iterator
import os
from dotenv import load_dotenv

load_dotenv()

# Use the new environment variable names for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_INFERENCE_MODEL = os.getenv("OPENAI_INFERENCE_MODEL", "gpt-4o-mini")
# Use OPENAI_VECTOR_DIMENSION for this provider
VECTOR_DIMENSION = int(os.getenv("OPENAI_VECTOR_DIMENSION", "1536"))

# Define data models
class Single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[Single]

@lru_cache(maxsize=128)
def cached_openai_call(prompt: str, model: Optional[str] = None) -> str:
    """
    Cached version of the OpenAI API call to avoid redundant calls.
    """
    if model is None:
        model = OPENAI_INFERENCE_MODEL

    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise graph relationship extractor. Extract all "
                    "relationships from the text and format them as a JSON object with this exact structure:\n"
                    '{ "graph": [ {"node": "Person/Entity", "target_node": "Related Entity", "relationship": "Type of Relationship"}, ... ] }\n'
                    "Include ALL relationships mentioned in the text, including implicit ones. Be thorough and precise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def openai_llm_parser(prompt: str) -> GraphComponents:
    """Parse text into graph components using OpenAI."""
    content = cached_openai_call(prompt)
    return GraphComponents.model_validate_json(content)

def openai_embeddings(text: str) -> List[float]:
    """Generate embeddings for a single text string using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL
    )
    return response.data[0].embedding

def openai_embeddings_batch(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """
    Get embeddings for a list of texts in batches to reduce API calls.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []

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
            all_embeddings.extend([[0] * VECTOR_DIMENSION] * len(batch))

    return all_embeddings

def graphrag_query(graph_context: Dict[str, List[str]], user_query: str,
                   model: Optional[str] = None, stream: bool = True) -> Union[str, Iterator[str]]:
    """
    Run RAG with the graph context using OpenAI.
    """
    if model is None:
        model = OPENAI_INFERENCE_MODEL

    client = OpenAI(api_key=OPENAI_API_KEY)
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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ],
            stream=stream
        )

        if stream:
            return response  # Return streaming response generator
        else:
            return response.choices[0].message
    except Exception as e:
        error_msg = f"Error querying LLM: {str(e)}"
        if stream:
            def error_generator():
                yield error_msg
            return error_generator()
        else:
            return error_msg
