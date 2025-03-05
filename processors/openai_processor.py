"""
OpenAI processor for graph component extraction
"""

import uuid
from functools import lru_cache
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel
from openai import OpenAI
import os
from processors.base_processor import BaseProcessor


class SingleRelationship(BaseModel):
    """Single relationship model for OpenAI response parsing"""
    node: str
    target_node: str
    relationship: str


class GraphComponents(BaseModel):
    """Graph components model for OpenAI response parsing"""
    graph: List[SingleRelationship]


class OpenAIProcessor(BaseProcessor):
    """
    Processor that uses OpenAI API to extract graph components from text.
    """
    
    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        Initialize the OpenAI processor.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.model = model
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    @lru_cache(maxsize=128)
    def _cached_openai_call(self, prompt, model=None):
        """
        Cached version of OpenAI API call to avoid redundant calls.
        
        Args:
            prompt: Prompt to send to OpenAI
            model: Optional model override
            
        Returns:
            Response content as string
        """
        if model is None:
            model = self.model
            
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
    
    def _openai_llm_parser(self, prompt):
        """
        Parse text into graph components using OpenAI.
        
        Args:
            prompt: Text prompt to process
            
        Returns:
            GraphComponents model
        """
        # Use the cached version
        content = self._cached_openai_call(prompt)
        return GraphComponents.model_validate_json(content)
    
    def process_chunk(self, chunk_text: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Process a chunk of text to extract nodes and relationships.
        
        Args:
            chunk_text: Text chunk to process
            
        Returns:
            Tuple of (nodes, relationships)
        """
        prompt = f"Extract nodes and relationships from the following text:\n{chunk_text}"
        parsed_response = self._openai_llm_parser(prompt).graph
        
        nodes = {}
        relationships = []
        
        for entry in parsed_response:
            node = entry.node
            target_node = entry.target_node
            relationship = entry.relationship
            
            # Add nodes to the dictionary with unique IDs
            source_id = self._add_node(node, nodes)
            
            if target_node:
                target_id = self._add_node(target_node, nodes)
                
                # Add relationship if target and relationship exist
                if relationship:
                    self._add_relationship(source_id, target_id, relationship, relationships)
                    
        return nodes, relationships
    
    def extract_components(self, text: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Extract graph components from the entire text.
        
        Args:
            text: Full text to process
            
        Returns:
            Tuple of (nodes, relationships)
        """
        return self.process_chunk(text) 