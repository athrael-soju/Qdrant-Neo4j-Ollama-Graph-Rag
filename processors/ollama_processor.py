"""
Ollama processor for graph component extraction using Qwen2.5:3b
"""

import uuid
import json
import re
import requests
import warnings
from typing import Dict, List, Tuple, Any, Optional
from processors.base_processor import BaseProcessor


class OllamaProcessor(BaseProcessor):
    """
    Ollama-based processor for extracting graph components from text
    Default model is Qwen2.5:3b
    """
    
    def __init__(self, 
                 model: str = "qwen2.5:3b", 
                 host: str = "localhost", 
                 port: int = 11434):
        """
        Initialize the Ollama processor
        
        Args:
            model: Ollama model to use (default: qwen2.5:3b)
            host: Ollama host (default: localhost)
            port: Ollama port (default: 11434)
        """
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
        # Check if model is available
        self._check_model_availability()
    
    def _check_model_availability(self) -> None:
        """
        Check if the specified model is available in Ollama.
        Provides a warning if the model is not found.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                if self.model not in available_models:
                    warnings.warn(
                        f"Model '{self.model}' not found in Ollama. Available models: {', '.join(available_models)}.\n"
                        f"You may need to pull it with: ollama pull {self.model}"
                    )
            else:
                warnings.warn(f"Failed to get model list from Ollama: {response.status_code}")
        except requests.RequestException as e:
            warnings.warn(f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running.")
    
    def _generate(self, prompt: str) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Prompt to send to Ollama
            
        Returns:
            Generated text
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more deterministic results
                "num_predict": 4096  # Longer output for comprehensive extraction
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=data)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                warnings.warn(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
        except requests.RequestException as e:
            warnings.warn(f"Error calling Ollama API: {str(e)}")
            return ""
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text to extract JSON from
            
        Returns:
            Extracted JSON object or None if extraction failed
        """
        # Try to find JSON between triple backticks
        json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find anything that looks like JSON
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                warnings.warn("No JSON found in response")
                return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues and try again
            try:
                # Replace single quotes with double quotes
                fixed_str = json_str.replace("'", '"')
                # Replace unquoted keys with quoted keys
                fixed_str = re.sub(r'(\s*)(\w+)(\s*):([^/])', r'\1"\2"\3:\4', fixed_str)
                return json.loads(fixed_str)
            except json.JSONDecodeError as e:
                warnings.warn(f"Failed to parse JSON: {str(e)}")
                return None
    
    def process_chunk(self, chunk_text: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Process a chunk of text to extract nodes and relationships.
        
        Args:
            chunk_text: Text chunk to process
            
        Returns:
            Tuple of (nodes, relationships)
        """
        system_prompt = """You are a precise graph relationship extractor. Extract all 
        relationships from the text and format them as a JSON object with this exact structure:
        {
            "graph": [
                {"node": "Person/Entity", 
                "target_node": "Related Entity", 
                "relationship": "Type of Relationship"},
                ...more relationships...
            ]
        }
        Include ALL relationships mentioned in the text, including implicit ones.
        Be thorough and precise. Return ONLY the JSON object with NO additional text."""
        
        prompt = f"{system_prompt}\n\nText to extract from:\n{chunk_text}"
        
        # Get the response from Ollama
        response_text = self._generate(prompt)
        
        # Extract JSON from the response
        json_response = self._extract_json(response_text)
        if not json_response or "graph" not in json_response:
            warnings.warn("Invalid response format or no relationships found")
            return {}, []
        
        # Process the response to extract nodes and relationships
        nodes = {}
        relationships = []
        
        for entry in json_response["graph"]:
            # Skip invalid entries
            if not isinstance(entry, dict) or "node" not in entry:
                continue
                
            node = entry.get("node", "").strip()
            target_node = entry.get("target_node", "").strip()
            relationship = entry.get("relationship", "").strip()
            
            # Skip entries with empty node names
            if not node:
                continue
                
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