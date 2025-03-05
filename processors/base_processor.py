"""
Base processor for graph component extraction
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel


class BaseProcessor(ABC):
    """
    Base class for all processors that extract graph components from text.
    All processor implementations should inherit from this class.
    """
    
    @abstractmethod
    def process_chunk(self, chunk_text: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Process a chunk of text to extract nodes and relationships.
        
        Args:
            chunk_text: Text chunk to process
            
        Returns:
            Tuple of (nodes, relationships), where:
            - nodes: Dictionary mapping node names to unique IDs
            - relationships: List of relationship dictionaries with source, target, and type
        """
        pass
    
    @abstractmethod
    def extract_components(self, text: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Extract graph components from the entire text.
        
        Args:
            text: Full text to process
            
        Returns:
            Tuple of (nodes, relationships), where:
            - nodes: Dictionary mapping node names to unique IDs
            - relationships: List of relationship dictionaries with source, target, and type
        """
        pass
        
    def _generate_node_id(self) -> str:
        """
        Generate a unique node ID.
        
        Returns:
            Unique node ID as string
        """
        return str(uuid.uuid4())
        
    def _add_node(self, node_name: str, nodes: Dict[str, str]) -> str:
        """
        Add a node to the nodes dictionary if it doesn't exist.
        
        Args:
            node_name: Name of the node
            nodes: Dictionary of existing nodes
            
        Returns:
            Node ID
        """
        if node_name not in nodes:
            nodes[node_name] = self._generate_node_id()
        return nodes[node_name]
        
    def _add_relationship(self, source_id: str, target_id: str, relationship_type: str, 
                          relationships: List[Dict[str, Any]]) -> None:
        """
        Add a relationship to the relationships list.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            relationships: List of existing relationships
        """
        relationships.append({
            "source": source_id,
            "target": target_id,
            "type": relationship_type
        }) 