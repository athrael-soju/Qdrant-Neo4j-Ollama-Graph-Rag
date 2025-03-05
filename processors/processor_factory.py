"""
Factory for creating processors
"""

from typing import Dict, Any, Optional, Type
from processors.base_processor import BaseProcessor
from processors.openai_processor import OpenAIProcessor
from processors.ollama_processor import OllamaProcessor


class ProcessorFactory:
    """
    Factory class for creating graph component processors.
    """
    
    @staticmethod
    def create_processor(processor_type: str = "ollama", config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
        """
        Create a processor instance based on the specified type and configuration.
        
        Args:
            processor_type: Type of processor to create (default: "ollama")
            config: Optional configuration dictionary
            
        Returns:
            Instance of a BaseProcessor subclass
        """
        if config is None:
            config = {}
            
        processor_type = processor_type.lower()
        
        if processor_type == "openai":
            model = config.get("model", "gpt-4o-mini")
            api_key = config.get("api_key", None)
            return OpenAIProcessor(model=model, api_key=api_key)
        elif processor_type == "ollama":
            model = config.get("model", "qwen2.5:3b")
            host = config.get("host", "localhost")
            port = config.get("port", 11434)
            return OllamaProcessor(model=model, host=host, port=port)
        else:
            # Default to Ollama processor
            print(f"Unknown processor type: {processor_type}. Using Ollama processor instead.")
            model = config.get("model", "qwen2.5:3b")
            host = config.get("host", "localhost")
            port = config.get("port", 11434)
            return OllamaProcessor(model=model, host=host, port=port)
    
    @staticmethod
    def get_available_processors() -> Dict[str, str]:
        """
        Get a dictionary of available processors and their descriptions.
        
        Returns:
            Dictionary mapping processor types to descriptions
        """
        return {
            "openai": "OpenAI-based processor (requires API key)",
            "ollama": "Ollama-based processor (requires Ollama service)"
        } 