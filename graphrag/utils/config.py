"""
Configuration utilities for GraphRAG.
"""

import os
import json
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "testpassword"
    },
    "qdrant": {
        "host": "localhost",
        "port": 6333
    },
    "embedding": {
        "model_name": "intfloat/e5-base",
        "device": "cpu"
    },
    "triplet_extraction": {
        "model_name": "google/flan-t5-base",
        "device": "cpu"
    },
    "data_dir": "data"
}

CONFIG_FILE_PATH = os.path.expanduser("~/.graphrag/config.json")

def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Returns:
        Configuration dictionary
    """
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r") as f:
                user_config = json.load(f)
                # Merge with defaults
                config = DEFAULT_CONFIG.copy()
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            return DEFAULT_CONFIG
    else:
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
    """
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump(config, f, indent=2)

def get_config_value(key: str, default: Optional[Any] = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Configuration key (can use dot notation for nested keys)
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = load_config()
    keys = key.split(".")
    
    current = config
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
            
    return current 