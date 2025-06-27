"""
Configuration utilities for refinire-rag-chroma

Provides configuration classes and utilities following the refinire-rag plugin development guide.
Supports environment variable-based configuration.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for ChromaVectorStore
    
    Returns:
        Dictionary containing default configuration values
    """
    return {
        "collection_name": "refinire_documents",
        "persist_directory": None,
        "distance_metric": "cosine",
        "batch_size": 100,
        "max_retries": 3,
        "auto_create_collection": True,
        "auto_clear_on_init": False
    }


def load_config_from_environment() -> Dict[str, Any]:
    """
    Load configuration from environment variables
    
    Returns:
        Dictionary containing configuration from environment variables
    """
    def parse_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on")
    
    config = get_default_config()
    
    # Override with environment variables
    if os.getenv("REFINIRE_RAG_CHROMA_COLLECTION_NAME"):
        config["collection_name"] = os.getenv("REFINIRE_RAG_CHROMA_COLLECTION_NAME")
    
    if os.getenv("REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY"):
        value = os.getenv("REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY")
        config["persist_directory"] = None if value == "" else value
    
    if os.getenv("REFINIRE_RAG_CHROMA_DISTANCE_METRIC"):
        config["distance_metric"] = os.getenv("REFINIRE_RAG_CHROMA_DISTANCE_METRIC")
    
    if os.getenv("REFINIRE_RAG_CHROMA_BATCH_SIZE"):
        try:
            config["batch_size"] = int(os.getenv("REFINIRE_RAG_CHROMA_BATCH_SIZE"))
        except ValueError:
            logger.warning(f"Invalid batch_size value, using default: {config['batch_size']}")
    
    if os.getenv("REFINIRE_RAG_CHROMA_MAX_RETRIES"):
        try:
            config["max_retries"] = int(os.getenv("REFINIRE_RAG_CHROMA_MAX_RETRIES"))
        except ValueError:
            logger.warning(f"Invalid max_retries value, using default: {config['max_retries']}")
    
    if os.getenv("REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION"):
        config["auto_create_collection"] = parse_bool(os.getenv("REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION"))
    
    if os.getenv("REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT"):
        config["auto_clear_on_init"] = parse_bool(os.getenv("REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT"))
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config.get("collection_name") or not config["collection_name"].strip():
        raise ValueError("Collection name cannot be empty")
    
    if config.get("distance_metric") not in ["cosine", "l2", "ip"]:
        raise ValueError(f"Invalid distance metric: {config.get('distance_metric')}. Must be one of: cosine, l2, ip")
    
    if config.get("batch_size", 0) <= 0:
        raise ValueError(f"Batch size must be positive, got: {config.get('batch_size')}")
    
    if config.get("max_retries", -1) < 0:
        raise ValueError(f"Max retries must be non-negative, got: {config.get('max_retries')}")
    
    persist_directory = config.get("persist_directory")
    if persist_directory is not None:
        # Validate directory is writable if it exists, or can be created
        try:
            if os.path.exists(persist_directory):
                if not os.path.isdir(persist_directory):
                    raise ValueError(f"Persist directory is not a directory: {persist_directory}")
                if not os.access(persist_directory, os.W_OK):
                    raise ValueError(f"Persist directory is not writable: {persist_directory}")
            else:
                # Try to create parent directory if it doesn't exist
                parent_dir = os.path.dirname(persist_directory)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot access persist directory {persist_directory}: {e}")


def get_env_template() -> str:
    """
    Get environment variable template
    
    Returns:
        String containing environment variable template
    """
    return """# ChromaDB VectorStore Configuration
# Set these environment variables to configure the ChromaDB plugin

# Collection name for ChromaDB
REFINIRE_RAG_CHROMA_COLLECTION_NAME=refinire_documents

# Directory for persistent storage (leave empty or comment out for in-memory storage)
REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=./chroma_db

# Distance metric for similarity search (cosine, l2, ip)
REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine

# Batch size for bulk operations
REFINIRE_RAG_CHROMA_BATCH_SIZE=100

# Maximum retry attempts for failed operations
REFINIRE_RAG_CHROMA_MAX_RETRIES=3

# Auto-create collection if it doesn't exist (true/false)
REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION=true

# Clear collection on initialization - use only for testing (true/false)
REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT=false
"""