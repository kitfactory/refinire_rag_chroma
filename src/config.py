"""
Configuration utilities for refinire-rag-chroma

Provides configuration classes and utilities following the refinire-rag plugin development guide.
Supports environment variable-based configuration with oneenv integration.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChromaVectorStoreConfig:
    """
    Configuration class for ChromaVectorStore plugin
    
    Follows refinire-rag plugin development guide requirements:
    - Property-based configuration
    - Environment variable integration
    - Validation support
    """
    
    @property
    def collection_name(self) -> str:
        """ChromaDB collection name"""
        return get_env_str("REFINIRE_RAG_CHROMA_COLLECTION_NAME", "refinire_documents")
    
    @property
    def persist_directory(self) -> Optional[str]:
        """ChromaDB persistence directory (None for in-memory)"""
        return get_env_optional_str("REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY")
    
    @property
    def distance_metric(self) -> str:
        """Distance metric for similarity search"""
        return get_env_str("REFINIRE_RAG_CHROMA_DISTANCE_METRIC", "cosine")
    
    @property
    def batch_size(self) -> int:
        """Batch size for bulk operations"""
        return get_env_int("REFINIRE_RAG_CHROMA_BATCH_SIZE", 100)
    
    @property
    def max_retries(self) -> int:
        """Maximum retry attempts for failed operations"""
        return get_env_int("REFINIRE_RAG_CHROMA_MAX_RETRIES", 3)
    
    @property
    def auto_create_collection(self) -> bool:
        """Auto-create collection if it doesn't exist"""
        return get_env_bool("REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION", True)
    
    @property
    def auto_clear_on_init(self) -> bool:
        """Clear collection on initialization (testing only)"""
        return get_env_bool("REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT", False)
    
    def validate(self) -> None:
        """Validate configuration values"""
        if not self.collection_name or not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if self.distance_metric not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Invalid distance metric: {self.distance_metric}. Must be one of: cosine, l2, ip")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {self.batch_size}")
        
        if self.max_retries < 0:
            raise ValueError(f"Max retries must be non-negative, got: {self.max_retries}")
        
        if self.persist_directory is not None and not validate_directory(self.persist_directory):
            raise ValueError(f"Invalid persist directory: {self.persist_directory}")


def get_env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean value from environment variable"""
    value = os.getenv(key, str(default).lower())
    return value.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int) -> int:
    """Parse integer value from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid integer value for {key}, using default: {default}")
        return default


def get_env_str(key: str, default: str) -> str:
    """Get string value from environment variable"""
    return os.getenv(key, default)


def get_env_optional_str(key: str) -> Optional[str]:
    """Get optional string value from environment variable"""
    value = os.getenv(key)
    return None if value == "" else value


def validate_directory(path: str) -> bool:
    """Validate that directory is accessible"""
    try:
        if os.path.exists(path):
            return os.path.isdir(path) and os.access(path, os.W_OK)
        else:
            # Try to create parent directory
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            return True
    except (OSError, PermissionError):
        return False