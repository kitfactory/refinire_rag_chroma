"""
refinire-rag-chroma: ChromaDB VectorStore plugin for refinire-rag
"""

__version__ = "0.0.1"
__author__ = "refinire-rag-chroma contributors"
__description__ = "ChromaDB VectorStore plugin for refinire-rag"

from .chroma_vector_store import ChromaVectorStore

# Try to import oneenv-related utilities
try:
    from .config import load_chroma_config, load_chroma_config_safe, get_current_profile
    __all__ = [
        "__version__",
        "ChromaVectorStore",
        "load_chroma_config",
        "load_chroma_config_safe", 
        "get_current_profile"
    ]
except ImportError:
    __all__ = [
        "__version__",
        "ChromaVectorStore"
    ]