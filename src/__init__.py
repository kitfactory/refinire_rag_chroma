"""
refinire-rag-chroma: ChromaDB VectorStore plugin for refinire-rag
"""

__version__ = "0.0.1"
__author__ = "refinire-rag-chroma contributors"
__description__ = "ChromaDB VectorStore plugin for refinire-rag"

from .models import VectorStore, VectorDocument, VectorSearchQuery, VectorSearchResult, CollectionConfig
from .service import ChromaVectorStore, ChromaService

__all__ = [
    "__version__",
    "VectorStore",
    "VectorDocument", 
    "VectorSearchQuery",
    "VectorSearchResult",
    "CollectionConfig",
    "ChromaVectorStore",
    "ChromaService"
]