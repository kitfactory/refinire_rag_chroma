"""
refinire-rag-chroma: ChromaDB VectorStore plugin for refinire-rag

This plugin provides ChromaDB integration for the refinire-rag system,
following the new plugin development guidelines.
"""

__version__ = "0.0.6"
__author__ = "refinire-rag-chroma contributors"
__description__ = "ChromaDB VectorStore plugin for refinire-rag"

from .chroma_vector_store import ChromaVectorStore

__all__ = [
    "__version__",
    "ChromaVectorStore"
]