"""
refinire-rag-chroma: ChromaDB VectorStore plugin for refinire-rag
"""

__version__ = "0.0.6"
__author__ = "refinire-rag-chroma contributors"
__description__ = "ChromaDB VectorStore plugin for refinire-rag"

# Import from the proper package location
from .refinire_rag_chroma import ChromaVectorStore

__all__ = [
    "__version__",
    "ChromaVectorStore"
]