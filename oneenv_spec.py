"""
Environment variables specification for refinire-rag-chroma
This file defines the environment variables used by this plugin.
"""

from oneenv import OneEnv


class ChromaVectorStoreOneEnv(OneEnv):
    """Environment variables specification for ChromaVectorStore"""
    
    def get_template(self) -> dict:
        """Return environment variables template"""
        return {
            # CRITICAL Variables - Essential for basic functionality
            "REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY": {
                "description": "Directory for persistent storage (None for in-memory)",
                "type": "string",
                "default": None,
                "required": False,
                "importance": "CRITICAL",
                "examples": ["/data/chroma", "./data/chroma", ""],
                "validation": "Must be a writable directory path or empty for in-memory storage"
            },
            
            # IMPORTANT Variables - Significantly affects behavior and performance
            "REFINIRE_RAG_CHROMA_COLLECTION_NAME": {
                "description": "Name of the ChromaDB collection",
                "type": "string",
                "default": "refinire_documents",
                "required": False,
                "importance": "IMPORTANT",
                "examples": ["production_docs", "dev_documents", "test_collection"],
                "validation": "Must be a non-empty string"
            },
            
            # OPTIONAL Variables - Performance optimizations and convenience features
            "REFINIRE_RAG_CHROMA_DISTANCE_METRIC": {
                "description": "Distance metric for similarity search",
                "type": "enum",
                "default": "cosine",
                "required": False,
                "importance": "OPTIONAL",
                "choices": ["cosine", "l2", "ip"],
                "examples": ["cosine", "l2", "ip"],
                "validation": "Must be one of: cosine, l2, ip"
            },
            
            "REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION": {
                "description": "Automatically create collection if it doesn't exist",
                "type": "boolean",
                "default": True,
                "required": False,
                "importance": "OPTIONAL",
                "examples": ["true", "false"],
                "validation": "Must be a boolean value (true/false)"
            },
            "REFINIRE_RAG_CHROMA_BATCH_SIZE": {
                "description": "Batch size for bulk operations",
                "type": "integer",
                "default": 100,
                "required": False,
                "importance": "OPTIONAL",
                "examples": [50, 100, 200, 500],
                "validation": "Must be a positive integer"
            },
            
            "REFINIRE_RAG_CHROMA_MAX_RETRIES": {
                "description": "Maximum retry attempts for failed operations",
                "type": "integer",
                "default": 3,
                "required": False,
                "importance": "OPTIONAL",
                "examples": [0, 3, 5, 10],
                "validation": "Must be a non-negative integer"
            },
            
            "REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT": {
                "description": "Clear collection on initialization (for testing)",
                "type": "boolean",
                "default": False,
                "required": False,
                "importance": "OPTIONAL",
                "examples": ["false", "true"],
                "validation": "Must be a boolean value (true/false)"
            }
        }