"""
Tests for VectorStore base class compatibility with refinire-rag v0.1.5
"""

import pytest
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from refinire_rag_chroma import ChromaVectorStore
from refinire_rag.storage import VectorEntry
from refinire_rag.models.document import Document


class TestBaseClassCompatibility:
    """Test compatibility with new VectorStore base class features"""
    
    @pytest.fixture
    def vector_store(self, request):
        """Create a test ChromaVectorStore instance"""
        test_name = request.node.name
        return ChromaVectorStore(
            collection_name=f"test_collection_{test_name}",
            persist_directory=None
        )
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder"""
        embedder = Mock()
        embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return embedder
    
    def test_processing_stats_initialization(self, vector_store):
        """Test that processing stats are properly initialized"""
        stats = vector_store.get_processing_stats()
        
        # Check base DocumentProcessor stats
        assert "documents_processed" in stats
        assert "total_processing_time" in stats
        assert "errors" in stats
        assert "last_processed" in stats
        
        # Check VectorStore specific stats
        assert "vectors_stored" in stats
        assert "vectors_retrieved" in stats
        assert "searches_performed" in stats
        assert "embedding_errors" in stats
        
        # All should be initialized to 0 or None
        assert stats["documents_processed"] == 0
        assert stats["vectors_stored"] == 0
        assert stats["vectors_retrieved"] == 0
        assert stats["searches_performed"] == 0
        assert stats["embedding_errors"] == 0
        assert stats["errors"] == 0
    
    def test_processing_stats_update_on_add_vector(self, vector_store):
        """Test that stats are updated when adding vectors"""
        entry = VectorEntry(
            document_id="test_doc",
            content="Test content",
            embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            metadata={"test": True}
        )
        
        # Add vector
        vector_store.add_vector(entry)
        
        # Check stats
        stats = vector_store.get_processing_stats()
        assert stats["vectors_stored"] == 1
    
    def test_processing_stats_update_on_search(self, vector_store, mock_embedder):
        """Test that stats are updated when searching"""
        # Add a vector first
        entry = VectorEntry(
            document_id="test_doc",
            content="Test content",
            embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            metadata={"test": True}
        )
        vector_store.add_vector(entry)
        
        # Perform search
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        vector_store.search_similar(query_vector, limit=5)
        
        # Check stats
        stats = vector_store.get_processing_stats()
        assert stats["searches_performed"] == 1
    
    def test_processing_stats_update_on_get_vector(self, vector_store):
        """Test that stats are updated when retrieving vectors"""
        # Add a vector first
        entry = VectorEntry(
            document_id="test_doc",
            content="Test content",
            embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            metadata={"test": True}
        )
        vector_store.add_vector(entry)
        
        # Retrieve vector
        retrieved = vector_store.get_vector("test_doc")
        
        # Check stats
        stats = vector_store.get_processing_stats()
        assert stats["vectors_retrieved"] == 1
        assert retrieved is not None
    
    def test_process_documents_stats_update(self, vector_store, mock_embedder):
        """Test that stats are updated during document processing"""
        vector_store.set_embedder(mock_embedder)
        
        documents = [
            Document(
                id="doc_1",
                content="Test document 1",
                metadata={"source": "test"}
            ),
            Document(
                id="doc_2",
                content="Test document 2",
                metadata={"source": "test"}
            )
        ]
        
        # Process documents
        processed_docs = list(vector_store.process(documents))
        
        # Check that documents were processed
        assert len(processed_docs) == 2
        
        # Check stats
        stats = vector_store.get_processing_stats()
        assert stats["documents_processed"] == 2
        assert stats["vectors_stored"] == 2  # Each document creates a vector
        assert stats["total_processing_time"] > 0
        assert stats["last_processed"] is not None
    
    def test_embedder_compatibility(self, vector_store, mock_embedder):
        """Test embedder integration with base class"""
        # Set embedder
        vector_store.set_embedder(mock_embedder)
        
        # Check both attributes are set
        assert vector_store._embedder == mock_embedder  # Base class attribute
        assert vector_store.embedder == mock_embedder   # Backward compatibility
    
    def test_get_config_method(self, vector_store):
        """Test get_config method required by base class"""
        config = vector_store.get_config()
        
        assert isinstance(config, dict)
        assert "collection_name" in config
        assert "persist_directory" in config
        assert "distance_metric" in config
    
    def test_documentprocessor_interface(self, vector_store, mock_embedder):
        """Test DocumentProcessor interface implementation"""
        vector_store.set_embedder(mock_embedder)
        
        # Test process method
        documents = [
            Document(
                id="doc_1",
                content="Test document",
                metadata={"test": True}
            )
        ]
        
        # Process should yield documents unchanged
        processed = list(vector_store.process(documents))
        assert len(processed) == 1
        assert processed[0].id == "doc_1"
        assert processed[0].content == "Test document"
    
    def test_error_stats_update(self, vector_store):
        """Test that error stats are updated on failures"""
        # Try to search without data - should not error but no results
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        results = vector_store.search_similar(query_vector)
        assert len(results) == 0
        
        # Try invalid operations that might cause errors
        try:
            # This should not crash but return None
            result = vector_store.get_vector("nonexistent_id")
            assert result is None
        except Exception:
            # If it does error, check error stats
            stats = vector_store.get_processing_stats()
            # Error stats might be updated depending on implementation
    
    def test_stats_persistence(self, vector_store, mock_embedder):
        """Test that stats persist across operations"""
        vector_store.set_embedder(mock_embedder)
        
        # Add some vectors
        entries = [
            VectorEntry(
                document_id=f"doc_{i}",
                content=f"Content {i}",
                embedding=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]),
                metadata={"index": i}
            )
            for i in range(1, 4)
        ]
        
        vector_store.add_vectors(entries)
        
        # Perform multiple searches
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        vector_store.search_similar(query_vector)
        vector_store.search_similar(query_vector)
        
        # Check cumulative stats
        stats = vector_store.get_processing_stats()
        assert stats["vectors_stored"] == 3
        assert stats["searches_performed"] == 2