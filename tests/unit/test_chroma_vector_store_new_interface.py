"""
Tests for ChromaVectorStore with new refinire-rag v0.1.1+ interface
"""

import pytest
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from refinire_rag_chroma import ChromaVectorStore
from refinire_rag.storage import VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.models.document import Document


class TestChromeVectorStoreNewInterface:
    """Test the new VectorStore interface implementation"""
    
    @pytest.fixture
    def vector_store(self, request):
        """Create a test ChromaVectorStore instance"""
        # Use test name to create unique collection
        test_name = request.node.name
        return ChromaVectorStore(
            collection_name=f"test_collection_{test_name}",
            persist_directory=None  # In-memory for testing
        )
    
    @pytest.fixture
    def sample_vector_entry(self):
        """Create a sample VectorEntry for testing"""
        return VectorEntry(
            document_id="test_doc_1",
            content="This is a test document content",
            embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            metadata={"source": "test", "type": "example"}
        )
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder"""
        embedder = Mock()
        embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return embedder
    
    def test_add_vector_new_interface(self, vector_store, sample_vector_entry):
        """Test add_vector with new VectorEntry interface"""
        # Test adding a single vector
        result_id = vector_store.add_vector(sample_vector_entry)
        
        assert result_id == sample_vector_entry.document_id
        assert vector_store.count_vectors() == 1
    
    def test_add_vectors_new_interface(self, vector_store):
        """Test add_vectors with new VectorEntry interface"""
        # Create multiple vector entries (same dimension as sample)
        entries = [
            VectorEntry(
                document_id=f"doc_{i}",
                content=f"Content {i}",
                embedding=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]),
                metadata={"index": i}
            )
            for i in range(1, 4)
        ]
        
        # Test adding multiple vectors
        result_ids = vector_store.add_vectors(entries)
        
        assert len(result_ids) == 3
        assert result_ids == ["doc_1", "doc_2", "doc_3"]
        assert vector_store.count_vectors() == 3
    
    def test_get_vector_new_interface(self, vector_store, sample_vector_entry):
        """Test get_vector with new interface"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        
        # Retrieve it
        retrieved = vector_store.get_vector(sample_vector_entry.document_id)
        
        assert retrieved is not None
        assert retrieved.document_id == sample_vector_entry.document_id
        assert retrieved.content == sample_vector_entry.content
        assert np.allclose(retrieved.embedding, sample_vector_entry.embedding)
        assert retrieved.metadata == sample_vector_entry.metadata
    
    def test_search_similar_new_interface(self, vector_store, sample_vector_entry):
        """Test search_similar with numpy array interface"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        
        # Search with numpy array
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        results = vector_store.search_similar(query_vector, limit=5)
        
        assert len(results) == 1
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].document_id == sample_vector_entry.document_id
        assert results[0].content == sample_vector_entry.content
        assert results[0].score > 0.0
    
    def test_search_similar_with_threshold(self, vector_store, sample_vector_entry):
        """Test search_similar with threshold parameter"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        
        # Search with high threshold (should return no results)
        query_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Different vector
        results = vector_store.search_similar(query_vector, limit=5, threshold=0.9)
        
        assert len(results) == 0  # Should be filtered out by threshold
    
    def test_search_by_metadata_new_interface(self, vector_store):
        """Test search_by_metadata with new interface"""
        # Add multiple vectors with different metadata
        entries = [
            VectorEntry(
                document_id="doc_1",
                content="Content 1",
                embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                metadata={"category": "A", "priority": 1}
            ),
            VectorEntry(
                document_id="doc_2", 
                content="Content 2",
                embedding=np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
                metadata={"category": "B", "priority": 2}
            )
        ]
        
        vector_store.add_vectors(entries)
        
        # Search by metadata
        results = vector_store.search_by_metadata({"category": "A"})
        
        assert len(results) == 1
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].document_id == "doc_1"
        assert results[0].score == 1.0  # Metadata-only search score
    
    def test_update_vector_new_interface(self, vector_store, sample_vector_entry):
        """Test update_vector with new VectorEntry interface"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        
        # Update it
        updated_entry = VectorEntry(
            document_id=sample_vector_entry.document_id,
            content="Updated content",
            embedding=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            metadata={"source": "updated", "type": "modified"}
        )
        
        success = vector_store.update_vector(updated_entry)
        assert success is True
        
        # Verify the update
        retrieved = vector_store.get_vector(updated_entry.document_id)
        assert retrieved.content == "Updated content"
        assert np.allclose(retrieved.embedding, updated_entry.embedding)
        assert retrieved.metadata["source"] == "updated"
    
    def test_delete_vector_new_interface(self, vector_store, sample_vector_entry):
        """Test delete_vector with new interface"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        assert vector_store.count_vectors() == 1
        
        # Delete it
        success = vector_store.delete_vector(sample_vector_entry.document_id)
        assert success is True
        assert vector_store.count_vectors() == 0
    
    def test_count_vectors_with_filters(self, vector_store):
        """Test count_vectors with filters parameter"""
        # Add vectors with different metadata
        entries = [
            VectorEntry(
                document_id="doc_1",
                content="Content 1", 
                embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                metadata={"type": "A"}
            ),
            VectorEntry(
                document_id="doc_2",
                content="Content 2",
                embedding=np.array([0.4, 0.5, 0.6, 0.7, 0.8]), 
                metadata={"type": "B"}
            ),
            VectorEntry(
                document_id="doc_3",
                content="Content 3",
                embedding=np.array([0.7, 0.8, 0.9, 1.0, 1.1]),
                metadata={"type": "A"}
            )
        ]
        
        vector_store.add_vectors(entries)
        
        # Count all vectors
        total_count = vector_store.count_vectors()
        assert total_count == 3
        
        # Count with filter
        filtered_count = vector_store.count_vectors({"type": "A"})
        assert filtered_count == 2
    
    def test_get_stats_new_interface(self, vector_store, sample_vector_entry):
        """Test get_stats returns VectorStoreStats"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        
        stats = vector_store.get_stats()
        
        assert isinstance(stats, VectorStoreStats)
        assert stats.total_vectors == 1
        assert stats.vector_dimension == 5  # Based on sample_vector_entry
        assert stats.storage_size_bytes == 0  # ChromaDB doesn't expose this
        assert stats.index_type == "approximate"
    
    def test_clear_returns_bool(self, vector_store, sample_vector_entry):
        """Test clear returns boolean"""
        # Add a vector first
        vector_store.add_vector(sample_vector_entry)
        assert vector_store.count_vectors() == 1
        
        # Clear the store
        success = vector_store.clear()
        assert success is True
        assert vector_store.count_vectors() == 0
    
    def test_get_config_class(self, vector_store):
        """Test get_config_class class method"""
        from typing import Dict
        config_class = ChromaVectorStore.get_config_class()
        assert config_class == Dict  # Should return Dict type
    
    def test_environment_variable_initialization(self):
        """Test initialization with environment variables"""
        # Set environment variables
        os.environ["REFINIRE_RAG_CHROMA_COLLECTION_NAME"] = "env_test_collection"
        os.environ["REFINIRE_RAG_CHROMA_DISTANCE_METRIC"] = "l2"
        os.environ["REFINIRE_RAG_CHROMA_BATCH_SIZE"] = "50"
        
        try:
            vector_store = ChromaVectorStore()
            
            assert vector_store.collection_name == "env_test_collection"
            assert vector_store.distance_metric == "l2"
            assert vector_store.batch_size == 50
        finally:
            # Clean up environment variables
            os.environ.pop("REFINIRE_RAG_CHROMA_COLLECTION_NAME", None)
            os.environ.pop("REFINIRE_RAG_CHROMA_DISTANCE_METRIC", None)
            os.environ.pop("REFINIRE_RAG_CHROMA_BATCH_SIZE", None)
    
    def test_parameter_override_env(self):
        """Test that parameters override environment variables"""
        # Set environment variable
        os.environ["REFINIRE_RAG_CHROMA_COLLECTION_NAME"] = "env_collection"
        
        try:
            vector_store = ChromaVectorStore(
                collection_name="param_collection"  # This should override env var
            )
            
            assert vector_store.collection_name == "param_collection"
        finally:
            # Clean up environment variable
            os.environ.pop("REFINIRE_RAG_CHROMA_COLLECTION_NAME", None)
            
    def test_default_values(self):
        """Test default configuration values"""
        # Ensure no environment variables are set
        env_vars_to_clear = [
            "REFINIRE_RAG_CHROMA_COLLECTION_NAME",
            "REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY", 
            "REFINIRE_RAG_CHROMA_DISTANCE_METRIC",
            "REFINIRE_RAG_CHROMA_BATCH_SIZE",
            "REFINIRE_RAG_CHROMA_MAX_RETRIES",
            "REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION",
            "REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT"
        ]
        
        original_values = {}
        for var in env_vars_to_clear:
            original_values[var] = os.environ.pop(var, None)
        
        try:
            vector_store = ChromaVectorStore()
            
            assert vector_store.collection_name == "refinire_documents"
            assert vector_store.persist_directory is None
            assert vector_store.distance_metric == "cosine"
            assert vector_store.batch_size == 100
            assert vector_store.max_retries == 3
            assert vector_store.auto_create_collection is True
            assert vector_store.auto_clear_on_init is False
        finally:
            # Restore original environment variables
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value
    
    def test_set_embedder(self, vector_store, mock_embedder):
        """Test set_embedder method"""
        vector_store.set_embedder(mock_embedder)
        assert vector_store.embedder == mock_embedder
    
    def test_process_documents(self, vector_store, mock_embedder):
        """Test DocumentProcessor process method"""
        # Set up embedder
        vector_store.set_embedder(mock_embedder)
        
        # Create test documents
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
        
        # Verify processing
        assert len(processed_docs) == 2
        assert processed_docs[0].id == "doc_1"
        assert processed_docs[1].id == "doc_2"
        
        # Verify documents were added to store
        assert vector_store.count_vectors() == 2
        
        # Verify embedder was called
        assert mock_embedder.embed_text.call_count == 2
    
    def test_process_without_embedder_raises_error(self, vector_store):
        """Test process raises error when no embedder is set"""
        documents = [
            Document(
                id="doc_1",
                content="Test document",
                metadata={}
            )
        ]
        
        with pytest.raises(Exception) as exc_info:
            list(vector_store.process(documents))
        
        assert "Embedder not set" in str(exc_info.value)