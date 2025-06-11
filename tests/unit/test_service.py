import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.service import ChromaService, ChromaVectorStore
from src.models import CollectionConfig, VectorDocument, VectorSearchQuery


class TestChromaService:
    def test_validate_collection_config_valid(self):
        service = ChromaService()
        config = CollectionConfig(
            name="valid_collection",
            dimension=384,
            distance_metric="cosine"
        )
        service.validate_collection_config(config)
    
    def test_validate_collection_config_invalid_dimension(self):
        service = ChromaService()
        with pytest.raises(ValueError):
            config = CollectionConfig(
                name="test",
                dimension=0,
                distance_metric="cosine"
            )
    
    def test_validate_collection_config_empty_name(self):
        service = ChromaService()
        config = CollectionConfig(
            name="  ",
            dimension=384,
            distance_metric="cosine"
        )
        with pytest.raises(ValueError, match="コレクション名は空であってはいけません"):
            service.validate_collection_config(config)
    
    def test_validate_collection_config_invalid_metric(self):
        service = ChromaService()
        config = CollectionConfig(
            name="test",
            dimension=384,
            distance_metric="invalid"
        )
        with pytest.raises(ValueError, match="距離メトリックはcosine, l2, ipのいずれかである必要があります"):
            service.validate_collection_config(config)
    
    @patch('src.service.chromadb.Client')
    def test_initialize_client_memory(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        service = ChromaService()
        client = service.initialize_client()
        
        assert client == mock_client
        mock_client_class.assert_called_once()
    
    @patch('src.service.chromadb.PersistentClient')
    def test_initialize_client_persistent(self, mock_persistent_client_class):
        mock_client = Mock()
        mock_persistent_client_class.return_value = mock_client
        
        service = ChromaService(persist_directory="/test/path")
        client = service.initialize_client()
        
        assert client == mock_client
        mock_persistent_client_class.assert_called_once_with(path="/test/path")


class TestChromaVectorStore:
    @patch('src.service.chromadb.Client')
    def test_create_collection_success(self, mock_client_class):
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore()
        config = CollectionConfig(
            name="test_collection",
            dimension=384,
            distance_metric="cosine"
        )
        
        result = store.create_collection(config)
        
        assert result is True
        mock_client.create_collection.assert_called_once_with(
            name="test_collection",
            metadata={}
        )
    
    @patch('src.service.chromadb.Client')
    def test_add_documents_success(self, mock_client_class):
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore()
        documents = [
            VectorDocument(
                id="doc1",
                content="テスト文書1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"type": "test"}
            ),
            VectorDocument(
                id="doc2",
                content="テスト文書2",
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        
        result = store.add_documents("test_collection", documents)
        
        assert result is True
        mock_collection.add.assert_called_once_with(
            ids=["doc1", "doc2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            metadatas=[{"type": "test"}, {}],
            documents=["テスト文書1", "テスト文書2"]
        )
    
    @patch('src.service.chromadb.Client')
    def test_search_success(self, mock_client_class):
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['テスト文書1', 'テスト文書2']],
            'embeddings': [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            'metadatas': [[{'type': 'test'}, {}]],
            'distances': [[0.1, 0.3]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore()
        query = VectorSearchQuery(
            query_embedding=[0.7, 0.8, 0.9],
            top_k=2
        )
        
        results = store.search("test_collection", query)
        
        assert len(results) == 2
        assert results[0].document.id == "doc1"
        assert results[0].document.content == "テスト文書1"
        assert results[0].similarity_score == pytest.approx(0.9, abs=0.01)
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.7, 0.8, 0.9]],
            n_results=2,
            where=None
        )
    
    @patch('src.service.chromadb.Client')
    def test_delete_collection_success(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore()
        result = store.delete_collection("test_collection")
        
        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")
    
    @patch('src.service.chromadb.Client')
    def test_list_collections_success(self, mock_client_class):
        mock_client = Mock()
        mock_collection1 = Mock()
        mock_collection1.name = "collection1"
        mock_collection2 = Mock()
        mock_collection2.name = "collection2"
        mock_client.list_collections.return_value = [mock_collection1, mock_collection2]
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore()
        collections = store.list_collections()
        
        assert collections == ["collection1", "collection2"]
        mock_client.list_collections.assert_called_once()
    
    @patch('src.service.chromadb.Client')
    def test_handle_errors(self, mock_client_class):
        mock_client = Mock()
        mock_client.create_collection.side_effect = Exception("Test error")
        mock_client_class.return_value = mock_client
        
        store = ChromaVectorStore()
        config = CollectionConfig(
            name="test_collection",
            dimension=384,
            distance_metric="cosine"
        )
        
        with pytest.raises(RuntimeError):
            store.create_collection(config)