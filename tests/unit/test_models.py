import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from pydantic import ValidationError
from src.models import VectorDocument, VectorSearchQuery, VectorSearchResult, CollectionConfig


class TestVectorDocument:
    def test_vector_document_creation(self):
        doc = VectorDocument(
            id="test_1",
            content="テストドキュメント",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"}
        )
        assert doc.id == "test_1"
        assert doc.content == "テストドキュメント"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"source": "test"}
    
    def test_vector_document_without_metadata(self):
        doc = VectorDocument(
            id="test_2",
            content="メタデータなし",
            embedding=[0.4, 0.5, 0.6]
        )
        assert doc.metadata is None


class TestVectorSearchQuery:
    def test_valid_query(self):
        query = VectorSearchQuery(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
            filter_metadata={"category": "test"}
        )
        assert query.query_embedding == [0.1, 0.2, 0.3]
        assert query.top_k == 5
        assert query.filter_metadata == {"category": "test"}
    
    def test_empty_embedding_validation(self):
        with pytest.raises(ValidationError):
            VectorSearchQuery(query_embedding=[])
    
    def test_top_k_validation(self):
        with pytest.raises(ValidationError):
            VectorSearchQuery(
                query_embedding=[0.1, 0.2, 0.3],
                top_k=0
            )
        
        with pytest.raises(ValidationError):
            VectorSearchQuery(
                query_embedding=[0.1, 0.2, 0.3],
                top_k=101
            )
    
    def test_default_values(self):
        query = VectorSearchQuery(query_embedding=[0.1, 0.2, 0.3])
        assert query.top_k == 5
        assert query.filter_metadata is None


class TestVectorSearchResult:
    def test_search_result_creation(self):
        doc = VectorDocument(
            id="result_1",
            content="結果ドキュメント",
            embedding=[0.7, 0.8, 0.9]
        )
        result = VectorSearchResult(
            document=doc,
            similarity_score=0.95
        )
        assert result.document == doc
        assert result.similarity_score == 0.95
    
    def test_similarity_score_validation(self):
        doc = VectorDocument(
            id="result_2",
            content="テスト",
            embedding=[0.1, 0.2, 0.3]
        )
        
        with pytest.raises(ValidationError):
            VectorSearchResult(document=doc, similarity_score=-0.1)
        
        with pytest.raises(ValidationError):
            VectorSearchResult(document=doc, similarity_score=1.1)


class TestCollectionConfig:
    def test_valid_config(self):
        config = CollectionConfig(
            name="test_collection",
            dimension=384,
            distance_metric="cosine",
            metadata_schema={"type": "str"}
        )
        assert config.name == "test_collection"
        assert config.dimension == 384
        assert config.distance_metric == "cosine"
        assert config.metadata_schema == {"type": "str"}
    
    def test_empty_name_validation(self):
        with pytest.raises(ValidationError):
            CollectionConfig(name="", dimension=384)
    
    def test_invalid_dimension(self):
        with pytest.raises(ValidationError):
            CollectionConfig(name="test", dimension=0)
    
    def test_default_values(self):
        config = CollectionConfig(name="test", dimension=128)
        assert config.distance_metric == "cosine"
        assert config.metadata_schema is None