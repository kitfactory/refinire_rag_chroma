"""
Tests for config utilities following new plugin guide
"""

import os
import pytest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from refinire_rag_chroma.config import get_default_config, load_config_from_environment, validate_config


class TestConfigUtilities:
    """Test configuration utilities"""
    
    def test_get_default_config(self):
        """Test default configuration values"""
        config = get_default_config()
        
        assert config["collection_name"] == "refinire_documents"
        assert config["persist_directory"] is None
        assert config["distance_metric"] == "cosine"
        assert config["batch_size"] == 100
        assert config["max_retries"] == 3
        assert config["auto_create_collection"] is True
        assert config["auto_clear_on_init"] is False
    
    def test_load_config_from_environment_defaults(self):
        """Test loading config with no environment variables"""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config_from_environment()
            
            assert config["collection_name"] == "refinire_documents"
            assert config["persist_directory"] is None
            assert config["distance_metric"] == "cosine"
            assert config["batch_size"] == 100
            assert config["max_retries"] == 3
            assert config["auto_create_collection"] is True
            assert config["auto_clear_on_init"] is False
    
    def test_load_config_from_environment_override(self):
        """Test configuration override from environment variables"""
        with patch.dict(os.environ, {
            "REFINIRE_RAG_CHROMA_COLLECTION_NAME": "test_collection",
            "REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY": "/tmp/test_chroma",
            "REFINIRE_RAG_CHROMA_DISTANCE_METRIC": "l2",
            "REFINIRE_RAG_CHROMA_BATCH_SIZE": "50",
            "REFINIRE_RAG_CHROMA_MAX_RETRIES": "5",
            "REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION": "false",
            "REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT": "true"
        }):
            config = load_config_from_environment()
            
            assert config["collection_name"] == "test_collection"
            assert config["persist_directory"] == "/tmp/test_chroma"
            assert config["distance_metric"] == "l2"
            assert config["batch_size"] == 50
            assert config["max_retries"] == 5
            assert config["auto_create_collection"] is False
            assert config["auto_clear_on_init"] is True
    
    def test_empty_persist_directory(self):
        """Test empty persist directory returns None"""
        with patch.dict(os.environ, {
            "REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY": ""
        }):
            config = load_config_from_environment()
            assert config["persist_directory"] is None
    
    def test_boolean_parsing_variations(self):
        """Test various boolean value formats"""
        # Test 'true' values
        for true_value in ["true", "True", "TRUE", "1", "yes", "on"]:
            with patch.dict(os.environ, {
                "REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION": true_value
            }):
                config = load_config_from_environment()
                assert config["auto_create_collection"] is True
        
        # Test 'false' values
        for false_value in ["false", "False", "FALSE", "0", "no", "off"]:
            with patch.dict(os.environ, {
                "REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION": false_value
            }):
                config = load_config_from_environment()
                assert config["auto_create_collection"] is False
    
    def test_validate_success(self):
        """Test successful validation"""
        config = get_default_config()
        # Should not raise exception
        validate_config(config)
    
    def test_validate_empty_collection_name(self):
        """Test validation fails for empty collection name"""
        config = get_default_config()
        config["collection_name"] = ""
        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            validate_config(config)
    
    def test_validate_invalid_distance_metric(self):
        """Test validation fails for invalid distance metric"""
        config = get_default_config()
        config["distance_metric"] = "invalid_metric"
        with pytest.raises(ValueError, match="Invalid distance metric"):
            validate_config(config)
    
    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch size"""
        config = get_default_config()
        config["batch_size"] = 0
        with pytest.raises(ValueError, match="Batch size must be positive"):
            validate_config(config)
    
    def test_validate_invalid_max_retries(self):
        """Test validation fails for negative max retries"""
        config = get_default_config()
        config["max_retries"] = -1
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            validate_config(config)
    
    def test_int_parsing_with_invalid_value(self):
        """Test integer parsing handles invalid values gracefully"""
        with patch.dict(os.environ, {
            "REFINIRE_RAG_CHROMA_BATCH_SIZE": "invalid_number"
        }):
            config = load_config_from_environment()
            # Should fall back to default value
            assert config["batch_size"] == 100