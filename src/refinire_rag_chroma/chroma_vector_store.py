"""
ChromaDB implementation of refinire-rag VectorStore

This module provides a ChromaDB-based implementation of the refinire-rag VectorStore interface,
following the new plugin development guidelines.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Iterable, Iterator
import numpy as np
import chromadb
from chromadb.api.models.Collection import Collection

from refinire_rag.storage import VectorStore, VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.exceptions import StorageError
from refinire_rag.models.document import Document

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of refinire-rag VectorStore
    
    This class provides a production-ready ChromaDB backend for refinire-rag,
    offering persistent storage and efficient similarity search capabilities.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize ChromaDB Vector Store
        
        Args:
            **kwargs: Configuration parameters
                - collection_name: ChromaDB collection name (default: refinire_documents)
                - persist_directory: Directory for persistent storage (default: None for in-memory)
                - distance_metric: Distance metric for similarity search (default: cosine)
                - batch_size: Batch size for bulk operations (default: 100)
                - max_retries: Maximum retry attempts (default: 3)
                - auto_create_collection: Auto-create collection if it doesn't exist (default: True)
                - auto_clear_on_init: Clear collection on initialization (default: False)
        """
        # Initialize configuration from kwargs and environment variables
        self._config = self._load_config(**kwargs)
        
        # Set configuration properties
        self.collection_name = self._config["collection_name"]
        self.persist_directory = self._config["persist_directory"]
        self.distance_metric = self._config["distance_metric"]
        self.batch_size = self._config["batch_size"]
        self.max_retries = self._config["max_retries"]
        self.auto_create_collection = self._config["auto_create_collection"]
        self.auto_clear_on_init = self._config["auto_clear_on_init"]
        
        # Validate configuration
        self._validate_config()
        
        # Initialize parent class with config
        super().__init__(self._config)
        
        # Initialize ChromaDB components
        self.client = None
        self.collection = None
        self._embedder = None  # ベースクラスが期待する属性名
        self.embedder = None   # 後方互換性のため
        
        logger.info(f"ChromaVectorStore initialized: collection={self.collection_name}, "
                   f"persist_dir={self.persist_directory or 'in-memory'}, metric={self.distance_metric}")
        
        self._initialize_client()
        self._initialize_collection()
    
    def _load_config(self, **kwargs) -> Dict[str, Any]:
        """
        Load configuration from kwargs and environment variables
        
        Priority: kwargs > environment variables > default values
        """
        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")
        
        config = {
            "collection_name": kwargs.get("collection_name") or os.getenv(
                "REFINIRE_RAG_CHROMA_COLLECTION_NAME", "refinire_documents"
            ),
            "persist_directory": kwargs.get("persist_directory") or os.getenv(
                "REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY"
            ),
            "distance_metric": kwargs.get("distance_metric") or os.getenv(
                "REFINIRE_RAG_CHROMA_DISTANCE_METRIC", "cosine"
            ),
            "batch_size": kwargs.get("batch_size") or int(os.getenv(
                "REFINIRE_RAG_CHROMA_BATCH_SIZE", "100"
            )),
            "max_retries": kwargs.get("max_retries") or int(os.getenv(
                "REFINIRE_RAG_CHROMA_MAX_RETRIES", "3"
            )),
            "auto_create_collection": kwargs.get("auto_create_collection") or parse_bool(os.getenv(
                "REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION", "true"
            )),
            "auto_clear_on_init": kwargs.get("auto_clear_on_init") or parse_bool(os.getenv(
                "REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT", "false"
            ))
        }
        
        # Handle empty string for persist_directory
        if config["persist_directory"] == "":
            config["persist_directory"] = None
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            Dictionary containing current configuration settings
        """
        return self._config.copy()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client"""
        try:
            if self.persist_directory:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info(f"ChromaDB persistent client initialized: {self.persist_directory}")
            else:
                self.client = chromadb.Client()
                logger.info("ChromaDB in-memory client initialized")
        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB client: {str(e)}")
    
    def _initialize_collection(self) -> None:
        """Initialize or get existing collection"""
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                # Create new collection if it doesn't exist
                metadata = {"distance_metric": self.distance_metric}
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=metadata
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            raise StorageError(f"Failed to initialize collection: {str(e)}")
    
    def add_vector(self, entry: VectorEntry) -> str:
        """
        Add a single vector to the store
        
        Args:
            entry: VectorEntry object containing vector data
            
        Returns:
            Document ID of the added vector
        """
        try:
            # Convert numpy array to list if needed
            embedding = entry.embedding.tolist() if hasattr(entry.embedding, 'tolist') else list(entry.embedding)
            
            # Ensure metadata is not empty (ChromaDB requirement)
            metadata = entry.metadata if entry.metadata else {"_empty": True}
            
            self.collection.add(
                ids=[entry.document_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[entry.content]
            )
            logger.debug(f"Added vector: {entry.document_id}")
            return entry.document_id
            
        except Exception as e:
            raise StorageError(f"Failed to add vector {entry.document_id}: {str(e)}")
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """
        Add multiple vectors to the store
        
        Args:
            entries: List of VectorEntry objects to add
            
        Returns:
            List of document IDs for the added vectors
        """
        if not entries:
            return []
        
        try:
            ids = [v.document_id for v in entries]
            embeddings = [v.embedding.tolist() if hasattr(v.embedding, 'tolist') else list(v.embedding) for v in entries]
            metadatas = [v.metadata if v.metadata else {"_empty": True} for v in entries]
            documents = [v.content for v in entries]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Added {len(entries)} vectors to collection")
            return ids
            
        except Exception as e:
            raise StorageError(f"Failed to add vectors: {str(e)}")
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10, 
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector as numpy array
            limit: Number of results to return
            threshold: Similarity threshold (optional)
            filters: Optional metadata filter
            
        Returns:
            List of VectorSearchResult objects
        """
        try:
            # Convert numpy array to list
            query_embedding = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            
            # ChromaDBでは複数条件の場合は$and演算子を使用
            where_clause = None
            if filters:
                if len(filters) > 1:
                    where_clause = {
                        "$and": [
                            {key: value} for key, value in filters.items()
                        ]
                    }
                else:
                    where_clause = filters
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # ChromaDBの距離を類似性スコアに変換
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    # 距離メトリックに応じて類似性スコアを計算
                    if self.distance_metric == "cosine":
                        # コサイン距離: 0=完全一致, 2=完全に異なる
                        similarity_score = max(0.0, 1.0 - (distance / 2.0))
                    elif self.distance_metric == "l2":
                        # ユークリッド距離: 小さいほど類似
                        # 正規化された距離として扱う（実際の範囲は文書に依存）
                        similarity_score = 1.0 / (1.0 + distance)
                    elif self.distance_metric == "ip":
                        # 内積: 大きいほど類似（負の場合もある）
                        similarity_score = max(0.0, min(1.0, (distance + 1.0) / 2.0))
                    else:
                        # デフォルト: 単純な逆変換
                        similarity_score = max(0.0, 1.0 - distance)
                    
                    # スコアを[0,1]範囲にクランプ
                    similarity_score = max(0.0, min(1.0, similarity_score))
                    
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    content = results['documents'][0][i] if results['documents'] else metadata.get('content', '')
                    
                    logger.debug(f"Document {results['ids'][0][i]}: distance={distance:.4f}, score={similarity_score:.4f}")
                    
                    search_result = VectorSearchResult(
                        document_id=results['ids'][0][i],
                        content=content,
                        metadata=metadata,
                        score=similarity_score,
                        embedding=None
                    )
                    # Apply threshold filter if specified
                    if threshold is None or similarity_score >= threshold:
                        search_results.append(search_result)
            
            logger.debug(f"Found {len(search_results)} similar vectors")
            return search_results
            
        except Exception as e:
            raise StorageError(f"Failed to search similar vectors: {str(e)}")
    
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        """
        Retrieve a vector by ID
        
        Args:
            document_id: Document identifier
            
        Returns:
            VectorEntry if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[document_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                embedding = []
                try:
                    embeddings_array = results.get('embeddings')
                    if embeddings_array is not None and len(embeddings_array) > 0:
                        emb = embeddings_array[0]
                        if emb is not None:
                            embedding = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                except (ValueError, TypeError):
                    # numpy配列の真偽値エラーを回避
                    pass
                
                metadata = {}
                try:
                    if results['metadatas'] and len(results['metadatas']) > 0:
                        metadata = results['metadatas'][0]
                except (ValueError, TypeError):
                    pass
                
                # Get content from documents or metadata
                content = ''
                try:
                    if results['documents'] and len(results['documents']) > 0:
                        content = results['documents'][0] or ''
                except (ValueError, TypeError):
                    pass
                
                if not content:
                    content = metadata.get('content', '')
                
                return VectorEntry(
                    document_id=document_id,
                    content=content,
                    embedding=np.array(embedding) if embedding else np.array([]),
                    metadata=metadata
                )
            
            return None
            
        except Exception as e:
            raise StorageError(f"Failed to get vector {document_id}: {str(e)}")
    
    def delete_vector(self, document_id: str) -> bool:
        """
        Delete a vector by ID
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted vector: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector {document_id}: {str(e)}")
            return False
    
    def update_vector(self, entry: VectorEntry) -> bool:
        """
        Update a vector's embedding and metadata
        
        Args:
            entry: VectorEntry with updated data
            
        Returns:
            True if updated successfully
        """
        try:
            # ChromaDB doesn't have direct update, so we delete and add
            self.delete_vector(entry.document_id)
            self.add_vector(entry)
            logger.debug(f"Updated vector: {entry.document_id}")
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to update vector {entry.document_id}: {str(e)}")
    
    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 100) -> List[VectorSearchResult]:
        """
        Search vectors by metadata only
        
        Args:
            filters: Metadata filter conditions
            limit: Maximum number of results to return
            
        Returns:
            List of matching VectorSearchResult objects
        """
        try:
            # ChromaDBでは複数条件の場合は$and演算子を使用
            where_clause = filters
            if len(filters) > 1:
                # 複数条件の場合は$and演算子でラップ
                where_clause = {
                    "$and": [
                        {key: value} for key, value in filters.items()
                    ]
                }
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
            
            search_results = []
            if results['ids']:
                for i, document_id in enumerate(results['ids']):
                    embedding = []
                    try:
                        embeddings_array = results.get('embeddings')
                        if embeddings_array is not None and len(embeddings_array) > i:
                            emb = embeddings_array[i]
                            if emb is not None:
                                embedding = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    except (ValueError, TypeError):
                        pass
                    
                    metadata = {}
                    try:
                        if results['metadatas'] and len(results['metadatas']) > i:
                            metadata = results['metadatas'][i]
                    except (ValueError, TypeError):
                        pass
                    
                    # Get content from documents or metadata
                    content = ''
                    try:
                        if results['documents'] and len(results['documents']) > i:
                            content = results['documents'][i] or ''
                    except (ValueError, TypeError):
                        pass
                    
                    if not content:
                        content = metadata.get('content', '')
                    
                    search_results.append(VectorSearchResult(
                        document_id=document_id,
                        content=content,
                        metadata=metadata,
                        score=1.0,  # No similarity score for metadata-only search
                        embedding=np.array(embedding) if embedding else None
                    ))
            
            logger.debug(f"Found {len(search_results)} vectors matching metadata filter")
            return search_results
            
        except Exception as e:
            raise StorageError(f"Failed to search by metadata: {str(e)}")
    
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Get the total number of vectors in the store
        
        Args:
            filters: Optional metadata filter to count specific vectors
            
        Returns:
            Number of vectors
        """
        try:
            if filters:
                # ChromaDBでは複数条件の場合は$and演算子を使用
                where_clause = filters
                if len(filters) > 1:
                    where_clause = {
                        "$and": [
                            {key: value} for key, value in filters.items()
                        ]
                    }
                results = self.collection.get(where=where_clause)
                return len(results['ids']) if results['ids'] else 0
            else:
                return self.collection.count()
        except Exception as e:
            raise StorageError(f"Failed to count vectors: {str(e)}")
    
    def clear(self) -> bool:
        """
        Clear all vectors from the store
        
        Returns:
            True if cleared successfully
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to clear collection: {str(e)}")
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store
        
        Returns:
            VectorStoreStats object with store statistics
        """
        try:
            total_vectors = self.count_vectors()
            
            # Get vector dimension from a sample vector
            dimension = 0
            try:
                results = self.collection.get(limit=1, include=['embeddings'])
                embeddings_array = results.get('embeddings')
                if embeddings_array is not None and len(embeddings_array) > 0:
                    # ChromaDB returns embeddings as numpy array
                    first_embedding = embeddings_array[0]
                    if first_embedding is not None:
                        dimension = len(first_embedding)
                        logger.debug(f"Detected vector dimension: {dimension}")
            except Exception as e:
                logger.debug(f"Could not detect vector dimension: {e}")
                pass
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                vector_dimension=dimension,
                storage_size_bytes=0,  # ChromaDB doesn't expose this directly
                index_type="approximate"  # ChromaDB uses approximate indexing
            )
            
        except Exception as e:
            raise StorageError(f"Failed to get stats: {str(e)}")
    
    def set_embedder(self, embedder: Any) -> None:
        """
        Set the embedder for generating vector embeddings
        
        Args:
            embedder: Embedder instance with embed_text() method
        """
        self._embedder = embedder  # ベースクラスが期待する属性名
        self.embedder = embedder   # 後方互換性のため
        logger.info(f"Set embedder: {type(embedder).__name__}")
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """
        Process documents by adding them to the vector store with embeddings
        
        Args:
            documents: Input documents to process and store
            config: Optional configuration for processing
            
        Returns:
            Iterator of processed documents (same as input)
        """
        if not self._embedder:
            raise StorageError("Embedder not set. Call set_embedder() first.")
        
        for document in documents:
            try:
                # Generate embedding for the document
                embedding = self._embedder.embed_text(document.content)
                
                # Convert embedding to numpy array if it isn't already
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Create VectorEntry and add to store
                vector_entry = VectorEntry(
                    document_id=document.id,
                    content=document.content,
                    embedding=embedding,
                    metadata=document.metadata
                )
                
                # Add to vector store
                self.add_vector(vector_entry)
                
                logger.debug(f"Processed and stored document: {document.id}")
                
                # Yield the original document
                yield document
                
            except Exception as e:
                logger.error(f"Failed to process document {document.id}: {str(e)}")
                raise StorageError(f"Failed to process document {document.id}: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        if not self.collection_name or not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if self.distance_metric not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Invalid distance metric: {self.distance_metric}. Must be one of: cosine, l2, ip")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {self.batch_size}")
        
        if self.max_retries < 0:
            raise ValueError(f"Max retries must be non-negative, got: {self.max_retries}")
        
        if self.persist_directory is not None:
            # Validate directory is writable if it exists, or can be created
            try:
                if os.path.exists(self.persist_directory):
                    if not os.path.isdir(self.persist_directory):
                        raise ValueError(f"Persist directory is not a directory: {self.persist_directory}")
                    if not os.access(self.persist_directory, os.W_OK):
                        raise ValueError(f"Persist directory is not writable: {self.persist_directory}")
                else:
                    # Try to create parent directory if it doesn't exist
                    parent_dir = os.path.dirname(self.persist_directory)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise ValueError(f"Cannot access persist directory {self.persist_directory}: {e}")
    
    @classmethod
    def get_config_class(cls):
        """
        Get the configuration class for this VectorStore
        
        Returns:
            Configuration class type
        """
        from typing import Dict
        return Dict