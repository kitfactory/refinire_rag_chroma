"""
ChromaDB implementation of refinire-rag VectorStore

This module provides a ChromaDB-based implementation of the refinire-rag VectorStore interface,
allowing seamless integration with the refinire-rag ecosystem.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.api.models.Collection import Collection

from refinire_rag import VectorStore, Document
from refinire_rag.storage import VectorEntry, VectorSearchResult, VectorStoreStats
from refinire_rag.exceptions import StorageError

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of refinire-rag VectorStore
    
    This class provides a production-ready ChromaDB backend for refinire-rag,
    offering persistent storage and efficient similarity search capabilities.
    """
    
    def __init__(
        self, 
        collection_name: str = "refinire_documents",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB Vector Store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage (None for in-memory)
            distance_metric: Distance metric for similarity search ("cosine", "l2", "ip")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self.client = None
        self.collection = None
        
        self._initialize_client()
        self._initialize_collection()
    
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
    
    def add_vector(self, vector_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Add a single vector to the store
        
        Args:
            vector_id: Unique identifier for the vector
            embedding: Vector embedding
            metadata: Associated metadata
        """
        try:
            # Extract document content from metadata if available
            document_content = metadata.get('content', '')
            
            self.collection.add(
                ids=[vector_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document_content]
            )
            logger.debug(f"Added vector: {vector_id}")
            
        except Exception as e:
            raise StorageError(f"Failed to add vector {vector_id}: {str(e)}")
    
    def add_vectors(self, vectors: List[VectorEntry]) -> None:
        """
        Add multiple vectors to the store
        
        Args:
            vectors: List of VectorEntry objects to add
        """
        if not vectors:
            return
        
        try:
            ids = [v.vector_id for v in vectors]
            embeddings = [v.embedding for v in vectors]
            metadatas = [v.metadata for v in vectors]
            documents = [v.metadata.get('content', '') for v in vectors]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Added {len(vectors)} vectors to collection")
            
        except Exception as e:
            raise StorageError(f"Failed to add vectors: {str(e)}")
    
    def add_documents_with_embeddings(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]]
    ) -> None:
        """
        Add documents with their precomputed embeddings
        
        Args:
            documents: List of Document objects
            embeddings: List of corresponding embeddings
        """
        if len(documents) != len(embeddings):
            raise StorageError("Number of documents must match number of embeddings")
        
        try:
            ids = [doc.id for doc in documents]
            metadatas = []
            
            for doc in documents:
                # Convert Document to metadata dict
                metadata = {
                    'content': doc.content,
                    'path': doc.metadata.get('path', ''),
                    'created_at': doc.metadata.get('created_at', ''),
                    'file_type': doc.metadata.get('file_type', ''),
                    'size_bytes': doc.metadata.get('size_bytes', 0),
                    **{k: v for k, v in doc.metadata.items() if k not in ['path', 'created_at', 'file_type', 'size_bytes']}
                }
                metadatas.append(metadata)
            
            documents_content = [doc.content for doc in documents]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            logger.info(f"Added {len(documents)} documents with embeddings")
            
        except Exception as e:
            raise StorageError(f"Failed to add documents with embeddings: {str(e)}")
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 10, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of VectorSearchResult objects
        """
        try:
            # ChromaDBでは複数条件の場合は$and演算子を使用
            where_clause = None
            if metadata_filter:
                if len(metadata_filter) > 1:
                    where_clause = {
                        "$and": [
                            {key: value} for key, value in metadata_filter.items()
                        ]
                    }
                else:
                    where_clause = metadata_filter
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
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
                    search_results.append(search_result)
            
            logger.debug(f"Found {len(search_results)} similar vectors")
            return search_results
            
        except Exception as e:
            raise StorageError(f"Failed to search similar vectors: {str(e)}")
    
    def search_similar_to_document(
        self, 
        document_id: str, 
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for vectors similar to a specific document
        
        Args:
            document_id: ID of the reference document
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of VectorSearchResult objects
        """
        try:
            # Get the document directly from ChromaDB with embeddings
            results = self.collection.get(
                ids=[document_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if not results['ids'] or len(results['ids']) == 0:
                raise StorageError(f"Document not found: {document_id}")
            
            # Extract embedding safely
            try:
                has_embeddings = (results['embeddings'] is not None and 
                                len(results['embeddings']) > 0 and 
                                results['embeddings'][0] is not None)
            except (ValueError, TypeError):
                has_embeddings = False
            
            if not has_embeddings:
                raise StorageError(f"No embedding found for document: {document_id}")
            
            query_embedding = results['embeddings'][0]
            
            # Convert numpy array to list if necessary
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            elif not isinstance(query_embedding, list):
                query_embedding = list(query_embedding)
            
            if len(query_embedding) == 0:
                raise StorageError(f"Empty embedding for document: {document_id}")
            
            logger.debug(f"Using embedding for {document_id} with dimension: {len(query_embedding)}")
            
            # Search using the document's embedding
            all_results = self.search_similar(
                query_embedding=query_embedding,
                top_k=top_k + 1,  # +1 to account for the document itself
                metadata_filter=metadata_filter
            )
            
            # Remove the document itself from results
            filtered_results = [r for r in all_results if r.document_id != document_id]
            return filtered_results[:top_k]
            
        except Exception as e:
            raise StorageError(f"Failed to search similar to document {document_id}: {str(e)}")
    
    def get_vector(self, vector_id: str) -> Optional[VectorEntry]:
        """
        Retrieve a vector by ID
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            VectorEntry if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[vector_id],
                include=['embeddings', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                embedding = []
                try:
                    if (results['embeddings'] and 
                        len(results['embeddings']) > 0 and 
                        results['embeddings'][0] is not None):
                        emb = results['embeddings'][0]
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
                
                # VectorEntryはdocument_idとcontentが必要
                content = metadata.get('content', '')
                
                return VectorEntry(
                    document_id=vector_id,
                    content=content,
                    embedding=np.array(embedding) if embedding else np.array([]),
                    metadata=metadata
                )
            
            return None
            
        except Exception as e:
            raise StorageError(f"Failed to get vector {vector_id}: {str(e)}")
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector by ID
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[vector_id])
            logger.debug(f"Deleted vector: {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {str(e)}")
            return False
    
    def update_vector(self, vector_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Update a vector's embedding and metadata
        
        Args:
            vector_id: Vector identifier
            embedding: New embedding
            metadata: New metadata
        """
        try:
            # ChromaDB doesn't have direct update, so we delete and add
            self.delete_vector(vector_id)
            self.add_vector(vector_id, embedding, metadata)
            logger.debug(f"Updated vector: {vector_id}")
            
        except Exception as e:
            raise StorageError(f"Failed to update vector {vector_id}: {str(e)}")
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[VectorEntry]:
        """
        Search vectors by metadata only
        
        Args:
            metadata_filter: Metadata filter conditions
            
        Returns:
            List of matching VectorEntry objects
        """
        try:
            # ChromaDBでは複数条件の場合は$and演算子を使用
            where_clause = metadata_filter
            if len(metadata_filter) > 1:
                # 複数条件の場合は$and演算子でラップ
                where_clause = {
                    "$and": [
                        {key: value} for key, value in metadata_filter.items()
                    ]
                }
            
            results = self.collection.get(
                where=where_clause,
                include=['embeddings', 'metadatas']
            )
            
            vectors = []
            if results['ids']:
                for i, vector_id in enumerate(results['ids']):
                    embedding = []
                    try:
                        if (results['embeddings'] and 
                            len(results['embeddings']) > i and 
                            results['embeddings'][i] is not None):
                            emb = results['embeddings'][i]
                            embedding = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    except (ValueError, TypeError):
                        pass
                    
                    metadata = {}
                    try:
                        if results['metadatas'] and len(results['metadatas']) > i:
                            metadata = results['metadatas'][i]
                    except (ValueError, TypeError):
                        pass
                    
                    content = metadata.get('content', '')
                    vectors.append(VectorEntry(
                        document_id=vector_id,
                        content=content,
                        embedding=np.array(embedding) if embedding else np.array([]),
                        metadata=metadata
                    ))
            
            logger.debug(f"Found {len(vectors)} vectors matching metadata filter")
            return vectors
            
        except Exception as e:
            raise StorageError(f"Failed to search by metadata: {str(e)}")
    
    def count_vectors(self) -> int:
        """
        Get the total number of vectors in the store
        
        Returns:
            Number of vectors
        """
        try:
            return self.collection.count()
        except Exception as e:
            raise StorageError(f"Failed to count vectors: {str(e)}")
    
    def get_vector_dimension(self) -> Optional[int]:
        """
        Get the dimension of vectors in the store
        
        Returns:
            Vector dimension if vectors exist, None otherwise
        """
        try:
            # Get a sample vector to determine dimension
            results = self.collection.get(limit=1, include=['embeddings'])
            
            try:
                has_embeddings = (results and 
                                'embeddings' in results and 
                                results['embeddings'] is not None and 
                                len(results['embeddings']) > 0)
            except (ValueError, TypeError):
                has_embeddings = False
                
            if has_embeddings:
                try:
                    emb = results['embeddings'][0]
                    if emb is not None and hasattr(emb, '__len__'):
                        dimension = len(emb)
                        logger.debug(f"Vector dimension detected: {dimension}")
                        return dimension
                except (ValueError, TypeError, IndexError):
                    pass
                    
            logger.debug("No vectors found or unable to determine dimension")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector dimension: {str(e)}")
            return None
    
    def get_all_vectors(self) -> List[VectorEntry]:
        """
        Retrieve all vectors from the store
        
        Returns:
            List of all VectorEntry objects
        """
        try:
            results = self.collection.get(include=['embeddings', 'metadatas'])
            
            vectors = []
            if results['ids']:
                for i, vector_id in enumerate(results['ids']):
                    embedding = []
                    try:
                        if (results['embeddings'] and 
                            len(results['embeddings']) > i and 
                            results['embeddings'][i] is not None):
                            emb = results['embeddings'][i]
                            embedding = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    except (ValueError, TypeError):
                        pass
                    
                    metadata = {}
                    try:
                        if results['metadatas'] and len(results['metadatas']) > i:
                            metadata = results['metadatas'][i]
                    except (ValueError, TypeError):
                        pass
                    
                    content = metadata.get('content', '')
                    vectors.append(VectorEntry(
                        document_id=vector_id,
                        content=content,
                        embedding=np.array(embedding) if embedding else np.array([]),
                        metadata=metadata
                    ))
            
            logger.info(f"Retrieved {len(vectors)} vectors")
            return vectors
            
        except Exception as e:
            raise StorageError(f"Failed to get all vectors: {str(e)}")
    
    def clear(self) -> None:
        """Clear all vectors from the store"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
            
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
            dimension = self.get_vector_dimension()
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                vector_dimension=dimension or 0,
                storage_size_bytes=0,  # ChromaDB doesn't expose this directly
                index_type="approximate"  # ChromaDB uses approximate indexing
            )
            
        except Exception as e:
            raise StorageError(f"Failed to get stats: {str(e)}")