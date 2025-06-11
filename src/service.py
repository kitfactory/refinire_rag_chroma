import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.api.models.Collection import Collection
from .models import (
    VectorStore, VectorDocument, VectorSearchQuery, 
    VectorSearchResult, CollectionConfig
)


logger = logging.getLogger(__name__)


class ChromaService:
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory
        self.client = None
        self._collections_cache: Dict[str, Collection] = {}
    
    def initialize_client(self) -> chromadb.ClientAPI:
        if self.client is None:
            if self.persist_directory:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self.client = chromadb.Client()
            logger.info("ChromaDBクライアント初期化完了")
        return self.client
    
    def validate_collection_config(self, config: CollectionConfig) -> None:
        if config.dimension <= 0:
            raise ValueError("次元数は正の値である必要があります")
        if not config.name.strip():
            raise ValueError("コレクション名は空であってはいけません")
        if config.distance_metric not in ["cosine", "l2", "ip"]:
            raise ValueError("距離メトリックはcosine, l2, ipのいずれかである必要があります")
    
    def handle_chroma_errors(self, operation: str, error: Exception) -> None:
        logger.error(f"ChromaDB操作エラー ({operation}): {str(error)}")
        if "already exists" in str(error).lower():
            raise ValueError(f"コレクションが既に存在します: {operation}")
        elif "not found" in str(error).lower():
            raise ValueError(f"コレクションが見つかりません: {operation}")
        else:
            raise RuntimeError(f"ChromaDB操作に失敗しました ({operation}): {str(error)}")


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: Optional[str] = None):
        self.service = ChromaService(persist_directory)
        self.client = self.service.initialize_client()
    
    def create_collection(self, config: CollectionConfig) -> bool:
        try:
            self.service.validate_collection_config(config)
            
            metadata = {}
            if config.metadata_schema:
                metadata.update(config.metadata_schema)
            
            collection = self.client.create_collection(
                name=config.name,
                metadata=metadata
            )
            
            self.service._collections_cache[config.name] = collection
            logger.info(f"コレクション作成成功: {config.name}")
            return True
            
        except Exception as e:
            self.service.handle_chroma_errors("create_collection", e)
            return False
    
    def add_documents(self, collection_name: str, documents: List[VectorDocument]) -> bool:
        try:
            if collection_name not in self.service._collections_cache:
                collection = self.client.get_collection(collection_name)
                self.service._collections_cache[collection_name] = collection
            else:
                collection = self.service._collections_cache[collection_name]
            
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = [doc.metadata or {} for doc in documents]
            documents_content = [doc.content for doc in documents]
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            
            logger.info(f"ドキュメント追加成功: {len(documents)}件")
            return True
            
        except Exception as e:
            self.service.handle_chroma_errors("add_documents", e)
            return False
    
    def search(self, collection_name: str, query: VectorSearchQuery) -> List[VectorSearchResult]:
        try:
            if collection_name not in self.service._collections_cache:
                collection = self.client.get_collection(collection_name)
                self.service._collections_cache[collection_name] = collection
            else:
                collection = self.service._collections_cache[collection_name]
            
            where_clause = query.filter_metadata if query.filter_metadata else None
            
            results = collection.query(
                query_embeddings=[query.query_embedding],
                n_results=query.top_k,
                where=where_clause
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc = VectorDocument(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i] if results['documents'] else "",
                        embedding=results['embeddings'][0][i] if results['embeddings'] else [],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                    
                    similarity_score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    search_results.append(VectorSearchResult(
                        document=doc,
                        similarity_score=max(0.0, min(1.0, similarity_score))
                    ))
            
            return search_results
            
        except Exception as e:
            self.service.handle_chroma_errors("search", e)
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self.service._collections_cache:
                del self.service._collections_cache[collection_name]
            
            logger.info(f"コレクション削除成功: {collection_name}")
            return True
            
        except Exception as e:
            self.service.handle_chroma_errors("delete_collection", e)
            return False
    
    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
            
        except Exception as e:
            self.service.handle_chroma_errors("list_collections", e)
            return []