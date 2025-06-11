"""
refinire-ragプロジェクトとの統合インターフェース

このモジュールは実際のrefinire-ragライブラリが利用可能になった時に
シームレスな統合を提供するためのアダプターパターンを実装します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import VectorDocument, VectorSearchQuery, VectorSearchResult, CollectionConfig
from .service import ChromaVectorStore


class EmbeddingModelInterface(ABC):
    """refinire-ragのEmbeddingModelとのインターフェース"""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embeddingの次元数を返す"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文書のリストをembeddingsに変換"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """クエリテキストをembeddingに変換"""
        pass


class DocumentInterface(ABC):
    """refinire-ragのDocumentとのインターフェース"""
    
    @property
    @abstractmethod
    def id(self) -> str:
        pass
    
    @property
    @abstractmethod
    def content(self) -> str:
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass


class RefinireRAGAdapter:
    """
    refinire-ragライブラリとChromaVectorStoreの統合アダプター
    
    実際のrefinire-ragライブラリが利用可能になった際に、
    このアダプターを介して統合することで、
    コードの変更を最小限に抑えることができます。
    """
    
    def __init__(
        self, 
        embedding_model: EmbeddingModelInterface,
        vector_store: ChromaVectorStore,
        collection_name: str = "refinire_rag_collection"
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.collection_name = collection_name
        
        # コレクション初期化
        self._initialize_collection()
    
    def _initialize_collection(self) -> None:
        """コレクションを初期化"""
        config = CollectionConfig(
            name=self.collection_name,
            dimension=self.embedding_model.dimension,
            distance_metric="cosine"
        )
        self.vector_store.create_collection(config)
    
    def add_documents(self, documents: List[DocumentInterface]) -> bool:
        """
        refinire-ragのDocumentをベクトルストアに追加
        
        Args:
            documents: refinire-ragのDocumentオブジェクトのリスト
            
        Returns:
            bool: 保存成功の場合True
        """
        if not documents:
            return True
        
        # 文書内容を抽出
        contents = [doc.content for doc in documents]
        
        # embeddingsを生成
        embeddings = self.embedding_model.embed_documents(contents)
        
        # VectorDocumentに変換
        vector_docs = [
            VectorDocument(
                id=doc.id,
                content=doc.content,
                embedding=embedding,
                metadata=doc.metadata
            )
            for doc, embedding in zip(documents, embeddings)
        ]
        
        # ベクトルストアに保存
        return self.vector_store.add_documents(self.collection_name, vector_docs)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        クエリに基づいて類似文書を検索
        
        Args:
            query: 検索クエリ
            top_k: 取得する結果数
            filter_metadata: メタデータフィルタ
            
        Returns:
            List[VectorSearchResult]: 検索結果のリスト
        """
        # クエリをembeddingに変換
        query_embedding = self.embedding_model.embed_query(query)
        
        # 検索クエリを作成
        search_query = VectorSearchQuery(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # 検索実行
        return self.vector_store.search(self.collection_name, search_query)
    
    def delete_collection(self) -> bool:
        """コレクションを削除"""
        return self.vector_store.delete_collection(self.collection_name)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得"""
        collections = self.vector_store.list_collections()
        return {
            "collection_name": self.collection_name,
            "exists": self.collection_name in collections,
            "all_collections": collections
        }


# 実際のrefinire-ragライブラリが利用可能になった際の統合例
def create_refinire_rag_integration(
    refinire_embedding_model,  # 実際のrefinire-ragのEmbeddingModel
    persist_directory: Optional[str] = None
) -> RefinireRAGAdapter:
    """
    実際のrefinire-ragライブラリとの統合を作成
    
    使用例:
    ```python
    # 実際のrefinire-ragが利用可能になった場合
    from refinire_rag import EmbeddingModel
    from refinire_rag_chroma.integration import create_refinire_rag_integration
    
    # embedding modelを初期化
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    
    # 統合アダプターを作成
    rag_system = create_refinire_rag_integration(
        embedding_model, 
        persist_directory="./rag_chroma_db"
    )
    
    # 文書を追加
    documents = load_documents_from_refinire_rag()
    rag_system.add_documents(documents)
    
    # 検索実行
    results = rag_system.search("機械学習について", top_k=5)
    ```
    
    Args:
        refinire_embedding_model: refinire-ragのEmbeddingModelインスタンス
        persist_directory: ChromaDBの永続化ディレクトリ
        
    Returns:
        RefinireRAGAdapter: 統合アダプターインスタンス
    """
    
    # ChromaVectorStoreを初期化
    vector_store = ChromaVectorStore(persist_directory=persist_directory)
    
    # アダプターを作成
    adapter = RefinireRAGAdapter(
        embedding_model=refinire_embedding_model,
        vector_store=vector_store
    )
    
    return adapter