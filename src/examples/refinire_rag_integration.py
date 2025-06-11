#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from src import ChromaVectorStore, VectorDocument, VectorSearchQuery, CollectionConfig


@dataclass
class MockDocument:
    """refinire-ragのDocumentクラスを模倣"""
    id: str
    content: str
    metadata: Dict[str, Any]


class MockEmbeddingModel:
    """refinire-ragのEmbeddingModelを模倣"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """テキストのリストをembeddingsに変換（実際はダミーデータ）"""
        print(f"📝 {len(texts)}件の文書をembedding中...")
        # 実際のrefinire-ragでは本物のembeddingが生成される
        return [np.random.random(self.dimension).tolist() for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """クエリテキストをembeddingに変換"""
        print(f"🔍 クエリをembedding中: {text[:50]}...")
        return np.random.random(self.dimension).tolist()


class MockRefinireRAG:
    """refinire-ragのメインクラスを模倣"""
    
    def __init__(self, embedding_model: MockEmbeddingModel, vector_store: ChromaVectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.collection_name = "refinire_rag_collection"
        
        # コレクション初期化
        config = CollectionConfig(
            name=self.collection_name,
            dimension=embedding_model.dimension,
            distance_metric="cosine",
            metadata_schema={
                "source": "str",
                "doc_type": "str", 
                "created_at": "str",
                "chunk_id": "str"
            }
        )
        self.vector_store.create_collection(config)
    
    def add_documents(self, documents: List[MockDocument]) -> bool:
        """文書をベクトルデータベースに追加"""
        print(f"📚 {len(documents)}件の文書を処理中...")
        
        # 文書内容からembeddingsを生成
        contents = [doc.content for doc in documents]
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
        success = self.vector_store.add_documents(self.collection_name, vector_docs)
        
        if success:
            print(f"✅ {len(documents)}件の文書を正常に保存しました")
        else:
            print(f"❌ 文書の保存に失敗しました")
        
        return success
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """クエリに基づいて類似文書を検索"""
        print(f"🔎 検索実行: '{query}' (上位{top_k}件)")
        
        # クエリをembeddingに変換
        query_embedding = self.embedding_model.embed_query(query)
        
        # 検索実行
        search_query = VectorSearchQuery(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        results = self.vector_store.search(self.collection_name, search_query)
        
        # 結果を辞書形式に変換
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.document.id,
                "content": result.document.content,
                "score": result.similarity_score,
                "metadata": result.document.metadata
            })
        
        print(f"📋 {len(formatted_results)}件の結果を取得しました")
        return formatted_results


def create_sample_documents() -> List[MockDocument]:
    """サンプル文書データを作成"""
    documents = [
        MockDocument(
            id="doc_001",
            content="機械学習は人工知能の一分野で、データからパターンを学習するアルゴリズムを研究します。",
            metadata={
                "source": "ml_textbook",
                "doc_type": "textbook",
                "created_at": "2024-01-15",
                "chunk_id": "chapter_1_section_1"
            }
        ),
        MockDocument(
            id="doc_002", 
            content="深層学習はニューラルネットワークを多層化した手法で、画像認識や自然言語処理で高い性能を発揮します。",
            metadata={
                "source": "dl_paper",
                "doc_type": "research_paper",
                "created_at": "2024-01-20",
                "chunk_id": "abstract"
            }
        ),
        MockDocument(
            id="doc_003",
            content="ベクトルデータベースは高次元ベクトルの類似性検索を効率的に行うためのデータベースシステムです。",
            metadata={
                "source": "vector_db_guide",
                "doc_type": "technical_guide",
                "created_at": "2024-01-25",
                "chunk_id": "introduction"
            }
        ),
        MockDocument(
            id="doc_004",
            content="ChromaDBは軽量でスケーラブルなベクトルデータベースで、embeddingの保存と検索に特化しています。",
            metadata={
                "source": "chromadb_docs",
                "doc_type": "documentation",
                "created_at": "2024-01-30",
                "chunk_id": "overview"
            }
        ),
        MockDocument(
            id="doc_005",
            content="RAG（Retrieval-Augmented Generation）は検索により取得した情報を生成モデルに与える手法です。",
            metadata={
                "source": "rag_tutorial",
                "doc_type": "tutorial",
                "created_at": "2024-02-01",
                "chunk_id": "concept_explanation"
            }
        )
    ]
    return documents


def main():
    print("🚀 refinire-rag + ChromaVectorStore 統合例")
    print("=" * 60)
    
    # 1. 初期化
    print("\n📋 1. システム初期化")
    embedding_model = MockEmbeddingModel()
    vector_store = ChromaVectorStore(persist_directory="./refinire_rag_chroma_db")
    rag_system = MockRefinireRAG(embedding_model, vector_store)
    
    print(f"✅ Embedding Model: {embedding_model.model_name}")
    print(f"✅ Vector Store: ChromaDB (dim={embedding_model.dimension})")
    print(f"✅ Collection: {rag_system.collection_name}")
    
    # 2. 文書の追加
    print("\n📚 2. 文書データの追加")
    documents = create_sample_documents()
    success = rag_system.add_documents(documents)
    
    if not success:
        print("❌ 文書の追加に失敗しました")
        return
    
    # 3. 検索例1: 基本検索
    print("\n🔍 3. 基本検索")
    query1 = "機械学習について教えて"
    results1 = rag_system.search(query1, top_k=3)
    
    print(f"\n📋 検索結果 (クエリ: '{query1}'):")
    for i, result in enumerate(results1, 1):
        print(f"  {i}. ID: {result['id']}")
        print(f"     スコア: {result['score']:.4f}")
        print(f"     内容: {result['content'][:80]}...")
        print(f"     ソース: {result['metadata']['source']}")
        print()
    
    # 4. 検索例2: フィルタ付き検索
    print("\n🔍 4. フィルタ付き検索")
    query2 = "データベースの使い方"
    filter_condition = {"doc_type": "technical_guide"}
    results2 = rag_system.search(query2, top_k=2, filter_metadata=filter_condition)
    
    print(f"\n📋 フィルタ検索結果 (クエリ: '{query2}', フィルタ: {filter_condition}):")
    for i, result in enumerate(results2, 1):
        print(f"  {i}. ID: {result['id']}")
        print(f"     スコア: {result['score']:.4f}")
        print(f"     内容: {result['content'][:80]}...")
        print(f"     文書タイプ: {result['metadata']['doc_type']}")
        print()
    
    # 5. 検索例3: 特定ソースからの検索
    print("\n🔍 5. 特定ソース検索")
    query3 = "RAGシステム"
    source_filter = {"source": "rag_tutorial"}
    results3 = rag_system.search(query3, top_k=1, filter_metadata=source_filter)
    
    print(f"\n📋 ソース指定検索結果 (クエリ: '{query3}', ソース: {source_filter['source']}):")
    for i, result in enumerate(results3, 1):
        print(f"  {i}. ID: {result['id']}")
        print(f"     スコア: {result['score']:.4f}")
        print(f"     内容: {result['content']}")
        print(f"     作成日: {result['metadata']['created_at']}")
        print()
    
    # 6. コレクション統計
    print("\n📊 6. データベース統計")
    collections = vector_store.list_collections()
    print(f"コレクション数: {len(collections)}")
    for collection in collections:
        print(f"  - {collection}")
    
    print("\n🎉 統合例の実行が完了しました！")
    print("\n💡 このサンプルでは以下を実演しました:")
    print("   • refinire-ragスタイルのembedding生成")
    print("   • ChromaVectorStoreへの文書保存")
    print("   • 意味的類似性検索")
    print("   • メタデータフィルタリング")
    print("   • 実用的なRAGワークフロー")


if __name__ == "__main__":
    main()