#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List
import tempfile
from datetime import datetime

# refinire-rag imports
from refinire_rag import Document, TFIDFEmbedder, TFIDFEmbeddingConfig

# Our ChromaDB vector store implementation
from src.chroma_vector_store import ChromaVectorStore


def create_sample_documents() -> List[Document]:
    """Create sample documents using refinire-rag Document class"""
    
    documents = [
        Document(
            id="doc_001",
            content="機械学習は人工知能の一分野で、データからパターンを学習するアルゴリズムを研究します。教師あり学習、教師なし学習、強化学習などの手法があります。",
            metadata={
                "path": "/docs/ml_intro.txt",
                "created_at": datetime.now().isoformat(),
                "file_type": "txt",
                "size_bytes": 150,
                "category": "machine_learning",
                "language": "japanese"
            }
        ),
        Document(
            id="doc_002",
            content="深層学習（ディープラーニング）はニューラルネットワークを多層化した手法です。画像認識、自然言語処理、音声認識などで高い性能を発揮します。",
            metadata={
                "path": "/docs/deep_learning.txt",
                "created_at": datetime.now().isoformat(),
                "file_type": "txt",
                "size_bytes": 120,
                "category": "deep_learning",
                "language": "japanese"
            }
        ),
        Document(
            id="doc_003",
            content="ベクトルデータベースは高次元ベクトルの類似性検索を効率的に行うためのデータベースシステムです。embedding技術と組み合わせて使用されます。",
            metadata={
                "path": "/docs/vector_db.txt",
                "created_at": datetime.now().isoformat(),
                "file_type": "txt",
                "size_bytes": 100,
                "category": "database",
                "language": "japanese"
            }
        ),
        Document(
            id="doc_004",
            content="ChromaDBは軽量でスケーラブルなベクトルデータベースです。Pythonで書かれており、embeddingの保存と検索に特化しています。オープンソースで提供されています。",
            metadata={
                "path": "/docs/chromadb_overview.txt",
                "created_at": datetime.now().isoformat(),
                "file_type": "txt",
                "size_bytes": 130,
                "category": "database",
                "language": "japanese"
            }
        ),
        Document(
            id="doc_005",
            content="RAG（Retrieval-Augmented Generation）は検索により取得した関連情報を生成モデルに与える手法です。知識ベースを活用して精度の高い回答を生成できます。",
            metadata={
                "path": "/docs/rag_explanation.txt",
                "created_at": datetime.now().isoformat(),
                "file_type": "txt",
                "size_bytes": 140,
                "category": "nlp",
                "language": "japanese"
            }
        )
    ]
    
    return documents


def main():
    print("🚀 refinire-rag + ChromaDB Vector Store 実統合例")
    print("=" * 70)
    
    # 1. 初期化
    print("\n📋 1. システム初期化")
    
    # TF-IDF Embedder を初期化（日本語対応）
    embedding_config = TFIDFEmbeddingConfig(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    embedder = TFIDFEmbedder(config=embedding_config)
    
    # ChromaDB Vector Store を初期化
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = ChromaVectorStore(
            collection_name="refinire_rag_real_demo",
            persist_directory=temp_dir,
            distance_metric="cosine"
        )
        
        print(f"✅ Embedder: TFIDFEmbedder (max_features={embedding_config.max_features})")
        print(f"✅ Vector Store: ChromaDB (collection: {vector_store.collection_name})")
        print(f"✅ Storage: {temp_dir}")
        
        # 2. サンプル文書を作成
        print("\n📚 2. サンプル文書の準備")
        documents = create_sample_documents()
        print(f"📄 {len(documents)}件の文書を準備しました")
        
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc.id}: {doc.content[:50]}...")
        
        # 3. 文書をembedding化
        print("\n🧮 3. Embedding生成")
        
        # TF-IDFモデルをトレーニング（文書コーパス全体でfitする必要がある）
        all_texts = [doc.content for doc in documents]
        embedder.fit(all_texts)
        print(f"✅ TF-IDFモデルをトレーニングしました（語彙数: {len(embedder.get_vocabulary())}）")
        
        # 各文書のembeddingを生成
        embedding_results = embedder.embed_documents(documents)
        embeddings = [result.vector.tolist() for result in embedding_results]
        print(f"✅ {len(embeddings)}件のembeddingを生成しました（次元: {embedder.get_embedding_dimension()}）")
        
        # 4. ベクトルストアに保存
        print("\n💾 4. ベクトルストアへの保存")
        vector_store.add_documents_with_embeddings(documents, embeddings)
        
        stats = vector_store.get_stats()
        print(f"✅ 保存完了:")
        print(f"   - 総ベクトル数: {stats.total_vectors}")
        print(f"   - ベクトル次元: {stats.vector_dimension}")
        print(f"   - インデックスタイプ: {stats.index_type}")
        print(f"   - コレクション名: {vector_store.collection_name}")
        
        # 5. 検索テスト1: 基本検索
        print("\n🔍 5. 類似検索テスト")
        
        query_text = "機械学習アルゴリズムについて"
        print(f"クエリ: '{query_text}'")
        
        # クエリをembedding化
        query_result = embedder.embed_text(query_text)
        query_embedding = query_result.vector.tolist()
        
        # 類似検索実行
        search_results = vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=3
        )
        
        print(f"\n📋 検索結果（上位{len(search_results)}件）:")
        for i, result in enumerate(search_results, 1):
            category = result.metadata.get('category', 'unknown')
            
            print(f"  {i}. ID: {result.document_id}")
            print(f"     スコア: {result.score:.4f}")
            print(f"     カテゴリ: {category}")
            print(f"     内容: {result.content[:80]}...")
            print()
        
        # 6. 検索テスト2: メタデータフィルタ
        print("\n🔍 6. メタデータフィルタ検索")
        
        filter_query = "データベースシステム"
        metadata_filter = {"category": "database"}
        
        print(f"クエリ: '{filter_query}'")
        print(f"フィルタ: {metadata_filter}")
        
        filter_query_result = embedder.embed_text(filter_query)
        filter_query_embedding = filter_query_result.vector.tolist()
        filtered_results = vector_store.search_similar(
            query_embedding=filter_query_embedding,
            top_k=5,
            metadata_filter=metadata_filter
        )
        
        print(f"\n📋 フィルタ検索結果（{len(filtered_results)}件）:")
        for i, result in enumerate(filtered_results, 1):
            category = result.metadata.get('category', 'unknown')
            
            print(f"  {i}. ID: {result.document_id}")
            print(f"     スコア: {result.score:.4f}")
            print(f"     カテゴリ: {category}")
            print(f"     内容: {result.content[:80]}...")
            print()
        
        # 7. 類似文書検索
        print("\n🔍 7. 文書間類似性検索")
        
        reference_doc_id = "doc_001"  # 機械学習の文書
        print(f"基準文書: {reference_doc_id}")
        
        similar_docs = vector_store.search_similar_to_document(
            document_id=reference_doc_id,
            top_k=3
        )
        
        print(f"\n📋 {reference_doc_id}に類似する文書（{len(similar_docs)}件）:")
        for i, result in enumerate(similar_docs, 1):
            category = result.metadata.get('category', 'unknown')
            
            print(f"  {i}. ID: {result.document_id}")
            print(f"     スコア: {result.score:.4f}")
            print(f"     カテゴリ: {category}")
            print(f"     内容: {result.content[:80]}...")
            print()
        
        # 8. メタデータ検索
        print("\n🔍 8. メタデータ検索")
        
        metadata_search_filter = {"language": "japanese", "category": "machine_learning"}
        print(f"メタデータ検索条件: {metadata_search_filter}")
        
        metadata_results = vector_store.search_by_metadata(metadata_search_filter)
        
        print(f"\n📋 メタデータ検索結果（{len(metadata_results)}件）:")
        for i, vector in enumerate(metadata_results, 1):
            category = vector.metadata.get('category', 'unknown')
            
            print(f"  {i}. ID: {vector.document_id}")
            print(f"     カテゴリ: {category}")
            print(f"     内容: {vector.content[:80]}...")
            print()
        
        # 9. Embedder情報表示
        print("\n📊 9. システム情報")
        
        try:
            embedder_info = embedder.get_embedder_info()
            print(f"Embedder情報:")
            print(f"  - タイプ: {embedder_info.get('type', 'TFIDFEmbedder')}")
            print(f"  - 次元数: {embedder.get_embedding_dimension()}")
            print(f"  - 語彙サイズ: {len(embedder.get_vocabulary()) if hasattr(embedder, 'get_vocabulary') else 'N/A'}")
        except Exception as e:
            print(f"Embedder情報取得エラー: {e}")
        
        try:
            embedding_stats = embedder.get_embedding_stats()
            print(f"\nEmbedding統計:")
            print(f"  - 処理済み文書数: {embedding_stats.get('documents_processed', 0)}")
            print(f"  - 処理済みテキスト数: {embedding_stats.get('texts_processed', 0)}")
        except Exception as e:
            print(f"Embedding統計取得エラー: {e}")
        
        final_stats = vector_store.get_stats()
        print(f"\nVector Store統計:")
        print(f"  - 総ベクトル数: {final_stats.total_vectors}")
        print(f"  - ベクトル次元: {final_stats.vector_dimension}")
        print(f"  - インデックスタイプ: {final_stats.index_type}")
        print(f"  - 距離メトリック: {vector_store.distance_metric}")
        
    print("\n🎉 実統合例の実行が完了しました！")
    print("\n💡 この例では以下を実演しました:")
    print("   • 実際のrefinire-rag Document/Embedderクラスの使用")
    print("   • ChromaDB Vector Storeの継承実装")
    print("   • TF-IDF埋め込み生成と保存")
    print("   • 各種検索機能（類似性・フィルタ・メタデータ）")
    print("   • refinire-ragエコシステムへの完全統合")


if __name__ == "__main__":
    main()