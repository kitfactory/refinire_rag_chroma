#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import List
from src import ChromaVectorStore, VectorDocument, VectorSearchQuery, CollectionConfig


def create_sample_embeddings(texts: List[str], dimension: int = 384) -> List[List[float]]:
    return [np.random.random(dimension).tolist() for _ in texts]


def main():
    vector_store = ChromaVectorStore(persist_directory="./chroma_db")
    
    config = CollectionConfig(
        name="sample_collection",
        dimension=384,
        distance_metric="cosine",
        metadata_schema={"source": "str", "category": "str"}
    )
    
    print("=== コレクション作成 ===")
    success = vector_store.create_collection(config)
    print(f"コレクション作成: {'成功' if success else '失敗'}")
    
    print("\n=== ドキュメント追加 ===")
    sample_texts = [
        "これはサンプルの文書です。",
        "機械学習について説明します。",
        "ベクトルデータベースの使用方法。"
    ]
    
    embeddings = create_sample_embeddings(sample_texts)
    
    documents = [
        VectorDocument(
            id=f"doc_{i}",
            content=text,
            embedding=emb,
            metadata={"source": "example", "category": f"type_{i%2}"}
        )
        for i, (text, emb) in enumerate(zip(sample_texts, embeddings))
    ]
    
    success = vector_store.add_documents("sample_collection", documents)
    print(f"ドキュメント追加: {'成功' if success else '失敗'}")
    
    print("\n=== 検索実行 ===")
    query_embedding = create_sample_embeddings(["機械学習の情報を探しています"], 384)[0]
    
    query = VectorSearchQuery(
        query_embedding=query_embedding,
        top_k=2,
        filter_metadata={"source": "example"}
    )
    
    results = vector_store.search("sample_collection", query)
    
    print(f"検索結果: {len(results)}件")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result.document.id}")
        print(f"     Content: {result.document.content}")
        print(f"     Score: {result.similarity_score:.4f}")
        print(f"     Metadata: {result.document.metadata}")
        print()
    
    print("\n=== コレクション一覧 ===")
    collections = vector_store.list_collections()
    print(f"コレクション数: {len(collections)}")
    for collection in collections:
        print(f"  - {collection}")


if __name__ == "__main__":
    main()