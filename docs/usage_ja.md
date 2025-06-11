# 使用方法ガイド

## 基本的な操作

### 1. セットアップ

```python
from refinire_rag import Document, TFIDFEmbedder, TFIDFEmbeddingConfig
from src.chroma_vector_store import ChromaVectorStore
import tempfile

# 一時ディレクトリを使用した設定
with tempfile.TemporaryDirectory() as temp_dir:
    vector_store = ChromaVectorStore(
        collection_name="日本語文書コレクション",
        persist_directory=temp_dir,
        distance_metric="cosine"
    )
```

### 2. 文書の準備と保存

```python
# 日本語文書の作成
documents = [
    Document(
        id="文書_001",
        content="機械学習は人工知能の重要な分野の一つです。データから学習し、パターンを発見する技術です。",
        metadata={
            "カテゴリ": "AI・機械学習",
            "作成日": "2024-01-15",
            "言語": "日本語",
            "レベル": "初級"
        }
    ),
    Document(
        id="文書_002",
        content="深層学習（ディープラーニング）は、多層のニューラルネットワークを使用した機械学習手法です。",
        metadata={
            "カテゴリ": "AI・機械学習", 
            "作成日": "2024-01-16",
            "言語": "日本語",
            "レベル": "中級"
        }
    ),
    Document(
        id="文書_003",
        content="自然言語処理（NLP）は、コンピューターが人間の言語を理解し処理する技術分野です。",
        metadata={
            "カテゴリ": "自然言語処理",
            "作成日": "2024-01-17", 
            "言語": "日本語",
            "レベル": "中級"
        }
    )
]

# TF-IDFEmbedderの設定
embedding_config = TFIDFEmbeddingConfig(
    max_features=1000,
    ngram_range=(1, 2),  # ユニグラムとバイグラム
    min_df=1,
    max_df=0.8
)
embedder = TFIDFEmbedder(config=embedding_config)

# モデルの訓練
corpus = [doc.content for doc in documents]
embedder.fit(corpus)

# Embeddingの生成
embedding_results = embedder.embed_documents(documents)
embeddings = [result.vector.tolist() for result in embedding_results]

# ベクトルストアに保存
vector_store.add_documents_with_embeddings(documents, embeddings)
```

### 3. 検索操作

#### 基本的な類似性検索

```python
# クエリのembedding化
query = "機械学習について詳しく教えてください"
query_result = embedder.embed_text(query)
query_embedding = query_result.vector.tolist()

# 類似文書の検索
results = vector_store.search_similar(
    query_embedding=query_embedding,
    top_k=3
)

# 結果の表示
print(f"クエリ: '{query}'")
print("=" * 50)
for i, result in enumerate(results, 1):
    print(f"{i}. 文書ID: {result.document_id}")
    print(f"   類似度: {result.score:.4f}")
    print(f"   内容: {result.content[:100]}...")
    print(f"   カテゴリ: {result.metadata.get('カテゴリ', 'N/A')}")
    print()
```

#### メタデータフィルタリング

```python
# 特定カテゴリでフィルタリング
ai_results = vector_store.search_similar(
    query_embedding=query_embedding,
    top_k=5,
    metadata_filter={"カテゴリ": "AI・機械学習"}
)

# 複数条件でフィルタリング
advanced_results = vector_store.search_similar(
    query_embedding=query_embedding,
    top_k=5,
    metadata_filter={
        "言語": "日本語",
        "レベル": {"$in": ["初級", "中級"]}
    }
)

# 日付範囲でフィルタリング（文字列比較）
recent_results = vector_store.search_similar(
    query_embedding=query_embedding,
    top_k=5,
    metadata_filter={
        "作成日": {"$gte": "2024-01-16"}
    }
)
```

#### 文書間類似性検索

```python
# 特定文書に類似する文書を検索
similar_to_doc = vector_store.search_similar_to_document(
    document_id="文書_001",
    top_k=2
)

print("文書_001 に類似する文書:")
for result in similar_to_doc:
    print(f"- {result.document_id}: {result.score:.4f}")
    print(f"  {result.content[:50]}...")
```

#### メタデータのみでの検索

```python
# カテゴリ別の文書取得
nlp_docs = vector_store.search_by_metadata({
    "カテゴリ": "自然言語処理"
})

# 複数条件での文書取得
beginner_docs = vector_store.search_by_metadata({
    "言語": "日本語",
    "レベル": "初級"
})
```

## 高度な使用パターン

### 1. バッチ処理

```python
def process_document_batch(documents_batch, vector_store, embedder):
    """文書の一括処理"""
    
    # 一括embedding生成
    embedding_results = embedder.embed_documents(documents_batch)
    embeddings = [result.vector.tolist() for result in embedding_results]
    
    # 一括保存
    vector_store.add_documents_with_embeddings(documents_batch, embeddings)
    
    return len(documents_batch)

# 大量文書の分割処理
batch_size = 100
total_processed = 0

for i in range(0, len(all_documents), batch_size):
    batch = all_documents[i:i+batch_size]
    processed = process_document_batch(batch, vector_store, embedder)
    total_processed += processed
    print(f"処理済み: {total_processed}/{len(all_documents)}")
```

### 2. 動的検索フィルタ

```python
def build_dynamic_filter(category=None, level=None, date_from=None, date_to=None):
    """動的にメタデータフィルタを構築"""
    
    conditions = []
    
    if category:
        conditions.append({"カテゴリ": category})
    
    if level:
        if isinstance(level, list):
            conditions.append({"レベル": {"$in": level}})
        else:
            conditions.append({"レベル": level})
    
    if date_from:
        conditions.append({"作成日": {"$gte": date_from}})
    
    if date_to:
        conditions.append({"作成日": {"$lte": date_to}})
    
    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

# 使用例
filter_condition = build_dynamic_filter(
    category="AI・機械学習",
    level=["初級", "中級"],
    date_from="2024-01-15"
)

results = vector_store.search_similar(
    query_embedding=query_embedding,
    metadata_filter=filter_condition
)
```

### 3. 検索結果の分析

```python
def analyze_search_results(results):
    """検索結果の統計分析"""
    
    if not results:
        return {"message": "検索結果がありません"}
    
    # カテゴリ別集計
    categories = {}
    levels = {}
    scores = []
    
    for result in results:
        # カテゴリ集計
        category = result.metadata.get('カテゴリ', 'その他')
        categories[category] = categories.get(category, 0) + 1
        
        # レベル集計
        level = result.metadata.get('レベル', 'その他')
        levels[level] = levels.get(level, 0) + 1
        
        # スコア収集
        scores.append(result.score)
    
    analysis = {
        "総件数": len(results),
        "カテゴリ別": categories,
        "レベル別": levels,
        "スコア統計": {
            "平均": sum(scores) / len(scores),
            "最高": max(scores),
            "最低": min(scores)
        }
    }
    
    return analysis

# 分析実行
results = vector_store.search_similar(query_embedding=query_embedding, top_k=10)
analysis = analyze_search_results(results)
print("検索結果分析:")
for key, value in analysis.items():
    print(f"  {key}: {value}")
```

## パフォーマンス最適化

### 1. Embedding次元の調整

```python
# 高精度用（重い）
high_accuracy_config = TFIDFEmbeddingConfig(
    max_features=5000,
    ngram_range=(1, 3)
)

# 高速処理用（軽い）
fast_config = TFIDFEmbeddingConfig(
    max_features=500,
    ngram_range=(1, 1)
)
```

### 2. バッチサイズの調整

```python
# メモリ使用量を抑えたい場合
small_batch_size = 50

# 処理速度を重視する場合
large_batch_size = 500
```

### 3. インデックス管理

```python
# 統計情報の確認
stats = vector_store.get_stats()
print(f"総ベクトル数: {stats.total_vectors}")
print(f"ベクトル次元: {stats.vector_dimension}")
print(f"インデックスタイプ: {stats.index_type}")

# 定期的なメンテナンス（必要に応じて）
if stats.total_vectors > 10000:
    print("大量データが蓄積されています。インデックス最適化を検討してください。")
```

## エラーハンドリング

```python
from refinire_rag.exceptions import StorageError

try:
    # 文書保存
    vector_store.add_documents_with_embeddings(documents, embeddings)
    
    # 検索実行
    results = vector_store.search_similar(query_embedding)
    
except StorageError as e:
    print(f"ストレージエラー: {e}")
    # エラー対応処理
    
except ValueError as e:
    print(f"入力値エラー: {e}")
    # 入力値の検証・修正
    
except Exception as e:
    print(f"予期しないエラー: {e}")
    # ログ出力、通知等
```