# API リファレンス

## ChromaVectorStore クラス

### 概要

`ChromaVectorStore`は`refinire_rag.VectorStore`を継承したChromaDB実装クラスです。ベクトル保存、類似性検索、メタデータフィルタリングなどの機能を提供します。

### クラス定義

```python
class ChromaVectorStore(VectorStore):
    def __init__(
        self, 
        collection_name: str = "refinire_documents",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine"
    )
```

## コンストラクタ

### `__init__(collection_name, persist_directory, distance_metric)`

ChromaVectorStoreインスタンスを初期化します。

**パラメータ:**

- `collection_name` (str, optional): ChromaDBコレクション名
  - デフォルト: `"refinire_documents"`
  - 例: `"日本語文書コレクション"`

- `persist_directory` (str, optional): 永続化ストレージのディレクトリパス
  - デフォルト: `None` (インメモリ)
  - 例: `"./chroma_db"`, `"/data/vector_store"`

- `distance_metric` (str, optional): 距離メトリック
  - デフォルト: `"cosine"`
  - 選択肢: `"cosine"`, `"l2"`, `"ip"`

**使用例:**

```python
# インメモリストレージ
vector_store = ChromaVectorStore()

# 永続化ストレージ
vector_store = ChromaVectorStore(
    collection_name="my_collection",
    persist_directory="/path/to/storage",
    distance_metric="cosine"
)
```

## メソッド

### 文書保存メソッド

#### `add_vector(vector_id, embedding, metadata)`

単一ベクトルを保存します。

**パラメータ:**
- `vector_id` (str): ベクトルの一意識別子
- `embedding` (List[float]): ベクトルembedding
- `metadata` (Dict[str, Any]): メタデータ

**戻り値:** None

**例外:** `StorageError`

#### `add_vectors(vectors)`

複数のベクトルを一括保存します。

**パラメータ:**
- `vectors` (List[VectorEntry]): VectorEntryオブジェクトのリスト

**戻り値:** None

**例外:** `StorageError`

#### `add_documents_with_embeddings(documents, embeddings)`

文書とembeddingを対応付けて保存します。

**パラメータ:**
- `documents` (List[Document]): refinire-ragのDocumentオブジェクト
- `embeddings` (List[List[float]]): 対応するembeddingのリスト

**戻り値:** None

**例外:** `StorageError`

**使用例:**

```python
from refinire_rag import Document

documents = [
    Document(
        id="doc1",
        content="サンプル文書",
        metadata={"category": "test"}
    )
]

embeddings = [[0.1, 0.2, 0.3, 0.4]]

vector_store.add_documents_with_embeddings(documents, embeddings)
```

### 検索メソッド

#### `search_similar(query_embedding, top_k, metadata_filter)`

クエリembeddingに基づく類似性検索を実行します。

**パラメータ:**
- `query_embedding` (List[float]): クエリのembedding
- `top_k` (int, optional): 取得する結果数 (デフォルト: 10)
- `metadata_filter` (Dict[str, Any], optional): メタデータフィルタ

**戻り値:** `List[VectorSearchResult]`

**例外:** `StorageError`

**使用例:**

```python
# 基本検索
results = vector_store.search_similar(
    query_embedding=[0.1, 0.2, 0.3],
    top_k=5
)

# フィルタ付き検索
results = vector_store.search_similar(
    query_embedding=[0.1, 0.2, 0.3],
    top_k=5,
    metadata_filter={"category": "AI"}
)
```

#### `search_similar_to_document(document_id, top_k, metadata_filter)`

指定した文書に類似する文書を検索します。

**パラメータ:**
- `document_id` (str): 基準となる文書ID
- `top_k` (int, optional): 取得する結果数 (デフォルト: 10)
- `metadata_filter` (Dict[str, Any], optional): メタデータフィルタ

**戻り値:** `List[VectorSearchResult]` (基準文書自体は除外)

**例外:** `StorageError`

**使用例:**

```python
similar_docs = vector_store.search_similar_to_document(
    document_id="doc_001",
    top_k=3
)
```

#### `search_by_metadata(metadata_filter)`

メタデータ条件のみで文書を検索します。

**パラメータ:**
- `metadata_filter` (Dict[str, Any]): メタデータフィルタ条件

**戻り値:** `List[VectorEntry]`

**例外:** `StorageError`

**使用例:**

```python
# 単一条件
results = vector_store.search_by_metadata({"category": "AI"})

# 複数条件
results = vector_store.search_by_metadata({
    "language": "japanese",
    "level": {"$in": ["beginner", "intermediate"]}
})
```

### データ操作メソッド

#### `get_vector(vector_id)`

指定IDのベクトルを取得します。

**パラメータ:**
- `vector_id` (str): ベクトルID

**戻り値:** `Optional[VectorEntry]` (見つからない場合はNone)

**例外:** `StorageError`

#### `delete_vector(vector_id)`

指定IDのベクトルを削除します。

**パラメータ:**
- `vector_id` (str): ベクトルID

**戻り値:** `bool` (削除成功時True)

**例外:** なし（エラー時はFalseを返す）

#### `update_vector(vector_id, embedding, metadata)`

ベクトルのembeddingとメタデータを更新します。

**パラメータ:**
- `vector_id` (str): ベクトルID
- `embedding` (List[float]): 新しいembedding
- `metadata` (Dict[str, Any]): 新しいメタデータ

**戻り値:** None

**例外:** `StorageError`

### 統計・管理メソッド

#### `count_vectors()`

保存されているベクトルの総数を取得します。

**戻り値:** `int`

**例外:** `StorageError`

#### `get_vector_dimension()`

ベクトルの次元数を取得します。

**戻り値:** `Optional[int]` (ベクトルが存在しない場合はNone)

**例外:** `StorageError`

#### `get_all_vectors()`

すべてのベクトルを取得します。

**戻り値:** `List[VectorEntry]`

**例外:** `StorageError`

**注意:** 大量データの場合は使用を避けてください。

#### `clear()`

コレクション内のすべてのベクトルを削除します。

**戻り値:** None

**例外:** `StorageError`

#### `get_stats()`

ベクトルストアの統計情報を取得します。

**戻り値:** `VectorStoreStats`

**例外:** `StorageError`

**使用例:**

```python
stats = vector_store.get_stats()
print(f"総ベクトル数: {stats.total_vectors}")
print(f"ベクトル次元: {stats.vector_dimension}")
print(f"インデックスタイプ: {stats.index_type}")
```

## データ型

### VectorSearchResult

検索結果を表すクラスです。

**属性:**
- `document_id` (str): 文書ID
- `content` (str): 文書内容
- `metadata` (Dict[str, Any]): メタデータ
- `score` (float): 類似性スコア (0.0-1.0)
- `embedding` (Optional[np.ndarray]): embedding（通常はNone）

### VectorEntry

ベクトルエントリを表すクラスです。

**属性:**
- `document_id` (str): 文書ID
- `content` (str): 文書内容
- `embedding` (np.ndarray): embedding
- `metadata` (Dict[str, Any]): メタデータ

### VectorStoreStats

ベクトルストア統計を表すクラスです。

**属性:**
- `total_vectors` (int): 総ベクトル数
- `vector_dimension` (int): ベクトル次元
- `storage_size_bytes` (int): ストレージサイズ（バイト）
- `index_type` (str): インデックスタイプ

## メタデータフィルタ演算子

### 基本演算子

| 演算子 | 説明 | 例 |
|--------|------|-----|
| 等価 | 完全一致 | `{"category": "AI"}` |
| `$ne` | 不等価 | `{"status": {"$ne": "deleted"}}` |
| `$gt` | より大きい | `{"score": {"$gt": 0.8}}` |
| `$gte` | 以上 | `{"score": {"$gte": 0.8}}` |
| `$lt` | より小さい | `{"score": {"$lt": 0.5}}` |
| `$lte` | 以下 | `{"score": {"$lte": 0.5}}` |

### 配列演算子

| 演算子 | 説明 | 例 |
|--------|------|-----|
| `$in` | 配列内の値と一致 | `{"category": {"$in": ["AI", "ML"]}}` |
| `$nin` | 配列内の値と不一致 | `{"status": {"$nin": ["draft", "deleted"]}}` |

### 論理演算子

| 演算子 | 説明 | 例 |
|--------|------|-----|
| `$and` | AND条件 | `{"$and": [{"a": 1}, {"b": 2}]}` |
| `$or` | OR条件 | `{"$or": [{"a": 1}, {"b": 2}]}` |
| `$not` | NOT条件 | `{"$not": {"category": "test"}}` |

### 複合例

```python
# 複雑なフィルタ例
complex_filter = {
    "$and": [
        {"language": "japanese"},
        {"$or": [
            {"category": "AI"},
            {"category": "ML"}
        ]},
        {"score": {"$gte": 0.7}},
        {"tags": {"$in": ["published", "reviewed"]}}
    ]
}

results = vector_store.search_similar(
    query_embedding=embedding,
    metadata_filter=complex_filter
)
```

## 例外

### StorageError

ベクトルストア操作でエラーが発生した場合に送出されます。

**継承:** `refinire_rag.exceptions.StorageError`

**発生条件:**
- ChromaDBクライアント初期化失敗
- コレクション操作失敗
- ベクトル保存・検索失敗
- データ整合性エラー

**処理例:**

```python
from refinire_rag.exceptions import StorageError

try:
    vector_store.add_documents_with_embeddings(documents, embeddings)
except StorageError as e:
    print(f"ストレージエラー: {e}")
    # エラーハンドリング処理
```

## ベストプラクティス

### 1. リソース管理

```python
# tempfileを使った一時ストレージ
import tempfile

with tempfile.TemporaryDirectory() as temp_dir:
    vector_store = ChromaVectorStore(persist_directory=temp_dir)
    # 処理...
    # 自動的にクリーンアップされる
```

### 2. エラーハンドリング

```python
def safe_vector_operation(vector_store, operation_func, *args, **kwargs):
    """安全なベクトル操作の実行"""
    try:
        return operation_func(*args, **kwargs)
    except StorageError as e:
        logger.error(f"ストレージエラー: {e}")
        return None
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        raise
```

### 3. バッチ処理

```python
def process_in_batches(items, batch_size=100):
    """バッチ処理でメモリ効率を向上"""
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

# 使用例
for document_batch in process_in_batches(all_documents, 50):
    vector_store.add_documents_with_embeddings(document_batch, embeddings_batch)
```