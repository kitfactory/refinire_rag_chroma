# refinire-rag-chroma

refinire-rag用ChromaDBベクトルストアプラグイン - 検索拡張生成システムのための本格的なベクトルデータベース統合

## 概要

`refinire-rag-chroma`は[refinire-rag](https://github.com/refinire/refinire-rag)エコシステム向けのシームレスなChromaDB統合を提供します。このプラグインは`VectorStore`インターフェースを実装し、ChromaDBをバックエンドとして効率的なベクトル保存、類似性検索、メタデータフィルタリングを可能にします。

## 機能

- ✅ **完全なrefinire-rag VectorStore実装**: `refinire_rag.VectorStore`を継承
- ✅ **文書保存・検索**: embeddingと共に文書を保存・検索
- ✅ **多様な検索タイプ**: 
  - クエリembeddingによる類似性検索
  - 文書間類似性検索
  - メタデータベースのフィルタリング・検索
- ✅ **柔軟なストレージオプション**: インメモリまたは永続化ストレージ
- ✅ **本格運用対応**: 包括的なエラーハンドリングとログ出力
- ✅ **完全なテストカバレッジ**: ユニットテストと統合例

## インストール

### 必要要件

- Python 3.10+
- refinire-rag >= 0.0.1
- chromadb >= 0.4.0

### 依存関係のインストール

```bash
# リポジトリをクローン
git clone https://github.com/your-org/refinire-rag-chroma.git
cd refinire-rag-chroma

# uvでインストール（推奨）
uv sync

# または pipでインストール
pip install -e .
```

## クイックスタート

### 基本的な使用方法

```python
from refinire_rag import Document, TFIDFEmbedder, TFIDFEmbeddingConfig
from src.chroma_vector_store import ChromaVectorStore

# Embedderを初期化
embedding_config = TFIDFEmbeddingConfig(max_features=1000, ngram_range=(1, 2))
embedder = TFIDFEmbedder(config=embedding_config)

# ChromaDBベクトルストアを初期化
vector_store = ChromaVectorStore(
    collection_name="my_documents",
    persist_directory="./chroma_db",  # インメモリの場合はNone
    distance_metric="cosine"
)

# 文書を作成
documents = [
    Document(
        id="doc_001",
        content="機械学習は人工知能の一分野です。",
        metadata={"category": "ai", "language": "japanese"}
    ),
    Document(
        id="doc_002", 
        content="ChromaDBは高性能なベクトルデータベースです。",
        metadata={"category": "database", "language": "japanese"}
    )
]

# Embedderを訓練してembeddingを生成
embedder.fit([doc.content for doc in documents])
embedding_results = embedder.embed_documents(documents)
embeddings = [result.vector.tolist() for result in embedding_results]

# 文書をembeddingと共に保存
vector_store.add_documents_with_embeddings(documents, embeddings)

# 類似文書を検索
query_result = embedder.embed_text("AIについて教えて")
search_results = vector_store.search_similar(
    query_embedding=query_result.vector.tolist(),
    top_k=5
)

# 結果を表示
for result in search_results:
    print(f"文書ID: {result.document_id}")
    print(f"スコア: {result.score:.4f}")
    print(f"内容: {result.content}")
    print("---")
```

### メタデータフィルタリングを使った高度な検索

```python
# メタデータフィルタ付き検索
filtered_results = vector_store.search_similar(
    query_embedding=query_embedding,
    top_k=3,
    metadata_filter={"category": "ai", "language": "japanese"}
)

# 文書間類似性検索
similar_docs = vector_store.search_similar_to_document(
    document_id="doc_001",
    top_k=3
)

# メタデータのみでの検索
metadata_results = vector_store.search_by_metadata(
    metadata_filter={"category": "database"}
)
```

## API リファレンス

### ChromaVectorStore

#### コンストラクタ

```python
ChromaVectorStore(
    collection_name: str = "refinire_documents",
    persist_directory: Optional[str] = None,
    distance_metric: str = "cosine"
)
```

**パラメータ:**
- `collection_name`: ChromaDBコレクション名
- `persist_directory`: 永続化ストレージのディレクトリ（インメモリの場合はNone）
- `distance_metric`: 距離メトリック（"cosine", "l2", "ip"）

#### メソッド

##### `add_documents_with_embeddings(documents, embeddings)`
事前計算されたembeddingと共に文書を保存します。

##### `search_similar(query_embedding, top_k=10, metadata_filter=None)`
クエリembeddingを使って類似ベクトルを検索します。

##### `search_similar_to_document(document_id, top_k=10, metadata_filter=None)`
特定の文書に類似する文書を検索します。

##### `search_by_metadata(metadata_filter)`
メタデータ条件のみで文書を検索します。

##### `get_stats()`
ベクトルストアの統計情報（総ベクトル数、次元数など）を取得します。

### メタデータフィルタ構文

ChromaDBは以下のフィルタ演算子をサポートしています：

```python
# 基本的な等価条件
{"category": "ai"}

# 比較演算子
{"size_bytes": {"$gt": 100}}     # より大きい
{"size_bytes": {"$gte": 100}}    # 以上
{"size_bytes": {"$lt": 1000}}    # より小さい
{"size_bytes": {"$lte": 1000}}   # 以下
{"status": {"$ne": "deleted"}}   # 等しくない

# 配列演算子
{"tags": {"$in": ["ai", "ml"]}}           # 配列内のいずれかと一致
{"category": {"$nin": ["test", "demo"]}}  # 配列内のいずれとも一致しない

# 論理演算子
{"$and": [{"category": "ai"}, {"language": "japanese"}]}
{"$or": [{"category": "ai"}, {"category": "ml"}]}
{"$not": {"category": "test"}}

# 複数条件（自動的に$andで結合）
{"category": "ai", "language": "japanese"}
```

## 使用例

### 完全統合例

包括的な例については [src/examples/real_refinire_rag_example.py](src/examples/real_refinire_rag_example.py) をご覧ください。以下をデモンストレーションしています：

- TF-IDFによる文書embedding
- 複数の検索シナリオ
- メタデータフィルタリング
- エラーハンドリング

例を実行する：

```bash
uv run python src/examples/real_refinire_rag_example.py
```

### レガシー統合例

モック統合例については [src/examples/refinire_rag_integration.py](src/examples/refinire_rag_integration.py) をご覧ください。

## 開発

### プロジェクト構造

```
refinire-rag-chroma/
├── src/
│   ├── chroma_vector_store.py    # メイン実装
│   ├── examples/                 # 使用例
│   └── ...
├── tests/
│   ├── unit/                     # ユニットテスト
│   └── e2e/                      # E2Eテスト
├── docs/                         # ドキュメント
├── pyproject.toml               # プロジェクト設定
└── README.md                    # このファイル
```

### テストの実行

```bash
# 全テストを実行
uv run pytest

# カバレッジ付きで実行
uv run pytest --cov=src --cov-report=html

# 特定のテストカテゴリを実行
uv run pytest tests/unit/
```

### コード品質

プロジェクトは厳格なコード品質ガイドラインに従っています：

- **単一責任原則**: 各クラスは一つの責任を持つ
- **DRY原則**: コードの重複なし
- **ドメイン駆動設計**: モデル、サービス、コントローラの明確な分離
- **包括的テスト**: 高いカバレッジのユニットテスト

## 貢献

1. リポジトリをフォーク
2. フィーチャーブランチを作成: `git checkout -b feature/your-feature-name`
3. 変更を加えテストを追加
4. テストを実行: `uv run pytest`
5. 変更をコミット: `git commit -am 'Add some feature'`
6. ブランチにプッシュ: `git push origin feature/your-feature-name`
7. プルリクエストを送信

## ライセンス

このプロジェクトはMITライセンスの下で公開されています - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 関連プロジェクト

- [refinire-rag](https://github.com/refinire/refinire-rag) - メインのRAGフレームワーク
- [ChromaDB](https://github.com/chroma-core/chroma) - ベクトルデータベースバックエンド

## サポート

質問やサポートについては：

1. [ドキュメント](docs/)をご確認ください
2. [既存のissue](https://github.com/your-org/refinire-rag-chroma/issues)をレビューしてください
3. 必要に応じて新しいissueを作成してください

## 変更履歴

### v0.0.1
- 初回リリース
- 完全なrefinire-rag VectorStore実装
- 主要検索操作のサポート
- 包括的テストスイート
- 本格運用対応エラーハンドリング

## 日本語での技術サポート

このプラグインは日本語文書処理に最適化されており、以下の機能を提供します：

- **日本語embedding対応**: TF-IDFEmbedderによる日本語テキスト処理
- **日本語メタデータ**: 日本語文字列をメタデータに含めた検索
- **文字エンコーディング**: UTF-8による完全な日本語サポート

### 日本語文書の例

```python
documents = [
    Document(
        id="tech_001",
        content="深層学習は多層ニューラルネットワークを用いた機械学習手法です。",
        metadata={
            "分野": "人工知能", 
            "言語": "日本語",
            "難易度": "中級"
        }
    ),
    Document(
        id="tech_002",
        content="RAGシステムは検索拡張生成により高精度な回答を生成します。",
        metadata={
            "分野": "自然言語処理",
            "言語": "日本語", 
            "難易度": "上級"
        }
    )
]

# 日本語メタデータでの検索
results = vector_store.search_by_metadata({
    "言語": "日本語",
    "難易度": {"$in": ["中級", "上級"]}
})
```