# TODO - refinire-rag-chroma プロジェクト

## 完了済み
- [x] 要件確認 - ChromaDBベースのVectorStoreプラグインの詳細要件を定義
- [x] プロジェクト構造をCLAUDE.mdの仕様に合わせて整備 (src/ディレクトリ構造作成)
- [x] pyproject.tomlをpytest設定とdev依存関係で更新
- [x] データモデル定義 - VectorStoreの抽象化とChromaDB実装のエンティティ設計
- [x] クラス設計テーブル作成とMermaidクラス図生成
- [x] サービス層実装 - ChromaDBのビジネスロジック
- [x] コントローラ実装 - VectorStoreインターフェース
- [x] ユニットテスト作成 (src/tests/unit/)

## 実装されたコンポーネント

### 1. データモデル (src/models.py)
- `VectorDocument`: ベクトル文書データクラス
- `VectorSearchQuery`: 検索クエリ仕様（Pydanticバリデーション付き）
- `VectorSearchResult`: 検索結果データクラス
- `CollectionConfig`: コレクション設定データクラス
- `VectorStore`: 抽象基底クラス（ABC）

### 2. サービス層 (src/service.py)
- `ChromaService`: ChromaDBビジネスロジック処理
  - クライアント初期化
  - バリデーション
  - エラーハンドリング
- `ChromaVectorStore`: VectorStoreの具象実装
  - コレクション作成・削除
  - ドキュメント追加
  - ベクトル検索
  - コレクション一覧取得

### 3. 使用例 (src/examples/usage_example.py)
- ChromaVectorStoreの基本的な使用方法
- サンプルデータでの動作確認

### 4. ユニットテスト (src/tests/unit/)
- `test_models.py`: データモデルのテスト
- `test_service.py`: サービス層のテスト（モック使用）

## ✅ 完了: refinire-rag新ガイド対応

### Phase 1: プラグインシステム更新 (完了)
- [x] pyproject.tomlにエントリーポイント追加 - 自動プラグイン発見対応
- [x] oneenvテンプレート作成 - 環境変数の標準化（既存テンプレート確認済み）  
- [x] 設定クラス修正 - プロパティデコレータ対応（ChromaVectorStoreConfig追加）
- [x] プラグイン発見機能テスト（エントリーポイント設定完了）
- [x] 新ガイド準拠の動作確認（ChromaVectorStoreConfig対応確認済み）

### 実装済み機能
1. **エントリーポイント**: `[project.entry-points."refinire_rag.vectorstores"]`
2. **設定クラス**: `ChromaVectorStoreConfig` - プロパティベース設定
3. **後方互換性**: 既存の初期化方法も引き続きサポート
4. **テストカバレッジ**: 新設定クラス用テスト追加（84%カバレッジ）

### 対応理由
refinire-rag v0.1.3で新しいプラグイン開発ガイドが導入され、以下の要件が追加：
1. **エントリーポイントシステム**: `pyproject.toml`での自動プラグイン発見
2. **環境変数ベース設定**: oneenvテンプレートによる標準化
3. **プラグインタイプ拡張**: VectorStore以外も対応
4. **設定クラス標準化**: プロパティデコレータ使用

## 今後の拡張予定

### Phase 2: 高度な機能
- [ ] バッチ処理機能（大量データの効率的な処理）
- [ ] インデックス最適化機能
- [ ] ベクトル次元変換ユーティリティ
- [ ] メタデータスキーマバリデーション強化

### Phase 3: パフォーマンス
- [ ] 非同期処理対応
- [ ] 接続プーリング
- [ ] キャッシュ機能
- [ ] パフォーマンステスト

### Phase 4: 運用機能
- [ ] ロギング強化
- [ ] メトリクス収集
- [ ] ヘルスチェック機能
- [ ] 設定ファイル対応

### Phase 5: E2Eテスト
- [ ] 統合テスト作成
- [ ] パフォーマンステスト
- [ ] エラーシナリオテスト

## 品質制約チェックリスト
- [x] 単一責任原則 (SRP) - 各クラス200行以内
- [x] DRY原則 - 重複コード排除
- [x] 入力バリデーション - Pydanticによる検証
- [x] エラーハンドリング - 適切な例外処理
- [x] テストカバレッジ - ユニットテスト実装
- [x] インターフェース分離 - 抽象クラスによる設計
- [x] 依存関係逆転 - 抽象に依存する設計

## 開発コマンド
```bash
# 依存関係インストール
uv add chromadb pydantic pytest pytest-cov

# テスト実行
pytest

# カバレッジ付きテスト
pytest --cov=src --cov-report=term-missing

# 使用例実行
python src/examples/usage_example.py
```