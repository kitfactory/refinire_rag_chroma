from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator


@dataclass
class VectorDocument:
    id: str
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class VectorSearchQuery(BaseModel):
    query_embedding: List[float] = Field(..., description="クエリのベクトル表現")
    top_k: int = Field(default=5, ge=1, le=100, description="取得する類似文書数")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="メタデータフィルタ")
    
    @field_validator('query_embedding')
    @classmethod
    def validate_embedding(cls, v):
        if not v or len(v) == 0:
            raise ValueError("クエリベクトルは空であってはいけません")
        return v


class VectorSearchResult(BaseModel):
    document: VectorDocument
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class CollectionConfig(BaseModel):
    name: str = Field(..., min_length=1, description="コレクション名")
    dimension: int = Field(..., ge=1, description="ベクトル次元数")
    distance_metric: str = Field(default="cosine", description="距離メトリック")
    metadata_schema: Optional[Dict[str, str]] = Field(default=None, description="メタデータスキーマ")


class VectorStore(ABC):
    @abstractmethod
    def create_collection(self, config: CollectionConfig) -> bool:
        pass
    
    @abstractmethod
    def add_documents(self, collection_name: str, documents: List[VectorDocument]) -> bool:
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query: VectorSearchQuery) -> List[VectorSearchResult]:
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        pass