# refinire-rag-chroma

ChromaDB VectorStore plugin for refinire-rag, providing seamless integration with ChromaDB for vector storage and similarity search.

## Features

- ✅ **Zero Configuration**: Works out of the box with sensible defaults
- ✅ **Environment Variable Configuration**: Configure via `REFINIRE_RAG_CHROMA_*` environment variables
- ✅ **Full refinire-rag v0.1.1+ Compatibility**: Implements the complete VectorStore interface
- ✅ **DocumentProcessor Integration**: Supports refinire-rag processing pipelines
- ✅ **Persistent and In-Memory Storage**: Choose between persistent disk storage or in-memory
- ✅ **Multiple Distance Metrics**: Support for cosine, L2, and inner product distance
- ✅ **Production Ready**: Comprehensive error handling, logging, and validation

## Quick Start

### Zero Configuration Usage

```python
from refinire_rag_chroma import ChromaVectorStore

# Works immediately with default settings
vector_store = ChromaVectorStore()

# Add documents (requires embedder)
vector_store.set_embedder(your_embedder)
processed_docs = list(vector_store.process(documents))
```

### Environment Variable Configuration

```bash
# Set environment variables
export REFINIRE_RAG_CHROMA_COLLECTION_NAME="my_documents"
export REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY="/data/chroma"
export REFINIRE_RAG_CHROMA_DISTANCE_METRIC="cosine"
```

```python
from refinire_rag_chroma import ChromaVectorStore

# Automatically uses environment variables
vector_store = ChromaVectorStore()
```

## Installation

```bash
pip install refinire-rag-chroma
```

Or with uv:

```bash
uv add refinire-rag-chroma
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REFINIRE_RAG_CHROMA_COLLECTION_NAME` | `"refinire_documents"` | Collection name |
| `REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY` | `None` | Storage directory (None = in-memory) |
| `REFINIRE_RAG_CHROMA_DISTANCE_METRIC` | `"cosine"` | Distance metric (`"cosine"`, `"l2"`, `"ip"`) |
| `REFINIRE_RAG_CHROMA_BATCH_SIZE` | `100` | Batch size for operations |
| `REFINIRE_RAG_CHROMA_MAX_RETRIES` | `3` | Maximum retry attempts |
| `REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION` | `"true"` | Auto-create collection |
| `REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT` | `"false"` | Clear on initialization |

### Parameter-based Configuration

```python
from refinire_rag_chroma import ChromaVectorStore

# Override specific settings with parameters
vector_store = ChromaVectorStore(
    collection_name="custom_collection",
    persist_directory="/path/to/storage",
    distance_metric="l2"
)
```

## Usage Examples

### Basic Vector Operations

```python
import numpy as np
from refinire_rag_chroma import ChromaVectorStore
from refinire_rag.storage import VectorEntry

vector_store = ChromaVectorStore()

# Add a vector
entry = VectorEntry(
    document_id="doc1",
    content="Sample document",
    embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    metadata={"source": "example"}
)
vector_store.add_vector(entry)

# Search similar vectors
query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
results = vector_store.search_similar(query_vector, limit=10)
```

### Document Processing Pipeline

```python
from refinire_rag_chroma import ChromaVectorStore
from refinire_rag.models.document import Document

# Set up vector store with embedder
vector_store = ChromaVectorStore()
vector_store.set_embedder(your_embedder)

# Process documents
documents = [
    Document(id="1", content="First document", metadata={}),
    Document(id="2", content="Second document", metadata={})
]

# Documents are automatically embedded and stored
processed_docs = list(vector_store.process(documents))
```

### Metadata Filtering

```python
# Search by metadata
results = vector_store.search_by_metadata(
    filters={"source": "wikipedia"},
    limit=50
)

# Count vectors with filters
count = vector_store.count_vectors(filters={"category": "science"})
```

## Docker Usage

```dockerfile
FROM python:3.10

ENV REFINIRE_RAG_CHROMA_COLLECTION_NAME=production_docs
ENV REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=/data/chroma
ENV REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine

COPY . /app
WORKDIR /app
RUN pip install refinire-rag-chroma

CMD ["python", "app.py"]
```

## Docker Compose

```yaml
version: '3.8'
services:
  app:
    image: my-app:latest
    environment:
      - REFINIRE_RAG_CHROMA_COLLECTION_NAME=production_docs
      - REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=/data/chroma
      - REFINIRE_RAG_CHROMA_BATCH_SIZE=200
    volumes:
      - chroma_data:/data/chroma

volumes:
  chroma_data:
```

## API Reference

### ChromaVectorStore

The main vector store implementation that supports both the VectorStore and DocumentProcessor interfaces.

#### Methods

- `add_vector(entry: VectorEntry) -> str`: Add a single vector
- `add_vectors(entries: List[VectorEntry]) -> List[str]`: Add multiple vectors
- `get_vector(document_id: str) -> Optional[VectorEntry]`: Retrieve a vector
- `update_vector(entry: VectorEntry) -> bool`: Update a vector
- `delete_vector(document_id: str) -> bool`: Delete a vector
- `search_similar(query_vector: np.ndarray, limit: int, threshold: Optional[float], filters: Optional[Dict]) -> List[VectorSearchResult]`: Search similar vectors
- `search_by_metadata(filters: Dict, limit: int) -> List[VectorSearchResult]`: Search by metadata only
- `count_vectors(filters: Optional[Dict]) -> int`: Count vectors
- `get_stats() -> VectorStoreStats`: Get store statistics
- `clear() -> bool`: Clear all vectors
- `set_embedder(embedder: Any) -> None`: Set embedder for processing
- `process(documents: Iterable[Document], config: Optional[Any]) -> Iterator[Document]`: Process documents

### Configuration

ChromaVectorStore automatically reads configuration from environment variables with sensible defaults. You can override specific settings by passing parameters to the constructor.

## Development

### Setup

```bash
git clone https://github.com/your-repo/refinire-rag-chroma
cd refinire-rag-chroma
uv install
```

### Testing

```bash
uv run pytest
```

### With Coverage

```bash
uv run pytest --cov=src
```

## Requirements

- Python 3.8+
- refinire-rag >=0.1.1
- chromadb >=0.4.0
- numpy

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and ensure they pass
6. Submit a pull request

## Changelog

### v0.0.1

- Initial release
- Full refinire-rag v0.1.1+ compatibility
- Environment variable configuration system
- Zero-configuration deployment support
- DocumentProcessor integration
- Comprehensive test suite