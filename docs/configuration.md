# Configuration Guide

refinire-rag-chroma uses environment variables for configuration to provide zero-configuration deployment with sensible defaults. No separate configuration class is needed - just set environment variables or pass parameters to the constructor.

## Environment Variables

All configuration is done through environment variables with the `REFINIRE_RAG_CHROMA_` prefix:

### Core Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `REFINIRE_RAG_CHROMA_COLLECTION_NAME` | `"refinire_documents"` | Name of the ChromaDB collection |
| `REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY` | `None` | Directory for persistent storage (None = in-memory) |
| `REFINIRE_RAG_CHROMA_DISTANCE_METRIC` | `"cosine"` | Distance metric: `"cosine"`, `"l2"`, or `"ip"` |

### Performance Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `REFINIRE_RAG_CHROMA_BATCH_SIZE` | `100` | Batch size for bulk operations |
| `REFINIRE_RAG_CHROMA_MAX_RETRIES` | `3` | Maximum retry attempts for failed operations |

### Behavior Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION` | `"true"` | Automatically create collection if it doesn't exist |
| `REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT` | `"false"` | Clear collection on initialization (useful for testing) |

## Usage Examples

### Default Configuration (Zero Config)

```python
from refinire_rag_chroma import ChromaVectorStore

# Uses all default values from environment variables
vector_store = ChromaVectorStore()
```

### Environment Variable Configuration

```bash
# Set environment variables
export REFINIRE_RAG_CHROMA_COLLECTION_NAME="my_documents"
export REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY="/data/chroma"
export REFINIRE_RAG_CHROMA_DISTANCE_METRIC="l2"
export REFINIRE_RAG_CHROMA_BATCH_SIZE="200"
```

```python
from refinire_rag_chroma import ChromaVectorStore

# Automatically reads from environment variables
vector_store = ChromaVectorStore()
```

### Parameter-based Configuration

```python
from refinire_rag_chroma import ChromaVectorStore

# Override specific settings with parameters
vector_store = ChromaVectorStore(
    collection_name="custom_collection",
    persist_directory="/path/to/storage",
    distance_metric="cosine"
)
```

### Mixed Configuration (Environment + Override)

```python
from refinire_rag_chroma import ChromaVectorStore

# Use environment variables but override specific values
vector_store = ChromaVectorStore(
    collection_name="override_collection"  # Overrides env var
    # Other values come from environment variables
)
```

## Docker Configuration

```dockerfile
FROM python:3.10

# Set environment variables
ENV REFINIRE_RAG_CHROMA_COLLECTION_NAME=production_docs
ENV REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=/app/data/chroma
ENV REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine
ENV REFINIRE_RAG_CHROMA_BATCH_SIZE=200

COPY . /app
WORKDIR /app
RUN pip install refinire-rag-chroma

CMD ["python", "app.py"]
```

## Docker Compose Configuration

```yaml
version: '3.8'
services:
  app:
    image: my-app:latest
    environment:
      - REFINIRE_RAG_CHROMA_COLLECTION_NAME=production_docs
      - REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=/data/chroma
      - REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine
      - REFINIRE_RAG_CHROMA_BATCH_SIZE=200
      - REFINIRE_RAG_CHROMA_MAX_RETRIES=5
    volumes:
      - chroma_data:/data/chroma

volumes:
  chroma_data:
```

## Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chroma-config
data:
  REFINIRE_RAG_CHROMA_COLLECTION_NAME: "production_docs"
  REFINIRE_RAG_CHROMA_DISTANCE_METRIC: "cosine"
  REFINIRE_RAG_CHROMA_BATCH_SIZE: "200"
  REFINIRE_RAG_CHROMA_MAX_RETRIES: "5"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: my-app:latest
        envFrom:
        - configMapRef:
            name: chroma-config
        env:
        - name: REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY
          value: "/data/chroma"
        volumeMounts:
        - name: chroma-storage
          mountPath: /data/chroma
      volumes:
      - name: chroma-storage
        persistentVolumeClaim:
          claimName: chroma-pvc
```

## Configuration Validation

The configuration system automatically validates all settings during initialization:

- **Collection name**: Must not be empty
- **Distance metric**: Must be one of `"cosine"`, `"l2"`, `"ip"`
- **Batch size**: Must be positive
- **Max retries**: Must be non-negative
- **Persist directory**: Must be writable (creates parent directories if needed)

```python
from refinire_rag_chroma import ChromaVectorStore

try:
    # This will raise ValueError due to invalid settings
    vector_store = ChromaVectorStore(
        collection_name="",  # Invalid - empty name
        distance_metric="invalid"  # Invalid - not supported
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

1. **Use environment variables** for production deployments
2. **Set persist_directory** for production to enable data persistence
3. **Adjust batch_size** based on your memory constraints and performance needs
4. **Use meaningful collection names** to organize different document sets
5. **Choose appropriate distance metrics**:
   - `"cosine"`: Good for normalized embeddings (most common)
   - `"l2"`: Euclidean distance, good for raw embeddings
   - `"ip"`: Inner product, good for specific use cases

## Testing Configuration

For testing, you might want to use in-memory storage and auto-clear:

```bash
export REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=""  # In-memory
export REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT="true"  # Clear on startup
export REFINIRE_RAG_CHROMA_COLLECTION_NAME="test_collection"
```