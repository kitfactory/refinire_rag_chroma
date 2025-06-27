# Environment Variables Specification

This document defines all environment variables used by refinire-rag-chroma and their importance levels for oneenv integration.

## Importance Levels

- **CRITICAL (Level 3)**: Essential for basic functionality. Application may fail without these.
- **IMPORTANT (Level 2)**: Affects performance, behavior, or specific features significantly.
- **OPTIONAL (Level 1)**: Nice-to-have settings that provide minor optimizations or convenience.

## Environment Variables

### Core Configuration

| Variable | Type | Default | Importance | Description |
|----------|------|---------|------------|-------------|
| `REFINIRE_RAG_CHROMA_COLLECTION_NAME` | string | `"refinire_documents"` | **IMPORTANT** | Name of the ChromaDB collection. Critical for data organization in multi-tenant environments. |
| `REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY` | string | `None` | **CRITICAL** | Directory for persistent storage. Essential for production data persistence vs in-memory storage. |
| `REFINIRE_RAG_CHROMA_DISTANCE_METRIC` | enum | `"cosine"` | **IMPORTANT** | Distance metric for similarity search. Significantly affects search accuracy and performance. |

### Performance Settings

| Variable | Type | Default | Importance | Description |
|----------|------|---------|------------|-------------|
| `REFINIRE_RAG_CHROMA_BATCH_SIZE` | integer | `100` | **OPTIONAL** | Batch size for bulk operations. Performance optimization that rarely needs tuning. |
| `REFINIRE_RAG_CHROMA_MAX_RETRIES` | integer | `3` | **OPTIONAL** | Maximum retry attempts for failed operations. Resilience setting with sensible default. |

### Behavior Control

| Variable | Type | Default | Importance | Description |
|----------|------|---------|------------|-------------|
| `REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION` | boolean | `true` | **IMPORTANT** | Automatically create collection if it doesn't exist. Critical for deployment automation. |
| `REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT` | boolean | `false` | **OPTIONAL** | Clear collection on initialization. Primarily for testing/development scenarios. |

## Detailed Analysis

### CRITICAL Variables

#### `REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY`
- **Impact**: Determines data persistence vs in-memory storage
- **Production Impact**: HIGH - Data loss without proper setting
- **Default Behavior**: In-memory storage (data lost on restart)
- **Validation**: Directory must be writable, creates parent directories if needed
- **Example Values**: 
  - `/data/chroma` (production)
  - `./data/chroma` (development)
  - `None` or empty (in-memory)

### IMPORTANT Variables

#### `REFINIRE_RAG_CHROMA_COLLECTION_NAME`
- **Impact**: Data organization and multi-tenancy
- **Production Impact**: MEDIUM - Affects data isolation
- **Default Behavior**: Uses "refinire_documents" collection
- **Validation**: Must not be empty or whitespace-only
- **Example Values**:
  - `production_docs`
  - `staging_documents`
  - `user_123_collection`

#### `REFINIRE_RAG_CHROMA_DISTANCE_METRIC`
- **Impact**: Search accuracy and performance characteristics
- **Production Impact**: MEDIUM - Affects search quality
- **Default Behavior**: Uses cosine similarity
- **Validation**: Must be one of "cosine", "l2", "ip"
- **Example Values**:
  - `cosine` (normalized embeddings, most common)
  - `l2` (Euclidean distance)
  - `ip` (inner product)

#### `REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION`
- **Impact**: Deployment automation and operational complexity
- **Production Impact**: MEDIUM - Affects deployment reliability
- **Default Behavior**: Automatically creates collections
- **Validation**: Boolean string parsing
- **Example Values**: `true`, `false`, `1`, `0`

### OPTIONAL Variables

#### `REFINIRE_RAG_CHROMA_BATCH_SIZE`
- **Impact**: Memory usage and processing speed
- **Production Impact**: LOW - Performance optimization
- **Default Behavior**: Processes in batches of 100
- **Validation**: Must be positive integer
- **Typical Range**: 50-500 depending on memory constraints

#### `REFINIRE_RAG_CHROMA_MAX_RETRIES`
- **Impact**: Resilience to transient failures
- **Production Impact**: LOW - Error handling optimization
- **Default Behavior**: Retries up to 3 times
- **Validation**: Must be non-negative integer
- **Typical Range**: 0-10

#### `REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT`
- **Impact**: Development and testing convenience
- **Production Impact**: LOW - Testing/development feature
- **Default Behavior**: Preserves existing data
- **Validation**: Boolean string parsing
- **Use Cases**: Unit tests, development reset

## Oneenv Integration Recommendations

### Environment Schema Definition

Based on oneenv capabilities, here's the recommended schema structure:

```python
# .oneenv/schema.py
from oneenv import EnvSchema, EnvVar
from typing import Literal

class ChromaVectorStoreEnv(EnvSchema):
    """Environment variables for ChromaVectorStore"""
    
    # CRITICAL Variables
    persist_directory: str = EnvVar(
        name="REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY",
        default=None,
        description="Directory for persistent storage (None for in-memory)",
        importance="CRITICAL",
        validation=lambda x: _validate_directory(x) if x else True
    )
    
    # IMPORTANT Variables
    collection_name: str = EnvVar(
        name="REFINIRE_RAG_CHROMA_COLLECTION_NAME",
        default="refinire_documents",
        description="Name of the ChromaDB collection",
        importance="IMPORTANT",
        validation=lambda x: len(x.strip()) > 0
    )
    
    distance_metric: Literal["cosine", "l2", "ip"] = EnvVar(
        name="REFINIRE_RAG_CHROMA_DISTANCE_METRIC",
        default="cosine",
        description="Distance metric for similarity search",
        importance="IMPORTANT",
        choices=["cosine", "l2", "ip"]
    )
    
    auto_create_collection: bool = EnvVar(
        name="REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION",
        default=True,
        description="Automatically create collection if it doesn't exist",
        importance="IMPORTANT"
    )
    
    # OPTIONAL Variables
    batch_size: int = EnvVar(
        name="REFINIRE_RAG_CHROMA_BATCH_SIZE",
        default=100,
        description="Batch size for bulk operations",
        importance="OPTIONAL",
        validation=lambda x: x > 0
    )
    
    max_retries: int = EnvVar(
        name="REFINIRE_RAG_CHROMA_MAX_RETRIES",
        default=3,
        description="Maximum retry attempts for failed operations",
        importance="OPTIONAL",
        validation=lambda x: x >= 0
    )
    
    auto_clear_on_init: bool = EnvVar(
        name="REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT",
        default=False,
        description="Clear collection on initialization (for testing)",
        importance="OPTIONAL"
    )

def _validate_directory(path: str) -> bool:
    """Validate that directory is accessible"""
    import os
    try:
        if os.path.exists(path):
            return os.path.isdir(path) and os.access(path, os.W_OK)
        else:
            # Try to create parent directory
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            return True
    except (OSError, PermissionError):
        return False
```

### Environment Profiles

#### Production Profile (.oneenv/production.env)
```env
# CRITICAL - Data persistence for production
REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=/data/chroma

# IMPORTANT - Production configuration
REFINIRE_RAG_CHROMA_COLLECTION_NAME=production_docs
REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine
REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION=true

# OPTIONAL - Performance tuning
REFINIRE_RAG_CHROMA_BATCH_SIZE=200
REFINIRE_RAG_CHROMA_MAX_RETRIES=5
```

#### Development Profile (.oneenv/development.env)
```env
# Local development with persistence
REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=./data/chroma
REFINIRE_RAG_CHROMA_COLLECTION_NAME=dev_documents
REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine

# Development optimizations
REFINIRE_RAG_CHROMA_BATCH_SIZE=50
REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT=false
```

#### Testing Profile (.oneenv/testing.env)
```env
# In-memory for testing (no persist_directory)
REFINIRE_RAG_CHROMA_COLLECTION_NAME=test_collection
REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION=true
REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT=true

# Small batch for faster tests
REFINIRE_RAG_CHROMA_BATCH_SIZE=10
REFINIRE_RAG_CHROMA_MAX_RETRIES=1
```

#### CI/CD Profile (.oneenv/ci.env)
```env
# Minimal config for CI/CD
REFINIRE_RAG_CHROMA_COLLECTION_NAME=ci_test_collection
REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION=true
REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT=true
REFINIRE_RAG_CHROMA_BATCH_SIZE=5
REFINIRE_RAG_CHROMA_MAX_RETRIES=0
```

### Validation Rules

#### Type Validation
- **String**: collection_name, persist_directory, distance_metric
- **Integer**: batch_size, max_retries
- **Boolean**: auto_create_collection, auto_clear_on_init

#### Custom Validation
- **distance_metric**: Must be in ["cosine", "l2", "ip"]
- **collection_name**: Must not be empty/whitespace
- **batch_size**: Must be > 0
- **max_retries**: Must be >= 0
- **persist_directory**: Must be writable if provided

### Environment Variable Patterns

#### Naming Convention
- Prefix: `REFINIRE_RAG_CHROMA_`
- Pattern: `{PREFIX}{CATEGORY}_{SETTING}`
- Categories: Core settings use direct names, performance settings have descriptive names

#### Boolean Parsing
- True values: `"true"`, `"1"`, `"yes"`, `"on"` (case-insensitive)
- False values: `"false"`, `"0"`, `"no"`, `"off"`, empty string

#### Default Behavior
- All variables have sensible defaults
- Application works with zero configuration
- Production-ready defaults for most scenarios

### Usage Examples with Oneenv

#### Basic Usage
```python
from oneenv import load_env
from .schema import ChromaVectorStoreEnv

# Load environment with validation
env = load_env(ChromaVectorStoreEnv)

# Use in ChromaVectorStore
vector_store = ChromaVectorStore(
    collection_name=env.collection_name,
    persist_directory=env.persist_directory,
    distance_metric=env.distance_metric
)
```

#### Profile-based Usage
```python
from oneenv import load_env_profile
from .schema import ChromaVectorStoreEnv

# Load specific profile
env = load_env_profile(ChromaVectorStoreEnv, profile="production")

# Or load based on environment
import os
profile = os.getenv("ENV", "development")
env = load_env_profile(ChromaVectorStoreEnv, profile=profile)
```

#### Validation and Error Handling
```python
from oneenv import load_env, ValidationError
from .schema import ChromaVectorStoreEnv

try:
    env = load_env(ChromaVectorStoreEnv)
    vector_store = ChromaVectorStore(
        collection_name=env.collection_name,
        persist_directory=env.persist_directory,
        distance_metric=env.distance_metric
    )
except ValidationError as e:
    logger.error(f"Environment validation failed: {e}")
    raise ConfigurationError(f"Invalid configuration: {e}")
```

### Integration with ChromaVectorStore

#### Modified Constructor
```python
class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        env: Optional[ChromaVectorStoreEnv] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        distance_metric: Optional[str] = None
    ):
        """
        Initialize ChromaDB Vector Store
        
        Args:
            env: Pre-loaded environment configuration (preferred)
            collection_name: Override collection name
            persist_directory: Override persist directory  
            distance_metric: Override distance metric
        """
        # Load environment if not provided
        if env is None:
            try:
                env = load_env(ChromaVectorStoreEnv)
            except ValidationError as e:
                logger.error(f"Environment validation failed: {e}")
                # Fallback to current implementation
                env = None
        
        # Use environment or fallback to parameters/defaults
        if env is not None:
            self.collection_name = collection_name or env.collection_name
            self.persist_directory = persist_directory or env.persist_directory
            self.distance_metric = distance_metric or env.distance_metric
            self.batch_size = env.batch_size
            self.max_retries = env.max_retries
            self.auto_create_collection = env.auto_create_collection
            self.auto_clear_on_init = env.auto_clear_on_init
        else:
            # Fallback to current implementation
            self._load_from_os_environ(collection_name, persist_directory, distance_metric)
```

### Project Structure for Oneenv

```
refinire-rag-chroma/
├── .oneenv/
│   ├── schema.py              # Environment schema definition
│   ├── production.env         # Production environment
│   ├── development.env        # Development environment
│   ├── testing.env           # Testing environment
│   ├── ci.env                # CI/CD environment
│   └── template.env          # Template with all variables
├── src/
│   └── refinire_rag_chroma/
│       ├── __init__.py
│       ├── chroma_vector_store.py
│       └── config.py         # Oneenv integration utilities
└── environments.md           # This documentation
```

### Configuration Templates

#### Template File (.oneenv/template.env)
```env
# Environment Configuration Template for refinire-rag-chroma
# Copy this file and customize for your environment

# ============================================================================
# CRITICAL SETTINGS - Required for production
# ============================================================================

# Data persistence directory (None for in-memory)
# Examples: /data/chroma, ./data/chroma, or leave empty for in-memory
REFINIRE_RAG_CHROMA_PERSIST_DIRECTORY=

# ============================================================================
# IMPORTANT SETTINGS - Affects behavior and performance
# ============================================================================

# Collection name for document organization
REFINIRE_RAG_CHROMA_COLLECTION_NAME=refinire_documents

# Distance metric for similarity search
# Options: cosine (recommended), l2, ip
REFINIRE_RAG_CHROMA_DISTANCE_METRIC=cosine

# Auto-create collection if it doesn't exist
# Options: true (recommended), false
REFINIRE_RAG_CHROMA_AUTO_CREATE_COLLECTION=true

# ============================================================================
# OPTIONAL SETTINGS - Performance and convenience
# ============================================================================

# Batch size for bulk operations (affects memory usage)
REFINIRE_RAG_CHROMA_BATCH_SIZE=100

# Maximum retry attempts for failed operations
REFINIRE_RAG_CHROMA_MAX_RETRIES=3

# Clear collection on initialization (testing only)
# Options: false (recommended), true
REFINIRE_RAG_CHROMA_AUTO_CLEAR_ON_INIT=false
```

## Migration Path

1. **Phase 1**: Current implementation (direct os.getenv()) ✅ CURRENT
2. **Phase 2**: Add oneenv dependency and schema definition
3. **Phase 3**: Modify ChromaVectorStore to accept oneenv config
4. **Phase 4**: Create environment profiles and templates
5. **Phase 5**: Enhanced validation and error messages
6. **Phase 6**: Advanced features (profile inheritance, dynamic validation)

## Security Considerations

- **No Sensitive Data**: None of these variables contain secrets
- **Path Validation**: persist_directory is validated for security
- **Injection Prevention**: All values are validated before use
- **Logging**: Sensitive paths are masked in logs

## Implementation Priority

### High Priority (Immediate Benefits)
1. **Schema Definition** - Type safety and validation
2. **Profile Management** - Environment-specific configurations
3. **Error Handling** - Better validation error messages

### Medium Priority (Enhanced Features)
4. **Template Generation** - Automatic .env file generation
5. **Profile Inheritance** - Base profiles with overrides
6. **Dynamic Validation** - Runtime validation improvements

### Low Priority (Advanced Features)
7. **Auto-completion** - IDE support for environment variables
8. **Configuration UI** - Web interface for environment management
9. **Migration Tools** - Automated environment migration

## Oneenv Specific Benefits

### Type Safety
- **Compile-time validation**: Catch configuration errors early
- **IDE support**: Auto-completion and type hints
- **Documentation**: Self-documenting configuration schema

### Profile Management
- **Environment isolation**: Separate configs for dev/staging/prod
- **Template sharing**: Consistent configuration across teams
- **Version control**: Track configuration changes

### Validation
- **Custom validators**: Domain-specific validation logic
- **Dependency validation**: Cross-variable validation rules
- **Error messages**: Clear, actionable error descriptions

### Developer Experience
- **Zero configuration**: Works without .env files
- **Gradual adoption**: Backward compatible with current implementation
- **Testing support**: Easy mocking and test configuration

## Oneenv Integration Checklist

- [ ] Add oneenv dependency to pyproject.toml
- [ ] Create .oneenv/schema.py with ChromaVectorStoreEnv
- [ ] Implement environment profiles (production, development, testing, ci)
- [ ] Update ChromaVectorStore constructor to accept env parameter
- [ ] Add backward compatibility with current os.getenv() approach
- [ ] Create configuration templates and documentation
- [ ] Add validation tests for all environment variables
- [ ] Update CI/CD to use oneenv profiles
- [ ] Document migration path for existing deployments

## Operational Notes

- **Docker**: All variables work well in containerized environments
- **Kubernetes**: Compatible with ConfigMaps and environment variable injection
- **CI/CD**: Easy to override for different deployment stages with oneenv profiles
- **Monitoring**: Enhanced validation provides better alerting on invalid configurations
- **Documentation**: Oneenv schema serves as living documentation
- **Team Collaboration**: Shared templates ensure consistent configuration across developers