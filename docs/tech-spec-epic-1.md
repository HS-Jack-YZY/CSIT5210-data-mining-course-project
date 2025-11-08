# Epic Technical Specification: Foundation & Environment Setup

Date: 2025-11-09
Author: Jack YUAN
Epic ID: epic-1
Status: Draft

---

## Overview

Epic 1 establishes the foundational infrastructure required for the Context-Aware Multi-Agent System project. This epic creates the necessary scaffolding for semantic clustering and cost optimization experiments by setting up the development environment, data pipeline, API integrations, and configuration management. Without proper foundation, no clustering, classification, or cost analysis can proceed.

The epic delivers a fully configured Python ML project based on Cookiecutter Data Science structure, integrated with Google Gemini Embedding API for semantic vector generation, loaded with the AG News dataset (120K training documents across 4 categories), and equipped with centralized configuration management. This foundation enables all subsequent epics to focus on core data mining algorithms rather than infrastructure concerns.

## Objectives and Scope

**In Scope for Epic 1:**
- ✅ Complete Python project structure initialization using Cookiecutter Data Science v2 template
- ✅ Virtual environment setup with all required ML dependencies (scikit-learn, numpy, pandas, google-genai)
- ✅ Centralized YAML-based configuration system with validation and environment variable support
- ✅ AG News dataset loading from Hugging Face with validation and caching
- ✅ Google Gemini Embedding API integration with authentication, batch processing, and retry logic
- ✅ Secure API key management using environment variables (.env pattern)
- ✅ Project documentation (README.md with installation and usage instructions)
- ✅ Version control setup (.gitignore configured for Python ML projects)

**Out of Scope for Epic 1:**
- ❌ Embedding generation (moved to Epic 2 - requires this foundation first)
- ❌ Clustering algorithm implementation (Epic 2)
- ❌ Agent implementation (Epic 3)
- ❌ Classification and routing logic (Epic 3)
- ❌ Cost calculation and baseline comparison (Epic 4)
- ❌ Visualizations and experimental reporting (Epic 4)

**Success Criteria:**
- All dependencies install without errors in clean virtual environment
- Configuration system successfully loads and validates config.yaml
- AG News dataset loads with expected 120K training and 7.6K test samples
- Gemini API authentication succeeds with test embedding call
- Project structure matches architecture specification exactly
- All 4 stories in Epic 1 completed and validated

## System Architecture Alignment

Epic 1 implements the foundational layer of the architecture as defined in [architecture.md](./architecture.md):

**Project Structure Alignment:**
- Follows Cookiecutter Data Science v2 standard layout (ADR-001)
- Establishes `data/`, `src/`, `notebooks/`, `results/`, `reports/` hierarchy
- Creates module structure: `src/context_aware_multi_agent_system/` with submodules for data, features, models, evaluation, visualization, utils

**Technology Stack Implementation:**
- Python 3.10 as specified (compatible with scikit-learn 1.7.2+)
- Core ML stack: numpy 1.24+, pandas 2.0+, scikit-learn 1.7.2+
- Embedding service: google-genai SDK with gemini-embedding-001 model (768 dimensions)
- Dataset loader: Hugging Face `datasets` library for AG News
- Configuration: PyYAML for config.yaml parsing, python-dotenv for .env management

**Integration Points:**
- Google Gemini Embedding API: Implements authentication, batch API support, retry logic (ADR-003)
- Hugging Face Datasets: Implements AG News loading with automatic caching to `~/.cache/huggingface/`
- Configuration abstraction: Centralizes all parameters in config.yaml, no hardcoded values

**Cross-Cutting Concerns:**
- Security: API key management via environment variables (ADR-006), .gitignore prevents secret leaks
- Reproducibility: Sets foundation for fixed random seeds (random_state=42, ADR-004)
- Error Handling: Implements tenacity retry decorator for all API calls
- Logging: Establishes emoji-prefixed logging pattern (📊, ✅, ⚠️, ❌)

**Dependency on Architecture Decisions:**
- **ADR-001**: Uses Cookiecutter Data Science template (Story 1.1 runs `ccds` command)
- **ADR-002**: Prepares for 768-dimensional embeddings (Gemini API configured for gemini-embedding-001)
- **ADR-003**: Configures Gemini Batch API support (50% cost savings, $0.075/M tokens)
- **ADR-004**: Configuration includes random_state=42 for reproducibility
- **ADR-006**: Separates config.yaml (committed) from .env (ignored) for security

## Detailed Design

### Services and Modules

Epic 1 establishes the following core modules and services:

| Module/Service | Location | Responsibilities | Inputs | Outputs | Owner Story |
|---------------|----------|------------------|--------|---------|-------------|
| **Config** | `src/config.py` | Load and validate configuration from config.yaml and .env; provide typed access to all parameters | config.yaml, .env | Config object with typed accessors | Story 1.2 |
| **Paths** | `src/config.py` | Define all project paths consistently; ensure directories exist | Project root path | Paths object with all directory paths | Story 1.2 |
| **DatasetLoader** | `src/data/load_dataset.py` | Load AG News from Hugging Face; validate structure; cache locally | Dataset name, split | Train/test datasets with 4 categories | Story 1.3 |
| **EmbeddingService** | `src/features/embedding_service.py` | Interface with Gemini Embedding API; handle authentication, batch processing, retry logic | API key, documents list, batch size | Embeddings array (n×768 float32) | Story 1.4 |
| **EmbeddingCache** | `src/features/embedding_cache.py` | Manage embedding storage/retrieval; prevent redundant API calls | Cache directory, embeddings array | Load/save operations | Story 1.4 |
| **Logger** | `src/utils/logger.py` | Unified logging setup with emoji prefixes; configure output format | Module name, log level | Logger instance | Story 1.1 |
| **Reproducibility** | `src/utils/reproducibility.py` | Set random seeds globally for numpy, random, sklearn | Random seed value | None (side effect: deterministic behavior) | Story 1.2 |

**Module Dependencies:**
```
Config ← EmbeddingService ← EmbeddingCache
Config ← DatasetLoader
Logger ← All modules
Reproducibility ← All scripts
```

**Code Organization Pattern:**
- **Data Layer**: `src/data/` - Dataset loading and preprocessing
- **Feature Layer**: `src/features/` - Embedding generation and caching
- **Utilities Layer**: `src/utils/` - Cross-cutting concerns (logging, reproducibility, helpers)
- **Configuration Layer**: `src/config.py` - Centralized configuration and path management

### Data Models and Contracts

**Configuration Schema (config.yaml):**
```yaml
# Dataset Configuration
dataset:
  name: str = "ag_news"           # Hugging Face dataset identifier
  categories: int = 4              # Expected number of categories
  sample_size: Optional[int] = None  # null = full dataset, int = sample size

# Clustering Configuration (used in Epic 2)
clustering:
  algorithm: str = "kmeans"
  n_clusters: int = 4
  random_state: int = 42
  max_iter: int = 300
  init: str = "k-means++"

# Embedding Configuration
embedding:
  model: str = "gemini-embedding-001"
  batch_size: int = 100
  cache_dir: str = "data/embeddings"
  output_dimensionality: int = 768

# Classification Configuration (used in Epic 3)
classification:
  method: str = "cosine_similarity"
  threshold: float = 0.7

# Metrics Configuration (used in Epic 4)
metrics:
  cost_per_1M_tokens_under_200k: float = 3.0
  cost_per_1M_tokens_over_200k: float = 6.0
  target_cost_reduction: float = 0.90
```

**Environment Variables Schema (.env):**
```bash
# Required
GEMINI_API_KEY=<your-api-key-here>

# Optional (defaults provided)
LOG_LEVEL=INFO
CACHE_DIR=data/embeddings
```

**Dataset Data Model:**
```python
# AG News Dataset Structure (from Hugging Face)
AGNewsDataset = {
    "train": {
        "text": List[str],        # Combined title + description
        "label": List[int],       # 0=World, 1=Sports, 2=Business, 3=Sci/Tech
    },
    "test": {
        "text": List[str],
        "label": List[int],
    }
}

# Expected sizes
train_size: int = 120_000
test_size: int = 7_600
num_categories: int = 4
category_names: List[str] = ["World", "Sports", "Business", "Sci/Tech"]
```

**Embedding Data Model:**
```python
# Embedding Array
Embedding = np.ndarray  # Shape: (n_documents, 768), dtype: float32

# Embedding Metadata (saved alongside .npy files)
EmbeddingMetadata = {
    "model": str,                # "gemini-embedding-001"
    "dimensions": int,           # 768
    "num_documents": int,        # Number of embedded documents
    "timestamp": str,            # ISO 8601 timestamp
    "dataset": str,              # "ag_news"
    "split": str,                # "train" or "test"
    "api_calls": int,            # Number of API calls made
    "estimated_cost": float,     # USD
}
```

**Path Definitions:**
```python
class Paths:
    project_root: Path
    data: Path
    data_raw: Path = data / "raw"
    data_embeddings: Path = data / "embeddings"
    data_interim: Path = data / "interim"
    data_processed: Path = data / "processed"
    models: Path
    notebooks: Path
    reports: Path
    reports_figures: Path = reports / "figures"
    results: Path
    src: Path
```

### APIs and Interfaces

**Config API:**
```python
class Config:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file and environment variables."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "clustering.n_clusters")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> n_clusters = config.get("clustering.n_clusters")  # Returns 4
        """
        ...

    def validate(self) -> bool:
        """Validate configuration schema and required fields."""
        ...

    @property
    def gemini_api_key(self) -> str:
        """Get Gemini API key from environment variable."""
        ...
```

**DatasetLoader API:**
```python
class DatasetLoader:
    def __init__(self, config: Config):
        """Initialize dataset loader with configuration."""
        ...

    def load_ag_news(self) -> Tuple[Dataset, Dataset]:
        """
        Load AG News dataset from Hugging Face.

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            DatasetLoadError: If dataset cannot be loaded or validated
        """
        ...

    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate dataset structure and content.

        Checks:
        - Contains expected fields (text, label)
        - Has expected number of categories
        - No missing values
        - Label range is valid [0-3]
        """
        ...

    def get_category_distribution(self, dataset: Dataset) -> Dict[int, int]:
        """Get document count per category."""
        ...
```

**EmbeddingService API:**
```python
class EmbeddingService:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        """
        Initialize Gemini Embedding API client.

        Args:
            api_key: Gemini API key from environment
            model: Embedding model name

        Raises:
            AuthenticationError: If API key is invalid
        """
        ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single document.

        Args:
            text: Document text

        Returns:
            Embedding vector of shape (768,), dtype float32
        """
        ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
    def generate_batch(
        self,
        documents: List[str],
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents using Batch API.

        Args:
            documents: List of document texts
            batch_size: Number of documents per API call

        Returns:
            Embeddings array of shape (n_documents, 768), dtype float32

        Note:
            Uses Gemini Batch API for 50% cost reduction
        """
        ...

    def test_connection(self) -> bool:
        """Test API authentication with a simple embedding call."""
        ...
```

**EmbeddingCache API:**
```python
class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        """Initialize embedding cache manager."""
        ...

    def save(
        self,
        embeddings: np.ndarray,
        split: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save embeddings to cache with metadata.

        Args:
            embeddings: Array of shape (n_documents, 768)
            split: "train" or "test"
            metadata: Embedding generation metadata

        Returns:
            Path to saved .npy file
        """
        ...

    def load(self, split: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings from cache.

        Args:
            split: "train" or "test"

        Returns:
            Tuple of (embeddings array, metadata dict)

        Raises:
            CacheNotFoundError: If cache file doesn't exist
        """
        ...

    def exists(self, split: str) -> bool:
        """Check if cache exists for given split."""
        ...

    def clear(self, split: Optional[str] = None) -> None:
        """Clear cache (all or specific split)."""
        ...
```

### Workflows and Sequencing

**Story 1.1: Project Initialization Workflow**
```
1. User runs: ccds (Cookiecutter Data Science CLI)
2. Template prompts for configuration:
   - Project name: "Context-Aware Multi-Agent System"
   - Author: "Jack YUAN"
   - Python version: 3.10
   - Environment: virtualenv
   - Dependencies: requirements.txt
   - Packages: basic (numpy, pandas, scikit-learn, matplotlib, jupyter)
3. Cookiecutter generates complete project structure
4. User creates virtual environment:
   python3.10 -m venv venv
   source venv/bin/activate
5. User installs dependencies:
   pip install -r requirements.txt
6. Structure validation:
   - Verify all directories created
   - Verify __init__.py files in src modules
   - Verify .gitignore includes .env, data/, *.pyc
```

**Story 1.2: Configuration Setup Workflow**
```
1. Create config.yaml in project root with all parameters
2. Create .env.example template:
   GEMINI_API_KEY=your-gemini-api-key
3. User copies .env.example → .env and adds real API key
4. Implement Config class:
   a. Load YAML using PyYAML
   b. Load environment variables using python-dotenv
   c. Merge configurations (env vars override YAML)
   d. Validate required fields
   e. Provide typed accessors with dot notation
5. Implement Paths class:
   a. Define all directory paths
   b. Create directories if missing (mkdir -p behavior)
6. Implement set_seed() utility:
   a. Set numpy.random.seed()
   b. Set random.seed()
   c. Set sklearn random_state via config
7. Test configuration loading:
   python -c "from config import Config; print(Config().get('clustering.n_clusters'))"
```

**Story 1.3: Dataset Loading Workflow**
```
1. Implement DatasetLoader class
2. Load AG News from Hugging Face:
   from datasets import load_dataset
   dataset = load_dataset("ag_news")
3. Validate dataset structure:
   a. Check fields: text, label
   b. Verify 4 categories (labels 0-3)
   c. Check sizes: train ~120K, test ~7.6K
   d. Log category distribution
4. Extract and preprocess text:
   a. Combine title + description fields
   b. Strip whitespace
   c. Validate no missing values
5. Cache dataset locally:
   a. Hugging Face auto-caches to ~/.cache/huggingface/
   b. Subsequent loads are instant (offline-capable)
6. Log dataset statistics:
   - Total samples per split
   - Category distribution
   - Sample text examples
```

**Story 1.4: Gemini API Integration Workflow**
```
1. Implement EmbeddingService class
2. Initialize Gemini client:
   import google.generativeai as genai
   genai.configure(api_key=config.gemini_api_key)
3. Test authentication:
   a. Generate test embedding for "Hello world"
   b. Verify response shape (768,)
   c. Catch authentication errors with helpful messages
4. Implement retry logic using tenacity:
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
5. Implement batch processing:
   a. Split documents into batches of 100
   b. Call Batch API for each batch
   c. Log progress: "Processing batch X/Y"
   d. Concatenate results
6. Implement EmbeddingCache:
   a. Save embeddings as .npy (float32)
   b. Save metadata as JSON
   c. Load cached embeddings on subsequent runs
7. Handle errors:
   - Network errors → retry with exponential backoff
   - Rate limiting → automatic backoff
   - API errors → log and raise with context
8. Track API usage:
   a. Count API calls
   b. Estimate tokens consumed
   c. Calculate estimated cost
   d. Log cost information
```

**End-to-End Epic 1 Sequence:**
```
Developer Workflow:
1. Run ccds to initialize project → Project structure created
2. Activate virtual environment → venv active
3. Install dependencies → All packages installed
4. Create config.yaml → Configuration defined
5. Copy .env.example → .env → API key secured
6. Run python -c "from config import Config; Config().validate()" → Config validated
7. Run dataset loader script → AG News downloaded and validated
8. Run embedding service test → API authentication verified
9. Epic 1 complete → Foundation ready for Epic 2

Data Flow:
config.yaml + .env
  → Config object
  → DatasetLoader
  → AG News dataset (120K train, 7.6K test)
  → EmbeddingService initialization (ready for Epic 2)
  → Cache infrastructure ready
```

## Non-Functional Requirements

### Performance

**Epic 1 Performance Targets:**

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Project initialization (ccds) | <5 minutes | One-time setup, includes user prompts |
| Dependency installation | <3 minutes | Standard pip install time for ~10 packages |
| Configuration loading | <100ms | YAML parsing is fast, minimal overhead |
| Dataset loading (first time) | <2 minutes | Hugging Face download from network |
| Dataset loading (cached) | <5 seconds | Local file read from cache |
| API authentication test | <2 seconds | Single embedding call with retry |

**Infrastructure Performance Characteristics:**
- **Configuration System**: In-memory caching of parsed YAML ensures subsequent `config.get()` calls are <1ms
- **Path Resolution**: All paths resolved at initialization, no repeated computation
- **Dataset Caching**: Hugging Face auto-caches to `~/.cache/huggingface/`, subsequent loads are disk I/O bound (~100MB/s)
- **API Retry Logic**: Exponential backoff (4s, 8s, 16s) prevents overwhelming API with failed requests

**Performance Non-Goals for Epic 1:**
- Embedding generation speed (addressed in Epic 2)
- Clustering performance (addressed in Epic 2)
- Classification latency (addressed in Epic 3)

**Acceptable Degradation:**
- Network latency for Hugging Face dataset download depends on connection speed (acceptable: 30s-5min)
- First API call may be slower due to SSL handshake (acceptable: up to 3s)

### Security

**API Key Protection (Critical):**
- ✅ GEMINI_API_KEY stored in `.env` file (NOT committed to git)
- ✅ `.gitignore` includes `.env` to prevent accidental commits
- ✅ `.env.example` template provided without real keys
- ✅ Config class loads from `os.getenv()` only, never logs API keys
- ✅ Error messages never expose API key values

**Secret Management Pattern:**
```python
# CORRECT: Load from environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in environment.\n"
        "Copy .env.example to .env and add your API key."
    )

# FORBIDDEN: Hardcoded key
# api_key = "AIzaSy..."  # NEVER DO THIS
```

**Git Security:**
- `.gitignore` includes:
  - `.env` (API keys)
  - `data/` (prevent committing large datasets)
  - `*.pyc`, `__pycache__/` (prevent bytecode commits)
  - `.ipynb_checkpoints/` (prevent notebook checkpoint commits)

**Dependency Security:**
- All dependencies pinned to specific versions in `requirements.txt`
- Use official package sources (PyPI) only
- No third-party repositories or untrusted sources

**Data Privacy:**
- AG News is public dataset (no PII)
- No user data collection
- Embeddings cached locally (never transmitted except to Gemini API)

**Authentication Security:**
- API authentication test call uses minimal data ("Hello world")
- Failed auth provides helpful error without exposing credentials
- Retry logic prevents brute-force API key testing

### Reliability/Availability

**Error Handling Coverage:**

| Error Category | Handling Strategy | User Experience |
|----------------|-------------------|-----------------|
| Missing API key | Raise ValueError with setup instructions | Clear error message with next steps |
| Invalid API key | Catch Gemini AuthError, provide diagnostic message | "Invalid API key. Check .env file." |
| Network failure | Retry up to 3 times with exponential backoff | Automatic recovery, log attempts |
| Rate limiting | Tenacity auto-retry with backoff | Transparent to user, logged |
| Missing dataset | Let Hugging Face auto-download | Progress bar shown to user |
| Invalid config | Validate schema, raise with specific field errors | "clustering.n_clusters must be int" |
| File system errors | Check directory permissions, create if missing | "Cannot write to data/. Check permissions." |

**Retry Logic Pattern:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((NetworkError, RateLimitError))
)
def api_call_with_retry():
    # API call implementation
    pass
```

**Graceful Degradation:**
- If embedding cache missing → regenerate (log warning, not error)
- If optional visualization fails → continue with metrics (log warning)
- If log directory unwritable → fall back to console logging

**Validation Checkpoints:**
1. **Config Validation**: Check all required fields before any operations
2. **Dataset Validation**: Verify structure immediately after loading
3. **API Validation**: Test authentication before bulk operations
4. **Path Validation**: Ensure directories exist/creatable at startup

**Availability Targets:**
- **Configuration System**: 100% (no external dependencies)
- **Dataset Loading**: 99.9% (depends on Hugging Face availability)
- **API Integration**: 99.5% (depends on Gemini API uptime)

**Recovery Procedures:**
- **Config Load Fails**: Check config.yaml syntax with PyYAML error message
- **Dataset Load Fails**: Clear Hugging Face cache (`~/.cache/huggingface/`), retry
- **API Auth Fails**: Verify GEMINI_API_KEY in .env, test at https://aistudio.google.com

### Observability

**Logging Strategy:**

Epic 1 establishes unified logging with emoji prefixes for quick visual parsing:

```python
logger.info("📊 Loading AG News dataset...")
logger.info("✅ Dataset loaded successfully: 120,000 train, 7,600 test")
logger.warning("⚠️ Using cached dataset from ~/.cache/huggingface/")
logger.error("❌ API authentication failed: Invalid API key")
```

**Log Levels:**
- **DEBUG**: Detailed traces (config values, API request/response details)
- **INFO**: Normal operations (dataset loading, API calls, progress)
- **WARNING**: Recoverable issues (cache miss, retry attempts)
- **ERROR**: Operation failures (auth errors, validation failures)

**Logged Events for Epic 1:**

| Event | Level | Example Message |
|-------|-------|-----------------|
| Project initialization start | INFO | "📊 Initializing project structure..." |
| Config loaded | INFO | "✅ Configuration loaded from config.yaml" |
| Config validation passed | INFO | "✅ Configuration validated: 4 clusters, 768 dimensions" |
| Dataset download start | INFO | "📊 Downloading AG News dataset..." |
| Dataset cached | WARNING | "⚠️ Using cached dataset from ~/.cache/" |
| Dataset validated | INFO | "✅ Dataset validated: 4 categories, 120K samples" |
| API auth test start | INFO | "📊 Testing Gemini API authentication..." |
| API auth success | INFO | "✅ API authentication successful" |
| API retry attempt | WARNING | "⚠️ API call failed, retrying (attempt 2/3)..." |
| API auth failure | ERROR | "❌ API authentication failed: Invalid key" |

**Metrics Tracked (logged, not yet persisted in Epic 1):**
- Configuration load time
- Dataset download time (if not cached)
- Dataset validation time
- API authentication test duration
- Number of retry attempts per API call

**Diagnostic Information:**
- Python version, OS, package versions logged at startup
- Config values logged (with API key redacted)
- Dataset statistics logged (sample counts, category distribution)
- File paths logged for debugging (config.yaml, .env, data directories)

**Structured Logging Format:**
```
2025-11-09 12:00:00 | INFO | config.py:45 | ✅ Configuration loaded from config.yaml
2025-11-09 12:00:01 | INFO | load_dataset.py:78 | 📊 Loading AG News dataset...
2025-11-09 12:00:05 | INFO | load_dataset.py:92 | ✅ Dataset loaded: 120,000 train, 7,600 test
```

**Observability for Debugging:**
- All file paths logged (absolute paths for clarity)
- Configuration values logged (API key masked as `GEMINI_API_KEY=***`)
- Exception stack traces logged for errors
- API call details logged (endpoint, payload size, response time)

**Future Observability (Epic 2+):**
- JSON-formatted metrics export
- API usage tracking (calls, tokens, cost)
- Performance profiling (embedding generation time, clustering duration)
- Visualization of metrics over time

## Dependencies and Integrations

### Python Dependencies (requirements.txt)

Epic 1 establishes the complete dependency stack for the project. All dependencies will be specified in `requirements.txt` with exact versions for reproducibility:

```txt
# Core ML and Data Science Stack
google-genai>=0.3.0              # Gemini Embedding API SDK
scikit-learn>=1.7.2              # K-Means clustering, PCA, metrics
numpy>=1.24.0                    # Array operations, efficient storage
pandas>=2.0.0                    # Data manipulation, analysis
datasets>=2.14.0                 # Hugging Face datasets (AG News)

# Visualization
matplotlib>=3.7.0                # Core plotting library
seaborn>=0.12.0                  # Statistical visualizations

# Configuration and Environment
PyYAML>=6.0                      # config.yaml parsing
python-dotenv>=1.0.0             # .env file management

# Reliability and Error Handling
tenacity>=8.0.0                  # Retry decorator for API calls

# Development Tools
pytest>=7.4.0                    # Unit testing framework
ruff>=0.1.0                      # Python linter and formatter
jupyter>=1.0.0                   # Interactive notebooks for analysis
```

**Dependency Constraints:**
- Python 3.10+ required (compatibility with scikit-learn 1.7.2+)
- All packages from official PyPI only (no third-party repos)
- Exact versions will be pinned after initial setup (e.g., `google-genai==0.3.2`)

**Installation Verification:**
```bash
pip install -r requirements.txt
python -c "import google.generativeai; import sklearn; import numpy; print('✅ All dependencies installed')"
```

### External Service Integrations

**1. Google Gemini Embedding API**

| Aspect | Details |
|--------|---------|
| **Purpose** | Generate 768-dimensional semantic embeddings for documents and queries |
| **Endpoint** | `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent` |
| **Authentication** | API key via `GEMINI_API_KEY` environment variable |
| **SDK** | `google-genai` Python package (v0.3.0+) |
| **Model** | `gemini-embedding-001` (768 dimensions, multilingual) |
| **Batch API** | Used for bulk embedding generation ($0.075/M tokens vs $0.15/M standard) |
| **Rate Limits** | Handled automatically by SDK + tenacity retry |
| **Error Handling** | Exponential backoff retry (max 3 attempts, 4s-16s wait) |
| **Cost Tracking** | Log all API calls, estimate token consumption, calculate cost |
| **Affected Stories** | Story 1.4 (API integration), Epic 2 (embedding generation) |

**Integration Pattern:**
```python
import google.generativeai as genai

# Configure with API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Single embedding
response = genai.embed_content(
    model="gemini-embedding-001",
    content="document text"
)
embedding = np.array(response['embedding'], dtype=np.float32)

# Batch embedding (cost-optimized)
batch_response = genai.embed_content_batch(
    model="gemini-embedding-001",
    requests=[{"content": text} for text in documents]
)
embeddings = np.array([r['embedding'] for r in batch_response], dtype=np.float32)
```

**2. Hugging Face Datasets**

| Aspect | Details |
|--------|---------|
| **Purpose** | Load AG News dataset (120K train, 7.6K test samples) |
| **Endpoint** | Hugging Face Hub dataset repository |
| **Authentication** | None required (public dataset) |
| **SDK** | `datasets` Python package (v2.14+) |
| **Dataset** | `ag_news` (4 categories: World, Sports, Business, Sci/Tech) |
| **Caching** | Auto-caches to `~/.cache/huggingface/datasets/` |
| **Offline Support** | Works offline after initial download |
| **Error Handling** | Automatic retry by datasets library |
| **Affected Stories** | Story 1.3 (dataset loading) |

**Integration Pattern:**
```python
from datasets import load_dataset

# Load dataset (downloads on first run, cached thereafter)
dataset = load_dataset("ag_news")

# Access splits
train_data = dataset["train"]  # 120,000 samples
test_data = dataset["test"]    # 7,600 samples

# Access fields
texts = train_data["text"]     # List of document texts
labels = train_data["label"]   # List of category labels (0-3)
```

### Internal Module Integration Points

**Config → All Modules:**
- All modules depend on Config for parameters (no hardcoded values allowed)
- Config loaded once at script startup, cached in memory
- Access pattern: `config.get("section.key")`

**Paths → All Modules:**
- All file I/O uses Paths for consistent directory access
- Paths created automatically if missing
- Access pattern: `paths.data_embeddings / "train_embeddings.npy"`

**Logger → All Modules:**
- All modules use centralized logger for consistency
- Emoji-prefixed messages for quick visual parsing
- Setup pattern: `logger = setup_logger(__name__)`

**Integration Dependencies Graph:**
```
External Services:
  Google Gemini API ← EmbeddingService
  Hugging Face Hub ← DatasetLoader

Configuration Layer:
  config.yaml + .env → Config → {EmbeddingService, DatasetLoader}
  Config → Paths → {All file I/O operations}

Utilities Layer:
  Logger ← All modules (logging)
  Reproducibility ← All scripts (random seed)

Data Flow:
  Hugging Face → DatasetLoader → AG News Dataset
  Config → EmbeddingService → Gemini API (ready for Epic 2)
  All modules → Logger → Console/File output
```

### Dependency Manifest Scanning

Epic 1 creates the initial `requirements.txt` file in the project root. No existing dependency manifests need scanning as this is project initialization.

**Post-Epic 1 Dependency Files:**
- `requirements.txt` - Python package dependencies
- `.python-version` (optional) - Specifies Python 3.10
- `pyproject.toml` (from Cookiecutter template) - Modern Python project configuration

**Dependency Validation:**
```bash
# Verify all dependencies installable
pip install --dry-run -r requirements.txt

# Check for security vulnerabilities
pip-audit -r requirements.txt  # Optional, requires pip-audit package
```

## Acceptance Criteria (Authoritative)

These criteria define completion for Epic 1. All must pass before moving to Epic 2.

### AC-1: Project Structure Initialized (Story 1.1)
**Given** I run the Cookiecutter Data Science initialization
**When** the template setup completes
**Then** the following structure exists:
- ✅ Project root contains: `data/`, `src/`, `notebooks/`, `reports/`, `results/`, `models/`
- ✅ `src/context_aware_multi_agent_system/` module exists with `__init__.py`
- ✅ Submodules exist: `src/.../data/`, `src/.../features/`, `src/.../models/`, `src/.../evaluation/`, `src/.../visualization/`, `src/.../utils/`
- ✅ `.gitignore` includes: `.env`, `data/`, `*.pyc`, `__pycache__/`, `.ipynb_checkpoints/`
- ✅ `requirements.txt` exists with all specified dependencies
- ✅ `README.md` exists with project overview

### AC-2: Virtual Environment Functional (Story 1.1)
**Given** the project structure is initialized
**When** I create and activate virtual environment
**Then**:
- ✅ `python3.10 -m venv venv` succeeds without errors
- ✅ Virtual environment activates successfully
- ✅ `pip install -r requirements.txt` completes without errors
- ✅ All imports succeed: `import google.generativeai, sklearn, numpy, pandas, datasets, matplotlib, seaborn, yaml, dotenv, tenacity`

### AC-3: Configuration System Functional (Story 1.2)
**Given** I have created `config.yaml` and `.env` files
**When** I load configuration
**Then**:
- ✅ `Config()` initializes without errors
- ✅ `config.get("clustering.n_clusters")` returns `4`
- ✅ `config.get("embedding.model")` returns `"gemini-embedding-001"`
- ✅ `config.get("embedding.output_dimensionality")` returns `768`
- ✅ `config.gemini_api_key` returns API key from `.env` (or raises clear error if missing)
- ✅ `config.validate()` returns `True` for valid config, raises informative error for invalid config
- ✅ All sections accessible: `dataset`, `clustering`, `embedding`, `classification`, `metrics`

### AC-4: Paths System Functional (Story 1.2)
**Given** Configuration is loaded
**When** I initialize Paths
**Then**:
- ✅ All path attributes exist: `data`, `data_embeddings`, `data_processed`, `models`, `reports`, `results`
- ✅ Directories are created if missing (mkdir -p behavior)
- ✅ Paths are absolute (not relative)
- ✅ Path operations use `pathlib.Path` (cross-platform compatible)

### AC-5: Reproducibility Utility Functional (Story 1.2)
**Given** I call `set_seed(42)`
**When** I run random operations
**Then**:
- ✅ `numpy.random.rand()` produces identical results across runs
- ✅ `random.random()` produces identical results across runs
- ✅ K-Means with `random_state=42` produces identical clusters (tested in Epic 2)

### AC-6: AG News Dataset Loaded (Story 1.3)
**Given** I call `DatasetLoader().load_ag_news()`
**When** the loading completes
**Then**:
- ✅ Returns tuple of (train_dataset, test_dataset)
- ✅ Train dataset has ~120,000 samples (exact: 120,000)
- ✅ Test dataset has ~7,600 samples (exact: 7,600)
- ✅ Dataset has fields: `text`, `label`
- ✅ Labels are in range [0-3] (4 categories)
- ✅ No missing values in `text` or `label` fields
- ✅ Category distribution is logged

### AC-7: Dataset Validated (Story 1.3)
**Given** Dataset is loaded
**When** I call `validate_dataset(dataset)`
**Then**:
- ✅ Returns `True` for valid AG News dataset
- ✅ Validates expected fields present
- ✅ Validates 4 categories exist
- ✅ Validates no missing values
- ✅ Validates label range [0-3]
- ✅ Raises `DatasetLoadError` with clear message for invalid dataset

### AC-8: Dataset Cached Locally (Story 1.3)
**Given** Dataset loaded once
**When** I load dataset again
**Then**:
- ✅ Loading completes in <5 seconds (uses cache)
- ✅ Cache location: `~/.cache/huggingface/datasets/ag_news/`
- ✅ No network calls made (works offline)
- ✅ Warning logged: "⚠️ Using cached dataset"

### AC-9: Gemini API Authentication Successful (Story 1.4)
**Given** Valid API key in `.env`
**When** I call `EmbeddingService(api_key).test_connection()`
**Then**:
- ✅ Returns `True` (authentication successful)
- ✅ Test embedding generated for "Hello world"
- ✅ Embedding shape is (768,)
- ✅ Embedding dtype is float32
- ✅ Logs: "✅ API authentication successful"

### AC-10: Gemini API Error Handling (Story 1.4)
**Given** Invalid or missing API key
**When** I call `EmbeddingService(api_key).test_connection()`
**Then**:
- ✅ Raises `AuthenticationError` with helpful message
- ✅ Error message includes: "GEMINI_API_KEY not found" or "Invalid API key"
- ✅ Error message includes next steps: "Copy .env.example to .env and add your API key"
- ✅ API key value never exposed in logs or error messages

### AC-11: Retry Logic Functional (Story 1.4)
**Given** Network error occurs during API call
**When** Retry logic activates
**Then**:
- ✅ Up to 3 retry attempts made
- ✅ Exponential backoff used: 4s, 8s, 16s (approximately)
- ✅ Each retry attempt logged: "⚠️ API call failed, retrying (attempt X/3)..."
- ✅ Success on retry logged: "✅ API call successful after X attempts"
- ✅ Failure after 3 attempts raises exception with context

### AC-12: Embedding Cache Functional (Story 1.4)
**Given** Embeddings generated
**When** I call `EmbeddingCache().save(embeddings, "train", metadata)`
**Then**:
- ✅ Embeddings saved to `data/embeddings/train_embeddings.npy`
- ✅ Metadata saved to `data/embeddings/train_metadata.json`
- ✅ Saved embeddings are float32 dtype
- ✅ Metadata includes: model, dimensions, num_documents, timestamp, dataset, split, api_calls, estimated_cost

**And when** I call `EmbeddingCache().load("train")`
**Then**:
- ✅ Returns tuple of (embeddings, metadata)
- ✅ Loaded embeddings match original (np.allclose check)
- ✅ Loaded metadata matches saved metadata

### AC-13: Logging Functional (All Stories)
**Given** Any operation executes
**When** I check logs
**Then**:
- ✅ Emoji prefixes used: 📊 (loading), ✅ (success), ⚠️ (warning), ❌ (error)
- ✅ Log format: `YYYY-MM-DD HH:MM:SS | LEVEL | module.py:line | message`
- ✅ All file paths logged as absolute paths
- ✅ API key never logged (masked as `GEMINI_API_KEY=***`)
- ✅ Exception stack traces included for errors

### AC-14: Epic 1 Complete (All Stories)
**Given** All above acceptance criteria pass
**When** I verify Epic 1 completion
**Then**:
- ✅ Project structure matches architecture specification
- ✅ All dependencies install and import successfully
- ✅ Configuration system loads and validates config
- ✅ AG News dataset loads with expected sample counts
- ✅ Gemini API authentication succeeds
- ✅ All 4 stories (1.1, 1.2, 1.3, 1.4) completed and tested
- ✅ Foundation ready for Epic 2 (embedding generation and clustering)

## Traceability Mapping

This table maps acceptance criteria to technical components and test approaches:

| AC | Story | PRD Requirement | Architecture Component | Implementation | Test Approach |
|----|-------|-----------------|------------------------|----------------|---------------|
| AC-1 | 1.1 | FR-13 (Reproducibility) | Project structure (ADR-001) | Cookiecutter Data Science template | Manual: verify directory tree exists |
| AC-2 | 1.1 | NFR-7 (Compatibility) | Virtual environment, dependencies | virtualenv + requirements.txt | Automated: `pip install` + import tests |
| AC-3 | 1.2 | FR-13, NFR-4 (Config) | Config class | config.py with PyYAML + dotenv | Unit test: config.get() for all sections |
| AC-4 | 1.2 | Architecture patterns | Paths class | config.py Paths with pathlib | Unit test: path existence, absoluteness |
| AC-5 | 1.2 | NFR-4 (Reproducibility) | set_seed() utility | utils/reproducibility.py | Unit test: identical random sequences |
| AC-6 | 1.3 | FR-1 (Dataset Loading) | DatasetLoader class | data/load_dataset.py | Integration test: load AG News, verify counts |
| AC-7 | 1.3 | FR-1 (Dataset Validation) | DatasetLoader.validate_dataset() | data/load_dataset.py | Unit test: valid/invalid dataset cases |
| AC-8 | 1.3 | NFR-1 (Performance) | Hugging Face caching | datasets library auto-cache | Integration test: load time <5s on 2nd load |
| AC-9 | 1.4 | FR-2 (Embedding API) | EmbeddingService.test_connection() | features/embedding_service.py | Integration test: API call, verify shape |
| AC-10 | 1.4 | NFR-8 (Security), NFR-3 (Reliability) | EmbeddingService auth error handling | features/embedding_service.py | Unit test: missing/invalid key scenarios |
| AC-11 | 1.4 | NFR-3 (Reliability) | @retry decorator | tenacity library | Unit test: mock failures, verify retry attempts |
| AC-12 | 1.4 | FR-2 (Embedding caching) | EmbeddingCache class | features/embedding_cache.py | Unit test: save/load roundtrip |
| AC-13 | All | NFR-4 (Observability) | Logger utility | utils/logger.py | Integration test: capture logs, verify format |
| AC-14 | All | Epic 1 scope | Complete Epic 1 | All modules | Integration test: full workflow |

**PRD → Epic 1 Coverage:**
- **FR-1**: Dataset Loading → AC-6, AC-7, AC-8
- **FR-2**: Embedding Generation (API setup only) → AC-9, AC-10, AC-11, AC-12
- **FR-13**: Documentation & Reproducibility → AC-1, AC-3, AC-5, AC-13
- **NFR-1**: Performance → AC-8
- **NFR-3**: Reliability → AC-10, AC-11
- **NFR-4**: Reproducibility → AC-3, AC-5
- **NFR-5**: Observability → AC-13
- **NFR-7**: Compatibility → AC-2
- **NFR-8**: Security → AC-10

## Risks, Assumptions, Open Questions

### Risks

| Risk ID | Risk Description | Likelihood | Impact | Mitigation Strategy | Owner |
|---------|------------------|------------|--------|---------------------|-------|
| R1-1 | Gemini API key invalid or expired | Medium | High | Provide clear error messages with setup instructions; test authentication at startup | Story 1.4 |
| R1-2 | Hugging Face dataset download fails (network issues) | Medium | Medium | Retry logic in datasets library; offline mode after first download | Story 1.3 |
| R1-3 | Dependency version conflicts during pip install | Low | Medium | Pin exact versions in requirements.txt; test in clean virtualenv | Story 1.1 |
| R1-4 | Python 3.10 not available on user's system | Medium | High | Document Python version requirement clearly; provide installation instructions | Story 1.1 |
| R1-5 | API rate limiting during bulk operations | Low | Low | Handled by tenacity retry + exponential backoff; batch API reduces calls | Story 1.4 |
| R1-6 | Config.yaml syntax errors cause cryptic failures | Medium | Low | Validate schema with PyYAML; provide clear error messages with line numbers | Story 1.2 |

### Assumptions

| Assumption ID | Assumption | Validation Method | Impact if Invalid |
|---------------|------------|-------------------|-------------------|
| A1-1 | User has Python 3.10+ installed or can install it | Check in README.md | Project cannot run; must document Python installation |
| A1-2 | User has internet connection for initial setup | Required for HF dataset + API key setup | Cannot download dataset or test API; provide offline instructions |
| A1-3 | Gemini API is available and stable (99.5% uptime) | Monitor API status during development | API failures handled by retry logic; acceptable degradation |
| A1-4 | AG News dataset structure unchanged on Hugging Face | Validate fields on load | Dataset validation will catch changes; fail fast with clear error |
| A1-5 | User has ~2GB disk space for dataset + embeddings | Document in README.md | Installation will fail; provide disk space requirements |
| A1-6 | Cookiecutter Data Science template v2 available | Test ccds command during implementation | Template defines structure; critical for Story 1.1 |

### Open Questions

| Question ID | Question | Potential Impact | Resolution Plan | Resolved? |
|-------------|----------|------------------|-----------------|-----------|
| Q1-1 | Should we support Python 3.9 or require 3.10+? | Broader compatibility vs simpler testing | Decision: Require 3.10+ (scikit-learn 1.7.2+ requirement) | ✅ Yes |
| Q1-2 | Should embedding cache be stored in project or user home? | Portability vs disk space | Decision: Project data/ (can .gitignore if large) | ✅ Yes |
| Q1-3 | What happens if API key has usage limits/quotas? | Embedding generation might fail mid-process | Implement cost tracking; warn user before bulk operations | ⬜ No (defer to Epic 2) |
| Q1-4 | Should we validate config schema strictly or loosely? | Strictness prevents errors vs flexibility | Decision: Strict validation (fail fast with clear errors) | ✅ Yes |
| Q1-5 | How to handle partial dataset downloads (interrupted)? | Corrupted cache could cause issues | Trust Hugging Face library handling; document cache clear procedure | ✅ Yes |

## Test Strategy Summary

### Unit Testing Strategy

**Modules to Unit Test:**
- `src/config.py`: Config loading, validation, get() method, API key retrieval
- `src/data/load_dataset.py`: Dataset validation logic (mock HF dataset)
- `src/features/embedding_service.py`: Retry logic (mock API failures)
- `src/features/embedding_cache.py`: Save/load roundtrip, cache existence checks
- `src/utils/reproducibility.py`: set_seed() determinism
- `src/utils/logger.py`: Log format, emoji prefixes

**Unit Test Framework:** pytest

**Test Coverage Goal:** >80% for Epic 1 modules (if time permits; optional for MVP)

**Example Unit Test:**
```python
def test_config_get_nested_key():
    """Test Config.get() with dot notation."""
    config = Config("tests/fixtures/config.yaml")
    assert config.get("clustering.n_clusters") == 4
    assert config.get("embedding.model") == "gemini-embedding-001"
```

### Integration Testing Strategy

**Integration Tests:**
1. **Full Dependency Installation**: Install all packages in fresh virtualenv, verify imports
2. **Dataset Loading**: Load actual AG News dataset, verify sample counts and structure
3. **API Authentication**: Real API call to Gemini (requires valid API key in test env)
4. **End-to-End Workflow**: Run all 4 stories sequentially, verify Epic 1 complete

**Test Environment:**
- Clean virtualenv with Python 3.10
- Temporary directory for project initialization
- Test `.env` file with valid API key (from secrets management)

**Example Integration Test:**
```python
def test_dataset_loading_integration():
    """Test full dataset loading workflow."""
    loader = DatasetLoader(config)
    train, test = loader.load_ag_news()

    assert len(train) == 120_000
    assert len(test) == 7_600
    assert loader.validate_dataset(train) == True
```

### Acceptance Test Strategy

**Acceptance Tests = Acceptance Criteria:**
- Each AC (AC-1 through AC-14) is tested explicitly
- Tests run in order (AC-1 → AC-2 → ... → AC-14)
- All ACs must pass for Epic 1 completion

**Test Execution:**
- Manual verification for AC-1 (project structure)
- Automated pytest for AC-2 through AC-13
- Manual checklist for AC-14 (epic completion)

**Test Data:**
- Real AG News dataset (public, no test data needed)
- Real Gemini API (requires valid key)
- Sample config.yaml (committed to repo)
- Sample .env.example (committed, no secrets)

### Edge Cases & Error Scenarios

**Error Scenarios to Test:**
1. Missing API key → Clear error message
2. Invalid API key → AuthenticationError with instructions
3. Network failure → Retry logic activates, succeeds or fails gracefully
4. Invalid config.yaml syntax → PyYAML error with line number
5. Missing config fields → Validation error with specific field name
6. Corrupted dataset cache → Clear cache, re-download
7. Insufficient disk space → OS error caught, logged

**Edge Cases:**
- Empty config value (e.g., `sample_size: null`)
- Very long API response time → Timeout handled
- Partial dataset download interrupted → Hugging Face library handles resume

### Test Automation

**Automated Tests (pytest):**
```bash
# Run all Epic 1 tests
pytest tests/epic1/ -v

# Run specific test suite
pytest tests/epic1/test_config.py -v

# Run with coverage
pytest tests/epic1/ --cov=src --cov-report=html
```

**Manual Tests:**
- Project structure verification (visual inspection)
- README.md clarity (read-through test)
- Epic 1 completion checklist (all ACs pass)

**CI/CD Integration (Optional):**
- GitHub Actions workflow to run pytest on every commit
- Automated dependency security scanning (pip-audit)
- Lint checking (ruff)
