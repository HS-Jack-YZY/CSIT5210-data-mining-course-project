# Architecture - Context-Aware Multi-Agent System

## Executive Summary

This architecture document defines a **Python-based machine learning system** that demonstrates how clustering and classification algorithms solve LLM cost optimization challenges. The system uses **K-Means clustering** to partition 120K news articles into 4 semantic groups, then routes queries using **cosine similarity classification** to specialized agents containing only relevant context—reducing API costs by 90%+.

**Architecture Approach:** Academic proof-of-concept combining proven data science methodologies (Cookiecutter Data Science structure) with modern ML tools (scikit-learn 1.7.2, Google Gemini Embeddings). The architecture prioritizes **reproducibility**, **clarity**, and **cost efficiency** over production scalability.

**Key Architectural Decisions:**
- **Starter Template:** Cookiecutter Data Science v2 (provides data/src/notebooks structure)
- **Embedding Service:** Google Gemini API with 768-dimensional embeddings and batch processing
- **Clustering:** scikit-learn K-Means with k-means++ initialization, random_state=42
- **Visualization:** matplotlib + seaborn for academic-quality figures (300 DPI)
- **Configuration:** YAML-based configuration with environment variable secrets
- **Cost Target:** <$10 total API costs, >90% cost reduction demonstration

---

## Project Initialization

**First Implementation Story: Project Setup**

This project uses the **Cookiecutter Data Science v2** template, which provides a proven structure for machine learning and data science projects.

### Initialization Command

```bash
# Install cookiecutter-data-science (one-time setup)
pipx install cookiecutter-data-science

# OR with pip
pip install cookiecutter-data-science

# Create project with recommended configuration
ccds
```

### Recommended Configuration Options

When running `ccds`, use the following configuration:

- **Project Name**: `Context-Aware Multi-Agent System`
- **Repo Name**: `context-aware-multi-agent-system` (auto-generated)
- **Module Name**: `context_aware_multi_agent_system` (auto-generated)
- **Author Name**: `Jack YUAN`
- **Description**: `Multi-Agent collaborative system for cost-efficient text classification using K-Means clustering and cosine similarity routing`
- **Python Version**: `3.10` (required for compatibility with scikit-learn 1.7.2+)
- **Environment Manager**: `virtualenv` (simple and widely compatible)
- **Dependency File**: `requirements.txt` (standard Python dependency specification)
- **Dataset Storage**: `none` (AG News dataset will be loaded from Hugging Face)
- **Pydata Packages**: `basic` (includes numpy, pandas, scikit-learn, jupyter, matplotlib - all required for this project)
- **Testing Framework**: `pytest` (for unit testing if time permits)
- **Linting and Formatting**: `ruff` (modern, fast Python linter and formatter)
- **Open Source License**: `MIT` (permissive license suitable for academic projects)
- **Documentation**: `mkdocs` (for project documentation)
- **Code Scaffold**: `Yes` (includes common data processing submodules)

### What the Starter Template Provides

The Cookiecutter Data Science template establishes these architectural decisions automatically:

| Category | Decision | Provided by Starter | Rationale |
|----------|----------|---------------------|-----------|
| **Project Structure** | Organized data/src/notebooks layout | ✅ Yes | Separates concerns (raw data, processed data, code, analysis) |
| **Python Version** | Python 3.10 | ✅ Yes | Compatible with all required libraries (scikit-learn 1.7.2+, google-genai) |
| **Environment Management** | virtualenv | ✅ Yes | Simple, reliable, widely supported |
| **Dependency Specification** | requirements.txt | ✅ Yes | Standard Python dependency management |
| **Code Quality** | Ruff (linting + formatting) | ✅ Yes | Fast, modern Python code quality tool |
| **Testing Framework** | pytest | ✅ Yes | Industry-standard Python testing framework |
| **Documentation** | mkdocs | ✅ Yes | Clean, markdown-based documentation |
| **Pydata Stack** | numpy, pandas, scikit-learn, matplotlib, jupyter | ✅ Yes | All ML/data science essentials included |
| **License** | MIT License | ✅ Yes | Open source, permissive, suitable for academic work |

---

## Decision Summary

All architectural decisions made for this project:

| Category | Decision | Version | Affects Epics | Rationale |
| -------- | -------- | ------- | ------------- | --------- |
| **Starter Template** | Cookiecutter Data Science v2 | Latest (2025) | All | Industry-standard ML project structure |
| **Python** | Python 3.10 | 3.10 | All | Compatible with all ML libraries |
| **Embedding Service** | Google Gemini Embedding API | google-genai latest | Epic 1 | State-of-art embeddings, batch API support |
| **Embedding Model** | gemini-embedding-001 | GA (2025-07) | Epic 1 | 768-dim, multilingual, $0.075/M tokens (batch) |
| **Embedding Dimension** | 768 | - | Epic 1, 2, 4 | Balance performance/storage, PRD requirement |
| **Dataset Loader** | Hugging Face datasets | ≥2.14.0 | Epic 1 | One-line AG News loading, auto-caching |
| **Data Processing** | pandas + datasets.map | pandas ≥2.0 | Epic 1 | Flexible analysis + efficient batch ops |
| **Embedding Storage** | numpy .npy files | numpy ≥1.24 | Epic 1 | Fast I/O, compact format |
| **Clustering Algorithm** | scikit-learn K-Means | scikit-learn ≥1.7.2 | Epic 2 | Standard implementation, proven reliability |
| **K-Means Config** | n_clusters=4, init='k-means++', random_state=42, max_iter=300 | - | Epic 2 | PRD requirements, reproducibility |
| **Cluster Evaluation** | Silhouette Score, Davies-Bouldin Index | sklearn.metrics | Epic 2, 6 | Standard cluster quality metrics |
| **Classification Method** | Cosine Similarity | sklearn.metrics.pairwise | Epic 4 | Fast, interpretable similarity measure |
| **Visualization** | matplotlib + seaborn | matplotlib ≥3.7, seaborn ≥0.12 | Epic 7 | Academic-quality plots, 300 DPI export |
| **Dimensionality Reduction** | PCA (2 components) | sklearn.decomposition | Epic 7 | Visualize 768D in 2D scatter plot |
| **Configuration Format** | YAML (config.yaml) | PyYAML ≥6.0 | All | Human-readable, supports comments |
| **Secret Management** | .env file | python-dotenv ≥1.0 | All | Secure API key storage |
| **Logging** | Python logging module | Built-in | All | Standard, emoji prefixes for clarity |
| **Error Handling** | tenacity retry decorator | tenacity ≥8.0 | All | Auto-retry with exponential backoff |
| **Code Style** | PEP 8 + Ruff | Ruff (from template) | All | Consistent formatting, fast linting |
| **Type Hints** | All function signatures | Built-in typing | All | Code clarity, IDE support |
| **Docstrings** | Google Style | - | All | Consistent documentation format |
| **Random Seed** | 42 (global set_seed) | - | Epic 2, 7 | Full reproducibility |
| **Cost Tracking** | JSON metrics export | Built-in json | Epic 5, 6 | Structured experiment results |

---

## Complete Project Structure

```
context-aware-multi-agent-system/
│
├── README.md                           # Project overview, installation, usage
├── LICENSE                             # MIT License
├── .gitignore                          # Python, data/, .env, etc.
├── .env                                # API keys (NOT committed to git)
├── .env.example                        # Environment variable template
├── config.yaml                         # Project configuration (committed)
├── requirements.txt                    # Exact dependency versions
├── pyproject.toml                      # Modern Python config
├── Makefile                            # Common commands
│
├── data/                               # Data directory (NOT committed)
│   ├── .gitkeep
│   ├── raw/                            # AG News (HF auto-cache)
│   ├── embeddings/                     # Embedding cache
│   │   ├── train_embeddings.npy       # (120K, 768) float32
│   │   ├── test_embeddings.npy        # (7.6K, 768) float32
│   │   └── metadata.json              # Embedding metadata
│   ├── interim/                        # Intermediate results
│   └── processed/                      # Final processed data
│       ├── cluster_assignments.csv     # Cluster labels
│       ├── centroids.npy               # (4, 768) centroids
│       ├── cluster_metadata.json       # Silhouette score, etc.
│       └── classification_results.csv  # Classification accuracy
│
├── models/                             # Trained models
│   └── kmeans_model.pkl                # Serialized KMeans (optional)
│
├── notebooks/                          # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_clustering_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── reports/                            # Generated reports
│   ├── experimental_report.md
│   └── figures/                        # Visualizations (300 DPI PNG)
│       ├── cluster_pca_visualization.png
│       ├── cost_comparison_chart.png
│       ├── confusion_matrix.png
│       ├── silhouette_analysis.png
│       └── classification_accuracy.png
│
├── results/                            # Experiment results (JSON)
│   ├── experiment_20251108_120000.json
│   └── metrics_summary.json
│
├── src/                                # Source code
│   └── context_aware_multi_agent_system/
│       ├── __init__.py
│       ├── config.py                   # Config and Paths classes
│       │
│       ├── data/                       # Data loading
│       │   ├── __init__.py
│       │   ├── load_dataset.py         # AG News loader
│       │   └── preprocess.py           # Data preprocessing
│       │
│       ├── features/                   # Feature engineering
│       │   ├── __init__.py
│       │   ├── embedding_service.py    # Gemini API wrapper
│       │   └── embedding_cache.py      # Cache management
│       │
│       ├── models/                     # Models
│       │   ├── __init__.py
│       │   ├── clustering.py           # K-Means clustering
│       │   ├── agent.py                # SpecializedAgent
│       │   └── router.py               # AgentRouter
│       │
│       ├── evaluation/                 # Metrics
│       │   ├── __init__.py
│       │   ├── clustering_metrics.py   # Silhouette, etc.
│       │   ├── classification_metrics.py  # Accuracy, confusion matrix
│       │   └── cost_calculator.py      # Cost estimation
│       │
│       ├── visualization/              # Plotting
│       │   ├── __init__.py
│       │   ├── cluster_plots.py        # PCA scatter plots
│       │   ├── cost_charts.py          # Cost comparison charts
│       │   └── metrics_plots.py        # Metric visualizations
│       │
│       └── utils/                      # Utilities
│           ├── __init__.py
│           ├── logger.py               # Logging setup
│           ├── reproducibility.py      # set_seed function
│           └── helpers.py              # General helpers
│
├── tests/                              # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_embedding_service.py
│   ├── test_clustering.py
│   ├── test_agent.py
│   └── test_router.py
│
├── scripts/                            # Executable scripts
│   ├── 01_generate_embeddings.py
│   ├── 02_train_clustering.py
│   ├── 03_evaluate_classification.py
│   ├── 04_baseline_comparison.py
│   └── 05_generate_report.py
│
└── docs/                               # Documentation (mkdocs)
    ├── index.md
    ├── installation.md
    ├── usage.md
    └── architecture.md
```

---

## Epic to Architecture Mapping

| Epic | Main Modules | Key Files | Architecture Support |
|------|-------------|-----------|---------------------|
| **Epic 1: Data Preparation & Embedding Generation** | `src/data/`, `src/features/` | `load_dataset.py`, `embedding_service.py`, `scripts/01_generate_embeddings.py` | Hugging Face datasets + Gemini API + caching |
| **Epic 2: K-Means Clustering** | `src/models/clustering.py` | `clustering.py`, `scripts/02_train_clustering.py` | scikit-learn K-Means, k-means++ init, random_state=42 |
| **Epic 3: Specialized Agents** | `src/models/agent.py` | `agent.py` | SpecializedAgent class, context reduction (1/K) |
| **Epic 4: Classification & Routing** | `src/models/router.py` | `router.py`, `scripts/03_evaluate_classification.py` | AgentRouter, cosine_similarity classification |
| **Epic 5: Baseline System** | `src/evaluation/cost_calculator.py` | `scripts/04_baseline_comparison.py` | Single-agent baseline, token counting |
| **Epic 6: Cost Metrics & Performance** | `src/evaluation/` | `clustering_metrics.py`, `classification_metrics.py`, `cost_calculator.py` | Silhouette, accuracy, cost calculation, JSON export |
| **Epic 7: Experimental Report & Visualization** | `src/visualization/`, `reports/` | `cluster_plots.py`, `cost_charts.py`, `scripts/05_generate_report.py` | matplotlib/seaborn, PCA, 300 DPI PNG |

---

## Technology Stack Details

### Core Technologies

**Programming Language:**
- Python 3.10 (compatible with all ML libraries)

**Machine Learning:**
- **scikit-learn 1.7.2+**: K-Means clustering, PCA, cosine similarity, metrics
- **numpy 1.24+**: Array operations, efficient storage (float32/int32)
- **pandas 2.0+**: Data manipulation, exploratory analysis

**Embedding Service:**
- **google-genai (latest GA)**: Official Gemini API SDK
- **Model**: gemini-embedding-001 (768 dimensions, multilingual)
- **Batch API**: 50% cost reduction ($0.075/M tokens)

**Data Pipeline:**
- **datasets 2.14+**: Hugging Face datasets library (AG News)
- **PyYAML 6.0+**: Configuration file parsing
- **python-dotenv 1.0+**: Environment variable management

**Visualization:**
- **matplotlib 3.7+**: Core plotting library
- **seaborn 0.12+**: Statistical visualizations
- **PCA**: sklearn.decomposition.PCA (768D → 2D)

**Development Tools:**
- **pytest**: Unit testing framework
- **ruff**: Fast Python linter and formatter (PEP 8)
- **tenacity 8.0+**: Retry decorator for API calls
- **mkdocs**: Documentation generator

### Integration Points

**External Integrations:**

1. **Google Gemini Embedding API**
   - **Entry Point**: `src/features/embedding_service.py`
   - **Authentication**: Environment variable `GEMINI_API_KEY`
   - **Error Handling**: tenacity retry (max 3 attempts, exponential backoff)
   - **Rate Limiting**: Batch API respects limits automatically
   - **Cost**: Batch API $0.075/M tokens, Standard $0.15/M tokens

2. **Hugging Face Datasets**
   - **Entry Point**: `src/data/load_dataset.py`
   - **Dataset**: AG News (120K train, 7.6K test)
   - **Caching**: Automatic to `~/.cache/huggingface/`
   - **Offline**: Works after initial download

**Internal API Contracts:**

```python
# Config API
config = Config('config.yaml')
n_clusters = config.get('clustering.n_clusters')  # Returns: 4
embedding_dim = config.get('embedding.output_dimensionality')  # Returns: 768

# EmbeddingService API
service = EmbeddingService(api_key=config.gemini_api_key)
embeddings = service.generate_batch(
    documents: List[str],
    batch_size: int = 100
) -> np.ndarray  # Shape: (n_documents, 768), dtype: float32

# Clustering API
kmeans = KMeansClustering(config.clustering_params)
labels, centroids = kmeans.fit_predict(
    embeddings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]
# Returns: labels (n_samples,) int32, centroids (4, 768) float32

# Agent API
agent = SpecializedAgent(cluster_id: int, documents: List[str])
response = agent.process_query(query: str) -> str

# Router API
router = AgentRouter(centroids: np.ndarray, agents: Dict[int, SpecializedAgent])
cluster_id, confidence = router.classify_query(
    query_embedding: np.ndarray
) -> Tuple[int, float]
selected_agent = router.route_query(
    query: str,
    query_embedding: np.ndarray
) -> SpecializedAgent
```

---

## Implementation Patterns

These patterns ensure all AI agents write compatible code:

### Naming Patterns

**Files:** snake_case
```
✅ embedding_service.py, cluster_plots.py, cost_calculator.py
❌ EmbeddingService.py, ClusterPlots.py
```

**Classes:** PascalCase
```python
✅ class EmbeddingService:
✅ class SpecializedAgent:
❌ class embedding_service:
```

**Functions/Methods:** snake_case
```python
✅ def generate_embeddings(documents: List[str]) -> np.ndarray:
✅ def classify_query(embedding: np.ndarray) -> int:
❌ def GenerateEmbeddings() or classifyQuery()
```

**Constants:** UPPER_SNAKE_CASE
```python
✅ MAX_RETRIES = 3
✅ DEFAULT_BATCH_SIZE = 100
❌ maxRetries = 3
```

### Structure Patterns

**Test Files:** `tests/test_*.py`
```
✅ tests/test_embedding_service.py
❌ src/.../test_embedding_service.py
```

**Configuration:** Root directory
```
✅ config.yaml, .env, .env.example (project root)
❌ src/config.yaml, config/config.yaml
```

**Data Organization:**
```
✅ data/embeddings/train_embeddings.npy
✅ data/embeddings/metadata.json
❌ data/train_embeddings.npy (mixed files)
```

### Format Patterns

**Data Types (MANDATORY):**
```python
✅ embeddings: np.ndarray = np.array(data, dtype=np.float32)  # Always float32
✅ labels: np.ndarray = np.array(data, dtype=np.int32)        # Always int32
✅ documents: List[str] = [doc.strip() for doc in raw_docs]  # Always List[str]
❌ embeddings = np.array(data, dtype=np.float64)  # Wrong dtype
```

**JSON Output (MANDATORY format):**
```python
# Standard experiment results structure
{
    "timestamp": "2025-11-08T12:00:00",
    "config": {...},
    "metrics": {
        "clustering": {...},
        "classification": {...},
        "cost": {...}
    }
}

# Always save with indent=2
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
```

**File Naming Timestamps:**
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"experiment_{timestamp}.json"
# Result: experiment_20251108_120000.json
```

### Communication Patterns

**Function Signatures (MANDATORY):**
```python
✅ ALWAYS include type hints and docstrings
def generate_embeddings(
    documents: List[str],
    model: str = "gemini-embedding-001",
    batch_size: int = 100
) -> np.ndarray:
    """
    Generate embeddings for documents using Gemini API.

    Args:
        documents: List of text documents
        model: Embedding model name
        batch_size: Batch size for API calls

    Returns:
        Embeddings array of shape (n_documents, 768)
    """
    ...
```

**Return Value Conventions:**
```python
# Clustering: return (labels, centroids)
def fit_clustering(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    return labels, centroids

# Classification: return (cluster_id, confidence)
def classify_query(embedding: np.ndarray) -> Tuple[int, float]:
    similarities = cosine_similarity(embedding, centroids)[0]
    cluster_id = int(similarities.argmax())
    confidence = float(similarities[cluster_id])
    return cluster_id, confidence
```

### Lifecycle Patterns

**Initialization Order (MANDATORY):**
```python
def main():
    # 1. Set random seed
    set_seed(42)

    # 2. Load configuration
    config = Config('config.yaml')
    paths = Paths()

    # 3. Setup logging
    logger = setup_logger(__name__)

    # 4. Validate configuration
    validate_config(config)

    # 5. Execute main logic
    ...

if __name__ == "__main__":
    main()
```

**Resource Management:**
```python
# ALWAYS use context managers
with open(path, 'w') as f:
    json.dump(data, f)
```

### Location Patterns

**Import Order (MANDATORY):**
```python
# 1. Standard library
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# 2. Third-party libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 3. Local modules
from config import Config, Paths
from utils.logger import setup_logger
```

**Configuration Access:**
```python
✅ config = Config()
✅ n_clusters = config.get('clustering.n_clusters')
❌ n_clusters = 4  # NO hardcoded values
```

### Consistency Patterns

**Error Messages:**
```python
# Consistent format with file path and next steps
if not cache_file.exists():
    raise FileNotFoundError(
        f"Embedding cache not found: {cache_file}\n"
        f"Run 'python scripts/01_generate_embeddings.py' first"
    )

logger.error(f"❌ Failed to load embeddings: {cache_file}")
```

**Progress Logging:**
```python
logger.info(f"📊 Processing batch {i+1}/{total_batches}...")
logger.info(f"✅ Completed: {processed}/{total} documents")
logger.warning(f"⚠️ Silhouette score {score:.3f} below target {target}")
logger.error(f"❌ API call failed: {error}")
```

---

## Cross-Cutting Concerns

These apply to **every Epic and Story**:

### Error Handling Strategy

**ALL external API calls MUST use retry:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def api_call_pattern(data):
    try:
        result = external_api.call(data)
        logger.info("✅ API call successful")
        return result
    except Exception as e:
        logger.error(f"❌ API call failed: {e}")
        raise
```

### Logging Approach

**Unified logger setup:**
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)

# Emoji prefixes for quick visual parsing
logger.info("📊 Loading AG News dataset...")
logger.info("✅ Dataset loaded successfully")
logger.warning("⚠️ Metric below threshold")
logger.error("❌ Operation failed")
```

### Date/Time Handling

**Consistent timestamp format:**
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Format: 20251108_120000
```

### Configuration Access

**NO hardcoded values allowed:**
```python
✅ config = Config()
✅ n_clusters = config.get('clustering.n_clusters')
✅ random_state = config.get('clustering.random_state')

❌ n_clusters = 4  # FORBIDDEN
❌ embedding_dim = 768  # FORBIDDEN
```

### Reproducibility

**Seed setting at start of every script:**
```python
from utils.reproducibility import set_seed
set_seed(42)  # Or: set_seed(config.get('clustering.random_state'))
```

---

## Data Architecture

### Data Models

**Embeddings:**
```python
Type: np.ndarray
Shape: (n_documents, 768)
Dtype: float32
Storage: .npy files
Example: data/embeddings/train_embeddings.npy
```

**Labels:**
```python
Type: np.ndarray
Shape: (n_documents,)
Dtype: int32
Values: 0-3 (cluster IDs)
Storage: .csv files
Example: data/processed/cluster_assignments.csv
```

**Centroids:**
```python
Type: np.ndarray
Shape: (4, 768)
Dtype: float32
Storage: .npy files
Example: data/processed/centroids.npy
```

**Documents:**
```python
Type: List[str]
Preprocessing: Strip whitespace
Source: Hugging Face datasets
```

### Data Relationships

```
AG News Dataset (120K documents, 4 categories)
    ↓
Embeddings (120K × 768 float32)
    ↓
K-Means Clustering
    ↓
Cluster Assignments (120K labels, 0-3)
    + Centroids (4 × 768)
    ↓
4 × SpecializedAgent (each ~30K documents)
    ↓
Query → Classification → Route to Agent
```

### Data Storage Strategy

**Caching Priority:**
1. Check cache exists
2. Load from cache if available
3. Generate and save to cache if missing

**Example:**
```python
cache_path = paths.data_embeddings / "train_embeddings.npy"
if cache_path.exists():
    embeddings = np.load(cache_path)
    logger.info(f"✅ Loaded from cache: {cache_path}")
else:
    embeddings = service.generate_batch(documents)
    np.save(cache_path, embeddings)
    logger.info(f"💾 Saved to cache: {cache_path}")
```

---

## Security Architecture

### API Key Management

**MANDATORY security rules:**

1. **NEVER hardcode API keys in source code**
2. **Use environment variables**
3. **Git ignore .env file**
4. **Provide .env.example template**

**Implementation:**
```python
# .env file (NOT committed to git)
GEMINI_API_KEY=your-actual-api-key-here

# .env.example file (committed as template)
GEMINI_API_KEY=your-gemini-api-key

# .gitignore (must include)
.env

# config.py
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment")
```

### Data Privacy

- AG News is public dataset (no privacy concerns)
- No user data collection
- Embeddings cached locally (not transmitted)
- API keys logged NEVER (logger setup excludes)

---

## Performance Considerations

### Optimization Strategies

**1. Embedding Generation:**
```python
# Use Batch API (50% cost reduction)
batch_size = 100
service.generate_batch(documents, batch_size=batch_size)
```

**2. Memory Management:**
```python
# Use memory mapping for large arrays
embeddings = np.load('file.npy', mmap_mode='r')  # Don't load to RAM
```

**3. Caching:**
```python
# Always check cache first
if cache_file.exists():
    return np.load(cache_file)
else:
    data = expensive_operation()
    np.save(cache_file, data)
    return data
```

**4. Parallel Processing:**
```python
# Batch API handles parallelization automatically
# No manual threading needed for API calls
```

### Performance Targets

- Query classification: <1 second
- K-Means clustering (120K docs): <5 minutes
- Batch embedding generation: <15 minutes total
- Silhouette Score calculation: <3 minutes

---

## Development Environment

### Prerequisites

**Required Software:**
- Python 3.10 or higher
- pip or pipx (for package installation)
- Git (for version control)
- Internet connection (for Gemini API, initial dataset download)

**Hardware Requirements:**
- RAM: 8GB minimum (16GB recommended)
- Disk: ~2GB for data + embeddings + results
- CPU: Any modern processor (no GPU required)

### Setup Commands

```bash
# 1. Clone repository
git clone <repository-url>
cd context-aware-multi-agent-system

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Verify setup
python -c "from config import Config; print('✅ Config loaded successfully')"

# 6. Run first script (generate embeddings)
python scripts/01_generate_embeddings.py
```

### Complete Dependency List

```txt
# requirements.txt (exact versions)
google-genai>=0.3.0
scikit-learn>=1.7.2
numpy>=1.24.0
pandas>=2.0.0
datasets>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
PyYAML>=6.0
python-dotenv>=1.0.0
tenacity>=8.0.0
pytest>=7.4.0
ruff>=0.1.0
```

---

## Architecture Decision Records (ADRs)

### ADR-001: Use Cookiecutter Data Science Template

**Context:** Need standardized ML project structure for 3-day timeline.

**Decision:** Use Cookiecutter Data Science v2 as project foundation.

**Rationale:**
- Industry-standard structure (9K+ GitHub stars)
- Separates data/code/notebooks/reports clearly
- Includes basic tooling (Ruff, pytest, mkdocs)
- Matches PRD recommended structure exactly
- Saves ~2 hours of manual setup

**Consequences:**
- First story must run `ccds` command
- Project structure predefined (flexibility reduced)
- All team members familiar with standard layout

---

### ADR-002: Use 768-Dimensional Embeddings

**Context:** Gemini supports 768, 1536, or 3072 dimensions.

**Decision:** Use 768 dimensions.

**Rationale:**
- Matches PRD specification
- Reduces storage: 120K × 768 × 4 bytes = ~370MB vs 1.4GB (3072-dim)
- Faster cosine similarity: O(d) scales linearly with dimensions
- Sufficient accuracy for 4-class classification
- Cost-neutral (pricing per token, not dimension)

**Consequences:**
- All embeddings consistently 768-dim
- Memory usage ~3x lower than 3072-dim
- Classification speed ~4x faster

---

### ADR-003: Use Gemini Batch API

**Context:** Need to embed 120K documents; costs add up.

**Decision:** Use Gemini Batch API for bulk embedding generation.

**Rationale:**
- 50% cost reduction: $0.075/M vs $0.15/M tokens
- 120K docs × 50 tokens × $0.075 / 1M = $0.45 (vs $0.90 standard)
- Asynchronous processing acceptable for one-time setup
- Stays well under <$10 budget

**Consequences:**
- Initial embedding generation slower (async)
- Caching required to avoid re-running
- Cost savings significant for academic project

---

### ADR-004: Use Fixed Random Seed (42)

**Context:** Need reproducible results for academic evaluation.

**Decision:** Set random_state=42 globally and in all randomized operations.

**Rationale:**
- PRD requires reproducibility
- Course instructor must verify results
- Fixed seed ensures identical clustering every run
- Standard practice in academic ML (42 is conventional)

**Consequences:**
- All experiments deterministic
- Results verifiable by instructor
- set_seed() must be called in all scripts

---

### ADR-005: Use JSON for Experiment Results

**Context:** Need to export metrics for report generation.

**Decision:** Store experiment results as JSON files with structured format.

**Rationale:**
- Human-readable (with indent=2)
- Easy to parse programmatically
- Standard format across all experiments
- Git-friendly (can track result changes)
- Includes full config for traceability

**Consequences:**
- All metrics exporters follow same JSON schema
- Results directory contains versioned experiments
- Easy to compare runs programmatically

---

### ADR-006: Separate Configuration and Secrets

**Context:** Need secure API key management + configurable parameters.

**Decision:** Use config.yaml for parameters, .env for secrets.

**Rationale:**
- config.yaml: committed to git, shareable
- .env: NOT committed, user-specific
- Prevents accidental API key leaks
- Standard practice (dotenv pattern)
- .env.example provides template

**Consequences:**
- Two files to manage (config.yaml + .env)
- Setup requires copying .env.example → .env
- Config class abstracts both sources

---

### ADR-007: Use matplotlib + seaborn for Visualizations

**Context:** Need academic-quality plots for report (300 DPI).

**Decision:** Use matplotlib (base) + seaborn (statistical plots).

**Rationale:**
- matplotlib: Most widely used, flexible, high-DPI export
- seaborn: Beautiful statistical plots (confusion matrix, distributions)
- Combined: Best of both worlds
- Avoid plotly complexity (interactive not needed)
- Matches academic paper standards

**Consequences:**
- All plots saved as PNG (300 DPI)
- Consistent style across visualizations
- Static plots (no interactivity)

---

## Summary

This architecture provides a **complete blueprint** for implementing a context-aware multi-agent system demonstrating LLM cost optimization through clustering and classification.

**Key Strengths:**
- ✅ All 7 Epics mapped to architecture
- ✅ All 14 Functional Requirements supported
- ✅ All 8 Non-Functional Requirements addressed
- ✅ Clear implementation patterns prevent AI agent conflicts
- ✅ Reproducible (fixed seeds, versioned dependencies)
- ✅ Cost-efficient (<$10 budget, 90%+ cost reduction)
- ✅ Academic-quality (clean code, 300 DPI plots, complete documentation)

**Next Steps:**
1. Run `ccds` to initialize project structure
2. Create config.yaml with specified parameters
3. Set up .env with GEMINI_API_KEY
4. Begin Epic 1: Generate embeddings (scripts/01_generate_embeddings.py)

---

_Generated by BMAD Decision Architecture Workflow v1.3.2_
_Date: 2025-11-08_
_For: Jack YUAN_
_Language: English (Documentation) / Chinese (Communication)_
