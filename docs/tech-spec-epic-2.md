# Epic Technical Specification: K-Means Clustering Implementation

Date: 2025-11-09
Author: Jack YUAN
Epic ID: 2
Status: Draft

---

## Overview

Epic 2 implements the **K-Means Clustering** component that partitions the AG News dataset (120K documents) into 4 semantic clusters based on their embeddings. This epic delivers the core algorithm that enables context reduction—dividing the full dataset into meaningful groups so that specialized agents only need to process ~25% of documents per query instead of 100%.

The clustering implementation uses scikit-learn's K-Means with k-means++ initialization (random_state=42) to ensure reproducible results for academic evaluation. The output includes cluster assignments for all documents, cluster centroids for query classification, and quality metrics (Silhouette Score >0.3) proving semantic separation. This epic bridges Epic 1 (embedding generation) and Epic 3 (specialized agent creation), providing the algorithmic foundation for the 90%+ cost reduction demonstrated in the final system.

## Objectives and Scope

### In Scope

**Core Functionality:**
- Implement K-Means clustering using scikit-learn (n_clusters=4, init='k-means++', random_state=42, max_iter=300)
- Load cached embeddings from Epic 1 (data/embeddings/train_embeddings.npy)
- Fit clustering model on 120K document embeddings (768 dimensions)
- Generate cluster assignments for all training documents
- Extract and store cluster centroids (4 × 768 float32 array)
- Calculate cluster quality metrics (Silhouette Score, Davies-Bouldin Index)
- Validate cluster distribution (balanced, no extreme skew)
- Export cluster assignments with metadata (CSV format)
- Generate PCA 2D visualization showing 4 separated clusters
- Save clustering artifacts for downstream epics

**Deliverables:**
- `src/models/clustering.py`: KMeansClustering class
- `scripts/02_train_clustering.py`: Clustering training script
- `data/processed/cluster_assignments.csv`: Cluster labels for all documents
- `data/processed/centroids.npy`: Cluster centroids (4 × 768)
- `data/processed/cluster_metadata.json`: Metrics and quality scores

### Out of Scope

**Explicitly Excluded:**
- Alternative clustering algorithms (MiniBatchKMeans, DBSCAN, hierarchical clustering)
- Automatic K selection (Elbow method, silhouette analysis across K values)
- Cluster labeling or interpretation (handled separately in Epic 2 story 2-5)
- Real-time cluster updates or online learning
- Cross-validation or cluster stability analysis
- Integration with specialized agents (Epic 3)
- Query classification or routing (Epic 4)

**Future Enhancements (Post-MVP):**
- Testing different K values (K=3, K=5, K=6)
- Alternative initialization methods comparison
- Incremental clustering for streaming data
- Hierarchical clustering for multi-level routing

## System Architecture Alignment

**Architecture Components Referenced:**

This epic implements the **Clustering Engine** component defined in the architecture (ADR-001, Decision Summary). It integrates with:

1. **Cookiecutter Data Science Structure**: Follows src/models/ for clustering logic, scripts/ for execution, data/processed/ for outputs
2. **Embedding Storage**: Consumes embeddings from `data/embeddings/train_embeddings.npy` (Epic 1 output)
3. **Configuration System**: Uses config.yaml for clustering parameters (n_clusters, random_state, max_iter, init method)
4. **Reproducibility Framework**: Enforces random_state=42 via utils/reproducibility.py (ADR-004)
5. **Data Architecture**: Produces cluster labels (int32), centroids (float32), following mandated data type patterns

**Constraints Applied:**
- **Performance**: K-Means convergence <5 minutes for 120K documents (NFR-1)
- **Reproducibility**: Fixed random seed ensures identical results across runs (ADR-004)
- **Data Types**: Labels as int32, centroids as float32 (Architecture Data Models section)
- **Logging**: Uses emoji-prefixed logging (📊, ✅, ⚠️, ❌) from utils/logger.py
- **Error Handling**: No external API calls in this epic, but validates embedding file existence

**Architectural Patterns Followed:**
- Initialization Order: set_seed → load config → setup logger → validate → execute
- Data Caching: Check cache → load if exists → generate and save if missing
- File Naming: snake_case for modules (clustering.py), PascalCase for classes (KMeansClustering)
- Configuration Access: No hardcoded values, all parameters from config.yaml

## Detailed Design

### Services and Modules

| Module | Responsibility | Inputs | Outputs | Owner |
|--------|---------------|--------|---------|-------|
| **KMeansClustering** (`src/models/clustering.py`) | K-Means clustering implementation wrapper | Embeddings (np.ndarray), clustering config | Cluster labels (int32), centroids (float32) | Epic 2 |
| **ClusteringMetrics** (`src/evaluation/clustering_metrics.py`) | Cluster quality evaluation | Embeddings, labels | Silhouette Score, Davies-Bouldin Index, cluster sizes | Epic 2 |
| **ClusterPlots** (`src/visualization/cluster_plots.py`) | PCA visualization generation | Embeddings, labels, centroids | PNG plots (300 DPI) | Epic 2 |
| **TrainClustering** (`scripts/02_train_clustering.py`) | Orchestration script | Cached embeddings, config | All clustering artifacts | Epic 2 |

**Module Interactions:**
```
scripts/02_train_clustering.py (orchestrator)
  ↓
  ├─→ Load embeddings from data/embeddings/train_embeddings.npy
  ├─→ KMeansClustering.fit_predict(embeddings) → labels, centroids
  ├─→ ClusteringMetrics.evaluate(embeddings, labels) → metrics
  ├─→ ClusterPlots.generate_pca_visualization(embeddings, labels, centroids) → PNG
  └─→ Save outputs: cluster_assignments.csv, centroids.npy, cluster_metadata.json
```

**Key Classes:**

**KMeansClustering** (src/models/clustering.py):
- Wraps scikit-learn KMeans with project-specific configuration
- Enforces reproducibility (random_state from config)
- Provides fit_predict() method returning labels and centroids
- Logs convergence information and cluster sizes

**ClusteringMetrics** (src/evaluation/clustering_metrics.py):
- Calculates Silhouette Score (target: >0.3)
- Calculates Davies-Bouldin Index (lower is better)
- Computes cluster size distribution
- Validates cluster balance (no cluster <10% or >50% of data)

**ClusterPlots** (src/visualization/cluster_plots.py):
- Applies PCA (768D → 2D) for visualization
- Generates scatter plot with 4 colored clusters
- Marks centroids on plot
- Exports 300 DPI PNG for report inclusion

### Data Models and Contracts

**Input Data Models:**

```python
# Embeddings (from Epic 1)
Type: np.ndarray
Shape: (120000, 768)  # 120K documents × 768 dimensions
Dtype: float32
Source: data/embeddings/train_embeddings.npy
Validation: Check shape[1] == 768, dtype == float32
```

**Output Data Models:**

```python
# Cluster Labels
Type: np.ndarray
Shape: (120000,)  # One label per document
Dtype: int32
Values: 0, 1, 2, 3 (cluster IDs)
Storage: data/processed/cluster_assignments.csv (with document_id)
Validation: All values in range [0, 3], no missing values

# Cluster Centroids
Type: np.ndarray
Shape: (4, 768)  # 4 clusters × 768 dimensions
Dtype: float32
Storage: data/processed/centroids.npy
Validation: Shape == (4, 768), dtype == float32

# Cluster Metadata
Type: dict (JSON)
Schema:
{
  "timestamp": str,               # ISO format
  "n_clusters": int,              # 4
  "n_documents": int,             # 120000
  "random_state": int,            # 42
  "n_iterations": int,            # Convergence iterations
  "inertia": float,               # Within-cluster sum of squares
  "silhouette_score": float,      # Target: >0.3
  "davies_bouldin_index": float,  # Lower is better
  "cluster_sizes": [int, ...],    # Size of each cluster
  "config": {...}                 # Full clustering config
}
Storage: data/processed/cluster_metadata.json
```

**CSV Schema (cluster_assignments.csv):**
```csv
document_id,cluster_id,category_label
0,2,World
1,1,Sports
2,0,Business
...
```

**API Contracts:**

```python
class KMeansClustering:
    def __init__(self, config: dict):
        """
        Initialize K-Means clustering with configuration.

        Args:
            config: Clustering parameters from config.yaml
                - n_clusters: int (default: 4)
                - random_state: int (default: 42)
                - max_iter: int (default: 300)
                - init: str (default: 'k-means++')
        """

    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit K-Means and return cluster assignments.

        Args:
            embeddings: Document embeddings (n_documents, 768) float32

        Returns:
            labels: Cluster assignments (n_documents,) int32
            centroids: Cluster centers (4, 768) float32

        Raises:
            ValueError: If embeddings shape invalid or dtype mismatch
        """
```

### APIs and Interfaces

**Public APIs:**

```python
# KMeansClustering API
from src.models.clustering import KMeansClustering

clustering = KMeansClustering(config.clustering_params)
labels, centroids = clustering.fit_predict(embeddings)

# Returns:
# - labels: np.ndarray (120000,) int32, values in [0, 3]
# - centroids: np.ndarray (4, 768) float32
```

```python
# ClusteringMetrics API
from src.evaluation.clustering_metrics import evaluate_clustering

metrics = evaluate_clustering(
    embeddings: np.ndarray,  # (120000, 768) float32
    labels: np.ndarray       # (120000,) int32
) -> dict

# Returns:
# {
#   "silhouette_score": float,        # Target: >0.3
#   "davies_bouldin_index": float,    # Lower is better
#   "cluster_sizes": [int, ...],      # 4 elements
#   "cluster_distribution": [float, ...]  # Percentages
# }
```

```python
# ClusterPlots API
from src.visualization.cluster_plots import generate_pca_visualization

plot_path = generate_pca_visualization(
    embeddings: np.ndarray,   # (120000, 768) float32
    labels: np.ndarray,       # (120000,) int32
    centroids: np.ndarray,    # (4, 768) float32
    output_path: Path         # PNG save location
) -> Path

# Saves 300 DPI PNG plot, returns path
```

**Internal Interfaces:**

```python
# Embedding loader (reused from Epic 1)
def load_embeddings(cache_path: Path) -> np.ndarray:
    """Load cached embeddings with validation."""
    embeddings = np.load(cache_path)
    assert embeddings.dtype == np.float32
    assert embeddings.shape[1] == 768
    return embeddings

# Cluster assignment exporter
def save_cluster_assignments(
    labels: np.ndarray,
    category_labels: List[str],
    output_path: Path
) -> None:
    """Export cluster assignments to CSV."""
    df = pd.DataFrame({
        'document_id': range(len(labels)),
        'cluster_id': labels,
        'category_label': category_labels
    })
    df.to_csv(output_path, index=False)
```

**File I/O Interfaces:**

| Operation | Path | Format | Direction |
|-----------|------|--------|-----------|
| Load embeddings | `data/embeddings/train_embeddings.npy` | NumPy binary | Read |
| Save cluster labels | `data/processed/cluster_assignments.csv` | CSV | Write |
| Save centroids | `data/processed/centroids.npy` | NumPy binary | Write |
| Save metadata | `data/processed/cluster_metadata.json` | JSON (indent=2) | Write |
| Save visualization | `reports/figures/cluster_pca_visualization.png` | PNG (300 DPI) | Write |

### Workflows and Sequencing

**Main Workflow (scripts/02_train_clustering.py):**

```
1. Initialization
   ├─→ set_seed(42)
   ├─→ load Config from config.yaml
   ├─→ setup logger
   └─→ validate configuration

2. Load Input Data
   ├─→ Check data/embeddings/train_embeddings.npy exists
   ├─→ Load embeddings (120000, 768) float32
   ├─→ Validate shape and dtype
   └─→ Log: "📊 Loaded 120K embeddings (768-dim)"

3. Fit K-Means Clustering
   ├─→ Initialize KMeansClustering(config.clustering_params)
   ├─→ Call fit_predict(embeddings)
   ├─→ K-Means algorithm runs:
   │   ├─→ k-means++ initialization (select 4 initial centroids)
   │   ├─→ Iterative optimization (max 300 iterations)
   │   ├─→ Convergence check (centroids stable)
   │   └─→ Return labels (120000,) int32, centroids (4, 768) float32
   └─→ Log: "✅ Clustering converged in {n} iterations"

4. Evaluate Cluster Quality
   ├─→ Call evaluate_clustering(embeddings, labels)
   ├─→ Calculate Silhouette Score (target: >0.3)
   ├─→ Calculate Davies-Bouldin Index (lower is better)
   ├─→ Compute cluster sizes and distribution
   └─→ Log metrics with warnings if below target

5. Generate Visualization
   ├─→ Apply PCA(n_components=2) to reduce 768D → 2D
   ├─→ Project embeddings and centroids to 2D space
   ├─→ Generate scatter plot:
   │   ├─→ 4 colored clusters (matplotlib colormap)
   │   ├─→ Mark centroids with larger markers
   │   ├─→ Add legend with cluster labels
   │   └─→ Professional formatting (300 DPI)
   └─→ Save to reports/figures/cluster_pca_visualization.png

6. Save Outputs
   ├─→ Save cluster_assignments.csv (document_id, cluster_id, category_label)
   ├─→ Save centroids.npy (4, 768) float32
   ├─→ Save cluster_metadata.json with:
   │   ├─→ timestamp, n_clusters, n_documents
   │   ├─→ silhouette_score, davies_bouldin_index
   │   ├─→ cluster_sizes, inertia, n_iterations
   │   └─→ full config for traceability
   └─→ Log: "💾 Saved clustering artifacts"

7. Validation and Summary
   ├─→ Validate all outputs exist and have correct schema
   ├─→ Check Silhouette Score meets target (>0.3)
   ├─→ Check cluster distribution is balanced
   └─→ Log summary report
```

**Sequence Diagram (K-Means Execution):**

```
User → Script: python scripts/02_train_clustering.py
Script → Config: Load clustering parameters
Script → Embeddings: Load cached embeddings (120K × 768)
Script → KMeansClustering: fit_predict(embeddings)
KMeansClustering → scikit-learn: KMeans(n_clusters=4, random_state=42, ...)
scikit-learn → KMeansClustering: labels, centroids
KMeansClustering → Script: Return (labels, centroids)
Script → ClusteringMetrics: evaluate_clustering(embeddings, labels)
ClusteringMetrics → Script: Return metrics dict
Script → ClusterPlots: generate_pca_visualization(embeddings, labels, centroids)
ClusterPlots → Script: Return plot_path
Script → Disk: Save all outputs (CSV, NPY, JSON, PNG)
Script → User: "✅ Clustering completed successfully"
```

**Error Handling Flow:**

```
Load embeddings
├─→ FileNotFoundError: "Embeddings not found. Run scripts/01_generate_embeddings.py first"
├─→ Shape mismatch: "Expected (*, 768), got {shape}"
└─→ Dtype mismatch: "Expected float32, got {dtype}"

Clustering convergence
├─→ Max iterations reached: Log warning, continue (clustering still valid)
└─→ Empty cluster: Re-run with different random_state (should not happen with k-means++)

Quality metrics
├─→ Silhouette < 0.3: Log warning "⚠️ Below target, but continuing"
└─→ Imbalanced clusters: Log warning if any cluster <10% or >50%

Save outputs
├─→ Directory not exists: Create automatically
└─→ Write permission denied: Raise error with clear message
```

## Non-Functional Requirements

### Performance

**Target Metrics:**
- K-Means clustering convergence: <5 minutes for 120K documents (768-dim embeddings)
- Silhouette Score calculation: <3 minutes
- PCA dimensionality reduction: <2 minutes
- Total script execution time: <10 minutes end-to-end

**Performance Optimization Strategies:**
- Use scikit-learn's optimized KMeans implementation (C/Cython backend)
- k-means++ initialization reduces iterations needed for convergence
- PCA uses randomized SVD for faster computation on large datasets
- Memory mapping for large embedding arrays (mmap_mode='r') if RAM limited
- Batch processing for metrics calculation if needed

**Memory Requirements:**
- Embeddings in memory: 120K × 768 × 4 bytes = ~370 MB
- K-Means model state: minimal overhead (<50 MB)
- PCA transformation: temporary 2D projection (~1 MB)
- Total peak memory: <500 MB (well within 8GB RAM minimum)

**Scalability Considerations:**
- Current implementation optimized for 120K documents
- For larger datasets (>1M documents), consider MiniBatchKMeans
- PCA can use IncrementalPCA for datasets exceeding RAM

### Security

**Applicable Security Considerations:**

This epic involves no external API calls, network access, or sensitive data handling. Security requirements are minimal:

**Data Integrity:**
- Validate embedding file integrity (shape, dtype) before processing
- Use checksums or metadata validation if embedding cache corruption suspected
- Ensure cluster assignments map correctly to document IDs

**File System Security:**
- Write outputs to project-controlled directories (data/processed/, reports/figures/)
- No user-supplied paths without validation
- Create directories with appropriate permissions (755 for dirs, 644 for files)

**No Security Risks:**
- No API keys or credentials required (this epic is local computation only)
- No network communication
- AG News dataset is public domain (no privacy concerns)
- No user input beyond configuration file (already validated)

**Configuration Validation:**
- Validate config.yaml parameters are within expected ranges (n_clusters > 0, random_state >= 0)
- Reject invalid parameter types with clear error messages

### Reliability/Availability

**Reliability Requirements:**

**Deterministic Execution:**
- Fixed random_state=42 ensures identical results across runs
- Same embeddings + same config = same cluster assignments (100% reproducible)
- K-Means convergence guaranteed (scikit-learn handles edge cases)

**Error Recovery:**
- If clustering fails, script exits with clear error message and suggested fix
- Partial outputs (e.g., only CSV saved) are overwritten on retry (no inconsistent state)
- Embedding validation catches corrupted cache early (fail-fast approach)

**Robustness:**
- K-Means with k-means++ initialization avoids empty clusters (very rare failure mode)
- If convergence doesn't complete in max_iter (300), algorithm returns best result with warning
- Silhouette Score calculation handles edge cases (single-cluster, identical points)

**Data Validation:**
- Pre-flight checks: embedding shape, dtype, finite values (no NaN/Inf)
- Post-clustering checks: all documents assigned, cluster IDs in valid range [0, 3]
- Cluster balance check: warn if any cluster <10% or >50% of data

**Availability:**
- No external dependencies (no network, no API calls)
- Script can run offline after initial embedding generation
- No scheduled downtime or maintenance windows needed
- Single-machine execution (no distributed coordination failures)

### Observability

**Logging Strategy:**

**Log Levels and Usage:**
```python
INFO:  Normal execution flow (start, progress, completion)
WARN:  Metrics below target, cluster imbalance, convergence warnings
ERROR: File not found, validation failures, unexpected exceptions
```

**Required Log Messages:**

**Startup:**
```
📊 Starting K-Means clustering...
📊 Loaded 120000 embeddings (768-dim) from cache
📊 Configuration: n_clusters=4, random_state=42, max_iter=300
```

**Progress:**
```
📊 Fitting K-Means clustering...
✅ Clustering converged in 47 iterations
📊 Calculating cluster quality metrics...
📊 Generating PCA visualization...
```

**Metrics:**
```
✅ Silhouette Score: 0.347 (target: >0.3)
✅ Davies-Bouldin Index: 1.234 (lower is better)
📊 Cluster sizes: [28934, 31245, 29876, 29945] (balanced)
```

**Warnings:**
```
⚠️ Silhouette Score 0.287 below target 0.3 (still acceptable)
⚠️ Cluster 2 contains 62% of documents (imbalanced)
⚠️ Convergence not reached in 300 iterations (using best result)
```

**Completion:**
```
💾 Saved cluster assignments: data/processed/cluster_assignments.csv
💾 Saved centroids: data/processed/centroids.npy
💾 Saved metadata: data/processed/cluster_metadata.json
💾 Saved visualization: reports/figures/cluster_pca_visualization.png
✅ Clustering completed successfully in 4m 32s
```

**Metrics Collection:**

**Exported to cluster_metadata.json:**
- Execution timestamp
- Convergence iterations
- Inertia (within-cluster sum of squares)
- Silhouette Score
- Davies-Bouldin Index
- Cluster sizes and distribution
- Configuration snapshot

**Traceability:**
- All outputs include timestamp
- Metadata links to config.yaml parameters
- Cluster assignments include document IDs for end-to-end tracking

**Monitoring:**
- Console output provides real-time progress
- Log files (if configured) capture full execution trace
- JSON metadata enables automated metric collection

## Dependencies and Integrations

### External Dependencies

**Core Python Libraries:**

| Library | Version | Purpose | Usage in Epic 2 |
|---------|---------|---------|-----------------|
| **scikit-learn** | >=1.7.2 | Machine learning algorithms | K-Means clustering, PCA, Silhouette Score, Davies-Bouldin Index |
| **numpy** | >=1.24.0 | Array operations | Embedding storage/manipulation, float32/int32 arrays |
| **pandas** | >=2.0.0 | Data manipulation | CSV export of cluster assignments |
| **matplotlib** | >=3.7.0 | Plotting library | PCA visualization base plotting |
| **seaborn** | >=0.12.0 | Statistical visualization | Enhanced cluster plot styling |
| **PyYAML** | >=6.0 | Configuration parsing | Load clustering parameters from config.yaml |

**Specific API Usage:**

```python
# scikit-learn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# numpy
import numpy as np
# Load/save embeddings, centroids as .npy files
# Array operations with dtype=float32, int32

# pandas
import pandas as pd
# Export cluster assignments to CSV

# matplotlib + seaborn
import matplotlib.pyplot as plt
import seaborn as sns
# Generate 300 DPI PNG visualizations
```

### Internal Dependencies (Epic 1 Outputs)

**Required Inputs from Epic 1:**

| Artifact | Path | Format | Validation |
|----------|------|--------|------------|
| **Train Embeddings** | `data/embeddings/train_embeddings.npy` | NumPy (120000, 768) float32 | Check exists, shape, dtype |
| **Embedding Metadata** | `data/embeddings/metadata.json` | JSON | Verify n_documents=120000, dimensions=768 |
| **Dataset Info** | Epic 1 output | Category labels for documents | Map cluster IDs to AG News categories |

**Dependency Validation:**
```python
# Pre-flight check in scripts/02_train_clustering.py
cache_path = paths.data_embeddings / "train_embeddings.npy"
if not cache_path.exists():
    raise FileNotFoundError(
        f"Embeddings not found: {cache_path}\n"
        f"Run 'python scripts/01_generate_embeddings.py' first"
    )
```

### Internal Module Dependencies

**Within Epic 2:**

```
scripts/02_train_clustering.py (main)
  ├─→ src/models/clustering.py (KMeansClustering)
  ├─→ src/evaluation/clustering_metrics.py (evaluate_clustering)
  ├─→ src/visualization/cluster_plots.py (generate_pca_visualization)
  └─→ src/config.py, src/utils/logger.py, src/utils/reproducibility.py
```

**Cross-Module Shared Utilities:**
- `src/config.py`: Config, Paths classes (shared across all epics)
- `src/utils/logger.py`: Emoji-prefixed logging (shared)
- `src/utils/reproducibility.py`: set_seed() function (shared)

### Downstream Integrations (Epic 2 Outputs → Other Epics)

**Epic 2 Outputs Consumed By:**

| Consumer Epic | Artifact Needed | Usage |
|---------------|-----------------|-------|
| **Epic 3: Specialized Agents** | `cluster_assignments.csv` | Assign documents to agents based on cluster |
| **Epic 3: Specialized Agents** | `centroids.npy` | Initialize agent registry with cluster metadata |
| **Epic 4: Classification & Routing** | `centroids.npy` | Classify queries via cosine similarity with centroids |
| **Epic 6: Cost Metrics** | `cluster_metadata.json` | Validate cluster quality metrics for report |
| **Epic 7: Experimental Report** | `cluster_pca_visualization.png` | Include cluster plot in report |
| **Epic 7: Experimental Report** | `cluster_metadata.json` | Report Silhouette Score, cluster sizes |

**Integration Contract:**

Epic 2 MUST guarantee:
- `centroids.npy` exists with shape (4, 768) dtype float32
- `cluster_assignments.csv` contains valid cluster IDs [0, 3] for all 120K documents
- `cluster_metadata.json` contains silhouette_score, davies_bouldin_index, cluster_sizes
- `cluster_pca_visualization.png` is 300 DPI, professionally formatted

### Configuration Dependencies

**Required config.yaml Parameters:**

```yaml
clustering:
  algorithm: "kmeans"           # MUST be "kmeans"
  n_clusters: 4                 # MUST be 4 for this project
  random_state: 42              # MUST be 42 for reproducibility
  max_iter: 300                 # Maximum iterations
  init: "k-means++"             # Initialization method
```

**Used by Epic 2:**
```python
config = Config('config.yaml')
n_clusters = config.get('clustering.n_clusters')       # 4
random_state = config.get('clustering.random_state')   # 42
max_iter = config.get('clustering.max_iter')           # 300
init_method = config.get('clustering.init')            # 'k-means++'
```

### No External Integrations

This epic has:
- ❌ No external API calls (Gemini API not used in Epic 2)
- ❌ No network communication
- ❌ No database connections
- ❌ No third-party services

All processing is local computation using cached data from Epic 1.

## Acceptance Criteria (Authoritative)

**CRITICAL: These acceptance criteria are the authoritative definition of "done" for Epic 2. All stories must satisfy these criteria before the epic is considered complete.**

### AC-1: K-Means Clustering Implementation

**Criteria:**
- ✅ KMeansClustering class implemented in `src/models/clustering.py`
- ✅ Uses scikit-learn KMeans with exact parameters: n_clusters=4, init='k-means++', random_state=42, max_iter=300
- ✅ `fit_predict()` method accepts embeddings (n, 768) float32 and returns (labels, centroids)
- ✅ Labels are np.ndarray shape (n,) dtype int32, values in [0, 3]
- ✅ Centroids are np.ndarray shape (4, 768) dtype float32
- ✅ Logs convergence information (iterations, inertia)

**Validation:**
```python
clustering = KMeansClustering(config.clustering_params)
labels, centroids = clustering.fit_predict(embeddings)
assert labels.shape == (120000,)
assert labels.dtype == np.int32
assert centroids.shape == (4, 768)
assert centroids.dtype == np.float32
assert all(0 <= label <= 3 for label in labels)
```

---

### AC-2: Cluster Quality Metrics

**Criteria:**
- ✅ Silhouette Score calculated and >0.3 (good cluster separation)
- ✅ Davies-Bouldin Index calculated (lower is better, no hard threshold)
- ✅ Cluster sizes computed and validated (no cluster <10% or >50% of data)
- ✅ All metrics exported to `cluster_metadata.json`
- ✅ Warning logged if Silhouette Score <0.3 (but epic still passes)

**Validation:**
```python
metrics = evaluate_clustering(embeddings, labels)
assert metrics['silhouette_score'] > 0.3  # Target met
assert 'davies_bouldin_index' in metrics
assert len(metrics['cluster_sizes']) == 4
assert all(size >= 0.1 * 120000 for size in metrics['cluster_sizes'])
assert all(size <= 0.5 * 120000 for size in metrics['cluster_sizes'])
```

---

### AC-3: PCA Visualization

**Criteria:**
- ✅ PCA reduces embeddings from 768D to 2D for visualization
- ✅ Scatter plot generated with 4 distinctly colored clusters
- ✅ Cluster centroids marked on plot (larger markers)
- ✅ Legend included with cluster labels (Cluster 0, 1, 2, 3)
- ✅ Professional formatting (axis labels, title, 300 DPI)
- ✅ Saved as PNG to `reports/figures/cluster_pca_visualization.png`

**Validation:**
```python
import matplotlib.image as mpimg
img = mpimg.imread('reports/figures/cluster_pca_visualization.png')
assert img.shape[0] >= 1800  # 300 DPI = 6" × 300 = 1800 pixels min
assert os.path.exists('reports/figures/cluster_pca_visualization.png')
```

---

### AC-4: Cluster Assignments Export

**Criteria:**
- ✅ All 120K documents assigned to exactly one cluster
- ✅ CSV file saved to `data/processed/cluster_assignments.csv`
- ✅ CSV contains columns: document_id, cluster_id, category_label
- ✅ Cluster IDs are valid integers [0, 3]
- ✅ Document IDs match AG News dataset order
- ✅ Category labels (World, Sports, Business, Sci/Tech) included for reference

**Validation:**
```python
df = pd.read_csv('data/processed/cluster_assignments.csv')
assert len(df) == 120000
assert set(df.columns) == {'document_id', 'cluster_id', 'category_label'}
assert df['cluster_id'].min() == 0
assert df['cluster_id'].max() == 3
assert df['document_id'].is_unique
```

---

### AC-5: Centroids Export

**Criteria:**
- ✅ Centroids saved to `data/processed/centroids.npy`
- ✅ Shape is exactly (4, 768)
- ✅ Dtype is float32
- ✅ No NaN or Inf values
- ✅ Centroids represent cluster centers computed by K-Means

**Validation:**
```python
centroids = np.load('data/processed/centroids.npy')
assert centroids.shape == (4, 768)
assert centroids.dtype == np.float32
assert not np.any(np.isnan(centroids))
assert not np.any(np.isinf(centroids))
```

---

### AC-6: Metadata Export

**Criteria:**
- ✅ Metadata saved to `data/processed/cluster_metadata.json`
- ✅ Contains all required fields: timestamp, n_clusters, n_documents, random_state, n_iterations, inertia, silhouette_score, davies_bouldin_index, cluster_sizes, config
- ✅ JSON formatted with indent=2 (human-readable)
- ✅ Timestamp in ISO format (YYYY-MM-DDTHH:MM:SS)

**Validation:**
```python
with open('data/processed/cluster_metadata.json') as f:
    metadata = json.load(f)
required_keys = {'timestamp', 'n_clusters', 'n_documents', 'random_state',
                 'n_iterations', 'inertia', 'silhouette_score',
                 'davies_bouldin_index', 'cluster_sizes', 'config'}
assert set(metadata.keys()) >= required_keys
assert metadata['n_clusters'] == 4
assert metadata['n_documents'] == 120000
assert metadata['random_state'] == 42
```

---

### AC-7: Reproducibility

**Criteria:**
- ✅ Fixed random_state=42 used in K-Means initialization
- ✅ Running script multiple times produces identical results
- ✅ Cluster assignments unchanged across runs (same embeddings → same clusters)
- ✅ set_seed(42) called at script start

**Validation:**
```bash
# Run script twice
python scripts/02_train_clustering.py
cp data/processed/cluster_assignments.csv run1.csv
python scripts/02_train_clustering.py
cp data/processed/cluster_assignments.csv run2.csv
diff run1.csv run2.csv  # Should be identical
```

---

### AC-8: Performance Requirements

**Criteria:**
- ✅ K-Means clustering completes in <5 minutes (120K documents)
- ✅ Silhouette Score calculation completes in <3 minutes
- ✅ PCA visualization generation completes in <2 minutes
- ✅ Total script execution time <10 minutes end-to-end
- ✅ Peak memory usage <500 MB (well within 8GB minimum)

**Validation:**
```python
import time
start = time.time()
# Run clustering script
elapsed = time.time() - start
assert elapsed < 600  # 10 minutes max
```

---

### AC-9: Logging and Observability

**Criteria:**
- ✅ Emoji-prefixed logs (📊, ✅, ⚠️, ❌) for visual clarity
- ✅ Log startup configuration (n_clusters, random_state, max_iter)
- ✅ Log convergence information (iterations, inertia)
- ✅ Log cluster quality metrics (Silhouette Score, Davies-Bouldin Index)
- ✅ Log warnings if metrics below target
- ✅ Log all file save operations with paths

**Validation:**
Check console output contains:
```
📊 Starting K-Means clustering...
📊 Loaded 120000 embeddings (768-dim) from cache
✅ Clustering converged in X iterations
✅ Silhouette Score: X.XXX (target: >0.3)
💾 Saved cluster assignments: data/processed/cluster_assignments.csv
✅ Clustering completed successfully
```

---

### AC-10: Error Handling

**Criteria:**
- ✅ Clear error if embeddings file missing (suggests running Epic 1 script)
- ✅ Validation error if embedding shape wrong (not (*, 768))
- ✅ Validation error if embedding dtype wrong (not float32)
- ✅ Warning if Silhouette Score <0.3 (but continues execution)
- ✅ Warning if cluster imbalance detected (any cluster <10% or >50%)
- ✅ Automatic directory creation if output paths don't exist

**Validation:**
```python
# Test missing embeddings
os.remove('data/embeddings/train_embeddings.npy')
result = subprocess.run(['python', 'scripts/02_train_clustering.py'],
                       capture_output=True)
assert b"Embeddings not found" in result.stderr
assert b"Run 'python scripts/01_generate_embeddings.py' first" in result.stderr
```

---

### Summary: Epic 2 Complete When

**All 10 acceptance criteria above are satisfied:**

1. ✅ KMeansClustering class implemented correctly
2. ✅ Cluster quality metrics calculated and meet targets
3. ✅ PCA visualization generated (300 DPI PNG)
4. ✅ Cluster assignments exported to CSV (120K rows)
5. ✅ Centroids exported to NPY (4×768 float32)
6. ✅ Metadata exported to JSON with all fields
7. ✅ Reproducibility guaranteed (random_state=42)
8. ✅ Performance targets met (<10 min total)
9. ✅ Logging comprehensive and emoji-prefixed
10. ✅ Error handling robust with clear messages

**Downstream Guarantee:**
Epic 3, 4, 6, 7 can proceed once Epic 2 outputs are validated against these criteria.

## Traceability Mapping

**This table maps acceptance criteria to spec sections, components, and test strategies, ensuring complete coverage.**

| AC ID | AC Name | Spec Section(s) | Component(s) | Test Strategy |
|-------|---------|----------------|--------------|---------------|
| **AC-1** | K-Means Clustering Implementation | Detailed Design → Services and Modules → KMeansClustering<br>APIs and Interfaces → KMeansClustering API | `src/models/clustering.py`<br>`KMeansClustering.fit_predict()` | Unit test: Verify fit_predict() returns correct shapes and dtypes<br>Integration test: Full clustering on test embeddings |
| **AC-2** | Cluster Quality Metrics | Detailed Design → Services and Modules → ClusteringMetrics<br>APIs and Interfaces → ClusteringMetrics API<br>NFR → Performance (Silhouette <3 min) | `src/evaluation/clustering_metrics.py`<br>`evaluate_clustering()` | Unit test: Metrics calculation on small sample<br>Integration test: Silhouette Score >0.3 on full dataset<br>Performance test: Timing validation |
| **AC-3** | PCA Visualization | Detailed Design → Services and Modules → ClusterPlots<br>Workflows and Sequencing → Step 5 (Generate Visualization)<br>APIs and Interfaces → ClusterPlots API | `src/visualization/cluster_plots.py`<br>`generate_pca_visualization()` | Visual inspection: Plot shows 4 colored clusters<br>Automated test: PNG exists, resolution ≥300 DPI<br>Manual review: Professional formatting |
| **AC-4** | Cluster Assignments Export | Data Models and Contracts → Output Data Models (Cluster Labels)<br>Workflows and Sequencing → Step 6 (Save Outputs)<br>APIs and Interfaces → File I/O Interfaces | `scripts/02_train_clustering.py`<br>`save_cluster_assignments()` | Integration test: CSV has 120K rows, correct schema<br>Data validation: All cluster IDs in [0, 3]<br>Correctness: Document IDs match dataset order |
| **AC-5** | Centroids Export | Data Models and Contracts → Output Data Models (Cluster Centroids)<br>Detailed Design → KMeansClustering (extract centroids)<br>Dependencies and Integrations → Downstream (Epic 4 uses centroids) | `src/models/clustering.py`<br>`scripts/02_train_clustering.py` | Unit test: Centroids shape (4, 768) float32<br>Integration test: No NaN/Inf values<br>Downstream test: Epic 4 can load and use centroids |
| **AC-6** | Metadata Export | Data Models and Contracts → Output Data Models (Cluster Metadata)<br>NFR → Observability (Metrics Collection) | `scripts/02_train_clustering.py`<br>JSON export logic | Schema validation: All required keys present<br>Value validation: n_clusters=4, random_state=42<br>Format check: JSON indent=2, timestamp ISO format |
| **AC-7** | Reproducibility | System Architecture Alignment → Reproducibility Framework<br>NFR → Reliability (Deterministic Execution)<br>Architecture ADR-004 | `src/utils/reproducibility.py`<br>`set_seed(42)` call in script | Repeatability test: Run script 3× → identical outputs<br>Hash comparison: cluster_assignments.csv MD5 identical<br>Centroid comparison: np.allclose(run1, run2) |
| **AC-8** | Performance Requirements | NFR → Performance (Target Metrics)<br>Workflows and Sequencing → Main Workflow (timed steps) | Full clustering pipeline | Benchmark test: Time each major step<br>End-to-end test: Total time <10 minutes<br>Memory profiling: Peak usage <500 MB |
| **AC-9** | Logging and Observability | NFR → Observability (Logging Strategy)<br>Workflows and Sequencing (log messages at each step) | `src/utils/logger.py`<br>All modules using logger | Log parsing: Check required messages present<br>Format validation: Emoji prefixes used correctly<br>Coverage check: All steps logged |
| **AC-10** | Error Handling | Workflows and Sequencing → Error Handling Flow<br>NFR → Reliability (Error Recovery) | All modules with validation logic | Negative test: Missing embeddings → clear error<br>Negative test: Wrong shape → validation error<br>Warning test: Low Silhouette → warning logged |

---

## Test Strategy Summary

### Unit Tests (Per Module)

**src/models/clustering.py:**
```python
def test_kmeans_clustering_fit_predict():
    # Small synthetic dataset
    embeddings = np.random.randn(1000, 768).astype(np.float32)
    config = {'n_clusters': 4, 'random_state': 42, 'max_iter': 300, 'init': 'k-means++'}
    clustering = KMeansClustering(config)
    labels, centroids = clustering.fit_predict(embeddings)
    assert labels.shape == (1000,)
    assert labels.dtype == np.int32
    assert centroids.shape == (4, 768)
    assert centroids.dtype == np.float32
```

**src/evaluation/clustering_metrics.py:**
```python
def test_evaluate_clustering():
    embeddings = np.random.randn(1000, 768).astype(np.float32)
    labels = np.random.randint(0, 4, 1000).astype(np.int32)
    metrics = evaluate_clustering(embeddings, labels)
    assert 'silhouette_score' in metrics
    assert 'davies_bouldin_index' in metrics
    assert len(metrics['cluster_sizes']) == 4
```

**src/visualization/cluster_plots.py:**
```python
def test_generate_pca_visualization(tmp_path):
    embeddings = np.random.randn(1000, 768).astype(np.float32)
    labels = np.random.randint(0, 4, 1000).astype(np.int32)
    centroids = np.random.randn(4, 768).astype(np.float32)
    output_path = tmp_path / "test_plot.png"
    result_path = generate_pca_visualization(embeddings, labels, centroids, output_path)
    assert result_path.exists()
```

### Integration Tests (Full Pipeline)

**Test 1: End-to-End Clustering:**
```python
def test_full_clustering_pipeline():
    # Run scripts/02_train_clustering.py
    result = subprocess.run(['python', 'scripts/02_train_clustering.py'],
                           capture_output=True)
    assert result.returncode == 0
    # Validate all outputs exist
    assert Path('data/processed/cluster_assignments.csv').exists()
    assert Path('data/processed/centroids.npy').exists()
    assert Path('data/processed/cluster_metadata.json').exists()
    assert Path('reports/figures/cluster_pca_visualization.png').exists()
```

**Test 2: Reproducibility:**
```python
def test_reproducibility():
    # Run twice, compare outputs
    subprocess.run(['python', 'scripts/02_train_clustering.py'])
    assignments1 = pd.read_csv('data/processed/cluster_assignments.csv')
    centroids1 = np.load('data/processed/centroids.npy')

    subprocess.run(['python', 'scripts/02_train_clustering.py'])
    assignments2 = pd.read_csv('data/processed/cluster_assignments.csv')
    centroids2 = np.load('data/processed/centroids.npy')

    assert assignments1.equals(assignments2)
    assert np.allclose(centroids1, centroids2)
```

### Performance Tests

**Benchmark Test:**
```python
def test_performance_targets():
    import time
    start = time.time()
    subprocess.run(['python', 'scripts/02_train_clustering.py'])
    elapsed = time.time() - start
    assert elapsed < 600  # 10 minutes max
```

### Acceptance Tests (Validate AC)

**AC Validation Script:**
```python
def validate_epic_2_complete():
    """Run all AC validations to confirm Epic 2 is done."""
    # AC-1: KMeansClustering implementation
    from src.models.clustering import KMeansClustering
    assert KMeansClustering is not None

    # AC-2: Cluster quality metrics
    metadata = json.load(open('data/processed/cluster_metadata.json'))
    assert metadata['silhouette_score'] > 0.3

    # AC-3: PCA visualization
    assert Path('reports/figures/cluster_pca_visualization.png').exists()

    # AC-4: Cluster assignments export
    df = pd.read_csv('data/processed/cluster_assignments.csv')
    assert len(df) == 120000

    # AC-5: Centroids export
    centroids = np.load('data/processed/centroids.npy')
    assert centroids.shape == (4, 768)

    # AC-6: Metadata export
    assert set(metadata.keys()) >= {'timestamp', 'n_clusters', 'silhouette_score'}

    # AC-7: Reproducibility
    # (Tested separately)

    # AC-8: Performance
    # (Tested separately)

    # AC-9: Logging
    # (Manual verification)

    # AC-10: Error handling
    # (Negative tests)

    print("✅ All Epic 2 acceptance criteria validated!")
```

---

## Coverage Matrix

| PRD Requirement | Tech Spec Section | AC Coverage | Test Coverage |
|-----------------|-------------------|-------------|---------------|
| FR-3: K-Means Clustering | Overview, Detailed Design | AC-1 | Unit + Integration |
| FR-4: Cluster Quality Evaluation | Detailed Design → ClusteringMetrics | AC-2 | Unit + Integration |
| FR-5: Cluster Visualization | Detailed Design → ClusterPlots | AC-3 | Integration + Visual |
| NFR-1: Performance (<5 min) | NFR → Performance | AC-8 | Benchmark |
| NFR-4: Reproducibility (random_state=42) | System Architecture Alignment | AC-7 | Repeatability |

**100% Coverage:** All PRD requirements for Epic 2 are traced to spec sections, acceptance criteria, and test strategies.

## Risks, Assumptions, Open Questions

### Risks

| Risk ID | Risk Description | Likelihood | Impact | Mitigation Strategy | Owner |
|---------|------------------|------------|--------|---------------------|-------|
| **R-2.1** | Silhouette Score <0.3 (cluster quality below target) | Medium | Medium | - Use k-means++ initialization (reduces risk)<br>- Accept scores ≥0.25 with warning<br>- Document in metadata for transparency<br>- Post-MVP: Try different K values if needed | Epic 2 Dev |
| **R-2.2** | Cluster imbalance (one cluster dominates) | Low | Low | - K-Means naturally balances with k-means++ init<br>- Log warning if any cluster >50% or <10%<br>- AG News dataset is balanced by design<br>- Epic 3 agents can handle varied cluster sizes | Epic 2 Dev |
| **R-2.3** | K-Means convergence failure (max_iter reached) | Very Low | Low | - max_iter=300 is generous for 120K documents<br>- scikit-learn returns best result even if not converged<br>- Log warning, continue execution<br>- Rare with k-means++ initialization | Epic 2 Dev |
| **R-2.4** | Memory overflow on machines with <8GB RAM | Low | High | - Current peak usage ~500MB (well under limit)<br>- Use memory mapping (mmap_mode='r') if needed<br>- Document minimum 8GB RAM requirement<br>- PCA can use IncrementalPCA if memory constrained | Epic 2 Dev |
| **R-2.5** | Embedding cache corruption (Epic 1 output invalid) | Very Low | High | - Pre-flight validation checks shape, dtype<br>- Check for NaN/Inf values before clustering<br>- Clear error message points to Epic 1 re-run<br>- Epic 1 should save metadata checksums | Epic 1 Owner |
| **R-2.6** | PCA visualization unclear (clusters overlap visually) | Medium | Low | - 2D projection inherently lossy (768D → 2D)<br>- Silhouette Score is authoritative metric<br>- Visual is supplementary, not critical<br>- If overlap severe, try different random seeds for PCA | Epic 2 Dev |

**Risk Summary:**
- **High Impact Risks**: 1 (R-2.4, R-2.5) - Both mitigated with validation
- **Medium Likelihood**: 2 (R-2.1, R-2.6) - Acceptable with mitigation
- **Overall Risk Level**: **LOW** - Most risks are low probability and well-mitigated

---

### Assumptions

| Assumption ID | Assumption | Validation | Impact if Invalid |
|---------------|------------|------------|-------------------|
| **A-2.1** | Epic 1 embeddings are already generated and cached | Check file exists at start | Script fails immediately with clear error |
| **A-2.2** | Embeddings are valid (shape, dtype, no corruption) | Pre-flight validation in script | Clustering fails early with validation error |
| **A-2.3** | K=4 is the optimal number of clusters for AG News | Based on PRD requirement (4 categories) | No issue - K=4 is fixed requirement, not optimization |
| **A-2.4** | scikit-learn 1.7.2+ is installed and working | Dependency check at import | Import error with clear message to install deps |
| **A-2.5** | config.yaml contains valid clustering parameters | Config validation at startup | Script fails early with parameter validation error |
| **A-2.6** | Output directories (data/processed/, reports/figures/) are writable | Script creates dirs automatically | Permission error if parent dir not writable |
| **A-2.7** | AG News dataset has balanced categories | Validated in Epic 1 | Cluster imbalance warning logged but acceptable |
| **A-2.8** | random_state=42 ensures reproducibility | Standard scikit-learn behavior | If broken, Epic 2 tests would fail immediately |
| **A-2.9** | Execution environment has ≥8GB RAM | Documented in architecture NFR | Memory error if violated (rare with 500MB peak) |
| **A-2.10** | No concurrent writes to output files | Single-threaded execution | No issue - script is not parallel |

**Assumptions Validation:**
- **Critical Assumptions (A-2.1, A-2.2)**: Validated at runtime with clear errors
- **Configuration Assumptions (A-2.3, A-2.5)**: Validated early in execution
- **Environment Assumptions (A-2.4, A-2.6, A-2.9)**: Documented in setup guide

---

### Open Questions

| Question ID | Question | Priority | Assigned To | Resolution Deadline | Status |
|-------------|----------|----------|-------------|---------------------|--------|
| **Q-2.1** | Should we persist the KMeans model object (.pkl) for reuse? | Low | Epic 2 Dev | Before implementation | **RESOLVED**: No - centroids are sufficient for Epic 4 classification. Model persistence adds complexity for no benefit. |
| **Q-2.2** | What if Silhouette Score is between 0.25-0.3 (borderline)? | Medium | Epic 2 Dev | Before validation | **RESOLVED**: Log warning, continue execution. Document score in metadata. No hard failure for borderline scores. |
| **Q-2.3** | Should cluster visualization include original category labels? | Low | Epic 2 Dev + UX | Before Epic 7 (report generation) | **OPEN**: Defer to Epic 7. Epic 2 generates basic plot, Epic 7 can enhance for report. |
| **Q-2.4** | How to handle if Epic 1 embeddings are updated (re-run)? | Low | Epic 1 + Epic 2 | Before Epic 3 | **RESOLVED**: Epic 2 script is idempotent - re-running overwrites outputs. Epic 3 onwards use latest clusterings. |
| **Q-2.5** | Should we validate cluster quality against ground truth AG News labels? | Medium | Epic 2 Dev | Before validation | **RESOLVED**: Yes - compute cluster purity as supplementary metric. Add to cluster_metadata.json. Not a pass/fail criterion. |
| **Q-2.6** | What format for cluster labeling (Epic 2 story 2-5)? | Medium | Epic 2 Story 2-5 | During story implementation | **OPEN**: Will be defined in story 2-5 implementation. Likely: analyze top terms per cluster, assign semantic labels. |

**Open Questions Status:**
- **Resolved**: 4 questions
- **Open**: 2 questions (low/medium priority, not blockers)
- **Blockers**: None

---

### Dependencies on Other Epics

**Epic 1 → Epic 2:**
- **CRITICAL**: train_embeddings.npy (120K × 768 float32)
- **Required**: metadata.json (document count, dimensions)
- **Optional**: Category labels for visualization enhancement

**Epic 2 → Epic 3:**
- **CRITICAL**: cluster_assignments.csv (document-to-cluster mapping)
- **CRITICAL**: centroids.npy (4 × 768 cluster centers)

**Epic 2 → Epic 4:**
- **CRITICAL**: centroids.npy (for query classification)

**Epic 2 → Epic 6, 7:**
- **Required**: cluster_metadata.json (metrics for report)
- **Required**: cluster_pca_visualization.png (figure for report)

---

### Technical Debt / Future Improvements

**Identified Technical Debt:**
- **TD-2.1**: Hardcoded K=4 (no dynamic K selection)
  - **Impact**: Low - K=4 is project requirement
  - **Effort**: Medium (implement Elbow method)
  - **Priority**: Post-MVP

- **TD-2.2**: No cluster stability validation (bootstrap, cross-validation)
  - **Impact**: Low - academic project, not production
  - **Effort**: High (requires multiple clustering runs)
  - **Priority**: Post-MVP

- **TD-2.3**: PCA 2D projection loses information (768D → 2D)
  - **Impact**: Low - visualization only, not used for decisions
  - **Effort**: Low (could add 3D plot or t-SNE alternative)
  - **Priority**: Post-MVP

**Future Enhancements (Post-Epic):**
1. **Experiment with different K values** (K=3, 5, 6) - Academic exploration
2. **Try alternative clustering algorithms** (DBSCAN, hierarchical) - Comparative study
3. **Implement cluster labeling automation** (top terms, LLM-based naming) - UX improvement
4. **Add cluster evolution tracking** (how clusters change if dataset grows) - Advanced analysis

---

## Test Strategy Summary

### Test Coverage Overview

| Test Level | Scope | Tools | Execution |
|------------|-------|-------|-----------|
| **Unit Tests** | Individual modules (KMeansClustering, ClusteringMetrics, ClusterPlots) | pytest | Automated (pre-commit, CI/CD) |
| **Integration Tests** | End-to-end clustering pipeline | pytest + subprocess | Automated (CI/CD) |
| **Performance Tests** | Execution time, memory usage | pytest + profiling | Automated (nightly) |
| **Acceptance Tests** | Validate all 10 ACs | Custom validation script | Manual + automated |
| **Reproducibility Tests** | Multiple runs → identical outputs | pytest + file comparison | Automated (CI/CD) |

---

### Testing Philosophy

**For Epic 2, we prioritize:**
1. **Correctness**: Cluster assignments are valid, centroids are accurate
2. **Reproducibility**: Same inputs → same outputs (critical for academic work)
3. **Performance**: Meets <10 minute target
4. **Integration**: Downstream epics can consume outputs successfully

**Test Data Strategy:**
- **Unit tests**: Small synthetic datasets (1000 documents, random embeddings)
- **Integration tests**: Full AG News embeddings (120K documents, real data)
- **Edge cases**: Empty clusters (shouldn't happen), NaN values, wrong dtypes

---

### Test Execution Plan

**Phase 1: Development Testing (Per Story)**
```bash
# Run unit tests for current module
pytest tests/test_clustering.py -v

# Run integration test for full pipeline
pytest tests/test_clustering_integration.py -v

# Quick smoke test
python scripts/02_train_clustering.py  # Should complete in <10 min
```

**Phase 2: Epic Validation (After All Stories Complete)**
```bash
# Run full test suite
pytest tests/ -v --cov=src.models.clustering --cov=src.evaluation.clustering_metrics

# Run acceptance validation
python tests/validate_epic_2_complete.py

# Run reproducibility test
./tests/test_reproducibility.sh
```

**Phase 3: Integration with Downstream Epics**
```bash
# Validate Epic 3 can load cluster_assignments.csv
python -c "import pandas as pd; df = pd.read_csv('data/processed/cluster_assignments.csv'); assert len(df) == 120000"

# Validate Epic 4 can load centroids.npy
python -c "import numpy as np; c = np.load('data/processed/centroids.npy'); assert c.shape == (4, 768)"
```

---

### Success Criteria for Epic 2 Completion

Epic 2 is **DONE** when:

1. ✅ All 10 acceptance criteria validated (see AC section above)
2. ✅ All unit tests passing (>90% code coverage)
3. ✅ Integration test passing (end-to-end pipeline works)
4. ✅ Performance test passing (execution <10 minutes)
5. ✅ Reproducibility test passing (3 runs → identical outputs)
6. ✅ Downstream epics can load and use Epic 2 outputs
7. ✅ Code reviewed and approved (if applicable)
8. ✅ Documentation complete (docstrings, README updated)
9. ✅ No open bugs or blockers
10. ✅ Epic 2 outputs committed to repository

**Final Validation Checklist:**
```
[ ] cluster_assignments.csv exists (120K rows, valid schema)
[ ] centroids.npy exists (4×768 float32, no NaN/Inf)
[ ] cluster_metadata.json exists (all required fields)
[ ] cluster_pca_visualization.png exists (300 DPI)
[ ] Silhouette Score >0.3
[ ] All tests passing
[ ] Reproducibility verified
[ ] Epic 3, 4 integration validated
```

---

**Epic 2 is now ready for implementation! 🚀**
