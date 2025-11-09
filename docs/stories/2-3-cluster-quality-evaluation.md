# Story 2.3: Cluster Quality Evaluation

Status: drafted

## Story

As a **data mining student**,
I want **to evaluate cluster quality using standard metrics**,
So that **I can prove the clustering produces good semantic separation**.

## Acceptance Criteria

### AC-1: Silhouette Score Calculation

**Given** K-Means clustering is complete
**When** I calculate the Silhouette Score
**Then**:
- ✅ Silhouette Score is computed for all 120K documents
- ✅ Score is calculated using `sklearn.metrics.silhouette_score`
- ✅ Target threshold: Silhouette Score >0.3 (good cluster separation)
- ✅ If score <0.3: log warning but continue execution
- ✅ Score is saved to `data/processed/cluster_metadata.json`

**Validation:**
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(embeddings, labels)
assert score > 0.3  # Target met
assert 'silhouette_score' in metadata
```

---

### AC-2: Davies-Bouldin Index Calculation

**Given** K-Means clustering is complete
**When** I calculate the Davies-Bouldin Index
**Then**:
- ✅ Davies-Bouldin Index is computed for all clusters
- ✅ Index is calculated using `sklearn.metrics.davies_bouldin_score`
- ✅ Lower values indicate better cluster separation (no hard threshold)
- ✅ Index is saved to `data/processed/cluster_metadata.json`

**Validation:**
```python
from sklearn.metrics import davies_bouldin_score
index = davies_bouldin_score(embeddings, labels)
assert 'davies_bouldin_index' in metadata
assert index > 0  # Valid range
```

---

### AC-3: Intra-Cluster Distance (Compactness)

**Given** K-Means clustering is complete
**When** I compute intra-cluster distances
**Then**:
- ✅ Average distance from documents to their cluster centroid is calculated for each cluster
- ✅ Overall intra-cluster distance (compactness metric) is computed
- ✅ Lower values indicate tighter, more compact clusters
- ✅ Per-cluster compactness is saved to `data/processed/cluster_metadata.json`

**Validation:**
```python
# For each cluster, compute mean distance to centroid
for cluster_id in range(4):
    cluster_mask = (labels == cluster_id)
    cluster_embeddings = embeddings[cluster_mask]
    distances = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)
    intra_cluster_dist = distances.mean()
    assert intra_cluster_dist > 0
```

---

### AC-4: Inter-Cluster Distance (Separation)

**Given** K-Means clustering is complete
**When** I compute inter-cluster distances
**Then**:
- ✅ Pairwise distances between all cluster centroids are calculated
- ✅ Minimum inter-cluster distance is identified (closest centroids)
- ✅ Maximum inter-cluster distance is identified (farthest centroids)
- ✅ Average inter-cluster distance is computed
- ✅ Higher values indicate better cluster separation
- ✅ Inter-cluster distances are saved to `data/processed/cluster_metadata.json`

**Validation:**
```python
# Compute pairwise centroid distances
from sklearn.metrics.pairwise import euclidean_distances
centroid_distances = euclidean_distances(centroids)
# Extract upper triangle (excluding diagonal)
inter_dists = centroid_distances[np.triu_indices(4, k=1)]
assert len(inter_dists) == 6  # 4 choose 2 = 6 pairs
assert inter_dists.min() > 0
```

---

### AC-5: Cluster Purity Against Ground Truth

**Given** K-Means clustering is complete and AG News ground truth categories are available
**When** I calculate cluster purity
**Then**:
- ✅ Cluster purity is computed by comparing with AG News ground truth categories (World, Sports, Business, Sci/Tech)
- ✅ For each cluster, identify the dominant AG News category
- ✅ Purity = (number of documents in dominant category) / (total documents in cluster)
- ✅ Overall purity (weighted average across clusters) is calculated
- ✅ Per-cluster purity breakdown is saved to `data/processed/cluster_metadata.json`
- ✅ Target: Average purity >0.7 (70% documents match dominant category)

**Validation:**
```python
# For each cluster, compute purity
for cluster_id in range(4):
    cluster_mask = (labels == cluster_id)
    cluster_categories = ground_truth_labels[cluster_mask]
    dominant_category = cluster_categories.mode()
    purity = (cluster_categories == dominant_category).mean()
    assert purity >= 0.25  # At least better than random (1/4)

# Overall purity (weighted by cluster size)
overall_purity = ...
assert overall_purity > 0.7  # Target threshold
```

---

### AC-6: Confusion Matrix (Cluster vs Ground Truth)

**Given** K-Means clustering is complete
**When** I generate a confusion matrix
**Then**:
- ✅ 4×4 confusion matrix is created:
  - Rows: AG News ground truth categories (World, Sports, Business, Sci/Tech)
  - Columns: Cluster IDs (0, 1, 2, 3)
  - Cells: Count of documents
- ✅ Matrix shows cluster-to-category alignment
- ✅ Confusion matrix is saved as numpy array to `data/processed/confusion_matrix.npy`
- ✅ Human-readable confusion matrix is logged to console

**Validation:**
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ground_truth_labels, labels)
assert cm.shape == (4, 4)
assert cm.sum() == 120000  # All documents accounted for
```

---

### AC-7: Cluster Size Validation

**Given** K-Means clustering is complete
**When** I validate cluster size distribution
**Then**:
- ✅ Cluster sizes are computed using `np.bincount(labels)`
- ✅ No cluster has <10% of documents (extreme imbalance)
- ✅ No cluster has >50% of documents (single cluster dominance)
- ✅ If imbalance detected: log warning with cluster sizes
- ✅ Cluster sizes are saved to `data/processed/cluster_metadata.json`

**Validation:**
```python
cluster_sizes = np.bincount(labels)
assert len(cluster_sizes) == 4
assert all(size >= 0.1 * 120000 for size in cluster_sizes)  # No cluster <10%
assert all(size <= 0.5 * 120000 for size in cluster_sizes)  # No cluster >50%
```

---

### AC-8: Cluster Quality Report Export

**Given** All cluster quality metrics are computed
**When** I export the results
**Then**:
- ✅ Results are saved to `data/processed/cluster_quality.json` with structure:
```json
{
  "silhouette_score": 0.35,
  "davies_bouldin_index": 1.2,
  "intra_cluster_distance": {
    "cluster_0": 45.2,
    "cluster_1": 43.8,
    "cluster_2": 46.1,
    "cluster_3": 44.5,
    "overall": 44.9
  },
  "inter_cluster_distance": {
    "min": 85.3,
    "max": 112.7,
    "mean": 98.4,
    "pairwise": [...]
  },
  "cluster_purity": {
    "cluster_0": 0.85,
    "cluster_1": 0.82,
    "cluster_2": 0.78,
    "cluster_3": 0.81,
    "overall": 0.82
  },
  "cluster_sizes": [30000, 29500, 30500, 30000]
}
```
- ✅ JSON formatted with `indent=2` (human-readable)
- ✅ All metrics included with descriptive keys
- ✅ Results also appended to existing `data/processed/cluster_metadata.json` from Story 2.2

**Validation:**
```python
import json
with open('data/processed/cluster_quality.json') as f:
    quality = json.load(f)
required_keys = {'silhouette_score', 'davies_bouldin_index', 'cluster_purity', 'cluster_sizes'}
assert set(quality.keys()) >= required_keys
```

---

### AC-9: Logging and Observability

**Given** Cluster quality evaluation is running
**When** Major metrics are computed
**Then**:
- ✅ Emoji-prefixed logs for visual clarity:
  - INFO: "📊 Calculating Silhouette Score..."
  - SUCCESS: "✅ Silhouette Score: 0.347 (target: >0.3)"
  - INFO: "📊 Computing Davies-Bouldin Index..."
  - SUCCESS: "✅ Davies-Bouldin Index: 1.234 (lower is better)"
  - INFO: "📊 Evaluating cluster purity..."
  - SUCCESS: "✅ Cluster purity: 82% (target: >70%)"
  - WARNING: "⚠️ Silhouette Score 0.287 below target 0.3 (still acceptable)"
  - WARNING: "⚠️ Cluster 2 imbalanced: 62% of documents"
- ✅ All major operations logged with clear descriptions
- ✅ Summary logged at completion:
```
✅ Cluster Quality Evaluation Complete
   - Silhouette Score: 0.347 (Good)
   - Davies-Bouldin Index: 1.234
   - Cluster Purity: 82%
   - Cluster Balance: Balanced
```

---

### AC-10: Error Handling

**Given** The cluster quality evaluation script is executed
**When** Errors may occur
**Then**:
- ✅ Clear error if cluster assignments file missing (suggests running Story 2.2 script)
- ✅ Clear error if embeddings file missing (suggests running Story 2.1 script)
- ✅ Clear error if ground truth labels unavailable
- ✅ Validation error if cluster labels have unexpected range (not [0,3])
- ✅ Validation error if embedding shapes don't match label count
- ✅ Automatic directory creation if output paths don't exist

**Validation:**
```python
# Test missing cluster assignments
if not Path('data/processed/cluster_assignments.csv').exists():
    raise FileNotFoundError(
        "Cluster assignments not found: data/processed/cluster_assignments.csv\n"
        "Run 'python scripts/02_train_clustering.py' first"
    )

# Test shape mismatch
assert len(embeddings) == len(labels), "Embeddings and labels count mismatch"
```

---

## Tasks / Subtasks

- [ ] Implement ClusteringMetrics class in `src/evaluation/clustering_metrics.py` (AC: #1, #2, #3, #4, #5, #6, #7)
  - [ ] Create ClusteringMetrics class with `__init__` accepting embeddings, labels, centroids, ground_truth
  - [ ] Implement `calculate_silhouette_score()` method
  - [ ] Implement `calculate_davies_bouldin_index()` method
  - [ ] Implement `calculate_intra_cluster_distance()` method (per-cluster and overall)
  - [ ] Implement `calculate_inter_cluster_distance()` method (pairwise centroid distances)
  - [ ] Implement `calculate_cluster_purity()` method (compare with ground truth)
  - [ ] Implement `generate_confusion_matrix()` method (4×4 cluster vs category)
  - [ ] Implement `validate_cluster_balance()` method (check for imbalance)
  - [ ] Add type hints: `calculate_silhouette_score(self) -> float`
  - [ ] Add Google-style docstrings with usage examples for all methods
  - [ ] Return structured dict with all metrics: `evaluate_all() -> dict`

- [ ] Create cluster quality evaluation script `scripts/03_evaluate_clustering.py` (AC: #8, #9, #10)
  - [ ] Import required modules: Config, Paths, ClusteringMetrics, logger
  - [ ] Implement set_seed(42) at script start for reproducibility
  - [ ] Load configuration from config.yaml
  - [ ] Setup logging with emoji prefixes
  - [ ] Load cluster assignments from `data/processed/cluster_assignments.csv`
  - [ ] Load embeddings from `data/embeddings/train_embeddings.npy`
  - [ ] Load centroids from `data/processed/centroids.npy`
  - [ ] Load ground truth labels from AG News dataset
  - [ ] Validate inputs: file existence, shape consistency, label range [0,3]
  - [ ] If files missing, raise FileNotFoundError with clear message and next steps
  - [ ] Initialize ClusteringMetrics with loaded data
  - [ ] Call `evaluate_all()` to compute all metrics
  - [ ] Log each metric as it's computed (Silhouette, Davies-Bouldin, purity, etc.)
  - [ ] Save results to `data/processed/cluster_quality.json` with indent=2
  - [ ] Append metrics to existing `cluster_metadata.json` from Story 2.2
  - [ ] Save confusion matrix to `data/processed/confusion_matrix.npy`
  - [ ] Create output directories if they don't exist
  - [ ] Log all save operations with file paths
  - [ ] Display final summary with all key metrics
  - [ ] Handle warnings for low Silhouette (<0.3) or cluster imbalance

- [ ] Implement Silhouette Score calculation (AC: #1)
  - [ ] Use `sklearn.metrics.silhouette_score(embeddings, labels)`
  - [ ] Pass metric='euclidean' for consistency with K-Means
  - [ ] Validate score is in range [-1, 1]
  - [ ] Log score with comparison to target (0.3)
  - [ ] Return float value

- [ ] Implement Davies-Bouldin Index calculation (AC: #2)
  - [ ] Use `sklearn.metrics.davies_bouldin_score(embeddings, labels)`
  - [ ] Validate index is non-negative
  - [ ] Log index value (lower is better, no hard threshold)
  - [ ] Return float value

- [ ] Implement intra-cluster distance calculation (AC: #3)
  - [ ] For each cluster (0-3):
    - Extract cluster embeddings using cluster mask
    - Compute distances to cluster centroid using `np.linalg.norm`
    - Calculate mean distance (compactness metric)
  - [ ] Compute overall weighted average intra-cluster distance
  - [ ] Return dict with per-cluster and overall values

- [ ] Implement inter-cluster distance calculation (AC: #4)
  - [ ] Use `sklearn.metrics.pairwise.euclidean_distances(centroids)`
  - [ ] Extract upper triangle (6 pairwise distances for 4 clusters)
  - [ ] Compute min, max, mean inter-cluster distances
  - [ ] Return dict with summary statistics

- [ ] Implement cluster purity calculation (AC: #5)
  - [ ] Load ground truth AG News labels (World=0, Sports=1, Business=2, Sci/Tech=3)
  - [ ] For each cluster:
    - Extract ground truth labels for cluster documents
    - Find dominant category (mode)
    - Calculate purity = count(dominant) / count(total)
  - [ ] Compute overall weighted purity (cluster size weights)
  - [ ] Return dict with per-cluster and overall purity

- [ ] Implement confusion matrix generation (AC: #6)
  - [ ] Use `sklearn.metrics.confusion_matrix(ground_truth, labels)`
  - [ ] Validate shape is (4, 4)
  - [ ] Validate sum equals total document count (120K)
  - [ ] Log confusion matrix to console in readable format
  - [ ] Save as numpy array to `data/processed/confusion_matrix.npy`

- [ ] Implement cluster balance validation (AC: #7)
  - [ ] Compute cluster sizes using `np.bincount(labels)`
  - [ ] Check if any cluster <10% of data (12K documents)
  - [ ] Check if any cluster >50% of data (60K documents)
  - [ ] If imbalance detected: log warning with cluster sizes
  - [ ] Return bool (balanced) and cluster_sizes dict

- [ ] Test cluster quality evaluation (AC: #1-#10)
  - [ ] Unit test: ClusteringMetrics methods on small synthetic dataset (1000 samples)
  - [ ] Unit test: Verify Silhouette Score in expected range [-1, 1]
  - [ ] Unit test: Verify Davies-Bouldin Index > 0
  - [ ] Unit test: Verify purity in range [0, 1]
  - [ ] Integration test: Run full script on actual cluster results from Story 2.2
  - [ ] Integration test: Verify all outputs exist and have correct schema
  - [ ] Integration test: Verify Silhouette Score >0.3 (target met)
  - [ ] Integration test: Verify cluster purity >0.7 (target met)
  - [ ] Negative test: Missing cluster assignments → FileNotFoundError
  - [ ] Negative test: Missing embeddings → FileNotFoundError
  - [ ] Negative test: Shape mismatch → ValueError

- [ ] Update project documentation (AC: all)
  - [ ] Update README.md with cluster quality evaluation script usage
  - [ ] Document script usage: `python scripts/03_evaluate_clustering.py`
  - [ ] Document expected outputs: cluster_quality.json, confusion_matrix.npy
  - [ ] Document key metrics and their interpretations
  - [ ] Add troubleshooting section for common errors
  - [ ] Document metric thresholds (Silhouette >0.3, Purity >0.7)

## Dev Notes

### Architecture Alignment

This story implements the **Cluster Quality Evaluation** component defined in the architecture. It integrates with:

1. **Cookiecutter Data Science Structure**: Follows src/evaluation/ for metrics logic, scripts/ for execution
2. **Story 2.2 Outputs**: Consumes cluster assignments, centroids, metadata from K-Means clustering
3. **Story 2.1 Outputs**: Uses embeddings from `data/embeddings/train_embeddings.npy`
4. **Configuration System**: Uses config.yaml for metric thresholds
5. **AG News Ground Truth**: Compares cluster assignments with original category labels
6. **Metrics Architecture**: Produces standardized quality metrics (Silhouette, Davies-Bouldin, purity)

**Constraints Applied:**
- **Performance**: Metric calculation <3 minutes for 120K documents (NFR-1 from PRD)
- **Reproducibility**: No randomization involved, deterministic metrics
- **Logging**: Uses emoji-prefixed logging (📊, ✅, ⚠️, ❌) from utils/logger.py
- **Error Handling**: Validates input file existence and data schema before computation

**Architectural Patterns Followed:**
- Initialization Order: set_seed → load config → setup logger → validate → execute
- Data Loading: Check file exists → load → validate → process
- File Naming: snake_case for modules (clustering_metrics.py), PascalCase for classes (ClusteringMetrics)
- Configuration Access: No hardcoded values, all thresholds from config.yaml

### Cluster Quality Metrics Strategy

**Why These Metrics:**

**1. Silhouette Score (Target: >0.3)**
- Measures how similar a document is to its own cluster vs. other clusters
- Range: [-1, 1], where >0 = good, >0.3 = clear separation, >0.5 = strong separation
- Interpretable: High score = documents well-matched to clusters
- Computational: O(n²) for full dataset, but scikit-learn optimizes for large data

**2. Davies-Bouldin Index (Lower is Better)**
- Ratio of within-cluster to between-cluster distances
- Range: [0, ∞), where lower = better separation
- No hard threshold: context-dependent, compare across runs
- Complements Silhouette: captures cluster compactness vs separation

**3. Intra-Cluster Distance (Compactness)**
- Average distance from documents to their cluster centroid
- Lower values = tighter, more coherent clusters
- Per-cluster metric helps identify loose clusters
- Directly interpretable in embedding space

**4. Inter-Cluster Distance (Separation)**
- Pairwise distances between cluster centroids
- Higher values = better separated clusters
- Minimum distance identifies "closest" clusters (potential overlap)
- Maximum distance identifies "farthest" clusters (most distinct)

**5. Cluster Purity (Target: >70%)**
- Percentage of documents matching dominant AG News category per cluster
- Range: [0, 1], where 1.0 = perfect alignment
- Validates semantic coherence: clusters capture meaningful categories
- Academic validation: connects unsupervised clustering to ground truth

**6. Confusion Matrix**
- Visual representation of cluster-category alignment
- Reveals which categories are confused (cross-cluster overlap)
- Diagonal dominance = good cluster-category correspondence
- Foundation for Epic 7 visualization (heatmap)

**Expected Behavior:**
- Silhouette >0.3 indicates clusters are well-separated and internally cohesive
- Davies-Bouldin Index ~1.0-1.5 typical for good clustering
- Cluster purity >70% means clusters align with AG News categories
- Balanced cluster sizes (no single cluster dominating) validates K-Means performance
- If metrics below targets: acceptable for MVP, document in report

### Data Models and Contracts

**Input Data:**
```python
# Embeddings (from Story 2.1)
Type: np.ndarray
Shape: (120000, 768)
Dtype: float32
Source: data/embeddings/train_embeddings.npy
Validation: Check shape[1] == 768, dtype == float32

# Cluster Labels (from Story 2.2)
Type: np.ndarray
Shape: (120000,)
Dtype: int32
Values: 0, 1, 2, 3
Source: data/processed/cluster_assignments.csv (column: cluster_id)
Validation: All values in [0, 3], no missing values

# Cluster Centroids (from Story 2.2)
Type: np.ndarray
Shape: (4, 768)
Dtype: float32
Source: data/processed/centroids.npy
Validation: Shape == (4, 768), dtype == float32, no NaN/Inf

# Ground Truth Labels (from AG News dataset)
Type: np.ndarray
Shape: (120000,)
Dtype: int32
Values: 0 (World), 1 (Sports), 2 (Business), 3 (Sci/Tech)
Source: Hugging Face datasets AG News train split
Validation: All values in [0, 3], count == 120000
```

**Output Data:**
```python
# Cluster Quality Metrics
Type: dict (JSON)
Schema:
{
  "silhouette_score": float,           # Target: >0.3
  "davies_bouldin_index": float,       # Lower is better
  "intra_cluster_distance": {
    "cluster_0": float,
    "cluster_1": float,
    "cluster_2": float,
    "cluster_3": float,
    "overall": float
  },
  "inter_cluster_distance": {
    "min": float,
    "max": float,
    "mean": float,
    "pairwise": [float, ...]  # 6 values for 4 clusters
  },
  "cluster_purity": {
    "cluster_0": float,
    "cluster_1": float,
    "cluster_2": float,
    "cluster_3": float,
    "overall": float
  },
  "cluster_sizes": [int, int, int, int]
}
Storage: data/processed/cluster_quality.json

# Confusion Matrix
Type: np.ndarray
Shape: (4, 4)
Dtype: int64
Storage: data/processed/confusion_matrix.npy
Rows: Ground truth AG News categories [World, Sports, Business, Sci/Tech]
Columns: Cluster IDs [0, 1, 2, 3]
```

**API Contracts:**
```python
class ClusteringMetrics:
    def __init__(
        self,
        embeddings: np.ndarray,     # (n_documents, 768) float32
        labels: np.ndarray,          # (n_documents,) int32
        centroids: np.ndarray,       # (4, 768) float32
        ground_truth: np.ndarray     # (n_documents,) int32
    ):
        """
        Initialize cluster quality evaluation.

        Args:
            embeddings: Document embeddings
            labels: Cluster assignments from K-Means
            centroids: Cluster centroids from K-Means
            ground_truth: AG News category labels (0-3)
        """

    def calculate_silhouette_score(self) -> float:
        """
        Calculate Silhouette Score for cluster quality.

        Returns:
            Silhouette Score in range [-1, 1]

        Target: >0.3 for good cluster separation
        """

    def calculate_davies_bouldin_index(self) -> float:
        """
        Calculate Davies-Bouldin Index for cluster quality.

        Returns:
            Davies-Bouldin Index (lower is better)
        """

    def calculate_cluster_purity(self) -> dict:
        """
        Calculate cluster purity against ground truth AG News labels.

        Returns:
            dict with per-cluster and overall purity scores

        Target: Overall purity >0.7
        """

    def evaluate_all(self) -> dict:
        """
        Compute all cluster quality metrics.

        Returns:
            Comprehensive metrics dict with all quality indicators
        """
```

### Project Structure Notes

After completion, the following files will be created/modified:

**New Files:**
- `src/evaluation/__init__.py` - Package init (if doesn't exist)
- `src/evaluation/clustering_metrics.py` - ClusteringMetrics class
- `scripts/03_evaluate_clustering.py` - Orchestration script for quality evaluation
- `data/processed/cluster_quality.json` - Cluster quality metrics (Silhouette, purity, etc.)
- `data/processed/confusion_matrix.npy` - 4×4 confusion matrix

**Modified Files:**
- `data/processed/cluster_metadata.json` - Append quality metrics to existing metadata from Story 2.2

**Expected Directory Structure:**
```
context-aware-multi-agent-system/
├── scripts/
│   ├── 01_generate_embeddings.py       # EXISTING: From Story 2.1
│   ├── 02_train_clustering.py          # EXISTING: From Story 2.2
│   └── 03_evaluate_clustering.py       # NEW: Quality evaluation
├── data/
│   ├── embeddings/                     # EXISTING: From Story 2.1
│   │   └── train_embeddings.npy        # INPUT: 120K embeddings
│   └── processed/                      # EXISTING: From Story 2.2
│       ├── cluster_assignments.csv     # INPUT: Cluster labels
│       ├── centroids.npy               # INPUT: 4 centroids
│       ├── cluster_metadata.json       # MODIFIED: Add quality metrics
│       ├── cluster_quality.json        # NEW: Quality metrics
│       └── confusion_matrix.npy        # NEW: 4×4 matrix
├── src/
│   ├── evaluation/                     # NEW: Metrics module
│   │   ├── __init__.py                 # NEW: Package init
│   │   └── clustering_metrics.py       # NEW: ClusteringMetrics class
│   └── utils/
│       ├── logger.py                   # EXISTING: Reused for logging
│       └── reproducibility.py          # EXISTING: Reused for set_seed(42)
└── config.yaml                         # EXISTING: May add metric thresholds
```

### Testing Standards

**Unit Tests:**
```python
# Test ClusteringMetrics.calculate_silhouette_score() on small dataset
def test_silhouette_score_calculation():
    embeddings = np.random.randn(1000, 768).astype(np.float32)
    labels = np.random.randint(0, 4, 1000).astype(np.int32)
    centroids = np.random.randn(4, 768).astype(np.float32)
    ground_truth = np.random.randint(0, 4, 1000).astype(np.int32)

    metrics = ClusteringMetrics(embeddings, labels, centroids, ground_truth)
    score = metrics.calculate_silhouette_score()

    assert -1.0 <= score <= 1.0
    assert isinstance(score, float)

# Test Davies-Bouldin Index
def test_davies_bouldin_index():
    metrics = ClusteringMetrics(embeddings, labels, centroids, ground_truth)
    index = metrics.calculate_davies_bouldin_index()
    assert index > 0  # Valid range
    assert isinstance(index, float)

# Test cluster purity
def test_cluster_purity():
    metrics = ClusteringMetrics(embeddings, labels, centroids, ground_truth)
    purity = metrics.calculate_cluster_purity()

    assert 'overall' in purity
    assert 0 <= purity['overall'] <= 1.0
    assert all(0 <= purity[f'cluster_{i}'] <= 1.0 for i in range(4))
```

**Integration Tests:**
```python
# Test full quality evaluation pipeline
def test_full_quality_evaluation():
    result = subprocess.run(['python', 'scripts/03_evaluate_clustering.py'],
                           capture_output=True)
    assert result.returncode == 0

    # Verify outputs exist
    assert Path('data/processed/cluster_quality.json').exists()
    assert Path('data/processed/confusion_matrix.npy').exists()

    # Verify metrics in valid ranges
    with open('data/processed/cluster_quality.json') as f:
        quality = json.load(f)
    assert -1.0 <= quality['silhouette_score'] <= 1.0
    assert quality['davies_bouldin_index'] > 0
    assert 0 <= quality['cluster_purity']['overall'] <= 1.0

# Test target metrics achieved
def test_quality_targets_met():
    with open('data/processed/cluster_quality.json') as f:
        quality = json.load(f)

    # Silhouette Score >0.3
    assert quality['silhouette_score'] > 0.3, "Silhouette Score below target"

    # Cluster purity >0.7
    assert quality['cluster_purity']['overall'] > 0.7, "Purity below target"

    # Cluster balance (no cluster <10% or >50%)
    cluster_sizes = quality['cluster_sizes']
    assert all(size >= 0.1 * 120000 for size in cluster_sizes)
    assert all(size <= 0.5 * 120000 for size in cluster_sizes)
```

**Performance Tests:**
```python
# Test quality evaluation completes in <3 minutes
def test_performance_targets():
    import time
    start = time.time()
    subprocess.run(['python', 'scripts/03_evaluate_clustering.py'])
    elapsed = time.time() - start
    assert elapsed < 180  # 3 minutes max
```

**Expected Test Coverage:**
- ClusteringMetrics class: all metric calculation methods
- Silhouette Score: range validation, target threshold
- Davies-Bouldin Index: non-negative validation
- Cluster purity: range [0,1], target >0.7
- Confusion matrix: shape (4,4), sum == 120K
- Error handling: missing files, shape mismatches
- Performance: execution time <3 minutes

### Learnings from Previous Story

**From Story 2-2-k-means-clustering-implementation (Status: done):**

- ✅ **Cluster Outputs Available**: Use cluster results from Story 2.2
  - Cluster assignments: `data/processed/cluster_assignments.csv`
  - Centroids: `data/processed/centroids.npy` (4 × 768 float32)
  - Metadata: `data/processed/cluster_metadata.json` (basic stats)
  - Validation: Check files exist before loading

- ✅ **Configuration Pattern**: Follow established config access pattern from Story 2.2
  - Use `config.get("evaluation.silhouette_threshold")` for threshold (if configured)
  - Use `config.get("evaluation.purity_threshold")` for purity target
  - Use `paths.data_processed` for processed data directory
  - Add evaluation section to config.yaml if needed

- ✅ **Logging Pattern**: Follow emoji-prefixed logging from Story 2.2
  - INFO: "📊 Calculating Silhouette Score..."
  - SUCCESS: "✅ Silhouette Score: 0.347 (target: >0.3)"
  - INFO: "📊 Computing Davies-Bouldin Index..."
  - SUCCESS: "✅ Davies-Bouldin Index: 1.234"
  - INFO: "📊 Evaluating cluster purity..."
  - SUCCESS: "✅ Cluster purity: 82% (target: >70%)"
  - WARNING: "⚠️ Silhouette Score 0.287 below target 0.3"
  - WARNING: "⚠️ Cluster 2 imbalanced: 62% of documents"
  - ERROR: "❌ Quality evaluation failed: {error_message}"

- ✅ **Reproducibility Pattern**: Reuse set_seed() from Story 2.2
  - Call set_seed(42) at script start (for consistency)
  - Metrics are deterministic (no randomization)
  - Ensures reproducible quality scores

- ✅ **Error Handling Pattern**: Follow Story 2.2 error handling approach
  - Clear error messages with troubleshooting guidance
  - FileNotFoundError if cluster assignments missing: suggest running Story 2.2 script
  - FileNotFoundError if embeddings missing: suggest running Story 2.1 script
  - ValueError for validation failures with helpful context
  - Provide actionable next steps

- ✅ **Type Hints and Docstrings**: Maintain Story 2.2 documentation standards
  - All methods have full type hints
  - Google-style docstrings with usage examples
  - Example: `def calculate_silhouette_score(self) -> float:`

- ✅ **Data Validation Pattern**: Follow Story 2.2 validation approach
  - Pre-flight checks: file exists, shape correct, dtype correct, no NaN/Inf
  - Fail-fast with clear error messages
  - Log validation success for debugging

- ✅ **Directory Creation**: Follow Story 2.2 pattern for output directories
  - Use `Path.mkdir(parents=True, exist_ok=True)` to create directories
  - Create data/processed/ if it doesn't exist
  - No errors if directories already exist

- ✅ **Testing Pattern**: Follow Story 2.2 comprehensive test approach
  - Create `tests/epic2/test_clustering_metrics.py`
  - Map tests to acceptance criteria (AC-1, AC-2, etc.)
  - Use pytest.raises() for exception testing
  - Use pytest fixtures for test setup (temp directories, mock data)
  - Test both unit (small synthetic data) and integration (full dataset)

**Files to Reuse (DO NOT RECREATE):**
- `src/utils/logger.py` - Use for emoji-prefixed logging
- `src/utils/reproducibility.py` - Use set_seed(42) function
- `src/config.py` - Load config for thresholds (if configured)
- `data/embeddings/train_embeddings.npy` - Input from Story 2.1
- `data/processed/cluster_assignments.csv` - Input from Story 2.2
- `data/processed/centroids.npy` - Input from Story 2.2
- `data/processed/cluster_metadata.json` - Extend with quality metrics

**Key Services from Previous Stories:**
- **Config class** (Story 1.2): Configuration management with get() method
- **Paths class** (Story 1.2): Path resolution
- **set_seed()** (Story 1.1): Reproducibility enforcement (though metrics are deterministic)
- **Logger** (Story 1.2): Emoji-prefixed structured logging
- **KMeansClustering** (Story 2.2): Cluster labels and centroids available

**Technical Debt from Story 2.2:**
- None affecting this story - Story 2.2 is complete and approved

**Review Findings from Story 2.2 to Apply:**
- ✅ Use comprehensive docstrings with usage examples
- ✅ Add type hints to all method signatures
- ✅ Include explicit validation checks with informative error messages
- ✅ Log all major operations for debugging
- ✅ Write tests covering all acceptance criteria
- ✅ Create helper functions to avoid code duplication

**New Patterns to Establish:**
- **Metric Calculation Pattern**: Compute metric → validate range → log result → return value
- **Quality Report Pattern**: Collect all metrics → structure as dict → save as JSON with indent=2
- **Threshold Validation Pattern**: If metric < threshold: log warning, continue (don't fail)
- **Ground Truth Alignment**: Load AG News labels → align with cluster labels → compute purity

[Source: stories/2-2-k-means-clustering-implementation.md#Dev-Agent-Record]

### References

- [Source: docs/tech-spec-epic-2.md#AC-2 - Cluster Quality Metrics]
- [Source: docs/tech-spec-epic-2.md#Detailed Design → Services and Modules → ClusteringMetrics]
- [Source: docs/tech-spec-epic-2.md#Data Models and Contracts → Output Data Models]
- [Source: docs/epics.md#Story 2.3 - Cluster Quality Evaluation]
- [Source: docs/PRD.md#FR-4 - Cluster Quality Evaluation]
- [Source: docs/architecture.md#Cluster Quality Evaluation]
- [Source: stories/2-2-k-means-clustering-implementation.md#Cluster Results Available]

## Dev Agent Record

### Context Reference

<!-- Path(s) to story context XML will be added here by context workflow -->

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
