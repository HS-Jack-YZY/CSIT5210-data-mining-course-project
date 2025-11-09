# Epic Technical Specification: Alternative Clustering Algorithms Exploration

Date: 2025-11-09
Author: Jack YUAN
Epic ID: 5
Status: Completed

---

## Overview

This epic implements and evaluates three alternative clustering algorithms (DBSCAN, Hierarchical Clustering, and Gaussian Mixture Models) to compare with the K-Means baseline implemented in Epic 2. The goal is to provide a comprehensive scientific analysis of which clustering approaches are most suitable for high-dimensional text embedding clustering tasks.

**Context:** Epic 2's K-Means implementation yielded poor results (Silhouette Score ≈0.0008, cluster purity ≈25.3%), indicating that the algorithm failed to discover meaningful semantic structure in the AG News embeddings. This epic systematically explores whether alternative clustering paradigms can achieve better performance or whether the task itself is fundamentally challenging due to the curse of dimensionality in 768-dimensional embedding space.

**Value Proposition:** By implementing and comparing four different clustering algorithms with rigorous evaluation, this epic demonstrates deep understanding of data mining methodologies and provides empirical evidence for algorithm selection in text clustering applications. The comprehensive comparison report will serve as a valuable academic contribution showing professional scientific integrity—documenting both successes and failures with equal rigor.

## Objectives and Scope

**In Scope:**

- **DBSCAN Implementation (Story 5.1):**
  - Apply density-based clustering to AG News embeddings
  - Parameter tuning for eps and min_samples
  - Handle noise points and variable cluster discovery
  - Evaluate with cosine distance metric (appropriate for text)

- **Hierarchical Clustering Implementation (Story 5.2):**
  - Apply agglomerative clustering with multiple linkage methods (ward, complete, average, single)
  - Generate dendrogram visualization showing hierarchical structure
  - Compare linkage methods for best performance
  - Evaluate cluster quality at n_clusters=4 for fair comparison

- **Gaussian Mixture Model Implementation (Story 5.3):**
  - Apply probabilistic soft clustering
  - Test multiple covariance types (full, tied, diag, spherical)
  - Extract both hard and soft cluster assignments
  - Analyze assignment confidence and uncertainty patterns

- **Comprehensive Algorithm Comparison (Story 5.4):**
  - Create comparison matrix across all algorithms and metrics
  - Generate side-by-side PCA visualizations
  - Analyze strengths, weaknesses, and use cases for each algorithm
  - Document recommendations for text clustering tasks
  - Produce comprehensive comparison report

**Out of Scope:**

- Spectral clustering or other advanced methods (time constraints)
- Ensemble clustering approaches (complexity beyond 3-day extension)
- Cross-validation of cluster stability (would require multiple runs)
- Implementation of custom clustering algorithms (use scikit-learn only)
- Testing on datasets other than AG News (focus on current dataset)
- Production optimization or deployment (academic study only)

## System Architecture Alignment

This epic extends the existing Cookiecutter Data Science project structure established in Epic 1 with minimal architectural changes:

**Reused Components:**
- `data/embeddings/train_embeddings.npy` - Same 120K × 768 embeddings from Epic 2
- `src/evaluation/clustering_metrics.py` - Same evaluation metrics (Silhouette Score, Davies-Bouldin Index, cluster purity)
- `src/visualization/cluster_plots.py` - Same PCA visualization infrastructure
- `config.yaml` - Extended with new algorithm-specific parameters

**New Components:**
- `src/models/dbscan_clustering.py` - DBSCAN implementation wrapper
- `src/models/hierarchical_clustering.py` - Hierarchical clustering wrapper
- `src/models/gmm_clustering.py` - Gaussian Mixture Model wrapper
- `src/evaluation/algorithm_comparison.py` - Cross-algorithm comparison utilities
- `scripts/06_alternative_clustering.py` - Main execution script for all three algorithms
- `scripts/07_compare_algorithms.py` - Comprehensive comparison and reporting
- `data/processed/dbscan_assignments.csv` - DBSCAN cluster labels
- `data/processed/hierarchical_assignments.csv` - Hierarchical cluster labels
- `data/processed/gmm_assignments.csv` - GMM cluster labels
- `reports/figures/algorithm_comparison.png` - Side-by-side algorithm visualizations
- `reports/clustering_comparison.md` - Comprehensive comparison report

**Architecture Constraints:**
- Must use same embeddings as K-Means (reproducibility)
- Must follow same evaluation methodology (fair comparison)
- Must maintain consistent data formats (.npy for arrays, .csv for assignments, .json for metrics)
- Must use scikit-learn implementations only (no custom algorithms)
- Must preserve random_state=42 for reproducibility where applicable

## Detailed Design

### Services and Modules

| Module | Responsibility | Inputs | Outputs | Owner |
|--------|---------------|--------|---------|-------|
| **DBSCANClustering** | Density-based clustering implementation | Embeddings (np.ndarray), eps (float), min_samples (int) | Labels (np.ndarray), core_samples (np.ndarray), metrics (dict) | Story 5.1 |
| **HierarchicalClustering** | Agglomerative clustering with dendrogram | Embeddings (np.ndarray), n_clusters (int), linkage (str) | Labels (np.ndarray), dendrogram_data (dict), metrics (dict) | Story 5.2 |
| **GMMClustering** | Probabilistic soft clustering | Embeddings (np.ndarray), n_components (int), covariance_type (str) | Labels (np.ndarray), probabilities (np.ndarray), BIC/AIC (float), metrics (dict) | Story 5.3 |
| **AlgorithmComparison** | Cross-algorithm comparison utilities | All algorithm results (dict) | Comparison matrix (DataFrame), visualization (Figure), report (markdown) | Story 5.4 |
| **ParameterTuner** | Automated parameter tuning for DBSCAN | Embeddings, parameter ranges | Best parameters (dict), tuning results (DataFrame) | Story 5.1 |

### Data Models and Contracts

**DBSCAN Assignments:**
```python
# data/processed/dbscan_assignments.csv
{
    "document_id": int,           # Document index (0-119999)
    "cluster_id": int,            # Cluster label (-1 for noise, 0+ for clusters)
    "ground_truth_category": str, # AG News category (World/Sports/Business/Sci-Tech)
    "is_core_sample": bool        # Whether document is a core sample
}
```

**Hierarchical Clustering Assignments:**
```python
# data/processed/hierarchical_assignments.csv
{
    "document_id": int,           # Document index
    "cluster_id": int,            # Cluster label (0-3)
    "ground_truth_category": str, # AG News category
    "linkage_method": str         # Linkage method used (ward/complete/average/single)
}
```

**GMM Assignments:**
```python
# data/processed/gmm_assignments.csv
{
    "document_id": int,                      # Document index
    "cluster_id": int,                       # Hard assignment (0-3)
    "cluster_0_prob": float,                 # Probability of belonging to cluster 0
    "cluster_1_prob": float,                 # Probability of belonging to cluster 1
    "cluster_2_prob": float,                 # Probability of belonging to cluster 2
    "cluster_3_prob": float,                 # Probability of belonging to cluster 3
    "assignment_confidence": float,          # Max probability (confidence score)
    "ground_truth_category": str,            # AG News category
    "covariance_type": str                   # Covariance type used
}
```

**Algorithm Comparison Metrics:**
```python
# results/algorithm_comparison.json
{
    "timestamp": "2025-11-09T12:00:00",
    "algorithms": {
        "kmeans": {
            "silhouette_score": float,
            "davies_bouldin_index": float,
            "cluster_purity": float,
            "n_clusters": int,
            "n_noise_points": int,
            "runtime_seconds": float,
            "convergence_iterations": int
        },
        "dbscan": {...},
        "hierarchical": {...},
        "gmm": {...}
    },
    "best_algorithm": {
        "overall": str,
        "by_silhouette": str,
        "by_purity": str,
        "by_speed": str
    }
}
```

### APIs and Interfaces

**DBSCAN Clustering API:**
```python
class DBSCANClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'cosine'):
        """Initialize DBSCAN with parameters."""

    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit DBSCAN and return labels and core samples.

        Args:
            embeddings: Document embeddings (n_samples, 768)

        Returns:
            labels: Cluster labels (-1 for noise, 0+ for clusters)
            core_samples: Boolean mask of core samples
        """

    def tune_parameters(
        self,
        embeddings: np.ndarray,
        eps_range: List[float] = [0.3, 0.5, 0.7, 1.0],
        min_samples_range: List[int] = [3, 5, 10]
    ) -> dict:
        """
        Auto-tune eps and min_samples by maximizing Silhouette Score.

        Returns:
            {
                "best_eps": float,
                "best_min_samples": int,
                "best_silhouette": float,
                "tuning_results": DataFrame
            }
        """
```

**Hierarchical Clustering API:**
```python
class HierarchicalClustering:
    def __init__(self, n_clusters: int = 4, linkage: str = 'ward'):
        """Initialize hierarchical clustering."""

    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Fit hierarchical clustering and return labels and dendrogram data.

        Returns:
            labels: Cluster labels (0 to n_clusters-1)
            dendrogram_data: Dictionary containing linkage matrix for dendrogram
        """

    def compare_linkage_methods(
        self,
        embeddings: np.ndarray,
        methods: List[str] = ['ward', 'complete', 'average', 'single']
    ) -> pd.DataFrame:
        """
        Compare multiple linkage methods.

        Returns:
            DataFrame with columns: linkage_method, silhouette_score, davies_bouldin, cluster_purity
        """
```

**GMM Clustering API:**
```python
class GMMClustering:
    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = 'full',
        random_state: int = 42
    ):
        """Initialize Gaussian Mixture Model."""

    def fit_predict(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Fit GMM and return hard labels, probabilities, BIC, AIC.

        Returns:
            labels: Hard cluster assignments (argmax of probabilities)
            probabilities: Soft assignments (n_samples, n_components)
            bic: Bayesian Information Criterion (lower is better)
            aic: Akaike Information Criterion (lower is better)
        """

    def compare_covariance_types(
        self,
        embeddings: np.ndarray,
        types: List[str] = ['full', 'tied', 'diag', 'spherical']
    ) -> pd.DataFrame:
        """
        Compare multiple covariance types.

        Returns:
            DataFrame with columns: covariance_type, bic, aic, silhouette_score, runtime
        """
```

**Algorithm Comparison API:**
```python
class AlgorithmComparison:
    def __init__(self, algorithms_results: Dict[str, dict]):
        """
        Initialize with results from all algorithms.

        Args:
            algorithms_results: {
                'kmeans': {...},
                'dbscan': {...},
                'hierarchical': {...},
                'gmm': {...}
            }
        """

    def generate_comparison_matrix(self) -> pd.DataFrame:
        """
        Create comparison matrix with all algorithms and metrics.

        Returns:
            DataFrame with algorithms as rows, metrics as columns
        """

    def generate_visualizations(self, embeddings: np.ndarray) -> Figure:
        """
        Generate side-by-side PCA plots for all algorithms.

        Returns:
            matplotlib Figure with 2x2 or 1x4 subplot layout
        """

    def generate_report(self, output_path: Path) -> None:
        """
        Generate comprehensive comparison report (markdown).

        Sections:
            1. Methodology
            2. Quantitative Results
            3. Visual Comparison
            4. Algorithm Analysis
            5. Recommendations
            6. Lessons Learned
        """
```

### Workflows and Sequencing

**Epic 5 Execution Flow:**

```
Start: Epic 5
    ↓
[Prerequisites Check]
    - Verify data/embeddings/train_embeddings.npy exists
    - Verify K-Means results from Epic 2 available
    - Load ground truth labels from AG News
    ↓
┌─────────────────────────────────────────────────┐
│ Story 5.1: DBSCAN Implementation                │
├─────────────────────────────────────────────────┤
│ 1. Load embeddings (120K × 768)                 │
│ 2. Parameter tuning loop:                       │
│    for eps in [0.3, 0.5, 0.7, 1.0]:            │
│      for min_samples in [3, 5, 10]:            │
│        - Fit DBSCAN(eps, min_samples, 'cosine')│
│        - Calculate Silhouette Score             │
│        - Track results                          │
│ 3. Select best parameters (max Silhouette)      │
│ 4. Fit final DBSCAN with best parameters        │
│ 5. Extract labels and core samples              │
│ 6. Evaluate: Silhouette, Davies-Bouldin, purity│
│ 7. Save assignments to CSV                      │
│ 8. Save metrics to JSON                         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Story 5.2: Hierarchical Clustering              │
├─────────────────────────────────────────────────┤
│ 1. Load embeddings                              │
│ 2. Linkage method comparison:                   │
│    for method in ['ward', 'complete',           │
│                    'average', 'single']:        │
│        - Fit AgglomerativeClustering            │
│        - Calculate metrics                      │
│        - Track results                          │
│ 3. Select best linkage method                   │
│ 4. Generate dendrogram visualization            │
│    - Use scipy.cluster.hierarchy.dendrogram     │
│    - Truncate if necessary (120K samples)       │
│    - Mark n_clusters=4 boundary                 │
│ 5. Fit final model with best method             │
│ 6. Evaluate cluster quality                     │
│ 7. Save assignments and dendrogram              │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Story 5.3: Gaussian Mixture Model               │
├─────────────────────────────────────────────────┤
│ 1. Load embeddings                              │
│ 2. Covariance type comparison:                  │
│    for cov_type in ['full', 'tied',             │
│                      'diag', 'spherical']:      │
│        - Fit GaussianMixture(random_state=42)   │
│        - Calculate BIC, AIC, Silhouette         │
│        - Measure runtime                        │
│        - Track results                          │
│ 3. Select best covariance type (min BIC)        │
│ 4. Extract hard labels (argmax probabilities)   │
│ 5. Extract soft labels (full probability dist)  │
│ 6. Analyze assignment confidence:               │
│    - Identify low-confidence docs (<0.5)        │
│    - Analyze confusion between cluster pairs    │
│ 7. Evaluate cluster quality                     │
│ 8. Save assignments with probabilities          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Story 5.4: Comprehensive Comparison             │
├─────────────────────────────────────────────────┤
│ 1. Load all algorithm results:                  │
│    - K-Means (Epic 2)                           │
│    - DBSCAN (Story 5.1)                         │
│    - Hierarchical (Story 5.2)                   │
│    - GMM (Story 5.3)                            │
│ 2. Generate comparison matrix (DataFrame)       │
│ 3. Generate side-by-side PCA visualizations:    │
│    - 2×2 subplot layout                         │
│    - Same PCA projection for all                │
│    - Consistent color mapping                   │
│ 4. Analyze per-algorithm:                       │
│    - Strengths and weaknesses                   │
│    - Best use cases                             │
│    - Parameter sensitivity                      │
│ 5. Ground truth alignment analysis:             │
│    - Confusion matrices for each algorithm      │
│    - Category-to-cluster mapping                │
│ 6. Dimensionality challenge analysis            │
│ 7. Generate comprehensive report (markdown)     │
│ 8. Export all results to JSON                   │
└─────────────────────────────────────────────────┘
    ↓
[Deliverables]
    - data/processed/dbscan_assignments.csv
    - data/processed/hierarchical_assignments.csv
    - data/processed/gmm_assignments.csv
    - results/algorithm_comparison.json
    - reports/figures/algorithm_comparison.png
    - reports/figures/dendrogram.png
    - reports/clustering_comparison.md
    ↓
End: Epic 5 Complete
```

**Timing Estimates:**
- Story 5.1 (DBSCAN): 4 hours (parameter tuning is time-intensive)
- Story 5.2 (Hierarchical): 4 hours (dendrogram generation, linkage comparison)
- Story 5.3 (GMM): 3 hours (faster convergence than DBSCAN)
- Story 5.4 (Comparison): 5 hours (comprehensive analysis and reporting)
- **Total**: ~16 hours (2 working days)

## Non-Functional Requirements

### Performance

**DBSCAN Performance Targets:**
- Single DBSCAN fit on 120K documents: <15 minutes
- Parameter tuning (12 combinations): <3 hours total
- Silhouette Score calculation: <3 minutes per run
- Memory usage: <4GB RAM (use memory mapping if needed)

**Hierarchical Clustering Performance Targets:**
- Single AgglomerativeClustering fit: <20 minutes (ward linkage)
- Linkage method comparison (4 methods): <2 hours total
- Dendrogram generation: May require sampling (10K samples) if full dataset too large
- Memory usage: <8GB RAM (hierarchical clustering is memory-intensive)

**GMM Performance Targets:**
- Single GMM fit: <10 minutes (faster than DBSCAN/Hierarchical)
- Covariance type comparison (4 types): <1 hour total
- Probability extraction: <1 minute
- Convergence: max_iter=100 should be sufficient

**Overall Performance Constraints:**
- No GPU required (all algorithms CPU-only)
- All algorithms must complete on standard laptop (16GB RAM recommended)
- If memory issues: implement sampling strategy (document in report)
- Progress logging required for all long-running operations (>5 minutes)

### Security

**No new security concerns for Epic 5:**
- No external API calls required (uses cached embeddings)
- No sensitive data beyond what was already handled in Epic 1
- All data processing is local
- Follow same API key management practices from architecture (though not used in this epic)

### Reliability/Availability

**Error Handling:**
- Memory overflow handling: If algorithm runs out of memory, implement sampling and document strategy
- Convergence failure handling: If GMM/Hierarchical fails to converge, log warning and document in results
- Invalid parameters handling: Validate parameter ranges before running expensive computations
- Data validation: Verify embeddings file integrity before processing

**Graceful Degradation:**
- If dendrogram visualization fails due to memory: use truncated dendrogram or skip visualization (document in report)
- If Silhouette calculation too slow: implement sampling strategy for evaluation
- If comparison report generation fails: export raw metrics to JSON as fallback

**Robustness:**
- All experiments must be deterministic (random_state=42 for GMM and sampling)
- Results must be reproducible by instructor
- Checkpointing after each algorithm (save intermediate results)

### Observability

**Logging Requirements:**
- Log all parameter combinations tested (DBSCAN eps/min_samples, Hierarchical linkage, GMM covariance)
- Log runtime for each algorithm variant
- Log convergence information (GMM iterations, Hierarchical merge steps)
- Log Silhouette Score after each run for comparison
- Progress bars for long-running operations (DBSCAN parameter tuning, Hierarchical fitting)

**Metrics Tracking:**
- All metrics exported to JSON with timestamps
- Comparison matrix saved as both CSV and JSON
- Parameter tuning results saved for reproducibility

**Debugging Support:**
- Verbose logging mode available (controlled by config)
- Intermediate results saved to data/interim/ (cluster assignments after each algorithm)
- Full stack traces logged on failures

## Dependencies and Integrations

**Python Library Dependencies:**

All required dependencies are already included in the project's requirements.txt from Epic 1:

```python
# Core Machine Learning (REQUIRED for Epic 5)
scikit-learn>=1.7.2  # Provides DBSCAN, AgglomerativeClustering, GaussianMixture
scipy>=1.9.0         # Required for dendrogram generation (scipy.cluster.hierarchy)
numpy>=1.24.0        # Array operations
pandas>=2.0.0        # DataFrames for comparison matrices

# Visualization (REQUIRED for Epic 5)
matplotlib>=3.7.0    # Base plotting for side-by-side visualizations
seaborn>=0.12.0      # Heatmaps for confusion matrices

# Configuration & Utilities (REUSED from Epic 1)
PyYAML>=6.0          # Config file parsing
python-dotenv>=1.0.0 # Environment variables
tenacity>=8.0.0      # Retry logic (not needed in Epic 5, but in requirements)

# Development (OPTIONAL for Epic 5)
pytest>=7.4.0        # Unit testing
ruff>=0.1.0          # Code formatting
jupyter>=1.0.0       # Analysis notebooks
```

**New Dependency (if not already included):**
```python
scipy>=1.9.0  # Required for hierarchical clustering dendrogram visualization
```

**Specific Module Usage:**

| Algorithm | scikit-learn Module | Additional Dependencies |
|-----------|-------------------|------------------------|
| **DBSCAN** | `sklearn.cluster.DBSCAN` | `sklearn.metrics.pairwise.cosine_distances` |
| **Hierarchical** | `sklearn.cluster.AgglomerativeClustering` | `scipy.cluster.hierarchy` (for dendrogram) |
| **GMM** | `sklearn.mixture.GaussianMixture` | None (self-contained) |
| **Metrics** | `sklearn.metrics` (silhouette_score, davies_bouldin_score) | None |

**Internal Dependencies (Epic Relationships):**

```
Epic 5 Dependencies:
    ← Epic 1: Embeddings (data/embeddings/train_embeddings.npy)
    ← Epic 2: K-Means results (for comparison baseline)
    ← Epic 2: AG News ground truth labels (for purity calculation)
    ← Architecture: Evaluation metrics infrastructure (src/evaluation/clustering_metrics.py)
    ← Architecture: PCA visualization infrastructure (src/visualization/cluster_plots.py)
```

**External Data Dependencies:**
- AG News dataset metadata (loaded from datasets library, already cached)
- Embedding cache must exist (generated in Epic 1)
- K-Means cluster assignments (from Epic 2, for comparison)

**No External API Dependencies:**
- Epic 5 does NOT require Gemini API (uses cached embeddings)
- All processing is local and offline
- No network connectivity required after initial setup

## Acceptance Criteria (Authoritative)

**Epic 5 is considered COMPLETE when ALL of the following criteria are met:**

### Story 5.1: DBSCAN Implementation
- ✅ DBSCAN clustering successfully runs on 120K embeddings with cosine distance metric
- ✅ Parameter tuning completed for eps ∈ {0.3, 0.5, 0.7, 1.0} and min_samples ∈ {3, 5, 10}
- ✅ Best parameters selected based on maximum Silhouette Score
- ✅ Cluster assignments saved to `data/processed/dbscan_assignments.csv` with columns: document_id, cluster_id, ground_truth_category, is_core_sample
- ✅ Number of clusters discovered (may be variable) and number of noise points logged
- ✅ Metrics calculated: Silhouette Score, Davies-Bouldin Index, cluster purity
- ✅ Results exported to JSON with timestamp and full parameter configuration
- ✅ Runtime performance: <3 hours for full parameter tuning

### Story 5.2: Hierarchical Clustering
- ✅ Hierarchical clustering successfully runs on 120K embeddings (or justified sample)
- ✅ Linkage method comparison completed for: ward, complete, average, single
- ✅ Best linkage method selected based on evaluation metrics
- ✅ Dendrogram visualization generated showing hierarchical structure
- ✅ Dendrogram shows n_clusters=4 boundary clearly marked
- ✅ Cluster assignments saved to `data/processed/hierarchical_assignments.csv`
- ✅ Dendrogram saved to `reports/figures/dendrogram.png` (300 DPI)
- ✅ Metrics calculated for best linkage method
- ✅ Linkage comparison results saved as DataFrame (CSV/JSON)
- ✅ Runtime performance: <2 hours for linkage method comparison

### Story 5.3: Gaussian Mixture Model
- ✅ GMM successfully runs on 120K embeddings with n_components=4
- ✅ Covariance type comparison completed for: full, tied, diag, spherical
- ✅ Best covariance type selected based on BIC (minimum value)
- ✅ Hard cluster assignments extracted (argmax of probabilities)
- ✅ Soft cluster assignments (full probability distribution) extracted
- ✅ Assignment confidence analyzed: low-confidence documents (<0.5) identified
- ✅ Cluster assignments with probabilities saved to `data/processed/gmm_assignments.csv`
- ✅ GMM-specific metrics calculated: BIC, AIC, log-likelihood, component weights
- ✅ Metrics calculated: Silhouette Score, Davies-Bouldin Index, cluster purity
- ✅ Covariance comparison results saved as DataFrame
- ✅ Runtime performance: <1 hour for covariance type comparison

### Story 5.4: Comprehensive Algorithm Comparison
- ✅ Comparison matrix generated with all 4 algorithms (K-Means, DBSCAN, Hierarchical, GMM)
- ✅ Comparison matrix includes metrics: Silhouette, Davies-Bouldin, Purity, Runtime, N_Clusters, N_Noise
- ✅ Side-by-side PCA visualizations generated (2×2 or 1×4 subplot layout)
- ✅ All algorithms use same PCA projection for fair visual comparison
- ✅ Visualizations saved to `reports/figures/algorithm_comparison.png` (300 DPI)
- ✅ Per-algorithm analysis documented: strengths, weaknesses, use cases, parameter sensitivity
- ✅ Ground truth alignment analyzed: confusion matrix for each algorithm
- ✅ Dimensionality challenge analysis included: curse of dimensionality discussion
- ✅ Comprehensive comparison report generated at `reports/clustering_comparison.md`
- ✅ Report includes sections: Methodology, Quantitative Results, Visual Comparison, Algorithm Analysis, Recommendations, Lessons Learned
- ✅ All results exported to `results/algorithm_comparison.json`
- ✅ Best algorithm identified (overall and per specific criterion)

### Cross-Story Quality Criteria
- ✅ All algorithms use same embeddings (data/embeddings/train_embeddings.npy)
- ✅ All algorithms use same evaluation methodology (fair comparison)
- ✅ Reproducibility: random_state=42 used where applicable (GMM, sampling)
- ✅ All cluster assignments have same format (CSV with ground_truth_category column)
- ✅ All metrics follow same JSON schema for consistency
- ✅ Progress logging implemented for all operations >5 minutes
- ✅ Code follows architecture patterns (PEP 8, type hints, docstrings)

## Traceability Mapping

| Acceptance Criterion | Spec Section | Component/Module | Test Strategy |
|---------------------|--------------|------------------|---------------|
| **DBSCAN runs on 120K embeddings** | APIs & Interfaces → DBSCANClustering.fit_predict() | src/models/dbscan_clustering.py | Integration test: Load embeddings, run DBSCAN, verify output shape |
| **Parameter tuning completed** | Workflows → Story 5.1 Parameter Loop | src/models/dbscan_clustering.py → tune_parameters() | Unit test: Verify all parameter combinations tested |
| **Best parameters selected** | Data Models → DBSCAN Assignments | scripts/06_alternative_clustering.py | Validation: Verify selected params have max Silhouette |
| **Cluster assignments saved** | Data Models → dbscan_assignments.csv | src/models/dbscan_clustering.py | File existence test + schema validation |
| **Metrics calculated (DBSCAN)** | Services & Modules → DBSCANClustering | src/evaluation/clustering_metrics.py | Unit test: Verify Silhouette, Davies-Bouldin, purity computed |
| **Hierarchical runs successfully** | APIs & Interfaces → HierarchicalClustering | src/models/hierarchical_clustering.py | Integration test: Run on sample data, verify convergence |
| **Linkage comparison completed** | Workflows → Story 5.2 Linkage Loop | src/models/hierarchical_clustering.py → compare_linkage_methods() | Unit test: Verify 4 linkage methods tested |
| **Dendrogram generated** | Workflows → Story 5.2 Step 4 | src/visualization/cluster_plots.py | Visual inspection + file existence test |
| **Dendrogram shows n_clusters=4** | APIs & Interfaces → HierarchicalClustering | scipy.cluster.hierarchy.dendrogram | Manual validation: Inspect visualization |
| **GMM runs successfully** | APIs & Interfaces → GMMClustering | src/models/gmm_clustering.py | Integration test: Verify convergence |
| **Covariance comparison completed** | Workflows → Story 5.3 Covariance Loop | src/models/gmm_clustering.py → compare_covariance_types() | Unit test: Verify 4 covariance types tested |
| **Hard/soft assignments extracted** | Data Models → GMM Assignments | src/models/gmm_clustering.py | Schema validation: Verify probabilities sum to 1.0 |
| **BIC/AIC calculated** | Services & Modules → GMMClustering | sklearn.mixture.GaussianMixture | Unit test: Verify BIC/AIC values returned |
| **Comparison matrix generated** | APIs & Interfaces → AlgorithmComparison.generate_comparison_matrix() | src/evaluation/algorithm_comparison.py | Schema validation: Verify all algorithms and metrics present |
| **Side-by-side visualizations** | Workflows → Story 5.4 Step 3 | src/evaluation/algorithm_comparison.py → generate_visualizations() | Visual inspection + file existence test |
| **Same PCA projection used** | Implementation Pattern | src/visualization/cluster_plots.py | Code review: Verify PCA fit once, transform all |
| **Per-algorithm analysis documented** | Workflows → Story 5.4 Step 4 | reports/clustering_comparison.md | Manual review: Verify strengths/weaknesses sections |
| **Ground truth alignment** | Workflows → Story 5.4 Step 5 | src/evaluation/clustering_metrics.py | Unit test: Verify confusion matrices for all algorithms |
| **Dimensionality analysis** | Workflows → Story 5.4 Step 6 | reports/clustering_comparison.md | Manual review: Verify curse of dimensionality discussion |
| **Comprehensive report generated** | APIs & Interfaces → AlgorithmComparison.generate_report() | src/evaluation/algorithm_comparison.py | File existence + section completeness check |
| **Best algorithm identified** | Data Models → algorithm_comparison.json | scripts/07_compare_algorithms.py | JSON schema validation: Verify best_algorithm keys |
| **Same embeddings used** | Architecture Alignment → Reused Components | data/embeddings/train_embeddings.npy | File hash comparison across all algorithm runs |
| **Same evaluation methodology** | Architecture Alignment → Constraints | src/evaluation/clustering_metrics.py | Code review: Verify same functions called |
| **Reproducibility (random_state=42)** | Architecture Alignment → Constraints | All algorithm classes | Unit test: Run twice, verify identical results |
| **Progress logging >5 min** | NFR → Observability | src/utils/logger.py | Manual testing: Verify progress bars appear |
| **Code follows patterns** | Architecture → Implementation Patterns | All Epic 5 code | Ruff linting + type checking passes |

## Risks, Assumptions, Open Questions

### Risks

**RISK-1: Memory Overflow (Hierarchical Clustering)**
- **Severity**: High
- **Probability**: Medium
- **Description**: Hierarchical clustering on 120K samples may exceed 16GB RAM due to O(n²) memory complexity
- **Mitigation**:
  - Implement sampling strategy (10K samples) if memory issues occur
  - Document sampling approach and impact on results
  - Use sparse distance matrix if available
- **Fallback**: Skip hierarchical clustering if infeasible, document limitation

**RISK-2: Poor Performance Across All Algorithms**
- **Severity**: Medium
- **Probability**: High (based on K-Means results)
- **Description**: All algorithms may fail to find semantic structure due to curse of dimensionality
- **Mitigation**:
  - Frame negative results as valuable scientific finding
  - Focus comparison report on "why clustering fails" analysis
  - Recommend dimensionality reduction preprocessing for future work
- **Impact**: Does not invalidate epic—honest reporting is academically valuable

**RISK-3: Parameter Tuning Too Slow (DBSCAN)**
- **Severity**: Medium
- **Probability**: Medium
- **Description**: 12 parameter combinations × 15 min per run = 3 hours (at edge of tolerance)
- **Mitigation**:
  - Reduce parameter grid if initial runs too slow
  - Use sampling for parameter tuning, full dataset for final run
  - Parallelize tuning if multiprocessing feasible
- **Fallback**: Use single parameter set from literature (eps=0.5, min_samples=5)

**RISK-4: Dendrogram Visualization Failure**
- **Severity**: Low
- **Probability**: Medium
- **Description**: scipy dendrogram may fail or produce unreadable plot for 120K samples
- **Mitigation**:
  - Use truncated dendrogram (scipy.cluster.hierarchy.dendrogram with truncate_mode)
  - Sample 10K documents for dendrogram visualization only
  - Document sampling strategy clearly
- **Fallback**: Skip dendrogram, provide textual description of hierarchical structure

### Assumptions

**ASSUMPTION-1: Epic 2 Completed Successfully**
- **Status**: Critical dependency
- **Validation**: Check for data/embeddings/train_embeddings.npy existence
- **Impact if false**: Cannot proceed with Epic 5
- **Mitigation**: Epic 2 must be completed first (enforced by sprint-status.yaml)

**ASSUMPTION-2: Embeddings Quality Sufficient**
- **Status**: Assumed true (based on Gemini API documentation)
- **Validation**: If all algorithms fail, may indicate embedding quality issue
- **Impact if false**: Entire clustering approach may be invalid
- **Mitigation**: Document finding, recommend alternative embedding models

**ASSUMPTION-3: 16GB RAM Available**
- **Status**: Hardware requirement
- **Validation**: Check system memory before running hierarchical clustering
- **Impact if false**: Must use sampling or skip hierarchical
- **Mitigation**: Document hardware requirements in Epic 5 README section

**ASSUMPTION-4: Cosine Distance Appropriate for DBSCAN**
- **Status**: Based on text clustering literature
- **Validation**: Compare cosine vs euclidean distance in parameter tuning
- **Impact if false**: DBSCAN performance may be suboptimal
- **Mitigation**: Test both metrics if time permits, document choice rationale

### Open Questions

**QUESTION-1: Should we test n_components > 4 for GMM?**
- **Context**: GMM can discover different number of components than ground truth
- **Decision Needed**: Test {3, 4, 5, 6} components or stick to 4?
- **Recommendation**: Stick to 4 for fair comparison with K-Means (time constraints)
- **Defer to**: Story 5.3 implementation

**QUESTION-2: How to handle DBSCAN discovering >4 clusters?**
- **Context**: DBSCAN may find 2, 10, or 100+ clusters depending on parameters
- **Decision Needed**: How to compare fairly with K-Means (fixed 4)?
- **Recommendation**: Report actual n_clusters, note in comparison that DBSCAN is variable
- **Defer to**: Story 5.4 comparison analysis

**QUESTION-3: Should comparison report include recommendations for this specific dataset?**
- **Context**: Report could be generic (all text clustering) or specific (AG News)
- **Decision Needed**: Scope of recommendations section
- **Recommendation**: Dataset-specific recommendations, generalization notes as "Future Work"
- **Defer to**: Story 5.4 report writing

**QUESTION-4: How to visualize variable cluster counts in side-by-side PCA?**
- **Context**: K-Means has 4 colors, DBSCAN may have 2-10+ colors plus noise (-1)
- **Decision Needed**: Color mapping strategy
- **Recommendation**: Use consistent 4-color palette, map DBSCAN clusters to colors by size
- **Defer to**: Story 5.4 visualization implementation

## Test Strategy Summary

### Unit Testing Strategy

**Module-Level Tests:**
```python
# tests/test_dbscan_clustering.py
def test_dbscan_initialization():
    """Verify DBSCAN initializes with correct parameters."""

def test_dbscan_fit_predict_shape():
    """Verify output shapes are correct (labels: (n,), core_samples: (n,))."""

def test_parameter_tuning_coverage():
    """Verify all parameter combinations tested."""

# tests/test_hierarchical_clustering.py
def test_hierarchical_linkage_methods():
    """Verify all 4 linkage methods return valid results."""

def test_dendrogram_data_structure():
    """Verify dendrogram data has required keys (linkage matrix)."""

# tests/test_gmm_clustering.py
def test_gmm_probabilities_sum_to_one():
    """Verify soft assignments sum to 1.0 for each sample."""

def test_covariance_comparison():
    """Verify BIC/AIC calculated for all covariance types."""

# tests/test_algorithm_comparison.py
def test_comparison_matrix_schema():
    """Verify comparison matrix has all required columns."""

def test_best_algorithm_selection():
    """Verify best algorithm logic (e.g., max Silhouette)."""
```

### Integration Testing Strategy

**End-to-End Workflows:**
1. **Epic 5 Full Pipeline Test** (scripts/test_epic5_integration.py):
   - Load embeddings
   - Run DBSCAN with default params → verify assignments saved
   - Run Hierarchical with ward linkage → verify assignments saved
   - Run GMM with full covariance → verify assignments + probabilities saved
   - Run comparison → verify report generated

2. **Reproducibility Test**:
   - Run all algorithms twice with same random_state
   - Verify byte-identical outputs (GMM, sampling)

3. **Performance Benchmark Test**:
   - Run each algorithm on 10K sample
   - Verify runtime < 5 minutes for sample size
   - Extrapolate to 120K, document if expected runtime exceeds targets

### Manual Testing Checklist

**Visual Validation:**
- [ ] Dendrogram shows clear hierarchical structure
- [ ] Side-by-side PCA visualizations are readable
- [ ] Cluster colors are distinguishable (colorblind-friendly)
- [ ] Algorithm comparison plot has consistent formatting

**Report Quality Review:**
- [ ] Comprehensive report markdown renders correctly
- [ ] All report sections present (6 sections from acceptance criteria)
- [ ] Recommendations section provides actionable guidance
- [ ] Quantitative results table is complete and accurate

**Data Validation:**
- [ ] All CSV files have correct schema (columns match data models)
- [ ] All JSON files are valid and parse correctly
- [ ] No NaN or Inf values in metrics
- [ ] Cluster IDs are within expected range

### Acceptance Testing Strategy

**Story 5.1 Acceptance:**
- Run: `python scripts/06_alternative_clustering.py --algorithm dbscan`
- Verify: data/processed/dbscan_assignments.csv exists with 120K rows
- Verify: results/dbscan_metrics.json contains Silhouette Score
- Verify: Best parameters logged with justification

**Story 5.2 Acceptance:**
- Run: `python scripts/06_alternative_clustering.py --algorithm hierarchical`
- Verify: data/processed/hierarchical_assignments.csv exists
- Verify: reports/figures/dendrogram.png exists (300 DPI)
- Verify: Linkage comparison DataFrame saved

**Story 5.3 Acceptance:**
- Run: `python scripts/06_alternative_clustering.py --algorithm gmm`
- Verify: data/processed/gmm_assignments.csv exists with probability columns
- Verify: All probabilities in [0, 1] and sum to ~1.0
- Verify: BIC/AIC values are finite and reasonable

**Story 5.4 Acceptance:**
- Run: `python scripts/07_compare_algorithms.py`
- Verify: reports/clustering_comparison.md exists and has 6 sections
- Verify: reports/figures/algorithm_comparison.png exists (2×2 layout)
- Verify: results/algorithm_comparison.json has all 4 algorithms
- Manual review: Report conclusions are data-driven and justified

### Performance Testing

**Benchmarks:**
- DBSCAN single run: <15 minutes
- Hierarchical single run: <20 minutes
- GMM single run: <10 minutes
- Full Epic 5 pipeline: <6 hours (including parameter tuning)

**Load Testing:**
- Test on 10K sample first (should complete in <10 minutes total)
- Extrapolate timings to 120K
- If projections exceed targets: implement sampling mitigation
