# Story 5.3: Gaussian Mixture Model Clustering

Status: ready-for-dev

## Story

As a **data mining student**,
I want **to apply Gaussian Mixture Models for probabilistic clustering**,
So that **I can compare soft clustering (probabilistic assignments) with hard clustering (K-Means) approaches and evaluate GMM performance on high-dimensional text embeddings**.

## Acceptance Criteria

**Given** document embeddings are generated and cached (from Story 2.1)
**When** I run GMM clustering algorithm
**Then** GMM is applied with parameters:
- n_components = 4 (for comparison with K-Means)
- covariance_type = 'full' (initial setting)
- random_state = 42 (reproducibility)
- max_iter = 100

**And** alternative covariance types are tested:
- 'full' (each component has own covariance matrix)
- 'tied' (all components share covariance)
- 'diag' (diagonal covariance)
- 'spherical' (single variance per component)
- Compare BIC/AIC scores across types

**And** probabilistic cluster assignments are extracted:
- Hard assignments: argmax of probability distribution
- Soft assignments: full probability distribution over clusters
- Assignment confidence: max probability value

**And** cluster assignments are saved to `data/processed/gmm_assignments.csv` with columns:
- document_id
- cluster_id (hard assignment)
- cluster_0_prob
- cluster_1_prob
- cluster_2_prob
- cluster_3_prob
- assignment_confidence (max probability)
- ground_truth_category
- covariance_type

**And** GMM-specific metrics are calculated:
- Log-likelihood of the model
- BIC (Bayesian Information Criterion)
- AIC (Akaike Information Criterion)
- Component weights (mixing coefficients)

**And** uncertainty analysis is performed:
- Identify documents with low assignment confidence (<0.5)
- Analyze confusion between cluster pairs
- Compare uncertainty patterns with ground truth categories

**And** standard clustering quality metrics are computed:
- Silhouette Score
- Davies-Bouldin Index
- Cluster purity (based on ground truth)

**And** convergence information is logged (iterations, final log-likelihood)

**And** all results are saved to `results/gmm_metrics.json`

## Tasks / Subtasks

- [ ] Task 1: Implement GMM Clustering Module (AC: 1-4)
  - [ ] 1.1: Create `src/models/gmm_clustering.py` with GMMClustering class
  - [ ] 1.2: Implement `__init__(n_components, covariance_type, random_state)` method
  - [ ] 1.3: Implement `fit_predict(embeddings)` returning labels, probabilities, BIC, AIC
  - [ ] 1.4: Implement `compare_covariance_types(embeddings, types)` for parameter comparison
  - [ ] 1.5: Add type hints and docstrings following project patterns

- [ ] Task 2: Covariance Type Comparison (AC: 2)
  - [ ] 2.1: Test 'full', 'tied', 'diag', 'spherical' covariance types
  - [ ] 2.2: Calculate BIC and AIC for each covariance type
  - [ ] 2.3: Measure runtime for each covariance type
  - [ ] 2.4: Select best covariance type based on minimum BIC
  - [ ] 2.5: Save comparison results to DataFrame/CSV

- [ ] Task 3: Extract Probabilistic Assignments (AC: 3-4)
  - [ ] 3.1: Extract hard cluster assignments using `predict()` or `argmax(predict_proba())`
  - [ ] 3.2: Extract soft assignments (full probability distribution) using `predict_proba()`
  - [ ] 3.3: Calculate assignment confidence (max probability per document)
  - [ ] 3.4: Validate probabilities sum to 1.0 for each document
  - [ ] 3.5: Save assignments to `data/processed/gmm_assignments.csv` with all probability columns

- [ ] Task 4: Uncertainty Analysis (AC: 6)
  - [ ] 4.1: Identify low-confidence documents (confidence < 0.5)
  - [ ] 4.2: Analyze which cluster pairs show highest confusion (similar probabilities)
  - [ ] 4.3: Compare uncertainty patterns with ground truth categories
  - [ ] 4.4: Generate uncertainty distribution statistics
  - [ ] 4.5: Document findings in results JSON

- [ ] Task 5: GMM-Specific Metrics Calculation (AC: 5)
  - [ ] 5.1: Extract BIC (Bayesian Information Criterion) from fitted model
  - [ ] 5.2: Extract AIC (Akaike Information Criterion) from fitted model
  - [ ] 5.3: Extract log-likelihood from model
  - [ ] 5.4: Extract component weights (mixing coefficients)
  - [ ] 5.5: Validate metrics are finite and reasonable

- [ ] Task 6: Standard Clustering Metrics (AC: 7)
  - [ ] 6.1: Calculate Silhouette Score using hard assignments
  - [ ] 6.2: Calculate Davies-Bouldin Index using hard assignments
  - [ ] 6.3: Calculate cluster purity using ground truth labels
  - [ ] 6.4: Use existing `src/evaluation/clustering_metrics.py` functions
  - [ ] 6.5: Ensure consistency with K-Means evaluation methodology

- [ ] Task 7: Create GMM Execution Script (AC: 8)
  - [ ] 7.1: Create `scripts/06_gmm_clustering.py` or integrate into existing alternative clustering script
  - [ ] 7.2: Load embeddings from cache
  - [ ] 7.3: Load ground truth labels from AG News
  - [ ] 7.4: Run covariance type comparison
  - [ ] 7.5: Fit final GMM with best covariance type
  - [ ] 7.6: Extract and save all assignments and metrics
  - [ ] 7.7: Log convergence information and performance stats
  - [ ] 7.8: Add progress logging for long operations

- [ ] Task 8: Save Results and Validate (AC: 9)
  - [ ] 8.1: Save all metrics to `results/gmm_metrics.json` with timestamp
  - [ ] 8.2: Validate CSV output schema matches spec
  - [ ] 8.3: Validate all probabilities in [0, 1] range
  - [ ] 8.4: Validate probability sums ≈ 1.0 for each document
  - [ ] 8.5: Log summary statistics to console

- [ ] Task 9: Testing and Validation
  - [ ] 9.1: Test GMM on small sample (1K documents) for quick validation
  - [ ] 9.2: Verify convergence on full dataset (120K documents)
  - [ ] 9.3: Validate runtime is within target (<1 hour for covariance comparison)
  - [ ] 9.4: Compare results with K-Means from Epic 2
  - [ ] 9.5: Document any convergence issues or warnings

## Dev Notes

### Algorithm Overview

**Gaussian Mixture Models (GMM):**
- Probabilistic model assuming data comes from mixture of K Gaussian distributions
- Each cluster represented by: mean vector (μ), covariance matrix (Σ), mixing coefficient (π)
- Soft clustering: each document has probability of belonging to each cluster
- Uses EM (Expectation-Maximization) algorithm for parameter estimation
- Model selection via BIC/AIC (lower is better)

**Key Differences from K-Means:**
- **K-Means**: Hard clustering, distance-based, deterministic centroids
- **GMM**: Soft clustering, probability-based, Gaussian distributions with covariance
- GMM captures uncertainty in cluster assignments
- GMM can model elliptical clusters (via covariance), K-Means assumes spherical

### Covariance Types Comparison

| Covariance Type | Description | Parameters | Use Case |
|----------------|-------------|------------|----------|
| **full** | Each component has own full covariance matrix | K × d × d | Most flexible, captures correlations |
| **tied** | All components share same covariance | 1 × d × d | Assumes similar cluster shapes |
| **diag** | Diagonal covariance (no correlations) | K × d | Faster, axis-aligned ellipsoids |
| **spherical** | Single variance per component | K × 1 | Fastest, similar to K-Means |

**For 768-dimensional embeddings:**
- 'full' is most expressive but computationally expensive (768 × 768 matrix per component)
- 'diag' or 'spherical' may be more practical for high dimensions
- BIC/AIC will penalize complexity, helping select appropriate type

### Expected Performance

**Runtime Targets (from tech spec):**
- Single GMM fit: <10 minutes (faster than DBSCAN/Hierarchical)
- Covariance type comparison (4 types): <1 hour total
- Probability extraction: <1 minute
- Convergence: max_iter=100 should be sufficient

**Memory Considerations:**
- 'full' covariance: ~3MB per component (768×768 floats) → 12MB for K=4
- Embeddings: 120K × 768 × 4 bytes ≈ 368MB
- Should fit comfortably in 16GB RAM

### Architecture Alignment

**Reused Components (from Epic 2):**
- `data/embeddings/train_embeddings.npy` - Same 120K × 768 embeddings
- `src/evaluation/clustering_metrics.py` - Silhouette Score, Davies-Bouldin, purity functions
- `config.yaml` - Extended with GMM-specific parameters

**New Components (this story):**
- `src/models/gmm_clustering.py` - GMMClustering implementation wrapper
- `data/processed/gmm_assignments.csv` - Cluster labels with probabilities
- `results/gmm_metrics.json` - GMM-specific metrics

**API Specification (from tech spec):**

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

### High-Dimensional Considerations

**Curse of Dimensionality:**
- In 768D space, Gaussian assumptions may not hold well
- "Full" covariance may be unstable (many parameters to estimate)
- Regularization (from sklearn) helps prevent singular matrices
- BIC will penalize over-parameterization

**Expected Challenges:**
- K-Means already failed (Silhouette ≈0.0008) due to high dimensions
- GMM may also struggle with meaningful cluster separation
- Covariance estimation may be unreliable
- This is expected and valuable finding for academic report

**Mitigation Strategies:**
- Use BIC/AIC to select simpler covariance types if 'full' overfits
- Document convergence warnings and handle gracefully
- Compare with K-Means to understand if probabilistic model helps
- Focus on "lessons learned" if results are poor

### Testing Strategy

**Unit Tests:**
- Test probability distributions sum to 1.0
- Test BIC/AIC values are finite
- Test all covariance types return valid results
- Test hard assignments match argmax(probabilities)

**Integration Tests:**
- Run on small sample (1K docs) for quick validation
- Verify full pipeline: load → fit → predict → save
- Validate CSV schema matches spec

**Acceptance Validation:**
- All 4 covariance types tested ✅
- Probabilities extracted and validated ✅
- Uncertainty analysis completed ✅
- Metrics saved to JSON ✅
- Runtime < 1 hour ✅

### Uncertainty Analysis Details

**Low-Confidence Documents:**
- Confidence < 0.5 means no single cluster dominates
- Indicates document lies between clusters
- May reveal limitations of K=4 assumption
- Example: document has [0.35, 0.30, 0.25, 0.10] → confidence = 0.35

**Cluster Pair Confusion:**
- Identify pairs where documents have similar probabilities
- Example: Sports vs World confusion → [0.4, 0.4, 0.1, 0.1]
- Compare with ground truth to understand semantic overlap
- Helps explain why clustering is challenging

### Project Structure Notes

**File Paths (from project structure):**
```
src/models/gmm_clustering.py          # New: GMM implementation
scripts/06_gmm_clustering.py           # New: Execution script (or integrate into 06_alternative_clustering.py)
data/processed/gmm_assignments.csv     # Output: Cluster assignments with probabilities
results/gmm_metrics.json               # Output: All GMM metrics
```

**Integration with Story 5.4:**
- Story 5.4 will load GMM results for comparison with K-Means, DBSCAN, Hierarchical
- Ensure JSON format consistent with other algorithms
- Provide same baseline metrics (Silhouette, Davies-Bouldin, purity)

### References

**Source Documents:**
- [Source: docs/tech-spec-epic-5.md#Story-5.3-GMM-Implementation]
- [Source: docs/epics.md#Story-5.3-Gaussian-Mixture-Model]
- [Source: docs/architecture.md#Evaluation-Metrics] (if exists)

**scikit-learn Documentation:**
- `sklearn.mixture.GaussianMixture` - Main GMM class
- `predict_proba()` - Soft cluster assignments
- `bic()` and `aic()` - Model selection criteria

**Academic Context:**
- This story demonstrates understanding of probabilistic clustering
- Comparison of covariance types shows model selection skills
- Uncertainty analysis shows critical thinking about cluster quality
- Negative results (if GMM also fails) are academically valuable

### Previous Story Learnings

**Note:** This is the first story in Epic 5 to be drafted. Epic 5 stories (5-1, 5-2) are still in backlog status. Therefore, there are no direct predecessor learnings from Epic 5.

**Learnings from Epic 2 (K-Means Clustering):**
- K-Means achieved poor results (Silhouette ≈0.0008, purity ≈25%)
- High-dimensional space (768D) presents challenges for clustering
- PCA visualization infrastructure available at `src/visualization/cluster_plots.py`
- Evaluation metrics infrastructure at `src/evaluation/clustering_metrics.py`
- Embeddings cached at `data/embeddings/train_embeddings.npy`

**Key Insights to Apply:**
- Use same embeddings for fair comparison
- Use same evaluation methodology (Silhouette, Davies-Bouldin, purity)
- Expect similar challenges with high-dimensional space
- GMM's probabilistic nature may provide better uncertainty quantification even if separation is poor
- Focus on robust implementation and clear documentation of findings

## Dev Agent Record

### Context Reference

- [Story Context XML](5-3-gaussian-mixture-model-clustering.context.xml)

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
