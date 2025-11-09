# Story 2.4: PCA Cluster Visualization

Status: drafted

## Story

As a **data mining student**,
I want **a clear 2D visualization showing 4 distinct semantic clusters**,
So that **I can demonstrate clustering effectiveness visually in my report**.

## Acceptance Criteria

### AC-1: PCA Dimensionality Reduction

**Given** K-Means clustering is complete with 120K document embeddings (768-dim)
**When** I apply PCA for dimensionality reduction
**Then**:
- ✅ PCA reduces 768-dimensional embeddings to 2D using `sklearn.decomposition.PCA(n_components=2)`
- ✅ Both document embeddings and cluster centroids are projected to 2D space
- ✅ Variance explained by PC1 and PC2 is calculated and logged
- ✅ Target: Combined variance explained >20% (validates meaningful projection)
- ✅ If variance <20%: log warning but continue (projection still useful for visualization)
- ✅ PCA transformation is fitted on embeddings and applied to both embeddings and centroids

**Validation:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)  # (120000, 2)
centroids_2d = pca.transform(centroids)  # (4, 2)
variance_explained = pca.explained_variance_ratio_.sum()
assert variance_explained > 0.20  # Target: >20%
```

---

### AC-2: Scatter Plot Generation with Cluster Colors

**Given** PCA projection is complete
**When** I generate the cluster visualization
**Then**:
- ✅ Scatter plot displays all 120K documents as points in 2D space
- ✅ Each cluster (0-3) uses a distinct color from colorblind-friendly palette
- ✅ Use `matplotlib.pyplot.scatter()` or `seaborn.scatterplot()` for rendering
- ✅ Point size is small enough to show density but visible (e.g., s=1 or s=5)
- ✅ Alpha transparency applied if points overlap (e.g., alpha=0.6)
- ✅ Optional: Subsample to 10K points if plot is too dense (preserves cluster distribution)

**Validation:**
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
for cluster_id in range(4):
    mask = (labels == cluster_id)
    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
               label=f'Cluster {cluster_id}', s=5, alpha=0.6)
```

---

### AC-3: Cluster Centroids Visualization

**Given** PCA projection includes centroids
**When** I add centroids to the plot
**Then**:
- ✅ Cluster centroids are marked with special symbols (e.g., stars ★, or large circles)
- ✅ Centroid markers are larger than data points (e.g., s=200-300)
- ✅ Centroid markers use same color as corresponding cluster
- ✅ Centroid markers have black edge for visibility (edgecolors='black', linewidth=2)
- ✅ All 4 centroids are visible and distinguishable on plot

**Validation:**
```python
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           marker='*', s=300, c=colors, edgecolors='black', linewidth=2,
           label='Centroids', zorder=10)
```

---

### AC-4: Plot Formatting and Labels

**Given** scatter plot and centroids are rendered
**When** I finalize plot formatting
**Then**:
- ✅ X-axis label: "PC1 (XX.X% variance explained)"
- ✅ Y-axis label: "PC2 (XX.X% variance explained)"
- ✅ Title: "K-Means Clustering of AG News (K=4, PCA Projection)"
- ✅ Legend shows all clusters (Cluster 0, 1, 2, 3) and centroids
- ✅ Legend positioned to not obscure data (e.g., loc='best' or loc='upper right')
- ✅ Grid enabled for easier reading (ax.grid(True, alpha=0.3))
- ✅ Tight layout applied to prevent label cutoff (plt.tight_layout())

**Validation:**
```python
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('K-Means Clustering of AG News (K=4, PCA Projection)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

---

### AC-5: High-Quality PNG Export

**Given** plot is fully formatted
**When** I save the visualization
**Then**:
- ✅ Visualization is saved to `visualizations/cluster_pca.png`
- ✅ Resolution: 300 DPI (publication quality)
- ✅ Format: PNG with transparency support
- ✅ File size reasonable (<5 MB for full 120K points, <1 MB if subsampled)
- ✅ Output directory `visualizations/` created if it doesn't exist
- ✅ Saved file path is logged

**Validation:**
```python
from pathlib import Path
output_path = Path('visualizations/cluster_pca.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
assert output_path.exists()
assert output_path.stat().st_size > 0
```

---

### AC-6: Optional Interactive Plotly Visualization

**Given** static PNG visualization is complete (optional enhancement)
**When** I generate an interactive version
**Then**:
- ✅ Plotly scatter plot is created with same data
- ✅ Hover tooltips show: cluster ID, document index (optional: text preview)
- ✅ Interactive legend allows toggling cluster visibility
- ✅ Zoom and pan enabled for exploring dense regions
- ✅ Saved as `visualizations/cluster_pca.html`
- ✅ HTML file is standalone (embeds all data, no external dependencies)

**Validation:**
```python
import plotly.graph_objects as go
fig = go.Figure()
for cluster_id in range(4):
    mask = (labels == cluster_id)
    fig.add_trace(go.Scatter(
        x=embeddings_2d[mask, 0], y=embeddings_2d[mask, 1],
        mode='markers', name=f'Cluster {cluster_id}'
    ))
fig.write_html('visualizations/cluster_pca.html')
```

---

### AC-7: Variance Explained Logging

**Given** PCA transformation is complete
**When** variance explained is calculated
**Then**:
- ✅ PC1 variance explained is logged (e.g., "PC1: 15.3%")
- ✅ PC2 variance explained is logged (e.g., "PC2: 8.7%")
- ✅ Combined variance (PC1 + PC2) is logged (e.g., "Total: 24.0%")
- ✅ If combined <20%: log warning "⚠️ Low variance explained (XX%), 2D projection may lose information"
- ✅ If combined >=20%: log success "✅ Good variance explained (XX%), 2D projection captures main structure"

**Validation:**
```python
pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100
total_var = pc1_var + pc2_var
logger.info(f"📊 PC1 variance: {pc1_var:.1f}%")
logger.info(f"📊 PC2 variance: {pc2_var:.1f}%")
logger.info(f"📊 Total variance explained: {total_var:.1f}%")
if total_var < 20:
    logger.warning(f"⚠️ Low variance explained ({total_var:.1f}%), 2D projection may lose information")
```

---

### AC-8: Colorblind-Friendly Palette

**Given** visualization uses colors to distinguish clusters
**When** I select the color palette
**Then**:
- ✅ Colors are distinguishable for colorblind viewers
- ✅ Use recommended palette: `seaborn.color_palette("colorblind", 4)` or matplotlib "tab10"
- ✅ Avoid red-green combinations (problematic for deuteranopia)
- ✅ Alternative: Use different markers per cluster (circle, square, triangle, diamond)
- ✅ Color legend is clear and readable

**Validation:**
```python
import seaborn as sns
colors = sns.color_palette("colorblind", 4)
# Or: colors = plt.cm.tab10.colors[:4]
```

---

### AC-9: Logging and Observability

**Given** PCA visualization script is running
**When** major operations are performed
**Then**:
- ✅ Emoji-prefixed logs for visual clarity:
  - INFO: "📊 Loading embeddings and cluster labels..."
  - SUCCESS: "✅ Loaded 120000 embeddings and labels"
  - INFO: "📊 Applying PCA dimensionality reduction (768D → 2D)..."
  - SUCCESS: "✅ PCA complete. Variance explained: XX.X%"
  - INFO: "📊 Generating cluster scatter plot..."
  - SUCCESS: "✅ Scatter plot rendered with 4 clusters"
  - INFO: "📊 Saving visualization to visualizations/cluster_pca.png..."
  - SUCCESS: "✅ Visualization saved (300 DPI PNG)"
- ✅ All major steps logged with timing information
- ✅ Summary logged at completion:
```
✅ PCA Cluster Visualization Complete
   - Documents visualized: 120,000
   - Variance explained: 24.0% (PC1: 15.3%, PC2: 8.7%)
   - Output: visualizations/cluster_pca.png (300 DPI)
```

---

### AC-10: Error Handling

**Given** the visualization script is executed
**When** errors may occur
**Then**:
- ✅ Clear error if embeddings file missing (suggests running Story 2.1 script)
- ✅ Clear error if cluster assignments file missing (suggests running Story 2.2 script)
- ✅ Clear error if centroids file missing (suggests running Story 2.2 script)
- ✅ Validation error if embedding shapes don't match label count
- ✅ Validation error if centroids shape not (4, 768)
- ✅ Automatic directory creation if output paths don't exist
- ✅ Graceful handling if Plotly not installed (skip interactive plot, log info message)

**Validation:**
```python
# Test missing embeddings
if not Path('data/embeddings/train_embeddings.npy').exists():
    raise FileNotFoundError(
        "Embeddings not found: data/embeddings/train_embeddings.npy\n"
        "Run 'python scripts/01_generate_embeddings.py' first"
    )

# Test shape mismatch
assert len(embeddings) == len(labels), "Embeddings and labels count mismatch"
assert centroids.shape == (4, 768), f"Expected centroids shape (4, 768), got {centroids.shape}"
```

---

## Tasks / Subtasks

- [ ] Implement PCAVisualizer class in `src/visualization/cluster_plots.py` (AC: #1, #2, #3, #4, #5, #8)
  - [ ] Create PCAVisualizer class with `__init__` accepting embeddings, labels, centroids
  - [ ] Implement `apply_pca(n_components=2)` method for dimensionality reduction
  - [ ] Implement `generate_scatter_plot()` method with cluster coloring
  - [ ] Implement `add_centroids_to_plot()` method with star markers
  - [ ] Implement `format_plot()` method for labels, title, legend, grid
  - [ ] Implement `save_visualization(output_path, dpi=300)` method
  - [ ] Add colorblind-friendly palette support (seaborn "colorblind" or matplotlib "tab10")
  - [ ] Add type hints: `apply_pca(self) -> tuple[np.ndarray, np.ndarray, float]`
  - [ ] Add Google-style docstrings with usage examples for all methods
  - [ ] Return variance explained: `get_variance_explained() -> tuple[float, float, float]`

- [ ] Create PCA visualization script `scripts/04_visualize_clusters.py` (AC: #7, #9, #10)
  - [ ] Import required modules: Config, Paths, PCAVisualizer, logger
  - [ ] Implement set_seed(42) at script start for reproducibility
  - [ ] Load configuration from config.yaml
  - [ ] Setup logging with emoji prefixes
  - [ ] Load embeddings from `data/embeddings/train_embeddings.npy`
  - [ ] Load cluster assignments from `data/processed/cluster_assignments.csv`
  - [ ] Load centroids from `data/processed/centroids.npy`
  - [ ] Validate inputs: file existence, shape consistency, label range [0,3]
  - [ ] If files missing, raise FileNotFoundError with clear message and next steps
  - [ ] Initialize PCAVisualizer with loaded data
  - [ ] Call `apply_pca()` to reduce dimensionality
  - [ ] Log variance explained (PC1, PC2, total)
  - [ ] Check if total variance >=20%, log warning if below threshold
  - [ ] Generate scatter plot with all clusters
  - [ ] Add centroids to plot
  - [ ] Format plot with labels, title, legend
  - [ ] Create `visualizations/` directory if doesn't exist
  - [ ] Save visualization to `visualizations/cluster_pca.png` (300 DPI)
  - [ ] Log save operation with file path
  - [ ] Display final summary with variance explained and output path

- [ ] Implement PCA dimensionality reduction (AC: #1)
  - [ ] Use `sklearn.decomposition.PCA(n_components=2, random_state=42)`
  - [ ] Fit PCA on embeddings: `pca.fit(embeddings)`
  - [ ] Transform embeddings: `embeddings_2d = pca.transform(embeddings)`
  - [ ] Transform centroids: `centroids_2d = pca.transform(centroids)`
  - [ ] Calculate variance explained: `pca.explained_variance_ratio_`
  - [ ] Validate variance in range [0, 1]
  - [ ] Return embeddings_2d (120000, 2), centroids_2d (4, 2), variance_total

- [ ] Implement scatter plot generation (AC: #2, #3, #4)
  - [ ] Create matplotlib figure: `fig, ax = plt.subplots(figsize=(10, 8))`
  - [ ] Get colorblind-friendly palette: `sns.color_palette("colorblind", 4)`
  - [ ] For each cluster (0-3):
    - [ ] Create cluster mask: `mask = (labels == cluster_id)`
    - [ ] Plot cluster points: `ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1])`
    - [ ] Set color, label, size (s=5), alpha (0.6)
  - [ ] Plot centroids with star markers: `ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='*', s=300)`
  - [ ] Set centroid edge: `edgecolors='black', linewidth=2, zorder=10`
  - [ ] Add axis labels with variance: `ax.set_xlabel(f'PC1 ({pc1_var:.1f}% variance)')`
  - [ ] Add title: "K-Means Clustering of AG News (K=4, PCA Projection)"
  - [ ] Add legend: `ax.legend(loc='best')`
  - [ ] Enable grid: `ax.grid(True, alpha=0.3)`
  - [ ] Apply tight layout: `plt.tight_layout()`

- [ ] Implement high-quality PNG export (AC: #5)
  - [ ] Resolve output path: `visualizations/cluster_pca.png`
  - [ ] Create output directory: `Path('visualizations').mkdir(parents=True, exist_ok=True)`
  - [ ] Save figure: `plt.savefig(output_path, dpi=300, bbox_inches='tight')`
  - [ ] Validate file exists and has non-zero size
  - [ ] Log save operation: "✅ Visualization saved to {output_path} (300 DPI)"

- [ ] Optional: Implement interactive Plotly visualization (AC: #6)
  - [ ] Check if plotly is installed: `try: import plotly.graph_objects as go`
  - [ ] If not installed: log info "ℹ️ Plotly not installed, skipping interactive plot"
  - [ ] If installed:
    - [ ] Create Plotly figure: `fig = go.Figure()`
    - [ ] For each cluster, add scatter trace with hover info
    - [ ] Update layout with axis labels, title
    - [ ] Save as HTML: `fig.write_html('visualizations/cluster_pca.html')`
    - [ ] Log: "✅ Interactive visualization saved to cluster_pca.html"

- [ ] Test PCA visualization (AC: #1-#10)
  - [ ] Unit test: PCAVisualizer.apply_pca() on small synthetic dataset (1000 samples)
  - [ ] Unit test: Verify PCA output shapes (embeddings_2d: (n, 2), centroids_2d: (4, 2))
  - [ ] Unit test: Verify variance explained in range [0, 1]
  - [ ] Unit test: Verify total variance >0
  - [ ] Integration test: Run full script on actual cluster results from Story 2.2
  - [ ] Integration test: Verify visualization file exists and has correct format (PNG, 300 DPI)
  - [ ] Integration test: Verify variance explained >20% (or log warning if below)
  - [ ] Visual inspection: Check scatter plot shows 4 distinct clusters
  - [ ] Visual inspection: Check centroids are marked with stars
  - [ ] Visual inspection: Check legend is readable and colors are distinguishable
  - [ ] Negative test: Missing embeddings → FileNotFoundError
  - [ ] Negative test: Missing cluster assignments → FileNotFoundError
  - [ ] Negative test: Shape mismatch → ValueError

- [ ] Update project documentation (AC: all)
  - [ ] Update README.md with PCA visualization script usage
  - [ ] Document script usage: `python scripts/04_visualize_clusters.py`
  - [ ] Document expected output: visualizations/cluster_pca.png (300 DPI)
  - [ ] Document variance explained interpretation (>20% is good)
  - [ ] Add troubleshooting section for common errors
  - [ ] Add note about optional Plotly interactive visualization

## Dev Notes

### Architecture Alignment

This story implements the **PCA Cluster Visualization** component defined in the architecture. It integrates with:

1. **Cookiecutter Data Science Structure**: Follows src/visualization/ for plotting logic, scripts/ for execution
2. **Story 2.2 Outputs**: Consumes cluster assignments, centroids from K-Means clustering
3. **Story 2.1 Outputs**: Uses embeddings from `data/embeddings/train_embeddings.npy`
4. **Configuration System**: Uses config.yaml for visualization parameters (DPI, figure size, palette)
5. **Reporting**: Produces publication-quality visualizations for academic report

**Constraints Applied:**
- **Performance**: PCA + plotting <5 minutes for 120K documents (NFR-1 from PRD)
- **Reproducibility**: Fixed random_state=42 for PCA ensures deterministic projections
- **Logging**: Uses emoji-prefixed logging (📊, ✅, ⚠️, ❌) from utils/logger.py
- **Error Handling**: Validates input file existence and data schema before visualization

**Architectural Patterns Followed:**
- Initialization Order: set_seed → load config → setup logger → validate → execute
- Data Loading: Check file exists → load → validate → process
- File Naming: snake_case for modules (cluster_plots.py), PascalCase for classes (PCAVisualizer)
- Configuration Access: No hardcoded values, all parameters from config.yaml

### PCA Visualization Strategy

**Why PCA for Dimensionality Reduction:**

**1. Linear Projection (768D → 2D)**
- PCA finds the two principal components (directions) that capture maximum variance
- These components are linear combinations of original 768 dimensions
- Preserves global structure: similar documents remain close after projection
- Fast computation: Randomized SVD for large datasets (120K × 768)

**2. Variance Explained (Target: >20%)**
- PC1 typically captures 10-15% of variance (dominant direction)
- PC2 captures 5-10% of variance (second most important direction)
- Combined >20% means 2D projection retains meaningful information
- If <20%: projection loses significant structure, but still useful for visualization

**3. Interpretability**
- PC1 and PC2 are orthogonal (independent) directions
- Axis labels show variance explained (e.g., "PC1: 15.3% variance")
- Users can assess how much information is preserved in 2D view

**4. Cluster Separation Visibility**
- Good clustering: clusters visually separated in 2D projection
- Poor clustering: clusters overlap or mixed in 2D projection
- Complements quantitative metrics (Silhouette Score) with visual validation

**Alternative Approaches (Not Used for MVP):**
- **t-SNE**: Better for local structure, but non-deterministic, slower, no variance explained metric
- **UMAP**: Preserves both local and global structure, but adds dependency, harder to interpret
- **3D PCA**: More variance captured, but harder to visualize in static report

**Expected Behavior:**
- Variance explained 20-30% typical for high-dimensional text embeddings
- 4 clusters should show some visual separation (not perfect, but distinguishable)
- Centroids should be positioned near cluster centers in 2D projection
- If clusters overlap in 2D: expected for complex semantic data, metrics are primary validation

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
```

**Output Data:**
```python
# PCA-Projected Embeddings (intermediate)
Type: np.ndarray
Shape: (120000, 2)
Dtype: float64 (PCA output)
Description: 2D coordinates for all documents

# PCA-Projected Centroids (intermediate)
Type: np.ndarray
Shape: (4, 2)
Dtype: float64 (PCA output)
Description: 2D coordinates for cluster centroids

# Visualization Output
Type: PNG image file
Path: visualizations/cluster_pca.png
Resolution: 300 DPI (publication quality)
Size: ~500KB - 5MB (depending on point density)
Format: RGB PNG with transparency

# Optional: Interactive HTML
Type: HTML file
Path: visualizations/cluster_pca.html
Format: Standalone HTML with embedded Plotly.js
Size: ~2-10MB (includes full data)
```

**API Contracts:**
```python
class PCAVisualizer:
    def __init__(
        self,
        embeddings: np.ndarray,     # (n_documents, 768) float32
        labels: np.ndarray,          # (n_documents,) int32
        centroids: np.ndarray        # (4, 768) float32
    ):
        """
        Initialize PCA visualizer.

        Args:
            embeddings: Document embeddings
            labels: Cluster assignments from K-Means
            centroids: Cluster centroids from K-Means
        """

    def apply_pca(self, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Apply PCA dimensionality reduction.

        Args:
            n_components: Number of principal components (default: 2)

        Returns:
            Tuple of (embeddings_2d, centroids_2d, variance_explained)
            - embeddings_2d: (n_documents, 2) float64
            - centroids_2d: (4, 2) float64
            - variance_explained: float (0-1, combined PC1+PC2 variance)
        """

    def generate_visualization(
        self,
        output_path: Path,
        dpi: int = 300,
        figsize: tuple[int, int] = (10, 8)
    ) -> Path:
        """
        Generate and save cluster visualization.

        Args:
            output_path: Path to save PNG file
            dpi: Resolution in dots per inch (default: 300)
            figsize: Figure size in inches (width, height)

        Returns:
            Path to saved visualization file

        Raises:
            FileNotFoundError: If input data files missing
            ValueError: If data validation fails
        """
```

### Project Structure Notes

After completion, the following files will be created/modified:

**New Files:**
- `src/visualization/__init__.py` - Package init (if doesn't exist)
- `src/visualization/cluster_plots.py` - PCAVisualizer class
- `scripts/04_visualize_clusters.py` - Orchestration script for PCA visualization
- `visualizations/cluster_pca.png` - Static cluster visualization (300 DPI)
- `visualizations/cluster_pca.html` - Interactive Plotly visualization (optional)

**No Modified Files** (this story only creates new visualization outputs)

**Expected Directory Structure:**
```
context-aware-multi-agent-system/
├── scripts/
│   ├── 01_generate_embeddings.py       # EXISTING: From Story 2.1
│   ├── 02_train_clustering.py          # EXISTING: From Story 2.2
│   ├── 03_evaluate_clustering.py       # EXISTING: From Story 2.3
│   └── 04_visualize_clusters.py        # NEW: PCA visualization
├── data/
│   ├── embeddings/                     # EXISTING: From Story 2.1
│   │   └── train_embeddings.npy        # INPUT: 120K embeddings
│   └── processed/                      # EXISTING: From Story 2.2
│       ├── cluster_assignments.csv     # INPUT: Cluster labels
│       └── centroids.npy               # INPUT: 4 centroids
├── src/
│   ├── visualization/                  # NEW: Visualization module
│   │   ├── __init__.py                 # NEW: Package init
│   │   └── cluster_plots.py            # NEW: PCAVisualizer class
│   └── utils/
│       ├── logger.py                   # EXISTING: Reused for logging
│       └── reproducibility.py          # EXISTING: Reused for set_seed(42)
├── visualizations/                     # NEW: Visualization outputs
│   ├── cluster_pca.png                 # NEW: Static visualization
│   └── cluster_pca.html                # NEW: Interactive visualization (optional)
└── config.yaml                         # EXISTING: May add visualization section
```

### Testing Standards

**Unit Tests:**
```python
# Test PCAVisualizer.apply_pca() on small dataset
def test_pca_dimensionality_reduction():
    embeddings = np.random.randn(1000, 768).astype(np.float32)
    labels = np.random.randint(0, 4, 1000).astype(np.int32)
    centroids = np.random.randn(4, 768).astype(np.float32)

    visualizer = PCAVisualizer(embeddings, labels, centroids)
    embeddings_2d, centroids_2d, variance = visualizer.apply_pca()

    assert embeddings_2d.shape == (1000, 2)
    assert centroids_2d.shape == (4, 2)
    assert 0 <= variance <= 1.0
    assert isinstance(variance, float)

# Test variance explained calculation
def test_variance_explained():
    visualizer = PCAVisualizer(embeddings, labels, centroids)
    embeddings_2d, centroids_2d, variance = visualizer.apply_pca()

    # Variance should be positive and less than 1
    assert variance > 0
    assert variance < 1.0

# Test colorblind palette
def test_colorblind_palette():
    import seaborn as sns
    colors = sns.color_palette("colorblind", 4)
    assert len(colors) == 4
```

**Integration Tests:**
```python
# Test full visualization pipeline
def test_full_visualization_pipeline():
    result = subprocess.run(['python', 'scripts/04_visualize_clusters.py'],
                           capture_output=True)
    assert result.returncode == 0

    # Verify output exists
    assert Path('visualizations/cluster_pca.png').exists()

    # Verify PNG format and DPI
    from PIL import Image
    img = Image.open('visualizations/cluster_pca.png')
    assert img.format == 'PNG'
    assert img.info['dpi'] == (300, 300)

# Test variance threshold logging
def test_variance_logging(caplog):
    # Run visualization and check logs
    # Should log variance explained and warning if <20%
    pass
```

**Visual Inspection Tests:**
```python
# These should be manually verified by reviewing the saved PNG
def test_visual_quality():
    """
    Manual inspection checklist:
    - [ ] 4 clusters visible with distinct colors
    - [ ] Centroids marked with star symbols
    - [ ] Legend shows all clusters
    - [ ] Axis labels include variance explained
    - [ ] Title is clear and descriptive
    - [ ] No overlapping labels
    - [ ] Grid is visible but not distracting
    """
    pass
```

**Expected Test Coverage:**
- PCAVisualizer class: all PCA and plotting methods
- Variance explained: range validation, threshold check
- File I/O: PNG save, path creation, file validation
- Error handling: missing files, shape mismatches
- Performance: execution time <5 minutes

### Learnings from Previous Story

**From Story 2-3-cluster-quality-evaluation (Status: done):**

- ✅ **Cluster Outputs Available**: Use cluster results from Story 2.2
  - Cluster assignments: `data/processed/cluster_assignments.csv`
  - Centroids: `data/processed/centroids.npy` (4 × 768 float32)
  - Validation: Check files exist before loading

- ✅ **Configuration Pattern**: Follow established config access pattern
  - Use `config.get("visualization.dpi")` for resolution (if configured)
  - Use `config.get("visualization.figsize")` for figure dimensions
  - Use `paths.visualizations` for output directory
  - Add visualization section to config.yaml if needed

- ✅ **Logging Pattern**: Follow emoji-prefixed logging from previous stories
  - INFO: "📊 Loading embeddings and cluster labels..."
  - SUCCESS: "✅ Loaded 120000 embeddings and labels"
  - INFO: "📊 Applying PCA dimensionality reduction (768D → 2D)..."
  - SUCCESS: "✅ PCA complete. Variance explained: 24.0%"
  - INFO: "📊 Generating cluster scatter plot..."
  - SUCCESS: "✅ Scatter plot rendered with 4 clusters"
  - INFO: "📊 Saving visualization to visualizations/cluster_pca.png..."
  - SUCCESS: "✅ Visualization saved (300 DPI PNG)"
  - WARNING: "⚠️ Low variance explained (18.5%), 2D projection may lose information"
  - ERROR: "❌ Visualization failed: {error_message}"

- ✅ **Reproducibility Pattern**: Reuse set_seed() from previous stories
  - Call set_seed(42) at script start (for consistency)
  - PCA uses random_state=42 for deterministic projections
  - Ensures reproducible visualizations

- ✅ **Error Handling Pattern**: Follow previous stories' error handling approach
  - Clear error messages with troubleshooting guidance
  - FileNotFoundError if embeddings missing: suggest running Story 2.1 script
  - FileNotFoundError if cluster assignments missing: suggest running Story 2.2 script
  - ValueError for validation failures with helpful context
  - Provide actionable next steps

- ✅ **Type Hints and Docstrings**: Maintain documentation standards
  - All methods have full type hints
  - Google-style docstrings with usage examples
  - Example: `def apply_pca(self, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, float]:`

- ✅ **Data Validation Pattern**: Follow validation approach
  - Pre-flight checks: file exists, shape correct, dtype correct, no NaN/Inf
  - Fail-fast with clear error messages
  - Log validation success for debugging

- ✅ **Directory Creation**: Follow pattern for output directories
  - Use `Path.mkdir(parents=True, exist_ok=True)` to create directories
  - Create visualizations/ if it doesn't exist
  - No errors if directories already exist

- ✅ **Testing Pattern**: Follow comprehensive test approach
  - Create `tests/epic2/test_cluster_plots.py`
  - Map tests to acceptance criteria (AC-1, AC-2, etc.)
  - Use pytest.raises() for exception testing
  - Use pytest fixtures for test setup (temp directories, mock data)
  - Test both unit (small synthetic data) and integration (full dataset)

**Files to Reuse (DO NOT RECREATE):**
- `src/utils/logger.py` - Use for emoji-prefixed logging
- `src/utils/reproducibility.py` - Use set_seed(42) function
- `src/config.py` - Load config for visualization parameters (if configured)
- `data/embeddings/train_embeddings.npy` - Input from Story 2.1
- `data/processed/cluster_assignments.csv` - Input from Story 2.2
- `data/processed/centroids.npy` - Input from Story 2.2

**Key Services from Previous Stories:**
- **Config class** (Story 1.2): Configuration management with get() method
- **Paths class** (Story 1.2): Path resolution
- **set_seed()** (Story 1.1): Reproducibility enforcement
- **Logger** (Story 1.2): Emoji-prefixed structured logging
- **KMeansClustering** (Story 2.2): Cluster labels and centroids available
- **ClusteringMetrics** (Story 2.3): Quality metrics already calculated

**Technical Debt from Previous Stories:**
- None affecting this story - All prerequisite stories are complete and approved

**Review Findings from Story 2.3 to Apply:**
- ✅ Use comprehensive docstrings with usage examples
- ✅ Add type hints to all method signatures
- ✅ Include explicit validation checks with informative error messages
- ✅ Log all major operations for debugging
- ✅ Write tests covering all acceptance criteria
- ✅ Use emoji-prefixed logging throughout (📊, ✅, ⚠️, ❌)
- ✅ Add threshold validation with warnings (variance <20%)

**New Patterns to Establish:**
- **PCA Visualization Pattern**: Load data → Apply PCA → Generate plot → Format → Save
- **Matplotlib Publication Quality**: 300 DPI, tight layout, colorblind-friendly palette
- **Variance Threshold Validation**: If variance < threshold: log warning, continue (don't fail)
- **Optional Enhancement Pattern**: Try import optional dependency → skip if not available → log info

[Source: stories/2-3-cluster-quality-evaluation.md#Dev-Agent-Record]

### References

- [Source: docs/tech-spec-epic-2.md#Story 2.4 - PCA Cluster Visualization]
- [Source: docs/tech-spec-epic-2.md#Detailed Design → Services and Modules → ClusterPlots]
- [Source: docs/tech-spec-epic-2.md#APIs and Interfaces → ClusterPlots API]
- [Source: docs/epics.md#Story 2.4 - PCA Cluster Visualization]
- [Source: docs/PRD.md#FR-5 - Cluster Visualization]
- [Source: docs/architecture.md#Visualization Components]
- [Source: stories/2-2-k-means-clustering-implementation.md#Cluster Results Available]
- [Source: stories/2-3-cluster-quality-evaluation.md#Learnings from Previous Story]

## Change Log

### 2025-11-09 - Story Drafted
- **Version:** v1.0
- **Changes:**
  - ✅ Story created from epics.md and tech-spec-epic-2.md
  - ✅ All 10 acceptance criteria defined with validation examples
  - ✅ Tasks and subtasks mapped to ACs
  - ✅ Dev notes include architecture alignment and learnings from Story 2.3
  - ✅ References to source documents included
- **Status:** backlog → drafted

---

## Dev Agent Record

### Context Reference

<!-- Path(s) to story context XML will be added here by context workflow -->

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
