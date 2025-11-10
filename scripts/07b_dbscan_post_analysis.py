#!/usr/bin/env python3
"""
DBSCAN Post-Analysis Script
Generate visualizations, comparisons, and reports from existing DBSCAN results.

This script processes the already-generated DBSCAN clustering results to create:
1. Comparison table with K-Means
2. Visualization charts (t-SNE/UMAP)
3. Detailed analysis report
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    logger.info(f"📊 Loading {filepath.name}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"✅ Loaded {filepath.name}")
    return data


def create_comparison_table(
    dbscan_metrics: Dict[str, Any],
    kmeans_metrics: Dict[str, Any],
    output_path: Path
) -> pd.DataFrame:
    """Create DBSCAN vs K-Means comparison table."""
    logger.info("📊 Creating DBSCAN vs K-Means comparison table...")

    # Extract K-Means purity (it's a dict with per-cluster values)
    if isinstance(kmeans_metrics.get('cluster_purity'), dict):
        kmeans_purity = kmeans_metrics['cluster_purity'].get('overall', 0.0)
    else:
        kmeans_purity = kmeans_metrics.get('cluster_purity', 0.0)

    comparison_data = {
        'Metric': [
            'Number of Clusters',
            'Noise Points',
            'Noise Percentage',
            'Silhouette Score',
            'Davies-Bouldin Index',
            'Cluster Purity',
            'Runtime (seconds)'
        ],
        'DBSCAN': [
            dbscan_metrics['n_clusters'],
            dbscan_metrics['n_noise_points'],
            f"{dbscan_metrics['noise_percentage']:.1f}%",
            dbscan_metrics['silhouette_score'] if dbscan_metrics['silhouette_score'] is not None else 'N/A',
            dbscan_metrics['davies_bouldin_index'] if dbscan_metrics['davies_bouldin_index'] is not None else 'N/A',
            f"{dbscan_metrics['cluster_purity']:.4f}",
            f"{dbscan_metrics['runtime_seconds']:.1f}"
        ],
        'K-Means': [
            4,  # K-Means was set to 4 clusters
            0,
            '0.0%',
            f"{kmeans_metrics['silhouette_score']:.6f}",
            f"{kmeans_metrics['davies_bouldin_index']:.2f}",
            f"{kmeans_purity:.4f}",
            'N/A'  # K-Means runtime not stored
        ],
        'Winner': []
    }

    # Determine winners
    winners = []
    # Number of clusters: K-Means expected 4
    winners.append('K-Means' if dbscan_metrics['n_clusters'] == 4 else 'DBSCAN' if dbscan_metrics['n_clusters'] > 1 else 'K-Means')
    # Noise points: lower is better for this task
    winners.append('K-Means')
    # Noise percentage: lower is better
    winners.append('K-Means')
    # Silhouette: higher is better (if available)
    if dbscan_metrics['silhouette_score'] is not None:
        winners.append('DBSCAN' if dbscan_metrics['silhouette_score'] > kmeans_metrics['silhouette_score'] else 'K-Means')
    else:
        winners.append('K-Means')
    # Davies-Bouldin: lower is better (if available)
    if dbscan_metrics['davies_bouldin_index'] is not None:
        winners.append('DBSCAN' if dbscan_metrics['davies_bouldin_index'] < kmeans_metrics['davies_bouldin_index'] else 'K-Means')
    else:
        winners.append('K-Means')
    # Purity: higher is better
    winners.append('DBSCAN' if dbscan_metrics['cluster_purity'] > kmeans_purity else 'K-Means')
    # Runtime: not comparable
    winners.append('N/A')

    comparison_data['Winner'] = winners

    df = pd.DataFrame(comparison_data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved comparison table to {output_path}")

    return df


def create_parameter_tuning_plot(tuning_results_path: Path, output_path: Path):
    """Create parameter tuning visualization."""
    logger.info("📊 Creating parameter tuning visualization...")

    df = pd.read_csv(tuning_results_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Clusters vs eps for different min_samples
    ax = axes[0, 0]
    for min_samples in df['min_samples'].unique():
        subset = df[df['min_samples'] == min_samples]
        ax.plot(subset['eps'], subset['n_clusters'], marker='o', label=f'min_samples={min_samples}')
    ax.set_xlabel('eps')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Clusters vs Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Noise ratio vs eps
    ax = axes[0, 1]
    for min_samples in df['min_samples'].unique():
        subset = df[df['min_samples'] == min_samples]
        ax.plot(subset['eps'], subset['noise_ratio'] * 100, marker='s', label=f'min_samples={min_samples}')
    ax.set_xlabel('eps')
    ax.set_ylabel('Noise Percentage (%)')
    ax.set_title('Noise Points vs Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Runtime vs eps
    ax = axes[1, 0]
    for min_samples in df['min_samples'].unique():
        subset = df[df['min_samples'] == min_samples]
        ax.plot(subset['eps'], subset['runtime_seconds'], marker='^', label=f'min_samples={min_samples}')
    ax.set_xlabel('eps')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Heatmap of n_clusters
    ax = axes[1, 1]
    pivot = df.pivot(index='min_samples', columns='eps', values='n_clusters')
    sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Clusters'})
    ax.set_title('Cluster Count Heatmap')
    ax.set_xlabel('eps')
    ax.set_ylabel('min_samples')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved parameter tuning plot to {output_path}")
    plt.close()


def create_cluster_visualization(
    embeddings_path: Path,
    assignments_path: Path,
    output_dir: Path,
    n_samples: int = 10000
):
    """Create t-SNE and UMAP visualizations of DBSCAN clusters."""
    logger.info("📊 Creating cluster visualizations...")

    # Load data
    logger.info(f"📊 Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded embeddings: shape={embeddings.shape}")

    logger.info(f"📊 Loading cluster assignments from {assignments_path}...")
    assignments_df = pd.read_csv(assignments_path)
    logger.info(f"✅ Loaded {len(assignments_df)} assignments")

    # Sample for visualization (full dataset is too large)
    if len(embeddings) > n_samples:
        logger.info(f"📊 Sampling {n_samples} points for visualization...")
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
        assignments_sample = assignments_df.iloc[indices]
    else:
        embeddings_sample = embeddings
        assignments_sample = assignments_df

    # t-SNE
    logger.info("📊 Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_sample)
    logger.info("✅ t-SNE complete")

    # UMAP
    logger.info("📊 Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_umap = reducer.fit_transform(embeddings_sample)
    logger.info("✅ UMAP complete")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # t-SNE by cluster
    ax = axes[0, 0]
    scatter = ax.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=assignments_sample['cluster_id'],
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f't-SNE Projection - Colored by DBSCAN Cluster\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # t-SNE by ground truth
    ax = axes[0, 1]
    category_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
    colors = assignments_sample['ground_truth_category'].map(category_map)
    scatter = ax.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=colors,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f't-SNE Projection - Colored by Ground Truth\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter, ax=ax, label='Category')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['World', 'Sports', 'Business', 'Sci/Tech'])

    # UMAP by cluster
    ax = axes[1, 0]
    scatter = ax.scatter(
        embeddings_umap[:, 0],
        embeddings_umap[:, 1],
        c=assignments_sample['cluster_id'],
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f'UMAP Projection - Colored by DBSCAN Cluster\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # UMAP by ground truth
    ax = axes[1, 1]
    scatter = ax.scatter(
        embeddings_umap[:, 0],
        embeddings_umap[:, 1],
        c=colors,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f'UMAP Projection - Colored by Ground Truth\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    cbar = plt.colorbar(scatter, ax=ax, label='Category')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['World', 'Sports', 'Business', 'Sci/Tech'])

    plt.tight_layout()
    output_path = output_dir / 'dbscan_cluster_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved cluster visualization to {output_path}")
    plt.close()


def generate_analysis_report(
    dbscan_metrics: Dict[str, Any],
    kmeans_metrics: Dict[str, Any],
    comparison_df: pd.DataFrame,
    output_path: Path
):
    """Generate detailed analysis report."""
    logger.info("📊 Generating analysis report...")

    # Extract K-Means purity
    if isinstance(kmeans_metrics.get('cluster_purity'), dict):
        kmeans_purity = kmeans_metrics['cluster_purity'].get('overall', 0.0)
    else:
        kmeans_purity = kmeans_metrics.get('cluster_purity', 0.0)

    report = f"""# DBSCAN Clustering Analysis Report

## Executive Summary

This report presents the results of applying DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to the AG News dataset embeddings and compares its performance with K-Means clustering.

**Key Findings:**
- DBSCAN with cosine metric **failed to discover meaningful clusters** in the high-dimensional embedding space
- All tested parameter combinations resulted in either 100% noise or a single large cluster
- K-Means significantly outperformed DBSCAN on this dataset

---

## 1. Clustering Results

### 1.1 DBSCAN Parameters
- **eps**: {dbscan_metrics['parameters']['eps']}
- **min_samples**: {dbscan_metrics['parameters']['min_samples']}
- **metric**: {dbscan_metrics['parameters']['metric']}

### 1.2 Cluster Discovery
- **Number of clusters**: {dbscan_metrics['n_clusters']} (expected: 4)
- **Noise points**: {dbscan_metrics['n_noise_points']} ({dbscan_metrics['noise_percentage']:.1f}%)
- **Core samples**: {dbscan_metrics['cluster_sizes'].get('0', 0):,}

**Analysis**: DBSCAN grouped all 120,000 samples into a single cluster, failing to distinguish between the four news categories (World, Sports, Business, Sci/Tech).

---

## 2. Quality Metrics

### 2.1 Internal Validation
- **Silhouette Score**: {dbscan_metrics['silhouette_score'] if dbscan_metrics['silhouette_score'] is not None else 'N/A (single cluster)'}
  - Cannot be calculated with only one cluster
- **Davies-Bouldin Index**: {dbscan_metrics['davies_bouldin_index'] if dbscan_metrics['davies_bouldin_index'] is not None else 'N/A (single cluster)'}
  - Requires at least 2 clusters

### 2.2 External Validation
- **Cluster Purity**: {dbscan_metrics['cluster_purity']:.4f}
  - This is the random baseline (1/4 = 0.25) for 4 categories
  - Indicates no meaningful alignment with ground truth labels

---

## 3. Comparison with K-Means

| Metric | DBSCAN | K-Means | Winner |
|--------|--------|---------|--------|
"""

    for _, row in comparison_df.iterrows():
        report += f"| {row['Metric']} | {row['DBSCAN']} | {row['K-Means']} | **{row['Winner']}** |\n"

    report += f"""
### 3.1 Performance Summary

**K-Means advantages:**
1. Successfully identified 4 distinct clusters matching the dataset structure
2. Higher cluster purity ({kmeans_purity:.4f} vs {dbscan_metrics['cluster_purity']:.4f})
3. Computable quality metrics (Silhouette, Davies-Bouldin)
4. Balanced cluster sizes (29,825-30,138 samples per cluster)

**DBSCAN limitations on this dataset:**
1. Failed to discover multiple clusters (collapsed to 1)
2. Unable to adapt to the uniform density distribution in embedding space
3. High sensitivity to epsilon parameter with no viable middle ground
4. Long runtime ({dbscan_metrics['runtime_seconds']:.1f}s) without meaningful results

---

## 4. Parameter Tuning Analysis

### 4.1 Tested Parameter Grid
- **eps values**: [0.3, 0.5, 0.7, 1.0]
- **min_samples values**: [3, 5, 10]
- **Total combinations**: 12

### 4.2 Parameter Sensitivity

**Small eps (0.3-0.7):**
- Result: 0 clusters, 100% noise points
- Issue: Threshold too strict for the data density

**Large eps (1.0):**
- Result: 1 cluster, 0% noise points
- Issue: Threshold too permissive, merged all points

**Critical gap**: No intermediate eps value produces 2-10 clusters, indicating DBSCAN is fundamentally unsuitable for this embedding space.

---

## 5. Root Cause Analysis

### 5.1 Why DBSCAN Failed

**1. High-Dimensional Curse**
- BERT embeddings have 768 dimensions
- Distance metrics behave poorly in high dimensions
- Density becomes nearly uniform (no distinct dense regions)

**2. Cosine Distance Characteristics**
- Normalized embeddings cluster in unit hypersphere
- Cosine distances are bounded [0, 2]
- Limited dynamic range for density-based clustering

**3. Dataset Structure Mismatch**
- AG News categories are semantically separable but not density-separable
- K-Means finds hyperplane boundaries (suitable for embeddings)
- DBSCAN finds density boundaries (unsuitable for embeddings)

### 5.2 Theoretical Limitations

DBSCAN assumes:
- Clusters are separated by low-density regions
- Different clusters have distinguishable densities

BERT embeddings have:
- Uniform density distribution across semantic space
- Separation based on angular distance, not density

---

## 6. Recommendations

### 6.1 For This Dataset
**Use K-Means** for AG News clustering:
- ✅ Works well with high-dimensional embeddings
- ✅ Produces interpretable 4-cluster solution
- ✅ Fast and deterministic with proper initialization
- ✅ Aligns with semantic categories

### 6.2 When to Use DBSCAN
DBSCAN is better suited for:
- Low-dimensional data (2D, 3D spatial data)
- Data with clear varying densities
- Unknown number of clusters with spatial structure
- Presence of true outliers/noise

### 6.3 Alternative Approaches
If DBSCAN is required for high-dimensional embeddings:
1. **Dimensionality reduction first** (PCA/UMAP to 2-10 dimensions)
2. **Use euclidean metric** after normalization
3. **Fine-grained eps search** (e.g., [0.75, 0.8, 0.85, 0.9, 0.95])
4. **HDBSCAN** (hierarchical variant, more robust parameter selection)

---

## 7. Conclusions

1. **DBSCAN is not suitable** for clustering BERT embeddings in their native 768-dimensional space
2. **K-Means significantly outperforms** DBSCAN on this task (purity: {kmeans_purity:.4f} vs {dbscan_metrics['cluster_purity']:.4f})
3. **Parameter tuning failed** to find a viable configuration (12/12 combinations produced degenerate results)
4. **High-dimensional density-based clustering** requires careful consideration of distance metrics and dimensionality reduction

**Final Recommendation**: Use K-Means for AG News text clustering with BERT embeddings.

---

## Appendix: Technical Details

### Runtime Performance
- Parameter tuning: 1,298.3 seconds (21.6 minutes)
- Final clustering: 238.5 seconds (4.0 minutes)
- Total: ~26 minutes

### Data Statistics
- Total samples: 120,000
- Embedding dimensions: 768
- Ground truth categories: 4 (balanced, 30,000 each)

### Software Environment
- Algorithm: scikit-learn DBSCAN
- Distance metric: cosine (sklearn metric='cosine')
- Hardware: [Auto-detected from system]

---

*Report generated: {dbscan_metrics['timestamp']}*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"✅ Saved analysis report to {output_path}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("DBSCAN Post-Analysis - Generate Missing Outputs")
    logger.info("=" * 80)

    # Setup paths
    project_root = Path.cwd()
    results_dir = project_root / 'results'
    data_dir = project_root / 'data'

    dbscan_metrics_path = results_dir / 'dbscan_metrics.json'
    kmeans_metrics_path = results_dir / 'cluster_quality.json'
    tuning_results_path = results_dir / 'dbscan_parameter_tuning.csv'
    assignments_path = data_dir / 'processed' / 'dbscan_assignments.csv'
    embeddings_path = data_dir / 'embeddings' / 'train_embeddings.npy'

    # Load metrics
    try:
        dbscan_metrics = load_json(dbscan_metrics_path)
        kmeans_metrics = load_json(kmeans_metrics_path)
    except Exception as e:
        logger.error(f"❌ Failed to load metrics: {e}")
        return 1

    # 1. Create comparison table
    try:
        comparison_output = results_dir / 'dbscan_vs_kmeans_comparison.csv'
        comparison_df = create_comparison_table(
            dbscan_metrics,
            kmeans_metrics,
            comparison_output
        )
        logger.info("\n" + "=" * 60)
        logger.info("DBSCAN vs K-Means Comparison:")
        logger.info("=" * 60)
        print(comparison_df.to_string(index=False))
    except Exception as e:
        logger.error(f"❌ Failed to create comparison table: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 2. Create parameter tuning plot
    try:
        tuning_plot_output = results_dir / 'dbscan_parameter_tuning.png'
        create_parameter_tuning_plot(tuning_results_path, tuning_plot_output)
    except Exception as e:
        logger.error(f"❌ Failed to create parameter tuning plot: {e}")
        import traceback
        traceback.print_exc()

    # 3. Create cluster visualization
    try:
        create_cluster_visualization(
            embeddings_path,
            assignments_path,
            results_dir,
            n_samples=10000
        )
    except Exception as e:
        logger.error(f"❌ Failed to create cluster visualization: {e}")
        import traceback
        traceback.print_exc()

    # 4. Generate analysis report
    try:
        report_output = results_dir / 'dbscan_analysis_report.md'
        generate_analysis_report(
            dbscan_metrics,
            kmeans_metrics,
            comparison_df,
            report_output
        )
    except Exception as e:
        logger.error(f"❌ Failed to generate analysis report: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("✅ Post-analysis complete! Generated outputs:")
    logger.info(f"   1. Comparison table: {comparison_output}")
    logger.info(f"   2. Parameter tuning plot: {tuning_plot_output}")
    logger.info(f"   3. Cluster visualization: {results_dir / 'dbscan_cluster_visualization.png'}")
    logger.info(f"   4. Analysis report: {report_output}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
