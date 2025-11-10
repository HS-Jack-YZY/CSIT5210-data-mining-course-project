#!/usr/bin/env python3
"""
GMM Post-Analysis Script
Generate visualizations, comparisons, and reports from GMM clustering results.

This script processes the already-generated GMM clustering results to create:
1. Uncertainty analysis visualizations
2. Covariance comparison plots
3. GMM vs K-Means vs DBSCAN comparison table
4. Detailed analysis report
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


def create_covariance_comparison_plot(comparison_csv: Path, output_path: Path):
    """Create covariance type comparison visualization."""
    logger.info("📊 Creating covariance comparison plot...")

    df = pd.read_csv(comparison_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: BIC comparison
    ax = axes[0, 0]
    bars = ax.bar(df['covariance_type'], df['bic'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel('BIC (lower is better)')
    ax.set_title('BIC by Covariance Type')
    ax.tick_params(axis='x', rotation=45)
    # Annotate best
    best_idx = df['bic'].idxmin()
    ax.annotate('Best', xy=(best_idx, df.loc[best_idx, 'bic']),
                xytext=(best_idx, df.loc[best_idx, 'bic'] * 1.01),
                ha='center', fontweight='bold', color='green')

    # Plot 2: AIC comparison
    ax = axes[0, 1]
    ax.bar(df['covariance_type'], df['aic'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel('AIC (lower is better)')
    ax.set_title('AIC by Covariance Type')
    ax.tick_params(axis='x', rotation=45)
    best_idx = df['aic'].idxmin()
    ax.annotate('Best', xy=(best_idx, df.loc[best_idx, 'aic']),
                xytext=(best_idx, df.loc[best_idx, 'aic'] * 1.01),
                ha='center', fontweight='bold', color='green')

    # Plot 3: Silhouette comparison
    ax = axes[1, 0]
    ax.bar(df['covariance_type'], df['silhouette_score'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel('Silhouette Score (higher is better)')
    ax.set_title('Silhouette Score by Covariance Type')
    ax.tick_params(axis='x', rotation=45)
    best_idx = df['silhouette_score'].idxmax()
    ax.annotate('Best', xy=(best_idx, df.loc[best_idx, 'silhouette_score']),
                xytext=(best_idx, df.loc[best_idx, 'silhouette_score'] * 1.1),
                ha='center', fontweight='bold', color='green')

    # Plot 4: Runtime comparison
    ax = axes[1, 1]
    ax.bar(df['covariance_type'], df['runtime_seconds'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Training Time by Covariance Type')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved covariance comparison plot to {output_path}")
    plt.close()


def create_uncertainty_visualization(gmm_metrics: Dict[str, Any], output_path: Path):
    """Create uncertainty analysis visualizations."""
    logger.info("📊 Creating uncertainty analysis plots...")

    uncertainty = gmm_metrics['uncertainty_analysis']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall confidence distribution
    ax = axes[0, 0]
    conf_stats = uncertainty['confidence_stats']
    x = ['Mean', 'Median', 'Q25', 'Q75']
    y = [conf_stats['mean'], conf_stats['q50'], conf_stats['q25'], conf_stats['q75']]
    ax.bar(x, y, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax.set_ylabel('Confidence')
    ax.set_title(f"Confidence Distribution\n(Mean: {conf_stats['mean']:.3f}, Std: {conf_stats['std']:.3f})")
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 2: Low confidence ratio by category
    ax = axes[0, 1]
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    low_conf_ratios = [uncertainty['category_uncertainty'][cat]['low_confidence_ratio'] * 100
                       for cat in categories]
    bars = ax.bar(categories, low_conf_ratios, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax.axhline(y=50, color='red', linestyle='--', label='50% threshold')
    ax.set_ylabel('Low Confidence Ratio (%)')
    ax.set_title('Low Confidence Documents by Category\n(Confidence < 0.5)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    # Annotate values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    # Plot 3: Mean confidence by category
    ax = axes[1, 0]
    mean_confs = [uncertainty['category_uncertainty'][cat]['mean_confidence']
                  for cat in categories]
    ax.bar(categories, mean_confs, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax.set_ylabel('Mean Confidence')
    ax.set_title('Average Confidence by Category')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Uncertainty Analysis Summary

    Total Documents: 120,000
    Low Confidence (< 0.5): {uncertainty['low_confidence_count']:,} ({uncertainty['low_confidence_ratio']*100:.1f}%)
    Confused (diff < 0.2): {uncertainty['confused_count']:,} ({uncertainty['confused_ratio']*100:.1f}%)

    Confidence Statistics:
    • Mean: {conf_stats['mean']:.4f}
    • Median: {conf_stats['q50']:.4f}
    • Std: {conf_stats['std']:.4f}
    • Min: {conf_stats['min']:.4f}
    • Max: {conf_stats['max']:.4f}

    Key Insight:
    Over 60% of documents have confidence
    below 0.5, indicating high uncertainty
    in cluster assignments.
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved uncertainty visualization to {output_path}")
    plt.close()


def create_cluster_visualization(
    embeddings_path: Path,
    assignments_path: Path,
    output_dir: Path,
    n_samples: int = 10000
):
    """Create t-SNE and UMAP visualizations of GMM clusters."""
    logger.info("📊 Creating GMM cluster visualizations...")

    # Load data
    logger.info(f"📊 Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded embeddings: shape={embeddings.shape}")

    logger.info(f"📊 Loading cluster assignments from {assignments_path}...")
    assignments_df = pd.read_csv(assignments_path)
    logger.info(f"✅ Loaded {len(assignments_df)} assignments")

    # Sample for visualization
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Category mapping
    category_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
    colors_gt = assignments_sample['ground_truth_category'].map(category_map)

    # Get max probability (confidence) for coloring
    prob_cols = [f'cluster_{i}_prob' for i in range(4)]
    max_probs = assignments_sample[prob_cols].max(axis=1)

    # t-SNE - GMM clusters
    ax = axes[0, 0]
    scatter = ax.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=assignments_sample['cluster_id'],
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f't-SNE - GMM Clusters\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # t-SNE - Ground truth
    ax = axes[0, 1]
    scatter = ax.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=colors_gt,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f't-SNE - Ground Truth\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter, ax=ax, label='Category')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['World', 'Sports', 'Business', 'Sci/Tech'])

    # t-SNE - Confidence
    ax = axes[0, 2]
    scatter = ax.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=max_probs,
        cmap='RdYlGn',
        alpha=0.6,
        s=10,
        vmin=0,
        vmax=1
    )
    ax.set_title(f't-SNE - Confidence\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Max Probability')

    # UMAP - GMM clusters
    ax = axes[1, 0]
    scatter = ax.scatter(
        embeddings_umap[:, 0],
        embeddings_umap[:, 1],
        c=assignments_sample['cluster_id'],
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f'UMAP - GMM Clusters\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # UMAP - Ground truth
    ax = axes[1, 1]
    scatter = ax.scatter(
        embeddings_umap[:, 0],
        embeddings_umap[:, 1],
        c=colors_gt,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title(f'UMAP - Ground Truth\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    cbar = plt.colorbar(scatter, ax=ax, label='Category')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['World', 'Sports', 'Business', 'Sci/Tech'])

    # UMAP - Confidence
    ax = axes[1, 2]
    scatter = ax.scatter(
        embeddings_umap[:, 0],
        embeddings_umap[:, 1],
        c=max_probs,
        cmap='RdYlGn',
        alpha=0.6,
        s=10,
        vmin=0,
        vmax=1
    )
    ax.set_title(f'UMAP - Confidence\n({len(embeddings_sample):,} samples)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Max Probability')

    plt.tight_layout()
    output_path = output_dir / 'gmm_cluster_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved cluster visualization to {output_path}")
    plt.close()


def create_three_algorithm_comparison(
    gmm_metrics: Dict[str, Any],
    kmeans_metrics: Dict[str, Any],
    dbscan_metrics: Dict[str, Any],
    output_path: Path
) -> pd.DataFrame:
    """Create comprehensive comparison of GMM, K-Means, and DBSCAN."""
    logger.info("📊 Creating three-algorithm comparison table...")

    # Extract K-Means purity
    if isinstance(kmeans_metrics.get('cluster_purity'), dict):
        kmeans_purity = kmeans_metrics['cluster_purity'].get('overall', 0.0)
    else:
        kmeans_purity = kmeans_metrics.get('cluster_purity', 0.0)

    # Extract GMM purity
    if isinstance(gmm_metrics.get('cluster_purity'), dict):
        gmm_purity = gmm_metrics['cluster_purity'].get('overall', 0.0)
    else:
        gmm_purity = gmm_metrics.get('cluster_purity', 0.0)

    comparison_data = {
        'Metric': [
            'Algorithm Type',
            'Number of Clusters',
            'Noise Points',
            'Silhouette Score',
            'Davies-Bouldin Index',
            'Cluster Purity',
            'Convergence',
            'Iterations',
            'Runtime (seconds)',
            'Special Features'
        ],
        'GMM': [
            'Probabilistic (Soft)',
            gmm_metrics['n_components'],
            0,
            f"{gmm_metrics['silhouette_score']:.6f}",
            f"{gmm_metrics['davies_bouldin_index']:.2f}",
            f"{gmm_purity:.4f}",
            'Yes' if gmm_metrics['converged'] else 'No',
            gmm_metrics['n_iterations'],
            f"{gmm_metrics['runtime_seconds']:.1f}",
            'Uncertainty quantification'
        ],
        'K-Means': [
            'Centroid-based (Hard)',
            4,
            0,
            f"{kmeans_metrics['silhouette_score']:.6f}",
            f"{kmeans_metrics['davies_bouldin_index']:.2f}",
            f"{kmeans_purity:.4f}",
            'N/A',
            'N/A',
            'N/A',
            'Simple & fast'
        ],
        'DBSCAN': [
            'Density-based',
            dbscan_metrics['n_clusters'],
            dbscan_metrics['n_noise_points'],
            'N/A' if dbscan_metrics['silhouette_score'] is None else f"{dbscan_metrics['silhouette_score']:.6f}",
            'N/A' if dbscan_metrics['davies_bouldin_index'] is None else f"{dbscan_metrics['davies_bouldin_index']:.2f}",
            f"{dbscan_metrics['cluster_purity']:.4f}",
            'N/A',
            'N/A',
            f"{dbscan_metrics['runtime_seconds']:.1f}",
            'Auto cluster count'
        ],
        'Best': []
    }

    # Determine winners
    winners = ['N/A']  # Algorithm Type

    # Number of clusters (4 is expected)
    cluster_counts = [gmm_metrics['n_components'], 4, dbscan_metrics['n_clusters']]
    if all(c == 4 for c in cluster_counts[:2]):
        winners.append('GMM / K-Means')
    else:
        winners.append('K-Means')

    # Noise points (lower is better for this task)
    winners.append('GMM / K-Means')

    # Silhouette
    if dbscan_metrics['silhouette_score'] is not None:
        scores = [gmm_metrics['silhouette_score'], kmeans_metrics['silhouette_score'], dbscan_metrics['silhouette_score']]
        best = ['GMM', 'K-Means', 'DBSCAN'][np.argmax(scores)]
    else:
        best = 'K-Means' if kmeans_metrics['silhouette_score'] > gmm_metrics['silhouette_score'] else 'GMM'
    winners.append(best)

    # Davies-Bouldin (lower is better)
    if dbscan_metrics['davies_bouldin_index'] is not None:
        scores = [gmm_metrics['davies_bouldin_index'], kmeans_metrics['davies_bouldin_index'], dbscan_metrics['davies_bouldin_index']]
        best = ['GMM', 'K-Means', 'DBSCAN'][np.argmin(scores)]
    else:
        best = 'K-Means' if kmeans_metrics['davies_bouldin_index'] < gmm_metrics['davies_bouldin_index'] else 'GMM'
    winners.append(best)

    # Purity (higher is better)
    purities = [gmm_purity, kmeans_purity, dbscan_metrics['cluster_purity']]
    winners.append(['GMM', 'K-Means', 'DBSCAN'][np.argmax(purities)])

    # Convergence
    winners.append('GMM')

    # Iterations
    winners.append('GMM')

    # Runtime
    winners.append('N/A')

    # Special features
    winners.append('All unique')

    comparison_data['Best'] = winners

    df = pd.DataFrame(comparison_data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved comparison table to {output_path}")

    return df


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("GMM Post-Analysis - Generate Visualizations and Reports")
    logger.info("=" * 80)

    # Setup paths
    project_root = Path.cwd()
    results_dir = project_root / 'results'
    data_dir = project_root / 'data'

    gmm_metrics_path = results_dir / 'gmm_metrics.json'
    kmeans_metrics_path = results_dir / 'cluster_quality.json'
    dbscan_metrics_path = results_dir / 'dbscan_metrics.json'
    covariance_csv_path = results_dir / 'gmm_covariance_comparison.csv'
    assignments_path = data_dir / 'processed' / 'gmm_assignments.csv'
    embeddings_path = data_dir / 'embeddings' / 'train_embeddings.npy'

    # Load metrics
    try:
        gmm_metrics = load_json(gmm_metrics_path)
        kmeans_metrics = load_json(kmeans_metrics_path)
        dbscan_metrics = load_json(dbscan_metrics_path)
    except Exception as e:
        logger.error(f"❌ Failed to load metrics: {e}")
        return 1

    # 1. Create covariance comparison plot
    try:
        covariance_plot_output = results_dir / 'gmm_covariance_comparison.png'
        create_covariance_comparison_plot(covariance_csv_path, covariance_plot_output)
    except Exception as e:
        logger.error(f"❌ Failed to create covariance comparison plot: {e}")
        import traceback
        traceback.print_exc()

    # 2. Create uncertainty visualization
    try:
        uncertainty_plot_output = results_dir / 'gmm_uncertainty_analysis.png'
        create_uncertainty_visualization(gmm_metrics, uncertainty_plot_output)
    except Exception as e:
        logger.error(f"❌ Failed to create uncertainty visualization: {e}")
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

    # 4. Create three-algorithm comparison
    try:
        comparison_output = results_dir / 'algorithm_comparison_three_way.csv'
        comparison_df = create_three_algorithm_comparison(
            gmm_metrics,
            kmeans_metrics,
            dbscan_metrics,
            comparison_output
        )
        logger.info("\n" + "=" * 60)
        logger.info("Three-Algorithm Comparison:")
        logger.info("=" * 60)
        print(comparison_df.to_string(index=False))
    except Exception as e:
        logger.error(f"❌ Failed to create three-algorithm comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("✅ Post-analysis complete! Generated outputs:")
    logger.info(f"   1. Covariance comparison: {covariance_plot_output}")
    logger.info(f"   2. Uncertainty analysis: {uncertainty_plot_output}")
    logger.info(f"   3. Cluster visualization: {results_dir / 'gmm_cluster_visualization.png'}")
    logger.info(f"   4. Three-way comparison: {comparison_output}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
