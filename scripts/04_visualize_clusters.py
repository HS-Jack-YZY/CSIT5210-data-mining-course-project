#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA cluster visualization script for AG News K-Means clustering.

This script applies PCA dimensionality reduction (768D → 2D) to document embeddings
and generates a publication-quality scatter plot showing 4 semantic clusters.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.visualization.cluster_plots import PCAVisualizer
from context_aware_multi_agent_system.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for PCA cluster visualization."""
    start_time = time.time()

    logger.info("📊 Starting PCA cluster visualization...")

    # Set random seed for reproducibility
    set_seed(42)
    logger.info("✅ Set random seed to 42")

    # Initialize configuration
    config = Config()
    paths = Paths()

    # Load embeddings
    logger.info("📊 Loading embeddings and cluster labels...")
    embeddings_path = paths.data_embeddings / "train_embeddings.npy"
    if not embeddings_path.exists():
        logger.error(
            f"❌ Embeddings not found: {embeddings_path}\n"
            f"Run 'python scripts/01_generate_embeddings.py' first"
        )
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]}D)")

    # Validate embeddings shape
    if embeddings.shape[1] != 768:
        logger.error(
            f"❌ Expected embeddings with 768 dimensions, got {embeddings.shape[1]}"
        )
        sys.exit(1)

    # Load cluster assignments
    assignments_path = paths.data_processed / "cluster_assignments.csv"
    if not assignments_path.exists():
        logger.error(
            f"❌ Cluster assignments not found: {assignments_path}\n"
            f"Run 'python scripts/02_train_clustering.py' first"
        )
        sys.exit(1)

    df = pd.read_csv(assignments_path)
    labels = df['cluster_id'].values.astype(np.int32)
    logger.info(f"✅ Loaded {len(labels)} cluster assignments")

    # Validate label range
    unique_labels = np.unique(labels)
    if not np.all((unique_labels >= 0) & (unique_labels <= 3)):
        logger.error(
            f"❌ Expected cluster labels in range [0, 3], got {unique_labels}"
        )
        sys.exit(1)

    # Validate embeddings and labels count match
    if len(embeddings) != len(labels):
        logger.error(
            f"❌ Embeddings count ({len(embeddings)}) does not match "
            f"labels count ({len(labels)})"
        )
        sys.exit(1)

    # Load centroids
    centroids_path = paths.data_processed / "centroids.npy"
    if not centroids_path.exists():
        logger.error(
            f"❌ Centroids not found: {centroids_path}\n"
            f"Run 'python scripts/02_train_clustering.py' first"
        )
        sys.exit(1)

    centroids = np.load(centroids_path)
    logger.info(f"✅ Loaded {centroids.shape[0]} centroids")

    # Validate centroids shape
    expected_shape = (4, 768)
    if centroids.shape != expected_shape:
        logger.error(
            f"❌ Expected centroids shape {expected_shape}, got {centroids.shape}"
        )
        sys.exit(1)

    # Initialize PCA visualizer
    try:
        visualizer = PCAVisualizer(embeddings, labels, centroids)
    except ValueError as e:
        logger.error(f"❌ Failed to initialize PCAVisualizer: {e}")
        sys.exit(1)

    # Apply PCA dimensionality reduction
    logger.info("📊 Applying PCA dimensionality reduction (768D → 2D)...")
    embeddings_2d, centroids_2d, variance_explained = visualizer.apply_pca()

    # Get individual variance components
    pc1_var, pc2_var, total_var = visualizer.get_variance_explained()
    logger.info(f"📊 PC1 variance: {pc1_var*100:.1f}%")
    logger.info(f"📊 PC2 variance: {pc2_var*100:.1f}%")
    logger.info(f"📊 Total variance explained: {total_var*100:.1f}%")

    # Check variance threshold (AC-1, AC-7)
    if total_var < 0.20:
        logger.warning(
            f"⚠️ Low variance explained ({total_var*100:.1f}%), "
            f"2D projection may lose information"
        )
    else:
        logger.info(
            f"✅ Good variance explained ({total_var*100:.1f}%), "
            f"2D projection captures main structure"
        )

    # Generate visualization
    logger.info("📊 Generating cluster scatter plot...")
    output_path = paths.project_root / "visualizations" / "cluster_pca.png"

    try:
        saved_path = visualizer.generate_visualization(
            output_path=output_path,
            dpi=300,
            figsize=(10, 8)
        )
    except Exception as e:
        logger.error(f"❌ Failed to generate visualization: {e}")
        sys.exit(1)

    # Validate output file
    if not saved_path.exists():
        logger.error(f"❌ Visualization file not created: {saved_path}")
        sys.exit(1)

    file_size_mb = saved_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Visualization saved: {saved_path} ({file_size_mb:.2f} MB)")

    # Optional: Generate interactive Plotly visualization
    try:
        import plotly.graph_objects as go

        logger.info("📊 Generating interactive Plotly visualization...")

        # Create Plotly figure
        fig = go.Figure()

        # Add traces for each cluster
        n_clusters = len(np.unique(labels))
        for cluster_id in range(n_clusters):
            mask = (labels == cluster_id)
            fig.add_trace(go.Scatter(
                x=embeddings_2d[mask, 0],
                y=embeddings_2d[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=3, opacity=0.6),
                hovertemplate=f'Cluster {cluster_id}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            ))

        # Add centroids
        fig.add_trace(go.Scatter(
            x=centroids_2d[:, 0],
            y=centroids_2d[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(
                size=15,
                symbol='star',
                line=dict(width=2, color='black')
            ),
            hovertemplate='Centroid<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title='K-Means Clustering of AG News (K=4, PCA Projection)',
            xaxis_title=f'PC1 ({pc1_var*100:.1f}% variance)',
            yaxis_title=f'PC2 ({pc2_var*100:.1f}% variance)',
            hovermode='closest',
            width=1000,
            height=800
        )

        # Save interactive HTML
        html_path = paths.project_root / "visualizations" / "cluster_pca.html"
        fig.write_html(html_path)
        logger.info(f"✅ Interactive visualization saved: {html_path}")

    except ImportError:
        logger.info("ℹ️ Plotly not installed, skipping interactive visualization")
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate interactive visualization: {e}")

    # Display summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("✅ PCA Cluster Visualization Complete")
    logger.info(f"   - Documents visualized: {len(embeddings):,}")
    logger.info(f"   - Variance explained: {total_var*100:.1f}% (PC1: {pc1_var*100:.1f}%, PC2: {pc2_var*100:.1f}%)")
    logger.info(f"   - Output: {saved_path} (300 DPI)")
    logger.info(f"   - Execution time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ PCA visualization failed: {e}", exc_info=True)
        sys.exit(1)
