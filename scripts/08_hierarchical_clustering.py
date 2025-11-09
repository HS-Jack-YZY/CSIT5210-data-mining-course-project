"""
Hierarchical Agglomerative Clustering script for AG News dataset.

This script applies hierarchical clustering to embeddings from Epic 1,
compares linkage methods, generates dendrogram, and evaluates cluster quality.

Usage:
    python scripts/08_hierarchical_clustering.py

Features:
    - Load cached embeddings from Epic 1
    - Compare linkage methods: ward, complete, average, single
    - Generate dendrogram visualization with cluster boundaries
    - Export cluster assignments (CSV), metrics (JSON), linkage comparison (CSV)
    - Comprehensive logging with emoji prefixes
"""

# CRITICAL: Set environment variables for reproducibility BEFORE importing numpy
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

import json
import logging
import sys
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.models.hierarchical_clustering import HierarchicalClustering
from context_aware_multi_agent_system.visualization.dendrogram_plot import generate_dendrogram
from context_aware_multi_agent_system.features.embedding_cache import EmbeddingCache
from context_aware_multi_agent_system.data.load_dataset import DatasetLoader
from context_aware_multi_agent_system.utils.reproducibility import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_embeddings(embeddings: np.ndarray, config: Config) -> None:
    """
    Validate embeddings shape, dtype, and content.

    Args:
        embeddings: Embeddings array to validate
        config: Configuration object

    Raises:
        ValueError: If validation fails
    """
    expected_dim = config.get('embedding.output_dimensionality', 768)

    # Check shape
    if len(embeddings.shape) != 2:
        raise ValueError(
            f"Embeddings must be 2D array, got {len(embeddings.shape)}D\n"
            f"Expected shape: (n_documents, {expected_dim})"
        )

    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Embeddings must have {expected_dim} dimensions\n"
            f"Expected shape: (*, {expected_dim}), got: {embeddings.shape}"
        )

    # Check dtype
    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Embeddings must have dtype float32, got {embeddings.dtype}"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(embeddings)):
        raise ValueError("Embeddings contain NaN values")

    if np.any(np.isinf(embeddings)):
        raise ValueError("Embeddings contain Inf values")

    logger.info(
        f"✅ Embeddings validation passed: shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}"
    )


def load_ground_truth_labels(config: Config) -> np.ndarray:
    """
    Load ground truth category labels from AG News dataset.

    Args:
        config: Configuration object

    Returns:
        Ground truth labels array (n_documents,) int32

    Raises:
        FileNotFoundError: If dataset cannot be loaded
    """
    logger.info("📊 Loading ground truth labels from AG News dataset...")

    try:
        dataset_loader = DatasetLoader(config)
        train_dataset = dataset_loader.load_dataset()

        # Extract labels as int32
        labels = np.array(train_dataset['label'], dtype=np.int32)

        logger.info(
            f"✅ Loaded {len(labels)} ground truth labels: "
            f"{np.unique(labels, return_counts=True)}"
        )

        return labels

    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load AG News dataset: {e}\n"
            f"Ensure dataset is downloaded and accessible."
        )


def save_cluster_assignments(
    labels: np.ndarray,
    ground_truth: np.ndarray,
    linkage_method: str,
    output_path: Path
) -> None:
    """
    Save cluster assignments to CSV file.

    Args:
        labels: Cluster labels (0-3)
        ground_truth: Ground truth category labels (int32)
        linkage_method: Linkage method used
        output_path: Output CSV file path

    Raises:
        IOError: If file save fails
    """
    logger.info(f"📊 Saving cluster assignments to {output_path}...")

    # Map ground truth to category names
    category_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    ground_truth_str = np.array([category_map[label] for label in ground_truth])

    # Create DataFrame
    assignments_df = pd.DataFrame({
        'document_id': np.arange(len(labels)),
        'cluster_id': labels,
        'ground_truth_category': ground_truth_str,
        'linkage_method': linkage_method
    })

    # Validate
    assert len(assignments_df) == len(labels), \
        f"Assignment count mismatch: expected {len(labels)}, got {len(assignments_df)}"
    assert set(assignments_df['cluster_id']) == {0, 1, 2, 3}, \
        f"Invalid cluster IDs: {set(assignments_df['cluster_id'])}"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    assignments_df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(
        f"✅ Saved {len(assignments_df)} cluster assignments: "
        f"columns={list(assignments_df.columns)}"
    )

    # Validate saved file
    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(labels), \
        f"Saved file row count mismatch: expected {len(labels)}, got {len(loaded)}"


def save_linkage_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save linkage method comparison results to CSV.

    Args:
        comparison_df: DataFrame with comparison results
        output_path: Output CSV file path
    """
    logger.info(f"📊 Saving linkage comparison to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    comparison_df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(
        f"✅ Saved linkage comparison: {len(comparison_df)} methods compared"
    )


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save cluster quality metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        output_path: Output JSON file path
    """
    logger.info(f"📊 Saving metrics to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Metrics saved: {output_path}")


def monitor_memory() -> Tuple[float, str]:
    """
    Monitor current memory usage.

    Returns:
        Tuple of (memory_gb, warning_message)
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)

    warning = ""
    if memory_gb > 16:
        warning = f"⚠️ Memory usage {memory_gb:.2f}GB exceeds 16GB threshold"

    return memory_gb, warning


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Hierarchical Agglomerative Clustering - AG News Dataset")
    logger.info("=" * 80)

    # Set seed for reproducibility (for sampling if needed)
    set_seed(42)

    # Initialize configuration and paths
    logger.info("📊 Loading configuration...")
    config = Config()
    paths = Paths()
    logger.info("✅ Configuration loaded")

    # Get clustering parameters
    n_clusters = config.get('clustering.n_clusters', 4)
    logger.info(f"📊 Clustering parameters: n_clusters={n_clusters}")

    # Monitor initial memory
    initial_memory, _ = monitor_memory()
    logger.info(f"💾 Initial memory usage: {initial_memory:.2f} GB")

    # Load embeddings
    logger.info("📊 Loading embeddings from cache...")
    embeddings_path = paths.data_embeddings / "train_embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            f"Run 'python scripts/01_generate_embeddings.py' first"
        )

    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded {len(embeddings):,} embeddings ({embeddings.shape[1]}D)")

    # Validate embeddings
    validate_embeddings(embeddings, config)

    # Check memory after loading embeddings
    current_memory, warning = monitor_memory()
    if warning:
        logger.warning(warning)
        logger.info(f"💾 Memory after loading embeddings: {current_memory:.2f} GB")

    # Load ground truth labels
    ground_truth = load_ground_truth_labels(config)

    # Validate ground truth matches embeddings
    if len(ground_truth) != len(embeddings):
        raise ValueError(
            f"Ground truth count {len(ground_truth)} != embeddings count {len(embeddings)}"
        )

    # Check if sampling is needed due to memory constraints
    # For dendrogram generation, we may need to sample
    sample_for_dendrogram = len(embeddings) > 10000

    if sample_for_dendrogram:
        logger.info(
            f"📊 Dataset size {len(embeddings):,} > 10,000 - will sample 10K for dendrogram"
        )
        # Sample 10K documents for dendrogram
        sample_indices = np.random.choice(len(embeddings), size=10000, replace=False)
        embeddings_sample = embeddings[sample_indices]
    else:
        embeddings_sample = embeddings

    # Initialize hierarchical clustering
    logger.info(f"📊 Initializing HierarchicalClustering...")
    clustering = HierarchicalClustering(n_clusters=n_clusters, linkage='ward')

    # Compare linkage methods
    logger.info("=" * 80)
    logger.info("LINKAGE METHOD COMPARISON")
    logger.info("=" * 80)

    comparison_start = time.time()
    comparison_df = clustering.compare_linkage_methods(
        embeddings,
        ground_truth,
        methods=['ward', 'complete', 'average', 'single']
    )
    comparison_end = time.time()
    comparison_runtime = comparison_end - comparison_start

    logger.info(f"⏱️ Linkage comparison runtime: {comparison_runtime:.1f}s")
    logger.info("\nLinkage Method Comparison Results:")
    logger.info("\n" + str(comparison_df.to_string(index=False)))

    # Save linkage comparison
    linkage_comparison_path = paths.results / "hierarchical_linkage_comparison.csv"
    save_linkage_comparison(comparison_df, linkage_comparison_path)

    # Select best linkage method
    best_method = comparison_df.iloc[0]['linkage_method']
    logger.info(f"\n🏆 Best linkage method: {best_method} (highest Silhouette Score)")

    # Fit final hierarchical clustering with best method
    logger.info("=" * 80)
    logger.info(f"FINAL CLUSTERING (Linkage: {best_method})")
    logger.info("=" * 80)

    final_clustering = HierarchicalClustering(n_clusters=n_clusters, linkage=best_method)

    start_time = time.time()
    labels, dendrogram_data = final_clustering.fit_predict(embeddings)
    end_time = time.time()
    runtime = end_time - start_time

    logger.info(f"⏱️ Clustering runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    # Check memory after clustering
    current_memory, warning = monitor_memory()
    if warning:
        logger.warning(warning)
    logger.info(f"💾 Memory after clustering: {current_memory:.2f} GB")

    # Generate dendrogram
    logger.info("=" * 80)
    logger.info("DENDROGRAM GENERATION")
    logger.info("=" * 80)

    dendrogram_path = paths.reports_figures / "hierarchical_dendrogram.png"
    dendrogram_output = generate_dendrogram(
        embeddings_sample,
        linkage_method=best_method,
        output_path=dendrogram_path,
        n_clusters=n_clusters,
        truncate_mode='lastp',
        p=30
    )

    logger.info(f"✅ Dendrogram saved: {dendrogram_output}")

    if sample_for_dendrogram:
        logger.info(
            f"ℹ️ Dendrogram generated from 10,000 sample documents "
            f"(full dataset: {len(embeddings):,})"
        )

    # Calculate metrics
    logger.info("=" * 80)
    logger.info("CLUSTER QUALITY METRICS")
    logger.info("=" * 80)

    metrics = final_clustering.calculate_metrics(labels, embeddings, ground_truth)

    # Add additional metadata
    metrics_full = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": "hierarchical",
        "linkage_method": best_method,
        "n_clusters": n_clusters,
        "n_documents": len(embeddings),
        "silhouette_score": metrics['silhouette_score'],
        "davies_bouldin_index": metrics['davies_bouldin_index'],
        "cluster_purity": metrics['cluster_purity'],
        "cluster_sizes": metrics['cluster_sizes'],
        "runtime_seconds": runtime,
        "comparison_runtime_seconds": comparison_runtime,
        "memory_usage_gb": current_memory,
        "dendrogram_sampled": sample_for_dendrogram,
        "dendrogram_sample_size": 10000 if sample_for_dendrogram else len(embeddings)
    }

    # Save cluster assignments
    assignments_path = paths.data_processed / "hierarchical_assignments.csv"
    save_cluster_assignments(labels, ground_truth, best_method, assignments_path)

    # Save metrics
    metrics_path = paths.results / "hierarchical_metrics.json"
    save_metrics(metrics_full, metrics_path)

    # Print summary
    logger.info("=" * 80)
    logger.info("✅ HIERARCHICAL CLUSTERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"   - Best linkage method: {best_method}")
    logger.info(f"   - Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"   - Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    logger.info(f"   - Cluster Purity: {metrics['cluster_purity']:.1%}")
    logger.info(f"   - Cluster Sizes: {metrics['cluster_sizes']}")
    logger.info(f"   - Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
    logger.info(f"   - Total Runtime (incl. comparison): {comparison_runtime + runtime:.1f}s")
    logger.info(f"   - Memory Usage: {current_memory:.2f} GB")
    logger.info("")
    logger.info("📁 Output Files:")
    logger.info(f"   - Assignments: {assignments_path}")
    logger.info(f"   - Dendrogram: {dendrogram_path}")
    logger.info(f"   - Metrics: {metrics_path}")
    logger.info(f"   - Linkage Comparison: {linkage_comparison_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)
