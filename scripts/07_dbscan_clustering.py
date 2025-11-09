"""
DBSCAN clustering script for AG News dataset.

This script applies DBSCAN density-based clustering to embeddings from Epic 1,
performs parameter tuning, and compares results with K-Means baseline.

Usage:
    python scripts/07_dbscan_clustering.py

Features:
    - Load cached embeddings from Epic 1
    - Compute cosine distance matrix for DBSCAN
    - Parameter tuning: eps × min_samples grid search
    - Export cluster assignments (CSV), metrics (JSON), comparisons (CSV)
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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.models.dbscan_clustering import DBSCANClustering
from context_aware_multi_agent_system.features.embedding_cache import EmbeddingCache
from context_aware_multi_agent_system.data.load_dataset import DatasetLoader
from context_aware_multi_agent_system.utils.reproducibility import set_seed
from context_aware_multi_agent_system.evaluation.clustering_metrics import calculate_purity

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
        Ground truth labels array (n_documents,)

    Raises:
        FileNotFoundError: If dataset cannot be loaded
    """
    logger.info("📊 Loading ground truth labels from AG News dataset...")

    try:
        dataset_loader = DatasetLoader(config)
        train_dataset, _ = dataset_loader.load_ag_news()

        # Extract labels
        labels = np.array(train_dataset['label'])

        # Map to category names
        category_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        labels_str = np.array([category_map[label] for label in labels])

        logger.info(
            f"✅ Loaded {len(labels_str)} ground truth labels: "
            f"{np.unique(labels_str, return_counts=True)}"
        )

        return labels_str

    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load AG News dataset: {e}\n"
            f"Ensure dataset is downloaded and accessible."
        )


def save_cluster_assignments(
    labels: np.ndarray,
    ground_truth: np.ndarray,
    core_samples_mask: np.ndarray,
    output_path: Path
) -> None:
    """
    Save cluster assignments to CSV file.

    Args:
        labels: Cluster labels (-1 for noise, 0+ for clusters)
        ground_truth: Ground truth category labels
        core_samples_mask: Boolean mask indicating core samples
        output_path: Output CSV file path

    Raises:
        IOError: If file save fails
    """
    logger.info(f"📊 Saving cluster assignments to {output_path}...")

    # Create DataFrame
    assignments_df = pd.DataFrame({
        'document_id': np.arange(len(labels)),
        'cluster_id': labels,
        'ground_truth_category': ground_truth,
        'is_core_sample': core_samples_mask
    })

    # Validate
    assert len(assignments_df) == len(labels), \
        f"Assignment count mismatch: expected {len(labels)}, got {len(assignments_df)}"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    assignments_df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(
        f"✅ Saved {len(assignments_df)} cluster assignments: "
        f"{set(assignments_df.columns)}"
    )

    # Validate saved file
    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(labels), \
        f"Saved file row count mismatch: expected {len(labels)}, got {len(loaded)}"


def calculate_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ground_truth: np.ndarray,
    runtime_seconds: float,
    best_eps: float,
    best_min_samples: int
) -> Dict[str, Any]:
    """
    Calculate cluster quality metrics for DBSCAN results.

    Args:
        embeddings: Document embeddings
        labels: Cluster labels (-1 for noise)
        ground_truth: Ground truth category labels
        runtime_seconds: Clustering runtime
        best_eps: Best eps parameter
        best_min_samples: Best min_samples parameter

    Returns:
        Dictionary of metrics
    """
    logger.info("📊 Calculating cluster quality metrics...")

    # Filter out noise points
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    non_noise_embeddings = embeddings[non_noise_mask]
    non_noise_ground_truth = ground_truth[non_noise_mask]

    # Count clusters and noise
    n_clusters = len(set(non_noise_labels)) if len(non_noise_labels) > 0 else 0
    n_noise = np.sum(labels == -1)
    noise_percentage = n_noise / len(labels)

    logger.info(
        f"📊 Clusters: {n_clusters}, Noise: {n_noise} ({noise_percentage*100:.1f}%)"
    )

    # Calculate metrics (only if >1 cluster and non-noise points exist)
    silhouette = None
    davies_bouldin = None

    if n_clusters > 1 and len(non_noise_labels) > 0:
        try:
            silhouette = silhouette_score(non_noise_embeddings, non_noise_labels)
            davies_bouldin = davies_bouldin_score(non_noise_embeddings, non_noise_labels)
            logger.info(
                f"✅ Silhouette Score: {silhouette:.4f}, "
                f"Davies-Bouldin Index: {davies_bouldin:.4f}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate Silhouette/Davies-Bouldin: {e}")
    else:
        reason = "only 1 cluster or all noise"
        logger.warning(
            f"⚠️ Cannot calculate Silhouette/Davies-Bouldin: {reason}"
        )

    # Calculate cluster purity (non-noise points only)
    purity = None
    if len(non_noise_labels) > 0:
        # Convert string labels to integers for calculate_purity
        category_map = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
        non_noise_ground_truth_int = np.array([category_map[cat] for cat in non_noise_ground_truth])
        purity = calculate_purity(non_noise_labels, non_noise_ground_truth_int)
        logger.info(f"📊 Cluster Purity: {purity:.3f}")
    else:
        logger.warning("⚠️ Cannot calculate purity: no non-noise points")

    # Cluster size distribution
    cluster_sizes = {}
    if n_clusters > 0:
        unique, counts = np.unique(non_noise_labels, return_counts=True)
        cluster_sizes = {int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)}

        cluster_size_stats = {
            'min': int(min(counts)),
            'max': int(max(counts)),
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts))
        }
    else:
        cluster_size_stats = {
            'min': 0,
            'max': 0,
            'mean': 0.0,
            'std': 0.0
        }

    # Package metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'algorithm': 'DBSCAN',
        'parameters': {
            'eps': float(best_eps),
            'min_samples': int(best_min_samples),
            'metric': 'cosine'
        },
        'n_clusters': int(n_clusters),
        'n_noise_points': int(n_noise),
        'noise_percentage': float(noise_percentage),
        'silhouette_score': float(silhouette) if silhouette is not None else None,
        'davies_bouldin_index': float(davies_bouldin) if davies_bouldin is not None else None,
        'cluster_purity': float(purity) if purity is not None else None,
        'cluster_sizes': cluster_sizes,
        'cluster_size_stats': cluster_size_stats,
        'runtime_seconds': float(runtime_seconds)
    }

    return metrics


def compare_with_kmeans(
    dbscan_metrics: Dict[str, Any],
    kmeans_metrics_path: Path
) -> pd.DataFrame:
    """
    Compare DBSCAN results with K-Means baseline.

    Args:
        dbscan_metrics: DBSCAN metrics dictionary
        kmeans_metrics_path: Path to K-Means metrics JSON

    Returns:
        Comparison DataFrame

    Raises:
        FileNotFoundError: If K-Means metrics file not found
    """
    logger.info(f"📊 Loading K-Means metrics from {kmeans_metrics_path}...")

    if not kmeans_metrics_path.exists():
        raise FileNotFoundError(
            f"K-Means metrics not found: {kmeans_metrics_path}\n"
            f"Run 'python scripts/03_evaluate_clustering.py' first"
        )

    # Load K-Means metrics
    with open(kmeans_metrics_path) as f:
        kmeans_metrics = json.load(f)

    logger.info("✅ K-Means metrics loaded")

    # Create comparison DataFrame
    comparison = pd.DataFrame([
        {
            'algorithm': 'K-Means',
            'silhouette_score': kmeans_metrics.get('silhouette_score'),
            'davies_bouldin_index': kmeans_metrics.get('davies_bouldin_index'),
            'cluster_purity': kmeans_metrics.get('cluster_purity'),
            'n_clusters': 4,
            'n_noise_points': 0,
            'runtime_seconds': kmeans_metrics.get('runtime_seconds')
        },
        {
            'algorithm': 'DBSCAN',
            'silhouette_score': dbscan_metrics['silhouette_score'],
            'davies_bouldin_index': dbscan_metrics['davies_bouldin_index'],
            'cluster_purity': dbscan_metrics['cluster_purity'],
            'n_clusters': dbscan_metrics['n_clusters'],
            'n_noise_points': dbscan_metrics['n_noise_points'],
            'runtime_seconds': dbscan_metrics['runtime_seconds']
        }
    ])

    # Log comparison summary
    logger.info("📊 DBSCAN vs K-Means Comparison:")

    if dbscan_metrics['silhouette_score'] is not None and kmeans_metrics.get('silhouette_score') is not None:
        dbscan_sil = dbscan_metrics['silhouette_score']
        kmeans_sil = kmeans_metrics['silhouette_score']
        better = "DBSCAN" if dbscan_sil > kmeans_sil else "K-Means"
        logger.info(
            f"  Silhouette Score: DBSCAN={dbscan_sil:.4f} vs K-Means={kmeans_sil:.4f} "
            f"({better} wins)"
        )

    if dbscan_metrics['cluster_purity'] is not None and kmeans_metrics.get('cluster_purity') is not None:
        dbscan_pur = dbscan_metrics['cluster_purity']
        kmeans_pur = kmeans_metrics['cluster_purity']
        better = "DBSCAN" if dbscan_pur > kmeans_pur else "K-Means"
        logger.info(
            f"  Cluster Purity: DBSCAN={dbscan_pur:.3f} vs K-Means={kmeans_pur:.3f} "
            f"({better} wins)"
        )

    logger.info(
        f"  Clusters: DBSCAN={dbscan_metrics['n_clusters']} (variable) vs K-Means=4 (fixed)"
    )
    logger.info(
        f"  Noise: DBSCAN={dbscan_metrics['n_noise_points']} vs K-Means=0"
    )

    # Add comparison to dbscan_metrics
    if kmeans_metrics.get('silhouette_score') is not None:
        dbscan_metrics['comparison'] = {
            'kmeans_silhouette': kmeans_metrics['silhouette_score'],
            'kmeans_purity': kmeans_metrics.get('cluster_purity'),
            'silhouette_improvement': (
                dbscan_metrics['silhouette_score'] - kmeans_metrics['silhouette_score']
                if dbscan_metrics['silhouette_score'] is not None else None
            ),
            'purity_improvement': (
                dbscan_metrics['cluster_purity'] - kmeans_metrics.get('cluster_purity')
                if dbscan_metrics['cluster_purity'] is not None and kmeans_metrics.get('cluster_purity') is not None
                else None
            )
        }

    return comparison


def main():
    """Main execution function for DBSCAN clustering."""
    logger.info("=" * 80)
    logger.info("DBSCAN Density-Based Clustering - AG News Dataset")
    logger.info("=" * 80)

    script_start_time = time.time()

    # CRITICAL: Set random seed for reproducibility
    set_seed(42)

    # Load configuration
    logger.info("📊 Loading configuration...")
    config = Config()
    paths = Paths()
    logger.info("✅ Configuration loaded")

    # Define paths
    embeddings_path = paths.data_embeddings / 'train_embeddings.npy'
    assignments_path = paths.data_processed / 'dbscan_assignments.csv'
    metrics_path = paths.results / 'dbscan_metrics.json'
    tuning_path = paths.results / 'dbscan_parameter_tuning.csv'
    comparison_path = paths.results / 'dbscan_vs_kmeans_comparison.csv'
    kmeans_metrics_path = paths.results / 'cluster_quality.json'

    # Ensure output directories exist (already created by Paths class, but double-check)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    logger.info(f"📊 Loading embeddings from {embeddings_path}...")

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            f"Run 'python scripts/01_generate_embeddings.py' first"
        )

    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # Validate embeddings
    validate_embeddings(embeddings, config)

    # Load ground truth labels
    ground_truth = load_ground_truth_labels(config)

    # Validate ground truth matches embeddings
    if len(ground_truth) != len(embeddings):
        raise ValueError(
            f"Ground truth count ({len(ground_truth)}) != embeddings count ({len(embeddings)})"
        )

    # Initialize DBSCAN clustering
    logger.info("📊 Initializing DBSCAN clustering...")
    clustering = DBSCANClustering(eps=0.5, min_samples=5, metric='cosine')
    logger.info("✅ DBSCAN initialized")

    # Parameter tuning
    logger.info("📊 Starting parameter tuning...")
    tuning_start_time = time.time()

    eps_range = [0.3, 0.5, 0.7, 1.0]
    min_samples_range = [3, 5, 10]

    best_eps, best_min_samples, tuning_df = clustering.tune_parameters(
        embeddings,
        eps_range=eps_range,
        min_samples_range=min_samples_range
    )

    tuning_time = time.time() - tuning_start_time
    logger.info(
        f"✅ Parameter tuning complete in {tuning_time:.1f}s ({tuning_time/60:.1f} minutes)"
    )

    # Verify tuning time budget
    if tuning_time > 10800:  # 3 hours = 10800 seconds
        logger.warning(
            f"⚠️ Parameter tuning time {tuning_time:.0f}s exceeds 3 hour budget"
        )

    # Save tuning results
    logger.info(f"📊 Saving parameter tuning results to {tuning_path}...")
    tuning_df.to_csv(tuning_path, index=False)
    logger.info("✅ Parameter tuning results saved")

    # Run final DBSCAN with best parameters
    logger.info(f"📊 Running final DBSCAN with best parameters: eps={best_eps}, min_samples={best_min_samples}...")
    final_start_time = time.time()

    labels, core_samples_mask = clustering.fit_predict(embeddings)

    final_runtime = time.time() - final_start_time
    logger.info(f"✅ DBSCAN complete in {final_runtime:.1f}s ({final_runtime/60:.1f} minutes)")

    # Verify runtime budget
    if final_runtime > 900:  # 15 minutes = 900 seconds
        logger.warning(
            f"⚠️ DBSCAN runtime {final_runtime:.1f}s exceeds 15 minute target"
        )

    # Save cluster assignments
    save_cluster_assignments(labels, ground_truth, core_samples_mask, assignments_path)

    # Calculate metrics
    metrics = calculate_cluster_metrics(
        embeddings,
        labels,
        ground_truth,
        final_runtime,
        best_eps,
        best_min_samples
    )

    # Save metrics
    logger.info(f"📊 Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("✅ Metrics saved")

    # Compare with K-Means
    try:
        comparison_df = compare_with_kmeans(metrics, kmeans_metrics_path)

        # Save comparison
        logger.info(f"📊 Saving comparison to {comparison_path}...")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("✅ Comparison saved")

        # Update metrics with comparison
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Could not compare with K-Means: {e}")

    # Final summary
    script_runtime = time.time() - script_start_time
    logger.info("=" * 80)
    logger.info("✅ DBSCAN Clustering Complete")
    logger.info(f"   - Algorithm: DBSCAN (density-based)")
    logger.info(
        f"   - Parameters: eps={best_eps}, min_samples={best_min_samples}, metric=cosine"
    )
    logger.info(f"   - Clusters discovered: {metrics['n_clusters']}")
    logger.info(
        f"   - Noise points: {metrics['n_noise_points']} ({metrics['noise_percentage']*100:.1f}%)"
    )
    if metrics['silhouette_score'] is not None:
        logger.info(f"   - Silhouette Score: {metrics['silhouette_score']:.4f}")
    if metrics['cluster_purity'] is not None:
        logger.info(f"   - Cluster Purity: {metrics['cluster_purity']:.3f}")
    logger.info(f"   - Runtime: {final_runtime:.1f}s ({final_runtime/60:.1f} minutes)")
    logger.info(f"   - Total script runtime: {script_runtime:.1f}s ({script_runtime/60:.1f} minutes)")
    logger.info(f"   - Assignments: {assignments_path}")
    logger.info(f"   - Metrics: {metrics_path}")
    logger.info(f"   - Tuning results: {tuning_path}")
    logger.info(f"   - Comparison: {comparison_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Script failed: {e}", exc_info=True)
        sys.exit(1)
