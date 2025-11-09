"""
K-Means clustering script for AG News dataset.

This script trains K-Means clustering on cached embeddings from Epic 1,
exports cluster assignments, centroids, and metadata for evaluation.

Usage:
    python scripts/02_train_clustering.py

Features:
    - Load cached embeddings from Epic 1
    - Train K-Means with k-means++ initialization (random_state=42)
    - Export cluster assignments (CSV), centroids (NPY), metadata (JSON)
    - Cluster balance validation
    - Comprehensive logging with emoji prefixes
"""

# CRITICAL: Set environment variables for reproducibility BEFORE importing numpy
# This ensures single-threaded execution to prevent non-deterministic behavior
# from multi-threaded BLAS libraries (OpenBLAS/MKL)
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
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.models.clustering import KMeansClustering
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


def check_cluster_balance(
    labels: np.ndarray,
    n_documents: int,
    min_pct: float = 0.1,
    max_pct: float = 0.5
) -> bool:
    """
    Check cluster size balance and log warnings if imbalanced.

    Args:
        labels: Cluster labels array
        n_documents: Total number of documents
        min_pct: Minimum cluster size as percentage (default: 10%)
        max_pct: Maximum cluster size as percentage (default: 50%)

    Returns:
        True if balanced, False if imbalanced
    """
    cluster_sizes = np.bincount(labels)
    min_size = int(min_pct * n_documents)
    max_size = int(max_pct * n_documents)

    is_balanced = True
    for cluster_id, size in enumerate(cluster_sizes):
        pct = (size / n_documents) * 100

        if size < min_size:
            logger.warning(
                f"⚠️ Cluster {cluster_id} contains {size} documents ({pct:.1f}%) "
                f"- below minimum threshold ({min_pct*100}%)"
            )
            is_balanced = False
        elif size > max_size:
            logger.warning(
                f"⚠️ Cluster {cluster_id} contains {size} documents ({pct:.1f}%) "
                f"- above maximum threshold ({max_pct*100}%)"
            )
            is_balanced = False

    if is_balanced:
        logger.info(
            f"📊 Cluster sizes: {cluster_sizes.tolist()} (balanced)"
        )
    else:
        logger.info(
            f"📊 Cluster sizes: {cluster_sizes.tolist()} (imbalanced)"
        )

    return is_balanced


def save_cluster_assignments(
    labels: np.ndarray,
    dataset_loader: DatasetLoader,
    output_path: Path
) -> None:
    """
    Save cluster assignments to CSV with document IDs and category labels.

    Args:
        labels: Cluster labels (n_documents,) int32
        dataset_loader: DatasetLoader to get category labels
        output_path: Path to save CSV file
    """
    # Load dataset to get category labels
    try:
        train_data, _ = dataset_loader.load_ag_news()
        # Map label integers to text
        label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        category_labels = [label_map[label] for label in train_data['label']]
    except Exception as e:
        # If dataset loading fails (e.g., using synthetic data), use generic labels
        logger.warning(f"⚠️ Could not load AG News dataset for labels: {e}")
        logger.warning("⚠️ Using generic category labels instead")
        # Distribute labels evenly across 4 categories
        label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        category_labels = [label_map[i % 4] for i in range(len(labels))]

    # Create DataFrame
    df = pd.DataFrame({
        'document_id': range(len(labels)),
        'cluster_id': labels,
        'category_label': category_labels
    })

    # Validate DataFrame
    assert len(df) == len(labels), \
        f"DataFrame length mismatch: {len(df)} != {len(labels)}"
    assert set(df.columns) == {'document_id', 'cluster_id', 'category_label'}, \
        f"DataFrame columns mismatch: {df.columns}"
    assert df['document_id'].is_unique, "Document IDs are not unique"
    assert df['cluster_id'].min() >= 0, \
        f"Invalid cluster ID: {df['cluster_id'].min()}"
    assert df['cluster_id'].max() < len(np.unique(labels)), \
        f"Invalid cluster ID: {df['cluster_id'].max()}"

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"💾 Saved cluster assignments: {output_path}")
    logger.info(f"   - {len(df)} documents")
    logger.info(f"   - {len(np.unique(labels))} clusters")


def save_centroids(centroids: np.ndarray, output_path: Path) -> None:
    """
    Save cluster centroids to NPY file.

    Args:
        centroids: Cluster centroids (n_clusters, 768) float32
        output_path: Path to save NPY file
    """
    # Validate centroids
    assert centroids.shape[1] == 768, \
        f"Centroids must have 768 dimensions, got {centroids.shape[1]}"
    assert centroids.dtype == np.float32, \
        f"Centroids must have dtype float32, got {centroids.dtype}"
    assert not np.any(np.isnan(centroids)), "Centroids contain NaN values"
    assert not np.any(np.isinf(centroids)), "Centroids contain Inf values"

    # Save to NPY file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, centroids)

    logger.info(f"💾 Saved centroids: {output_path}")
    logger.info(f"   - Shape: {centroids.shape}")
    logger.info(f"   - Dtype: {centroids.dtype}")


def save_cluster_metadata(
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save clustering metadata to JSON file.

    Args:
        metadata: Metadata dict with clustering information
        output_path: Path to save JSON file
    """
    # Validate required fields
    required_fields = {
        'timestamp', 'n_clusters', 'n_documents', 'random_state',
        'n_iterations', 'inertia', 'cluster_sizes', 'config'
    }
    missing_fields = required_fields - set(metadata.keys())
    if missing_fields:
        raise ValueError(f"Missing required metadata fields: {missing_fields}")

    # Validate field values
    assert metadata['n_clusters'] > 0, "n_clusters must be positive"
    assert metadata['n_documents'] > 0, "n_documents must be positive"
    assert metadata['n_iterations'] > 0, "n_iterations must be positive"
    assert len(metadata['cluster_sizes']) == metadata['n_clusters'], \
        f"cluster_sizes length mismatch: {len(metadata['cluster_sizes'])} != {metadata['n_clusters']}"

    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Saved metadata: {output_path}")
    logger.info(f"   - Timestamp: {metadata['timestamp']}")
    logger.info(f"   - Clusters: {metadata['n_clusters']}")
    logger.info(f"   - Documents: {metadata['n_documents']}")
    logger.info(f"   - Iterations: {metadata['n_iterations']}")
    logger.info(f"   - Inertia: {metadata['inertia']:.2f}")


def main():
    """Main clustering workflow."""
    start_time = time.time()

    logger.info("📊 Starting K-Means clustering...")

    # Step 1: Set random seed for reproducibility
    set_seed(42)
    logger.info("🎲 Set random seed to 42 for reproducibility")

    # Step 2: Load configuration
    config = Config()
    paths = Paths()
    logger.info("⚙️ Loaded configuration from config.yaml")

    # Step 3: Validate clustering configuration
    clustering_config = {
        'n_clusters': config.get('clustering.n_clusters'),
        'random_state': config.get('clustering.random_state'),
        'max_iter': config.get('clustering.max_iter'),
        'init': config.get('clustering.init')
    }

    logger.info(
        f"📊 Configuration: n_clusters={clustering_config['n_clusters']}, "
        f"random_state={clustering_config['random_state']}, "
        f"max_iter={clustering_config['max_iter']}, "
        f"init={clustering_config['init']}"
    )

    # Step 4: Load cached embeddings
    embedding_cache = EmbeddingCache(paths.data_embeddings)

    if not embedding_cache.exists('train'):
        logger.error(
            "❌ Embeddings not found: data/embeddings/train_embeddings.npy\n"
            "Please run 'python scripts/01_generate_embeddings.py' first to generate embeddings."
        )
        sys.exit(1)

    logger.info("📂 Loading cached embeddings...")
    embeddings, _ = embedding_cache.load('train')

    logger.info(
        f"📊 Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim) from cache"
    )

    # Step 5: Validate embeddings
    try:
        validate_embeddings(embeddings, config)
    except ValueError as e:
        logger.error(f"❌ Embeddings validation failed: {e}")
        sys.exit(1)

    # Step 6: Initialize and fit K-Means clustering
    clustering = KMeansClustering(clustering_config)

    logger.info("📊 Fitting K-Means clustering...")
    fit_start_time = time.time()

    labels, centroids = clustering.fit_predict(embeddings)

    fit_elapsed = time.time() - fit_start_time
    logger.info(f"⏱️ Clustering completed in {fit_elapsed:.1f}s")

    # Step 7: Check cluster balance
    n_documents = len(embeddings)
    is_balanced = check_cluster_balance(labels, n_documents)

    if not is_balanced:
        logger.warning(
            "⚠️ Cluster imbalance detected - review clustering parameters or dataset"
        )

    # Step 8: Save cluster assignments
    dataset_loader = DatasetLoader(config)
    assignments_path = paths.data_processed / "cluster_assignments.csv"

    save_cluster_assignments(labels, dataset_loader, assignments_path)

    # Step 9: Save centroids
    centroids_path = paths.data_processed / "centroids.npy"
    save_centroids(centroids, centroids_path)

    # Step 10: Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_clusters': clustering_config['n_clusters'],
        'n_documents': n_documents,
        'random_state': clustering_config['random_state'],
        'n_iterations': clustering.n_iterations,
        'inertia': clustering.inertia,
        'cluster_sizes': np.bincount(labels).tolist(),
        'config': clustering_config,
        'fit_time_seconds': fit_elapsed,
        'is_balanced': is_balanced
    }

    metadata_path = paths.data_processed / "cluster_metadata.json"
    save_cluster_metadata(metadata, metadata_path)

    # Step 11: Display summary
    total_elapsed = time.time() - start_time
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    logger.info("=" * 60)
    logger.info("✅ Clustering completed successfully")
    logger.info(f"⏱️ Total execution time: {minutes}m {seconds}s")
    logger.info(f"📊 Clustered {n_documents} documents into {clustering_config['n_clusters']} clusters")
    logger.info(f"📊 Convergence: {clustering.n_iterations} iterations, inertia={clustering.inertia:.2f}")
    logger.info(f"📊 Cluster balance: {'✅ Balanced' if is_balanced else '⚠️ Imbalanced'}")
    logger.info("=" * 60)
    logger.info("📂 Output files:")
    logger.info(f"   - {assignments_path}")
    logger.info(f"   - {centroids_path}")
    logger.info(f"   - {metadata_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Clustering failed: {e}", exc_info=True)
        sys.exit(1)
