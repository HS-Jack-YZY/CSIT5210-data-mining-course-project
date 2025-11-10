"""
Gaussian Mixture Model clustering script for AG News dataset.

This script applies GMM clustering on cached embeddings from Epic 1,
compares covariance types, and exports probabilistic assignments with uncertainty analysis.

Usage:
    python scripts/07_gmm_clustering.py

Features:
    - Load cached embeddings from Epic 1
    - Compare 4 covariance types (full, tied, diag, spherical)
    - Extract hard and soft cluster assignments
    - Perform uncertainty analysis (low-confidence documents)
    - Calculate GMM-specific metrics (BIC, AIC, log-likelihood)
    - Calculate standard clustering metrics (Silhouette, Davies-Bouldin, purity)
    - Export assignments (CSV), metrics (JSON), and covariance comparison (CSV)
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.models.gmm_clustering import GMMClustering
from context_aware_multi_agent_system.evaluation.clustering_metrics import ClusteringMetrics
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


def perform_uncertainty_analysis(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ground_truth: np.ndarray,
    n_components: int
) -> Dict[str, Any]:
    """
    Perform uncertainty analysis on GMM probabilistic assignments.

    Analyzes:
    - Low-confidence documents (confidence < 0.5)
    - Cluster pair confusion (documents with similar probabilities for multiple clusters)
    - Uncertainty patterns by ground truth category

    Args:
        probabilities: Soft assignments, shape (n_documents, n_components)
        labels: Hard assignments, shape (n_documents,)
        ground_truth: Ground truth labels, shape (n_documents,)
        n_components: Number of clusters

    Returns:
        Dictionary containing uncertainty analysis results
    """
    logger.info("📊 Performing uncertainty analysis...")

    # Calculate assignment confidence (max probability per document)
    confidence = probabilities.max(axis=1)

    # Identify low-confidence documents (confidence < 0.5)
    low_confidence_mask = confidence < 0.5
    low_confidence_count = low_confidence_mask.sum()
    low_confidence_ratio = low_confidence_count / len(confidence)

    logger.info(
        f"📊 Low-confidence documents (< 0.5): {low_confidence_count} ({low_confidence_ratio:.1%})"
    )

    # Analyze cluster pair confusion
    # For each document, find the two clusters with highest probabilities
    sorted_probs = np.sort(probabilities, axis=1)
    top2_probs = sorted_probs[:, -2:]  # Top 2 probabilities
    prob_diff = top2_probs[:, 1] - top2_probs[:, 0]  # Difference between top 2

    # Documents with small difference are confused between clusters
    confusion_threshold = 0.2
    confused_mask = prob_diff < confusion_threshold
    confused_count = confused_mask.sum()
    confused_ratio = confused_count / len(prob_diff)

    logger.info(
        f"📊 Confused documents (prob diff < {confusion_threshold}): "
        f"{confused_count} ({confused_ratio:.1%})"
    )

    # Analyze uncertainty by ground truth category
    category_uncertainty = {}
    category_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    for category_id in range(4):
        category_mask = ground_truth == category_id
        if category_mask.sum() == 0:
            continue

        category_confidence = confidence[category_mask]
        category_low_conf_ratio = (category_confidence < 0.5).sum() / category_mask.sum()

        category_uncertainty[category_names[category_id]] = {
            'total_documents': int(category_mask.sum()),
            'mean_confidence': float(category_confidence.mean()),
            'std_confidence': float(category_confidence.std()),
            'low_confidence_ratio': float(category_low_conf_ratio),
            'min_confidence': float(category_confidence.min()),
            'max_confidence': float(category_confidence.max())
        }

        logger.info(
            f"📊 {category_names[category_id]}: mean_conf={category_confidence.mean():.3f}, "
            f"low_conf_ratio={category_low_conf_ratio:.1%}"
        )

    # Confidence distribution statistics
    confidence_stats = {
        'mean': float(confidence.mean()),
        'std': float(confidence.std()),
        'min': float(confidence.min()),
        'max': float(confidence.max()),
        'q25': float(np.percentile(confidence, 25)),
        'q50': float(np.percentile(confidence, 50)),
        'q75': float(np.percentile(confidence, 75))
    }

    logger.info(
        f"✅ Confidence distribution: mean={confidence_stats['mean']:.3f}, "
        f"std={confidence_stats['std']:.3f}, median={confidence_stats['q50']:.3f}"
    )

    return {
        'low_confidence_count': int(low_confidence_count),
        'low_confidence_ratio': float(low_confidence_ratio),
        'confused_count': int(confused_count),
        'confused_ratio': float(confused_ratio),
        'confusion_threshold': confusion_threshold,
        'confidence_stats': confidence_stats,
        'category_uncertainty': category_uncertainty
    }


def save_assignments(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ground_truth: np.ndarray,
    covariance_type: str,
    output_path: Path
) -> None:
    """
    Save cluster assignments with probabilities to CSV.

    Args:
        probabilities: Soft assignments, shape (n_documents, n_components)
        labels: Hard assignments, shape (n_documents,)
        ground_truth: Ground truth labels, shape (n_documents,)
        covariance_type: Covariance type used
        output_path: Path to save CSV
    """
    logger.info(f"💾 Saving cluster assignments to {output_path}...")

    n_documents = len(labels)
    n_components = probabilities.shape[1]

    # Calculate assignment confidence
    confidence = probabilities.max(axis=1)

    # Build DataFrame
    data = {
        'document_id': np.arange(n_documents),
        'cluster_id': labels,
    }

    # Add probability columns for each cluster
    for cluster_id in range(n_components):
        data[f'cluster_{cluster_id}_prob'] = probabilities[:, cluster_id]

    data['assignment_confidence'] = confidence
    data['ground_truth_category'] = ground_truth
    data['covariance_type'] = covariance_type

    df = pd.DataFrame(data)

    # Validate schema
    expected_columns = [
        'document_id', 'cluster_id',
        'cluster_0_prob', 'cluster_1_prob', 'cluster_2_prob', 'cluster_3_prob',
        'assignment_confidence', 'ground_truth_category', 'covariance_type'
    ]

    assert list(df.columns) == expected_columns, \
        f"CSV schema mismatch. Expected: {expected_columns}, got: {list(df.columns)}"

    # Validate probabilities
    prob_columns = [f'cluster_{i}_prob' for i in range(n_components)]
    prob_sums = df[prob_columns].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    # Validate probability ranges
    for col in prob_columns:
        assert df[col].min() >= 0.0 and df[col].max() <= 1.0, \
            f"Probabilities in {col} are outside [0, 1] range"

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(f"✅ Saved {len(df)} assignments with {n_components} probability columns")


def main():
    """Main execution function for GMM clustering."""
    logger.info("=" * 80)
    logger.info("📊 Starting GMM Clustering Pipeline")
    logger.info("=" * 80)

    start_time = time.time()

    # Set reproducibility seed
    set_seed(42)
    logger.info("🎲 Set random seed to 42 for reproducibility")

    # Load configuration
    config = Config()
    paths = Paths()
    logger.info("✅ Loaded configuration")

    # Load cached embeddings
    logger.info("📂 Loading cached embeddings...")
    cache = EmbeddingCache(paths.data_embeddings)

    if not cache.exists('train'):
        raise FileNotFoundError(
            "Training embeddings cache not found. Please run 01_generate_embeddings.py first."
        )

    embeddings, metadata = cache.load('train')
    logger.info(f"✅ Loaded {len(embeddings)} training embeddings")

    # Validate embeddings
    validate_embeddings(embeddings, config)

    # Load ground truth labels
    logger.info("📂 Loading ground truth labels...")
    dataset_loader = DatasetLoader(config)
    train_data, _ = dataset_loader.load_ag_news()
    ground_truth = np.array(train_data['label'], dtype=np.int32)
    logger.info(f"✅ Loaded {len(ground_truth)} ground truth labels")

    assert len(embeddings) == len(ground_truth), \
        f"Embeddings and ground truth count mismatch: {len(embeddings)} vs {len(ground_truth)}"

    # Step 1: Compare covariance types
    logger.info("\n" + "=" * 80)
    logger.info("📊 Step 1: Comparing Covariance Types")
    logger.info("=" * 80)

    gmm = GMMClustering(
        n_components=4,
        covariance_type='full',  # Will be tested in comparison
        random_state=42,
        max_iter=100
    )

    covariance_types = ['full', 'tied', 'diag', 'spherical']
    comparison_df = gmm.compare_covariance_types(embeddings, covariance_types)

    # Save covariance comparison
    comparison_path = paths.results / 'gmm_covariance_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"💾 Saved covariance comparison to {comparison_path}")

    # Select best covariance type (lowest BIC)
    best_covariance_type = comparison_df.iloc[0]['covariance_type']
    best_bic = comparison_df.iloc[0]['bic']
    logger.info(f"🏆 Best covariance type: '{best_covariance_type}' (BIC={best_bic:.2f})")

    # Step 2: Fit final GMM with best covariance type
    logger.info("\n" + "=" * 80)
    logger.info(f"📊 Step 2: Fitting Final GMM (covariance_type='{best_covariance_type}')")
    logger.info("=" * 80)

    final_gmm = GMMClustering(
        n_components=4,
        covariance_type=best_covariance_type,
        random_state=42,
        max_iter=100
    )

    labels, probabilities, bic, aic = final_gmm.fit_predict(embeddings)

    logger.info(f"✅ GMM clustering complete")
    logger.info(f"📊 Convergence: {final_gmm.converged}, Iterations: {final_gmm.n_iterations}")
    logger.info(f"📊 Log-likelihood: {final_gmm.log_likelihood:.2f}")
    logger.info(f"📊 BIC: {bic:.2f}, AIC: {aic:.2f}")

    # Extract component weights (mixing coefficients)
    weights = final_gmm.weights
    logger.info(f"📊 Component weights: {weights}")

    # Step 3: Perform uncertainty analysis
    logger.info("\n" + "=" * 80)
    logger.info("📊 Step 3: Uncertainty Analysis")
    logger.info("=" * 80)

    uncertainty_results = perform_uncertainty_analysis(
        probabilities, labels, ground_truth, n_components=4
    )

    # Step 4: Calculate standard clustering metrics
    logger.info("\n" + "=" * 80)
    logger.info("📊 Step 4: Calculating Standard Clustering Metrics")
    logger.info("=" * 80)

    # Need to create dummy centroids for ClusteringMetrics API compatibility
    # (GMM doesn't have explicit centroids, but we can use cluster means)
    centroids = np.zeros((4, 768), dtype=np.float32)
    for cluster_id in range(4):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)

    metrics_calculator = ClusteringMetrics(
        embeddings=embeddings,
        labels=labels,
        centroids=centroids,
        ground_truth=ground_truth
    )

    standard_metrics = metrics_calculator.evaluate_all()

    logger.info(f"📊 Silhouette Score: {standard_metrics['silhouette_score']:.6f}")
    logger.info(f"📊 Davies-Bouldin Index: {standard_metrics['davies_bouldin_index']:.4f}")
    logger.info(f"📊 Cluster Purity: {standard_metrics['cluster_purity']['overall']:.1%}")

    # Step 5: Save assignments
    logger.info("\n" + "=" * 80)
    logger.info("📊 Step 5: Saving Results")
    logger.info("=" * 80)

    # Create output directories
    processed_dir = paths.data_processed
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save assignments
    assignments_path = processed_dir / 'gmm_assignments.csv'
    save_assignments(probabilities, labels, ground_truth, best_covariance_type, assignments_path)

    # Step 6: Save metrics to JSON
    metrics_output = {
        'algorithm': 'GMM',
        'n_components': 4,
        'covariance_type': best_covariance_type,
        'random_state': 42,
        'max_iter': 100,
        'timestamp': datetime.now().isoformat(),

        # Convergence information
        'converged': bool(final_gmm.converged),
        'n_iterations': int(final_gmm.n_iterations),
        'log_likelihood': float(final_gmm.log_likelihood),

        # GMM-specific metrics
        'bic': float(bic),
        'aic': float(aic),
        'component_weights': weights.tolist(),

        # Standard clustering metrics
        'silhouette_score': float(standard_metrics['silhouette_score']),
        'davies_bouldin_index': float(standard_metrics['davies_bouldin_index']),
        'cluster_purity': standard_metrics['cluster_purity'],

        # Cluster size and balance
        'cluster_sizes': standard_metrics['cluster_sizes'],
        'is_balanced': bool(standard_metrics['is_balanced']),

        # Distance metrics
        'intra_cluster_distance': standard_metrics['intra_cluster_distance'],
        'inter_cluster_distance': standard_metrics['inter_cluster_distance'],

        # Uncertainty analysis
        'uncertainty_analysis': uncertainty_results,

        # Covariance type comparison
        'covariance_comparison': comparison_df.to_dict(orient='records'),

        # Runtime
        'runtime_seconds': time.time() - start_time
    }

    metrics_path = paths.results / 'gmm_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)

    logger.info(f"💾 Saved metrics to {metrics_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ GMM Clustering Pipeline Complete")
    logger.info("=" * 80)
    logger.info(f"📊 Best covariance type: {best_covariance_type}")
    logger.info(f"📊 BIC: {bic:.2f}, AIC: {aic:.2f}")
    logger.info(f"📊 Silhouette Score: {standard_metrics['silhouette_score']:.6f}")
    logger.info(f"📊 Davies-Bouldin Index: {standard_metrics['davies_bouldin_index']:.4f}")
    logger.info(f"📊 Cluster Purity: {standard_metrics['cluster_purity']['overall']:.1%}")
    logger.info(f"📊 Low-confidence documents: {uncertainty_results['low_confidence_ratio']:.1%}")
    logger.info(f"⏱️  Total runtime: {time.time() - start_time:.2f} seconds")
    logger.info("\n📂 Output files:")
    logger.info(f"   - Assignments: {assignments_path}")
    logger.info(f"   - Metrics: {metrics_path}")
    logger.info(f"   - Covariance comparison: {comparison_path}")


if __name__ == '__main__':
    main()
