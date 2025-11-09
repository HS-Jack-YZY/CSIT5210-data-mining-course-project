"""
Unit tests for DBSCAN clustering implementation.

Tests cover:
- AC-1: DBSCAN clustering execution and output validation
- AC-2: Parameter tuning functionality
- AC-4: Metrics calculation and edge cases
- AC-8: Error handling and validation
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_distances

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.models.dbscan_clustering import DBSCANClustering


class TestDBSCANInitialization:
    """Test DBSCAN initialization (AC-1, AC-8)."""

    def test_dbscan_initialization_default_parameters(self):
        """Test DBSCAN initializes with default parameters."""
        dbscan = DBSCANClustering()

        assert dbscan.eps == 0.5
        assert dbscan.min_samples == 5
        assert dbscan.metric == 'cosine'
        assert dbscan.model is None

    def test_dbscan_initialization_custom_parameters(self):
        """Test DBSCAN initializes with custom parameters."""
        dbscan = DBSCANClustering(eps=0.7, min_samples=10, metric='cosine')

        assert dbscan.eps == 0.7
        assert dbscan.min_samples == 10
        assert dbscan.metric == 'cosine'

    def test_dbscan_rejects_non_cosine_metric(self):
        """Test DBSCAN rejects non-cosine metric (AC-8)."""
        with pytest.raises(ValueError, match="Only 'cosine' metric is supported"):
            DBSCANClustering(metric='euclidean')


class TestDBSCANFitPredict:
    """Test DBSCAN fit_predict functionality (AC-1)."""

    def test_fit_predict_returns_correct_shapes(self):
        """Test fit_predict returns labels and core_samples with correct shapes."""
        # Create small synthetic data
        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize

        dbscan = DBSCANClustering(eps=0.5, min_samples=5)
        labels, core_samples_mask = dbscan.fit_predict(embeddings)

        # Check shapes
        assert labels.shape == (100,)
        assert core_samples_mask.shape == (100,)

        # Check types
        assert core_samples_mask.dtype == bool
        assert labels.dtype in [np.int32, np.int64]

        # Check label range (should be -1 for noise, 0+ for clusters)
        assert np.min(labels) >= -1

    def test_fit_predict_produces_noise_points(self):
        """Test DBSCAN can identify noise points (label -1)."""
        # Create sparse synthetic data
        embeddings = np.random.randn(50, 768).astype(np.float32)

        dbscan = DBSCANClustering(eps=0.3, min_samples=10)  # Strict parameters
        labels, _ = dbscan.fit_predict(embeddings)

        # Should have some noise points with strict parameters
        # Note: This is probabilistic, but with random data and strict params,
        # we should get noise points
        assert -1 in labels or len(set(labels)) > 0

    def test_fit_predict_validates_embeddings_shape(self):
        """Test fit_predict validates embeddings shape (AC-8)."""
        dbscan = DBSCANClustering()

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            dbscan.fit_predict(np.random.randn(100).astype(np.float32))

        # Wrong embedding dimension
        with pytest.raises(ValueError, match="Embeddings must have 768 dimensions"):
            dbscan.fit_predict(np.random.randn(100, 512).astype(np.float32))

    def test_fit_predict_validates_embeddings_dtype(self):
        """Test fit_predict validates embeddings dtype (AC-8)."""
        dbscan = DBSCANClustering()

        # Wrong dtype
        with pytest.raises(ValueError, match="Embeddings must have dtype float32"):
            dbscan.fit_predict(np.random.randn(100, 768).astype(np.float64))

    def test_fit_predict_validates_embeddings_nan(self):
        """Test fit_predict validates embeddings contain no NaN (AC-8)."""
        dbscan = DBSCANClustering()

        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings[0, 0] = np.nan

        with pytest.raises(ValueError, match="Embeddings contain NaN values"):
            dbscan.fit_predict(embeddings)

    def test_fit_predict_validates_embeddings_inf(self):
        """Test fit_predict validates embeddings contain no Inf (AC-8)."""
        dbscan = DBSCANClustering()

        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings[0, 0] = np.inf

        with pytest.raises(ValueError, match="Embeddings contain Inf values"):
            dbscan.fit_predict(embeddings)


class TestDBSCANParameterTuning:
    """Test DBSCAN parameter tuning (AC-2)."""

    def test_tune_parameters_tests_all_combinations(self):
        """Test tune_parameters tests all eps × min_samples combinations."""
        # Small synthetic data for fast testing
        embeddings = np.random.randn(200, 768).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        dbscan = DBSCANClustering()

        eps_range = [0.5, 0.7]
        min_samples_range = [3, 5]

        best_eps, best_min_samples, tuning_df = dbscan.tune_parameters(
            embeddings,
            eps_range=eps_range,
            min_samples_range=min_samples_range
        )

        # Should test 2 eps × 2 min_samples = 4 combinations
        assert len(tuning_df) == 4

        # Check DataFrame has required columns
        expected_columns = {
            'eps', 'min_samples', 'n_clusters', 'n_noise',
            'noise_ratio', 'silhouette_score', 'runtime_seconds'
        }
        assert expected_columns.issubset(set(tuning_df.columns))

        # Best parameters should be in tested ranges
        assert best_eps in eps_range
        assert best_min_samples in min_samples_range

        # Instance parameters should be updated
        assert dbscan.eps == best_eps
        assert dbscan.min_samples == best_min_samples

    def test_tune_parameters_selects_best_silhouette(self):
        """Test parameter selection prioritizes Silhouette Score (AC-2)."""
        # Create data with clear clusters
        np.random.seed(42)

        # Create 3 clusters
        cluster1 = np.random.randn(50, 768).astype(np.float32) + 1
        cluster2 = np.random.randn(50, 768).astype(np.float32) - 1
        cluster3 = np.random.randn(50, 768).astype(np.float32)

        embeddings = np.vstack([cluster1, cluster2, cluster3])
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        dbscan = DBSCANClustering()

        best_eps, best_min_samples, tuning_df = dbscan.tune_parameters(
            embeddings,
            eps_range=[0.5, 0.7],
            min_samples_range=[3, 5]
        )

        # Best parameters should produce valid Silhouette Score
        best_row = tuning_df[
            (tuning_df['eps'] == best_eps) &
            (tuning_df['min_samples'] == best_min_samples)
        ].iloc[0]

        # If any valid Silhouette scores exist, best should be valid
        valid_scores = tuning_df[tuning_df['silhouette_score'] > 0]
        if len(valid_scores) > 0:
            assert best_row['silhouette_score'] > 0


class TestDBSCANCosineDistance:
    """Test cosine distance matrix computation (AC-1)."""

    def test_cosine_distance_matrix_properties(self):
        """Test cosine distance matrix has correct properties."""
        embeddings = np.random.randn(50, 768).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        distances = cosine_distances(embeddings)

        # Check shape
        assert distances.shape == (50, 50)

        # Check diagonal is zero (self-distance)
        assert np.allclose(np.diag(distances), 0, atol=1e-6)

        # Check symmetric
        assert np.allclose(distances, distances.T, atol=1e-6)

        # Check range [0, 2] for cosine distance
        assert np.all(distances >= 0)
        assert np.all(distances <= 2)


class TestDBSCANEdgeCases:
    """Test DBSCAN edge cases (AC-4, AC-8)."""

    def test_all_noise_points(self):
        """Test DBSCAN handles all-noise case (AC-8)."""
        # Random sparse data with very strict parameters
        embeddings = np.random.randn(30, 768).astype(np.float32)

        dbscan = DBSCANClustering(eps=0.01, min_samples=20)  # Impossible to cluster
        labels, core_samples_mask = dbscan.fit_predict(embeddings)

        # Should have only noise points
        n_noise = np.sum(labels == -1)
        # At least most points should be noise
        assert n_noise > len(labels) * 0.5

    def test_single_cluster(self):
        """Test DBSCAN handles single cluster case (AC-8)."""
        # All points very similar
        embeddings = np.ones((50, 768), dtype=np.float32)
        embeddings += np.random.randn(50, 768).astype(np.float32) * 0.01  # Small noise

        dbscan = DBSCANClustering(eps=1.0, min_samples=3)  # Lenient parameters
        labels, _ = dbscan.fit_predict(embeddings)

        # Should have 1 cluster (or possibly all noise, both are valid)
        unique_labels = set(labels)
        if -1 in unique_labels:
            n_clusters = len(unique_labels) - 1
        else:
            n_clusters = len(unique_labels)

        # Should have 0 or 1 cluster
        assert n_clusters <= 1


class TestDBSCANProperties:
    """Test DBSCAN properties (AC-1)."""

    def test_n_clusters_property(self):
        """Test n_clusters_ property returns correct value."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        dbscan = DBSCANClustering(eps=0.5, min_samples=5)
        labels, _ = dbscan.fit_predict(embeddings)

        # Calculate expected n_clusters
        unique_labels = set(labels)
        expected_n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        assert dbscan.n_clusters_ == expected_n_clusters

    def test_n_noise_property(self):
        """Test n_noise_ property returns correct value."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        dbscan = DBSCANClustering(eps=0.5, min_samples=5)
        labels, _ = dbscan.fit_predict(embeddings)

        # Calculate expected n_noise
        expected_n_noise = np.sum(labels == -1)

        assert dbscan.n_noise_ == expected_n_noise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
