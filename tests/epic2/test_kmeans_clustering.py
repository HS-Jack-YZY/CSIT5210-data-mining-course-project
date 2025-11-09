"""
Unit tests for KMeansClustering class.

This module tests the K-Means clustering implementation for correctness,
validation, and error handling.

Test Coverage:
    - AC-1: KMeansClustering.fit_predict() functionality
    - AC-1: Parameter validation
    - AC-1: Convergence and reproducibility
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.models.clustering import KMeansClustering


class TestKMeansClusteringInit:
    """Test KMeansClustering initialization."""

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = {
            'n_clusters': 4,
            'random_state': 42,
            'max_iter': 300,
            'init': 'k-means++'
        }
        clustering = KMeansClustering(config)

        assert clustering.config == config
        assert clustering.model is None

    def test_init_with_missing_parameters(self):
        """Test initialization fails with missing parameters."""
        config = {
            'n_clusters': 4,
            # Missing random_state, max_iter, init
        }

        with pytest.raises(ValueError) as exc_info:
            KMeansClustering(config)

        assert "Missing required clustering parameters" in str(exc_info.value)

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        config = {
            'n_clusters': 8,
            'random_state': 123,
            'max_iter': 500,
            'init': 'random'
        }
        clustering = KMeansClustering(config)

        assert clustering.config['n_clusters'] == 8
        assert clustering.config['random_state'] == 123


class TestKMeansClusteringFitPredict:
    """Test KMeansClustering.fit_predict() method."""

    @pytest.fixture
    def valid_config(self):
        """Provide valid clustering configuration."""
        return {
            'n_clusters': 4,
            'random_state': 42,
            'max_iter': 300,
            'init': 'k-means++'
        }

    @pytest.fixture
    def synthetic_embeddings(self):
        """Generate synthetic embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 768).astype(np.float32)

    def test_fit_predict_on_synthetic_data(self, valid_config, synthetic_embeddings):
        """Test fit_predict on synthetic dataset (AC-1)."""
        clustering = KMeansClustering(valid_config)
        labels, centroids = clustering.fit_predict(synthetic_embeddings)

        # Verify labels shape and dtype
        assert labels.shape == (1000,)
        assert labels.dtype == np.int32

        # Verify labels values are in valid range [0, 3]
        assert labels.min() >= 0
        assert labels.max() <= 3
        assert all(0 <= label <= 3 for label in labels)

        # Verify centroids shape and dtype
        assert centroids.shape == (4, 768)
        assert centroids.dtype == np.float32

        # Verify no NaN or Inf in centroids
        assert not np.any(np.isnan(centroids))
        assert not np.any(np.isinf(centroids))

    def test_fit_predict_convergence_properties(self, valid_config, synthetic_embeddings):
        """Test fit_predict convergence and model properties (AC-5)."""
        clustering = KMeansClustering(valid_config)
        labels, centroids = clustering.fit_predict(synthetic_embeddings)

        # Verify model is fitted
        assert clustering.model is not None

        # Verify convergence iterations < max_iter
        assert clustering.n_iterations is not None
        assert clustering.n_iterations < valid_config['max_iter']

        # Verify inertia is computed
        assert clustering.inertia is not None
        assert clustering.inertia > 0

    def test_fit_predict_reproducibility(self, valid_config, synthetic_embeddings):
        """Test fit_predict produces identical results with same seed (AC-1, AC-5)."""
        clustering1 = KMeansClustering(valid_config)
        labels1, centroids1 = clustering1.fit_predict(synthetic_embeddings)

        clustering2 = KMeansClustering(valid_config)
        labels2, centroids2 = clustering2.fit_predict(synthetic_embeddings)

        # Verify identical results
        assert np.array_equal(labels1, labels2)
        assert np.allclose(centroids1, centroids2, atol=1e-6)

    def test_fit_predict_invalid_shape_1d(self, valid_config):
        """Test fit_predict fails with 1D array (AC-7)."""
        invalid_embeddings = np.random.randn(1000).astype(np.float32)
        clustering = KMeansClustering(valid_config)

        with pytest.raises(ValueError) as exc_info:
            clustering.fit_predict(invalid_embeddings)

        assert "must be 2D array" in str(exc_info.value)

    def test_fit_predict_invalid_shape_wrong_dimensions(self, valid_config):
        """Test fit_predict fails with wrong embedding dimensions (AC-7)."""
        invalid_embeddings = np.random.randn(1000, 512).astype(np.float32)
        clustering = KMeansClustering(valid_config)

        with pytest.raises(ValueError) as exc_info:
            clustering.fit_predict(invalid_embeddings)

        assert "must have 768 dimensions" in str(exc_info.value)

    def test_fit_predict_invalid_dtype(self, valid_config):
        """Test fit_predict fails with wrong dtype (AC-7)."""
        invalid_embeddings = np.random.randn(1000, 768).astype(np.float64)
        clustering = KMeansClustering(valid_config)

        with pytest.raises(ValueError) as exc_info:
            clustering.fit_predict(invalid_embeddings)

        assert "must have dtype float32" in str(exc_info.value)

    def test_fit_predict_nan_values(self, valid_config):
        """Test fit_predict fails with NaN values (AC-7)."""
        invalid_embeddings = np.random.randn(1000, 768).astype(np.float32)
        invalid_embeddings[0, 0] = np.nan

        clustering = KMeansClustering(valid_config)

        with pytest.raises(ValueError) as exc_info:
            clustering.fit_predict(invalid_embeddings)

        assert "NaN values" in str(exc_info.value)

    def test_fit_predict_inf_values(self, valid_config):
        """Test fit_predict fails with Inf values (AC-7)."""
        invalid_embeddings = np.random.randn(1000, 768).astype(np.float32)
        invalid_embeddings[0, 0] = np.inf

        clustering = KMeansClustering(valid_config)

        with pytest.raises(ValueError) as exc_info:
            clustering.fit_predict(invalid_embeddings)

        assert "Inf values" in str(exc_info.value)

    def test_fit_predict_all_clusters_used(self, valid_config, synthetic_embeddings):
        """Test fit_predict assigns documents to all clusters (AC-4)."""
        clustering = KMeansClustering(valid_config)
        labels, centroids = clustering.fit_predict(synthetic_embeddings)

        # Verify all clusters have at least some documents
        unique_labels = np.unique(labels)
        assert len(unique_labels) == valid_config['n_clusters']

        # Verify cluster IDs are consecutive [0, 1, 2, 3]
        assert set(unique_labels) == set(range(valid_config['n_clusters']))


class TestKMeansClusteringProperties:
    """Test KMeansClustering properties."""

    @pytest.fixture
    def valid_config(self):
        """Provide valid clustering configuration."""
        return {
            'n_clusters': 4,
            'random_state': 42,
            'max_iter': 300,
            'init': 'k-means++'
        }

    def test_properties_before_fit(self, valid_config):
        """Test properties return None before fitting."""
        clustering = KMeansClustering(valid_config)

        assert clustering.n_iterations is None
        assert clustering.inertia is None

    def test_properties_after_fit(self, valid_config):
        """Test properties return values after fitting."""
        clustering = KMeansClustering(valid_config)
        embeddings = np.random.randn(1000, 768).astype(np.float32)

        clustering.fit_predict(embeddings)

        assert clustering.n_iterations is not None
        assert clustering.n_iterations > 0
        assert clustering.inertia is not None
        assert clustering.inertia > 0
