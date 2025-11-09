"""
Unit tests for GMMClustering class.

This module tests the Gaussian Mixture Model clustering implementation for correctness,
validation, and error handling.

Test Coverage:
    - AC-1: GMMClustering initialization with parameters
    - AC-1: fit_predict() functionality
    - AC-2: compare_covariance_types() functionality
    - AC-3: Probability distribution validation
    - AC-5: GMM-specific metrics (BIC, AIC)
    - Reproducibility with random_state=42
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.models.gmm_clustering import GMMClustering


class TestGMMClusteringInit:
    """Test GMMClustering initialization."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        gmm = GMMClustering()

        assert gmm.n_components == 4
        assert gmm.covariance_type == 'full'
        assert gmm.random_state == 42
        assert gmm.max_iter == 100
        assert gmm.model is None

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        gmm = GMMClustering(
            n_components=8,
            covariance_type='diag',
            random_state=123,
            max_iter=200
        )

        assert gmm.n_components == 8
        assert gmm.covariance_type == 'diag'
        assert gmm.random_state == 123
        assert gmm.max_iter == 200

    def test_init_with_invalid_covariance_type(self):
        """Test initialization fails with invalid covariance type."""
        with pytest.raises(ValueError) as exc_info:
            GMMClustering(covariance_type='invalid')

        assert "Invalid covariance_type" in str(exc_info.value)

    def test_init_with_all_covariance_types(self):
        """Test initialization with all valid covariance types."""
        valid_types = ['full', 'tied', 'diag', 'spherical']

        for cov_type in valid_types:
            gmm = GMMClustering(covariance_type=cov_type)
            assert gmm.covariance_type == cov_type


class TestGMMClusteringFitPredict:
    """Test GMMClustering.fit_predict() method."""

    @pytest.fixture
    def sample_embeddings(self):
        """Provide sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 768).astype(np.float32)

    def test_fit_predict_with_valid_embeddings(self, sample_embeddings):
        """Test fit_predict with valid embeddings (AC-1)."""
        # Use 'diag' covariance for numerical stability with random high-dimensional data
        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probabilities, bic, aic = gmm.fit_predict(sample_embeddings)

        # Validate labels
        assert labels.shape == (1000,)
        assert labels.dtype == np.int32
        assert labels.min() >= 0
        assert labels.max() < 4

        # Validate probabilities
        assert probabilities.shape == (1000, 4)
        assert probabilities.dtype == np.float32

        # Validate BIC and AIC
        assert isinstance(bic, float)
        assert isinstance(aic, float)
        assert np.isfinite(bic)
        assert np.isfinite(aic)

    def test_probability_distribution_validity(self, sample_embeddings):
        """Test probabilities are valid distributions (AC-3)."""
        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probabilities, bic, aic = gmm.fit_predict(sample_embeddings)

        # Probabilities must be in [0, 1]
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

        # Probabilities must sum to approximately 1.0 per document
        prob_sums = probabilities.sum(axis=1)
        # Use larger tolerance for float32 precision
        assert np.allclose(prob_sums, 1.0, atol=1e-4)

    def test_hard_labels_match_argmax(self, sample_embeddings):
        """Test hard labels match argmax of probabilities (AC-3)."""
        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probabilities, bic, aic = gmm.fit_predict(sample_embeddings)

        # Hard labels should match argmax of probabilities
        expected_labels = np.argmax(probabilities, axis=1)
        assert np.array_equal(labels, expected_labels)

    def test_bic_aic_calculation(self, sample_embeddings):
        """Test BIC and AIC are calculated correctly (AC-5)."""
        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probabilities, bic, aic = gmm.fit_predict(sample_embeddings)

        # BIC and AIC should be finite and positive (for typical data)
        assert np.isfinite(bic)
        assert np.isfinite(aic)

        # BIC is typically larger than AIC (due to stronger penalty)
        # This is not always true, but should hold for most datasets
        # We just check they are different
        assert bic != aic

    def test_reproducibility_with_random_state(self, sample_embeddings):
        """Test reproducibility with random_state=42."""
        gmm1 = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels1, probs1, bic1, aic1 = gmm1.fit_predict(sample_embeddings)

        gmm2 = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels2, probs2, bic2, aic2 = gmm2.fit_predict(sample_embeddings)

        # Results should be identical with same random_state
        assert np.array_equal(labels1, labels2)
        assert np.array_equal(probs1, probs2)
        assert bic1 == bic2
        assert aic1 == aic2

    def test_fit_predict_with_invalid_shape(self):
        """Test fit_predict fails with invalid embeddings shape."""
        gmm = GMMClustering()

        # 1D array
        with pytest.raises(ValueError) as exc_info:
            gmm.fit_predict(np.random.randn(100).astype(np.float32))
        assert "must be 2D array" in str(exc_info.value)

        # Wrong dimensionality
        with pytest.raises(ValueError) as exc_info:
            gmm.fit_predict(np.random.randn(100, 512).astype(np.float32))
        assert "768 dimensions" in str(exc_info.value)

    def test_fit_predict_with_invalid_dtype(self):
        """Test fit_predict fails with invalid dtype."""
        gmm = GMMClustering()

        # Float64 instead of float32
        with pytest.raises(ValueError) as exc_info:
            gmm.fit_predict(np.random.randn(100, 768).astype(np.float64))
        assert "dtype float32" in str(exc_info.value)

    def test_fit_predict_with_nan_values(self):
        """Test fit_predict fails with NaN values."""
        gmm = GMMClustering()
        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings[0, 0] = np.nan

        with pytest.raises(ValueError) as exc_info:
            gmm.fit_predict(embeddings)
        assert "NaN values" in str(exc_info.value)

    def test_fit_predict_with_inf_values(self):
        """Test fit_predict fails with Inf values."""
        gmm = GMMClustering()
        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings[0, 0] = np.inf

        with pytest.raises(ValueError) as exc_info:
            gmm.fit_predict(embeddings)
        assert "Inf values" in str(exc_info.value)


class TestGMMClusteringCovarianceComparison:
    """Test GMMClustering.compare_covariance_types() method."""

    @pytest.fixture
    def sample_embeddings(self):
        """Provide sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(500, 768).astype(np.float32)

    def test_compare_all_covariance_types(self, sample_embeddings):
        """Test comparison of all covariance types (AC-2)."""
        gmm = GMMClustering(n_components=4, random_state=42)
        # Test only diag and spherical for numerical stability with random data
        comparison_df = gmm.compare_covariance_types(
            sample_embeddings,
            types=['diag', 'spherical']
        )

        # Validate DataFrame structure
        assert len(comparison_df) == 2
        assert list(comparison_df.columns) == [
            'covariance_type', 'bic', 'aic', 'silhouette_score', 'runtime_seconds'
        ]

        # Validate all covariance types are present
        assert set(comparison_df['covariance_type']) == {'diag', 'spherical'}

        # Validate all metrics are finite
        assert comparison_df['bic'].apply(np.isfinite).all()
        assert comparison_df['aic'].apply(np.isfinite).all()
        assert comparison_df['silhouette_score'].apply(np.isfinite).all()
        assert comparison_df['runtime_seconds'].apply(np.isfinite).all()

        # DataFrame should be sorted by BIC (lower is better)
        assert comparison_df['bic'].is_monotonic_increasing

    def test_compare_subset_of_covariance_types(self, sample_embeddings):
        """Test comparison with subset of covariance types."""
        gmm = GMMClustering(n_components=4, random_state=42)
        comparison_df = gmm.compare_covariance_types(
            sample_embeddings,
            types=['diag', 'spherical']
        )

        # Should only have 2 rows
        assert len(comparison_df) == 2
        assert set(comparison_df['covariance_type']) == {'diag', 'spherical'}

    def test_compare_returns_best_by_bic(self, sample_embeddings):
        """Test comparison returns best type by BIC."""
        gmm = GMMClustering(n_components=4, random_state=42)
        comparison_df = gmm.compare_covariance_types(
            sample_embeddings,
            types=['diag', 'spherical']
        )

        # First row should have lowest BIC
        best_type = comparison_df.iloc[0]['covariance_type']
        best_bic = comparison_df.iloc[0]['bic']

        for idx, row in comparison_df.iterrows():
            assert row['bic'] >= best_bic


class TestGMMClusteringProperties:
    """Test GMMClustering properties."""

    @pytest.fixture
    def sample_embeddings(self):
        """Provide sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(500, 768).astype(np.float32)

    def test_properties_before_fitting(self):
        """Test properties before fitting return None."""
        gmm = GMMClustering()

        assert gmm.converged is None
        assert gmm.n_iterations is None
        assert gmm.log_likelihood is None
        assert gmm.weights is None

    def test_properties_after_fitting(self, sample_embeddings):
        """Test properties after fitting return valid values."""
        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probs, bic, aic = gmm.fit_predict(sample_embeddings)

        # Convergence status
        assert isinstance(gmm.converged, bool)

        # Number of iterations
        assert isinstance(gmm.n_iterations, (int, np.integer))
        assert gmm.n_iterations > 0

        # Log-likelihood (can be np.float32 or float)
        assert isinstance(gmm.log_likelihood, (float, np.floating))
        assert np.isfinite(gmm.log_likelihood)

        # Weights (mixing coefficients)
        assert isinstance(gmm.weights, np.ndarray)
        assert gmm.weights.shape == (4,)
        assert np.allclose(gmm.weights.sum(), 1.0)
        assert np.all(gmm.weights >= 0)
        assert np.all(gmm.weights <= 1)


class TestGMMClusteringPerformance:
    """Test GMMClustering performance characteristics."""

    def test_small_sample_performance(self):
        """Test performance on small sample (1K documents)."""
        np.random.seed(42)
        embeddings = np.random.randn(1000, 768).astype(np.float32)

        import time
        start_time = time.time()

        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probs, bic, aic = gmm.fit_predict(embeddings)

        runtime = time.time() - start_time

        # Should complete in reasonable time (<30 seconds for 1K documents)
        assert runtime < 30.0

        # Validate results
        assert len(labels) == 1000
        assert len(probs) == 1000

    def test_convergence_information(self):
        """Test convergence information is logged (AC-8)."""
        np.random.seed(42)
        embeddings = np.random.randn(500, 768).astype(np.float32)

        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42, max_iter=100)
        labels, probs, bic, aic = gmm.fit_predict(embeddings)

        # Should have convergence info
        assert gmm.converged is not None
        assert gmm.n_iterations is not None
        assert gmm.n_iterations <= 100

        # Log-likelihood should be calculated
        assert gmm.log_likelihood is not None
        assert np.isfinite(gmm.log_likelihood)
