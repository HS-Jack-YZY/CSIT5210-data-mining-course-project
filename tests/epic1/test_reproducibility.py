"""
Unit tests for reproducibility utilities.

Tests the set_seed() function for ensuring deterministic behavior
across numpy and Python's random module.
"""

import numpy as np
import random
import pytest

from src.context_aware_multi_agent_system.utils.reproducibility import set_seed


class TestSetSeed:
    """Test suite for set_seed() function."""

    def test_set_seed_produces_identical_numpy_results(self):
        """Test AC-5: set_seed(42) produces identical numpy.random results."""
        # First run
        set_seed(42)
        result1 = np.random.rand(10)

        # Second run with same seed
        set_seed(42)
        result2 = np.random.rand(10)

        # Results must be identical
        assert np.allclose(result1, result2), "numpy.random results should be identical"
        assert np.array_equal(result1, result2), "numpy.random results should be exactly equal"

    def test_set_seed_produces_identical_python_random_results(self):
        """Test AC-5: set_seed(42) produces identical random.random results."""
        # First run
        set_seed(42)
        result1 = [random.random() for _ in range(10)]

        # Second run with same seed
        set_seed(42)
        result2 = [random.random() for _ in range(10)]

        # Results must be identical
        assert result1 == result2, "random.random results should be identical"

    def test_set_seed_affects_numpy_operations(self):
        """Test AC-5: Subsequent numpy operations are deterministic."""
        # Test various numpy random operations
        set_seed(42)
        rand_int1 = np.random.randint(0, 100, size=5)
        rand_normal1 = np.random.randn(5)
        rand_choice1 = np.random.choice([1, 2, 3, 4, 5], size=5)

        set_seed(42)
        rand_int2 = np.random.randint(0, 100, size=5)
        rand_normal2 = np.random.randn(5)
        rand_choice2 = np.random.choice([1, 2, 3, 4, 5], size=5)

        assert np.array_equal(rand_int1, rand_int2)
        assert np.allclose(rand_normal1, rand_normal2)
        assert np.array_equal(rand_choice1, rand_choice2)

    def test_set_seed_affects_python_random_operations(self):
        """Test AC-5: Subsequent Python random operations are deterministic."""
        # Test various Python random operations
        set_seed(42)
        rand_int1 = [random.randint(0, 100) for _ in range(5)]
        rand_choice1 = [random.choice([1, 2, 3, 4, 5]) for _ in range(5)]

        set_seed(42)
        rand_int2 = [random.randint(0, 100) for _ in range(5)]
        rand_choice2 = [random.choice([1, 2, 3, 4, 5]) for _ in range(5)]

        assert rand_int1 == rand_int2
        assert rand_choice1 == rand_choice2

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random results."""
        # Seed 42
        set_seed(42)
        result1 = np.random.rand(10)

        # Seed 100 (different)
        set_seed(100)
        result2 = np.random.rand(10)

        # Results should be different
        assert not np.array_equal(result1, result2), "Different seeds should produce different results"

    def test_set_seed_with_zero(self):
        """Test set_seed() works with seed value of 0."""
        set_seed(0)
        result1 = np.random.rand(5)

        set_seed(0)
        result2 = np.random.rand(5)

        assert np.array_equal(result1, result2)

    def test_set_seed_with_large_number(self):
        """Test set_seed() works with large seed values."""
        large_seed = 999999

        set_seed(large_seed)
        result1 = np.random.rand(5)

        set_seed(large_seed)
        result2 = np.random.rand(5)

        assert np.array_equal(result1, result2)

    def test_set_seed_ensures_kmeans_reproducibility(self):
        """
        Test AC-5: Verify K-Means will produce identical clusters.

        Note: This test validates the concept. Actual K-Means testing
        will happen in Epic 2 with scikit-learn's random_state parameter.
        """
        from sklearn.cluster import KMeans

        # Generate sample data
        set_seed(42)
        X = np.random.rand(100, 10)

        # First K-Means run
        set_seed(42)
        kmeans1 = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels1 = kmeans1.fit_predict(X)

        # Second K-Means run
        set_seed(42)
        kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X)

        # Cluster labels should be identical
        assert np.array_equal(labels1, labels2), "K-Means should produce identical clusters with same seed"
