"""
Unit tests for Hierarchical Agglomerative Clustering implementation.

Tests cover:
- AC-1: Hierarchical clustering execution and output validation
- AC-2: Linkage method comparison functionality
- AC-4: Cluster quality metrics calculation
- AC-8: Error handling and input validation
- AC-9: Reproducibility and deterministic behavior
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.models.hierarchical_clustering import HierarchicalClustering


class TestHierarchicalInitialization:
    """Test HierarchicalClustering initialization (AC-1, AC-8)."""

    def test_hierarchical_initialization_default_parameters(self):
        """Test HierarchicalClustering initializes with default parameters."""
        clustering = HierarchicalClustering()

        assert clustering.n_clusters == 4
        assert clustering.linkage == 'ward'
        assert clustering.model is None

    def test_hierarchical_initialization_custom_parameters(self):
        """Test HierarchicalClustering initializes with custom parameters."""
        clustering = HierarchicalClustering(n_clusters=3, linkage='complete')

        assert clustering.n_clusters == 3
        assert clustering.linkage == 'complete'

    def test_hierarchical_rejects_invalid_linkage(self):
        """Test HierarchicalClustering rejects invalid linkage method (AC-8)."""
        with pytest.raises(ValueError, match="Invalid linkage method"):
            HierarchicalClustering(linkage='invalid')


class TestHierarchicalFitPredict:
    """Test HierarchicalClustering fit_predict functionality (AC-1)."""

    def test_fit_predict_returns_correct_shapes(self):
        """Test fit_predict returns labels and dendrogram data with correct shapes."""
        # Create small synthetic data
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels, dendrogram_data = clustering.fit_predict(embeddings)

        # Check labels shape and type
        assert labels.shape == (100,)
        assert labels.dtype == np.int32

        # Check label range (0 to n_clusters-1)
        assert np.min(labels) >= 0
        assert np.max(labels) < 4
        assert set(labels) == {0, 1, 2, 3}  # All 4 clusters present

        # Check dendrogram data structure
        assert isinstance(dendrogram_data, dict)
        assert 'linkage_matrix' in dendrogram_data
        assert 'linkage_method' in dendrogram_data
        assert 'n_clusters' in dendrogram_data
        assert dendrogram_data['linkage_method'] == 'ward'
        assert dendrogram_data['n_clusters'] == 4

    def test_fit_predict_validates_embeddings_shape(self):
        """Test fit_predict validates embeddings shape (AC-8)."""
        clustering = HierarchicalClustering()

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            clustering.fit_predict(np.random.randn(100).astype(np.float32))

        # Wrong embedding dimension
        with pytest.raises(ValueError, match="Embeddings must have 768 dimensions"):
            clustering.fit_predict(np.random.randn(100, 512).astype(np.float32))

    def test_fit_predict_validates_embeddings_dtype(self):
        """Test fit_predict validates embeddings dtype (AC-8)."""
        clustering = HierarchicalClustering()

        # Wrong dtype
        with pytest.raises(ValueError, match="Embeddings must have dtype float32"):
            clustering.fit_predict(np.random.randn(100, 768).astype(np.float64))

    def test_fit_predict_validates_embeddings_nan(self):
        """Test fit_predict validates embeddings contain no NaN (AC-8)."""
        clustering = HierarchicalClustering()

        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings[0, 0] = np.nan

        with pytest.raises(ValueError, match="Embeddings contain NaN values"):
            clustering.fit_predict(embeddings)

    def test_fit_predict_validates_embeddings_inf(self):
        """Test fit_predict validates embeddings contain no Inf (AC-8)."""
        clustering = HierarchicalClustering()

        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings[0, 0] = np.inf

        with pytest.raises(ValueError, match="Embeddings contain Inf values"):
            clustering.fit_predict(embeddings)

    def test_fit_predict_deterministic_ward(self):
        """Test fit_predict is deterministic for ward linkage (AC-9)."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering1 = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels1, _ = clustering1.fit_predict(embeddings)

        clustering2 = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels2, _ = clustering2.fit_predict(embeddings)

        # Results should be identical (deterministic)
        assert np.array_equal(labels1, labels2)

    def test_fit_predict_deterministic_complete(self):
        """Test fit_predict is deterministic for complete linkage (AC-9)."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering1 = HierarchicalClustering(n_clusters=4, linkage='complete')
        labels1, _ = clustering1.fit_predict(embeddings)

        clustering2 = HierarchicalClustering(n_clusters=4, linkage='complete')
        labels2, _ = clustering2.fit_predict(embeddings)

        # Results should be identical (deterministic)
        assert np.array_equal(labels1, labels2)


class TestLinkageMethodComparison:
    """Test linkage method comparison (AC-2)."""

    def test_compare_linkage_methods_returns_dataframe(self):
        """Test compare_linkage_methods returns DataFrame with correct structure."""
        # Small synthetic data for fast testing
        embeddings = np.random.randn(200, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, size=200).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4)
        comparison_df = clustering.compare_linkage_methods(embeddings, ground_truth)

        # Check DataFrame structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 4  # 4 linkage methods

        # Check columns
        expected_columns = [
            'linkage_method',
            'silhouette_score',
            'davies_bouldin_index',
            'cluster_purity',
            'runtime_seconds'
        ]
        assert list(comparison_df.columns) == expected_columns

        # Check linkage methods
        assert set(comparison_df['linkage_method']) == {'ward', 'complete', 'average', 'single'}

        # Check metrics are valid
        assert all(comparison_df['silhouette_score'] >= -1)
        assert all(comparison_df['silhouette_score'] <= 1)
        assert all(comparison_df['davies_bouldin_index'] >= 0)
        assert all(comparison_df['cluster_purity'] >= 0)
        assert all(comparison_df['cluster_purity'] <= 1)
        assert all(comparison_df['runtime_seconds'] >= 0)

    def test_compare_linkage_methods_sorted_by_silhouette(self):
        """Test compare_linkage_methods sorts results by Silhouette Score."""
        embeddings = np.random.randn(200, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, size=200).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4)
        comparison_df = clustering.compare_linkage_methods(embeddings, ground_truth)

        # Check sorted by silhouette_score (descending)
        silhouette_scores = comparison_df['silhouette_score'].values
        assert all(silhouette_scores[i] >= silhouette_scores[i+1]
                  for i in range(len(silhouette_scores)-1))

    def test_compare_linkage_methods_custom_methods_list(self):
        """Test compare_linkage_methods with custom methods list."""
        embeddings = np.random.randn(200, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, size=200).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4)
        comparison_df = clustering.compare_linkage_methods(
            embeddings,
            ground_truth,
            methods=['ward', 'complete']
        )

        # Check only 2 methods tested
        assert len(comparison_df) == 2
        assert set(comparison_df['linkage_method']) == {'ward', 'complete'}


class TestClusterMetricsCalculation:
    """Test cluster metrics calculation (AC-4)."""

    def test_calculate_metrics_returns_expected_structure(self):
        """Test calculate_metrics returns dict with all required fields."""
        embeddings = np.random.randn(200, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, size=200).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels, _ = clustering.fit_predict(embeddings)

        metrics = clustering.calculate_metrics(labels, embeddings, ground_truth)

        # Check dict structure
        assert isinstance(metrics, dict)
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'cluster_purity' in metrics
        assert 'cluster_sizes' in metrics

        # Check metric types and ranges
        assert isinstance(metrics['silhouette_score'], float)
        assert -1 <= metrics['silhouette_score'] <= 1

        assert isinstance(metrics['davies_bouldin_index'], float)
        assert metrics['davies_bouldin_index'] >= 0

        assert isinstance(metrics['cluster_purity'], float)
        assert 0 <= metrics['cluster_purity'] <= 1

        assert isinstance(metrics['cluster_sizes'], list)
        assert len(metrics['cluster_sizes']) == 4
        assert sum(metrics['cluster_sizes']) == len(embeddings)

    def test_calculate_metrics_cluster_sizes_sum_to_total(self):
        """Test calculate_metrics cluster sizes sum to total documents."""
        embeddings = np.random.randn(500, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, size=500).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels, _ = clustering.fit_predict(embeddings)

        metrics = clustering.calculate_metrics(labels, embeddings, ground_truth)

        # Cluster sizes should sum to total
        assert sum(metrics['cluster_sizes']) == 500

        # All clusters should have at least 1 document
        assert all(size > 0 for size in metrics['cluster_sizes'])


class TestPurityCalculation:
    """Test cluster purity calculation (AC-4)."""

    def test_calculate_purity_perfect_clustering(self):
        """Test purity calculation with perfect clustering."""
        # Create perfect clusters: each cluster = one category
        n_per_cluster = 50
        labels = np.concatenate([
            np.full(n_per_cluster, 0),
            np.full(n_per_cluster, 1),
            np.full(n_per_cluster, 2),
            np.full(n_per_cluster, 3)
        ]).astype(np.int32)

        ground_truth = np.concatenate([
            np.full(n_per_cluster, 0),
            np.full(n_per_cluster, 1),
            np.full(n_per_cluster, 2),
            np.full(n_per_cluster, 3)
        ]).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4)
        purity = clustering._calculate_purity(labels, ground_truth)

        # Perfect clustering should have purity = 1.0
        assert purity == 1.0

    def test_calculate_purity_random_clustering(self):
        """Test purity calculation with random clustering."""
        labels = np.random.randint(0, 4, size=400).astype(np.int32)
        ground_truth = np.random.randint(0, 4, size=400).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4)
        purity = clustering._calculate_purity(labels, ground_truth)

        # Random clustering should have purity around 0.25 (chance level)
        # Allow some variance due to randomness
        assert 0.15 <= purity <= 0.40

    def test_calculate_purity_returns_float(self):
        """Test purity calculation returns float type."""
        labels = np.random.randint(0, 4, size=200).astype(np.int32)
        ground_truth = np.random.randint(0, 4, size=200).astype(np.int32)

        clustering = HierarchicalClustering(n_clusters=4)
        purity = clustering._calculate_purity(labels, ground_truth)

        assert isinstance(purity, float)
        assert 0 <= purity <= 1


class TestDifferentLinkageMethods:
    """Test behavior with different linkage methods (AC-2)."""

    def test_ward_linkage_uses_euclidean_affinity(self):
        """Test ward linkage uses euclidean affinity."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels, _ = clustering.fit_predict(embeddings)

        # Should succeed without error (ward requires euclidean)
        assert labels is not None
        assert len(labels) == 100

    def test_complete_linkage_works(self):
        """Test complete linkage method works correctly."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='complete')
        labels, _ = clustering.fit_predict(embeddings)

        assert labels.shape == (100,)
        assert set(labels) == {0, 1, 2, 3}

    def test_average_linkage_works(self):
        """Test average linkage method works correctly."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='average')
        labels, _ = clustering.fit_predict(embeddings)

        assert labels.shape == (100,)
        assert set(labels) == {0, 1, 2, 3}

    def test_single_linkage_works(self):
        """Test single linkage method works correctly."""
        embeddings = np.random.randn(100, 768).astype(np.float32)

        clustering = HierarchicalClustering(n_clusters=4, linkage='single')
        labels, _ = clustering.fit_predict(embeddings)

        assert labels.shape == (100,)
        assert set(labels) == {0, 1, 2, 3}
