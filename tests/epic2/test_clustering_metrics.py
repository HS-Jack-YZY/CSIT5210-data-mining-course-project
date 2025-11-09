"""
Unit tests for ClusteringMetrics class.

Tests cover all acceptance criteria (AC-1 through AC-10) for cluster quality evaluation,
including Silhouette Score, Davies-Bouldin Index, cluster purity, confusion matrix,
and error handling.

Test Coverage:
    - AC-1: Silhouette Score calculation
    - AC-2: Davies-Bouldin Index calculation
    - AC-3: Intra-cluster distance calculation
    - AC-4: Inter-cluster distance calculation
    - AC-5: Cluster purity calculation
    - AC-6: Confusion matrix generation
    - AC-7: Cluster balance validation
    - AC-8: Comprehensive metric evaluation
    - AC-9: Error handling (invalid inputs, shape mismatches)
"""

import pytest
import numpy as np
from pathlib import Path

from context_aware_multi_agent_system.evaluation import ClusteringMetrics


@pytest.fixture
def synthetic_data():
    """
    Create synthetic clustering data for unit tests.

    Returns:
        Dictionary with embeddings, labels, centroids, and ground_truth
    """
    np.random.seed(42)

    n_documents = 1000
    n_clusters = 4
    embedding_dim = 768

    # Generate synthetic embeddings
    embeddings = np.random.randn(n_documents, embedding_dim).astype(np.float32)

    # Generate synthetic cluster labels (well-separated clusters)
    labels = np.random.randint(0, n_clusters, n_documents).astype(np.int32)

    # Generate synthetic centroids
    centroids = np.random.randn(n_clusters, embedding_dim).astype(np.float32)

    # Generate synthetic ground truth (AG News categories)
    ground_truth = np.random.randint(0, 4, n_documents).astype(np.int32)

    return {
        'embeddings': embeddings,
        'labels': labels,
        'centroids': centroids,
        'ground_truth': ground_truth,
        'n_documents': n_documents,
        'n_clusters': n_clusters
    }


class TestClusteringMetricsInitialization:
    """Test ClusteringMetrics initialization and input validation."""

    def test_valid_initialization(self, synthetic_data):
        """Test successful initialization with valid inputs (AC-10)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        assert metrics.n_clusters == synthetic_data['n_clusters']
        assert metrics.embeddings.shape[0] == synthetic_data['n_documents']

    def test_invalid_embeddings_shape(self, synthetic_data):
        """Test ValueError for invalid embeddings shape (AC-10)."""
        # Create 1D embeddings (invalid)
        invalid_embeddings = np.random.randn(1000).astype(np.float32)

        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            ClusteringMetrics(
                embeddings=invalid_embeddings,
                labels=synthetic_data['labels'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_invalid_embeddings_dimensions(self, synthetic_data):
        """Test ValueError for incorrect embedding dimensions (AC-10)."""
        # Create embeddings with wrong dimension size
        invalid_embeddings = np.random.randn(1000, 512).astype(np.float32)

        with pytest.raises(ValueError, match="Embeddings must have 768 dimensions"):
            ClusteringMetrics(
                embeddings=invalid_embeddings,
                labels=synthetic_data['labels'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_invalid_embeddings_dtype(self, synthetic_data):
        """Test ValueError for incorrect embeddings dtype (AC-10)."""
        # Create embeddings with wrong dtype
        invalid_embeddings = synthetic_data['embeddings'].astype(np.float64)

        with pytest.raises(ValueError, match="Embeddings must have dtype float32"):
            ClusteringMetrics(
                embeddings=invalid_embeddings,
                labels=synthetic_data['labels'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_embeddings_with_nan(self, synthetic_data):
        """Test ValueError for embeddings containing NaN (AC-10)."""
        invalid_embeddings = synthetic_data['embeddings'].copy()
        invalid_embeddings[0, 0] = np.nan

        with pytest.raises(ValueError, match="Embeddings contain NaN values"):
            ClusteringMetrics(
                embeddings=invalid_embeddings,
                labels=synthetic_data['labels'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_embeddings_with_inf(self, synthetic_data):
        """Test ValueError for embeddings containing Inf (AC-10)."""
        invalid_embeddings = synthetic_data['embeddings'].copy()
        invalid_embeddings[0, 0] = np.inf

        with pytest.raises(ValueError, match="Embeddings contain Inf values"):
            ClusteringMetrics(
                embeddings=invalid_embeddings,
                labels=synthetic_data['labels'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_labels_count_mismatch(self, synthetic_data):
        """Test ValueError for labels count mismatch (AC-10)."""
        # Create labels with wrong count
        invalid_labels = np.random.randint(0, 4, 500).astype(np.int32)

        with pytest.raises(ValueError, match="Labels count mismatch"):
            ClusteringMetrics(
                embeddings=synthetic_data['embeddings'],
                labels=invalid_labels,
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_invalid_label_range(self, synthetic_data):
        """Test ValueError for labels outside valid range (AC-10)."""
        # Create labels with invalid range
        invalid_labels = synthetic_data['labels'].copy()
        invalid_labels[0] = 10  # Out of range

        with pytest.raises(ValueError, match="Invalid cluster labels"):
            ClusteringMetrics(
                embeddings=synthetic_data['embeddings'],
                labels=invalid_labels,
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )


class TestSilhouetteScore:
    """Test Silhouette Score calculation (AC-1)."""

    def test_silhouette_score_range(self, synthetic_data):
        """Test Silhouette Score is in valid range [-1, 1] (AC-1)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        score = metrics.calculate_silhouette_score()

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_silhouette_score_reproducibility(self, synthetic_data):
        """Test Silhouette Score is deterministic (AC-1)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        score1 = metrics.calculate_silhouette_score()
        score2 = metrics.calculate_silhouette_score()

        assert score1 == score2


class TestDaviesBouldinIndex:
    """Test Davies-Bouldin Index calculation (AC-2)."""

    def test_davies_bouldin_positive(self, synthetic_data):
        """Test Davies-Bouldin Index is positive (AC-2)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        index = metrics.calculate_davies_bouldin_index()

        assert isinstance(index, float)
        assert index > 0

    def test_davies_bouldin_reproducibility(self, synthetic_data):
        """Test Davies-Bouldin Index is deterministic (AC-2)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        index1 = metrics.calculate_davies_bouldin_index()
        index2 = metrics.calculate_davies_bouldin_index()

        assert index1 == index2


class TestIntraClusterDistance:
    """Test intra-cluster distance calculation (AC-3)."""

    def test_intra_cluster_distance_structure(self, synthetic_data):
        """Test intra-cluster distance returns correct structure (AC-3)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        distances = metrics.calculate_intra_cluster_distance()

        # Check structure
        assert isinstance(distances, dict)
        assert 'overall' in distances

        # Check per-cluster distances
        for cluster_id in range(synthetic_data['n_clusters']):
            assert f'cluster_{cluster_id}' in distances
            assert isinstance(distances[f'cluster_{cluster_id}'], float)
            assert distances[f'cluster_{cluster_id}'] >= 0

    def test_intra_cluster_distance_overall(self, synthetic_data):
        """Test overall intra-cluster distance is positive (AC-3)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        distances = metrics.calculate_intra_cluster_distance()

        assert distances['overall'] > 0


class TestInterClusterDistance:
    """Test inter-cluster distance calculation (AC-4)."""

    def test_inter_cluster_distance_structure(self, synthetic_data):
        """Test inter-cluster distance returns correct structure (AC-4)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        distances = metrics.calculate_inter_cluster_distance()

        # Check structure
        assert isinstance(distances, dict)
        assert 'min' in distances
        assert 'max' in distances
        assert 'mean' in distances
        assert 'pairwise' in distances

        # Check values
        assert distances['min'] > 0
        assert distances['max'] >= distances['min']
        assert distances['mean'] > 0

        # Check pairwise distances count (n choose 2 = 6 for 4 clusters)
        n_pairs = (synthetic_data['n_clusters'] * (synthetic_data['n_clusters'] - 1)) // 2
        assert len(distances['pairwise']) == n_pairs


class TestClusterPurity:
    """Test cluster purity calculation (AC-5)."""

    def test_cluster_purity_range(self, synthetic_data):
        """Test cluster purity values are in range [0, 1] (AC-5)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        purity = metrics.calculate_cluster_purity()

        # Check overall purity
        assert 'overall' in purity
        assert 0 <= purity['overall'] <= 1.0

        # Check per-cluster purity
        for cluster_id in range(synthetic_data['n_clusters']):
            assert f'cluster_{cluster_id}' in purity
            assert 0 <= purity[f'cluster_{cluster_id}'] <= 1.0

    def test_cluster_purity_perfect_alignment(self):
        """Test cluster purity with perfect cluster-category alignment (AC-5)."""
        np.random.seed(42)

        # Create perfectly aligned clusters and ground truth
        n_documents = 1000
        n_clusters = 4
        embedding_dim = 768

        embeddings = np.random.randn(n_documents, embedding_dim).astype(np.float32)

        # Perfect alignment: cluster_id == category_id
        labels = np.repeat([0, 1, 2, 3], n_documents // n_clusters).astype(np.int32)
        ground_truth = labels.copy()

        centroids = np.random.randn(n_clusters, embedding_dim).astype(np.float32)

        metrics = ClusteringMetrics(embeddings, labels, centroids, ground_truth)
        purity = metrics.calculate_cluster_purity()

        # With perfect alignment, all purity scores should be 1.0
        assert purity['overall'] == 1.0
        for cluster_id in range(n_clusters):
            assert purity[f'cluster_{cluster_id}'] == 1.0


class TestConfusionMatrix:
    """Test confusion matrix generation (AC-6)."""

    def test_confusion_matrix_shape(self, synthetic_data):
        """Test confusion matrix has correct shape (AC-6)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        cm = metrics.generate_confusion_matrix()

        assert cm.shape == (4, 4)

    def test_confusion_matrix_sum(self, synthetic_data):
        """Test confusion matrix sum equals total documents (AC-6)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        cm = metrics.generate_confusion_matrix()

        assert cm.sum() == synthetic_data['n_documents']


class TestClusterBalance:
    """Test cluster balance validation (AC-7)."""

    def test_cluster_balance_structure(self, synthetic_data):
        """Test cluster balance returns correct structure (AC-7)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        is_balanced, cluster_sizes = metrics.validate_cluster_balance()

        assert isinstance(is_balanced, bool)
        assert isinstance(cluster_sizes, dict)
        assert len(cluster_sizes) == synthetic_data['n_clusters']

    def test_cluster_balance_imbalanced(self):
        """Test cluster balance detection for imbalanced clusters (AC-7)."""
        np.random.seed(42)

        n_documents = 1000
        n_clusters = 4
        embedding_dim = 768

        embeddings = np.random.randn(n_documents, embedding_dim).astype(np.float32)

        # Create imbalanced labels (cluster 0 has 70% of documents)
        labels = np.zeros(n_documents, dtype=np.int32)
        labels[:700] = 0  # 70% in cluster 0
        labels[700:800] = 1  # 10% in cluster 1
        labels[800:900] = 2  # 10% in cluster 2
        labels[900:] = 3    # 10% in cluster 3

        centroids = np.random.randn(n_clusters, embedding_dim).astype(np.float32)
        ground_truth = np.random.randint(0, 4, n_documents).astype(np.int32)

        metrics = ClusteringMetrics(embeddings, labels, centroids, ground_truth)
        is_balanced, cluster_sizes = metrics.validate_cluster_balance()

        # Should detect imbalance (cluster 0 > 50%)
        assert not is_balanced
        assert cluster_sizes[0] == 700


class TestEvaluateAll:
    """Test comprehensive evaluation (AC-8)."""

    def test_evaluate_all_structure(self, synthetic_data):
        """Test evaluate_all returns complete results structure (AC-8)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        results = metrics.evaluate_all()

        # Check required keys
        required_keys = {
            'silhouette_score',
            'davies_bouldin_index',
            'intra_cluster_distance',
            'inter_cluster_distance',
            'cluster_purity',
            'cluster_sizes',
            'is_balanced'
        }
        assert set(results.keys()) == required_keys

        # Check data types
        assert isinstance(results['silhouette_score'], float)
        assert isinstance(results['davies_bouldin_index'], float)
        assert isinstance(results['intra_cluster_distance'], dict)
        assert isinstance(results['inter_cluster_distance'], dict)
        assert isinstance(results['cluster_purity'], dict)
        assert isinstance(results['cluster_sizes'], list)
        assert isinstance(results['is_balanced'], bool)

    def test_evaluate_all_metrics_valid(self, synthetic_data):
        """Test all metrics in evaluate_all have valid values (AC-8)."""
        metrics = ClusteringMetrics(
            embeddings=synthetic_data['embeddings'],
            labels=synthetic_data['labels'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        results = metrics.evaluate_all()

        # Validate ranges
        assert -1.0 <= results['silhouette_score'] <= 1.0
        assert results['davies_bouldin_index'] > 0
        assert 0 <= results['cluster_purity']['overall'] <= 1.0
        assert len(results['cluster_sizes']) == synthetic_data['n_clusters']
        assert all(size >= 0 for size in results['cluster_sizes'])
