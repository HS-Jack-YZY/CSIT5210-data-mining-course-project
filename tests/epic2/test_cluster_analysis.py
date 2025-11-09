"""
Unit tests for ClusterAnalyzer class.

Tests cover:
- AC-1: Cluster-to-category mapping
- AC-2: Representative document extraction
- AC-3: Cluster purity calculation
- AC-6: Category distribution analysis
- AC-7: Representative document ranking
- AC-9: Error handling
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from context_aware_multi_agent_system.evaluation import ClusterAnalyzer


class TestClusterAnalyzer:
    """Test ClusterAnalyzer class for cluster analysis and labeling."""

    @pytest.fixture
    def synthetic_data(self):
        """Create small synthetic dataset for testing."""
        # Create 100 documents with 768 dimensions
        np.random.seed(42)
        embeddings = np.random.randn(100, 768).astype(np.float32)

        # Create 4 clusters
        labels = np.array([i % 4 for i in range(100)], dtype=np.int32)

        # Create centroids
        centroids = np.random.randn(4, 768).astype(np.float32)

        # Create ground truth with known purity
        # Cluster 0: 80% category 0, 20% category 1
        # Cluster 1: 100% category 1
        # Cluster 2: 70% category 2, 30% category 3
        # Cluster 3: 60% category 3, 40% category 0
        ground_truth = np.zeros(100, dtype=np.int32)

        for i in range(100):
            cluster_id = labels[i]
            if cluster_id == 0:
                ground_truth[i] = 0 if i % 5 != 0 else 1  # 80% cat 0, 20% cat 1
            elif cluster_id == 1:
                ground_truth[i] = 1  # 100% cat 1
            elif cluster_id == 2:
                ground_truth[i] = 2 if i % 10 < 7 else 3  # 70% cat 2, 30% cat 3
            else:  # cluster 3
                ground_truth[i] = 3 if i % 5 < 3 else 0  # 60% cat 3, 40% cat 0

        # Create titles
        titles = np.array([f"Document {i}" for i in range(100)])

        return {
            'embeddings': embeddings,
            'labels': labels,
            'centroids': centroids,
            'ground_truth': ground_truth,
            'titles': titles
        }

    def test_initialization(self, synthetic_data):
        """Test ClusterAnalyzer initialization validates inputs correctly."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        assert analyzer.n_clusters == 4
        assert len(analyzer.labels) == 100
        assert analyzer.embeddings.shape == (100, 768)

    def test_initialization_validation_errors(self, synthetic_data):
        """Test ClusterAnalyzer raises errors for invalid inputs (AC-9)."""
        # Test shape mismatch
        with pytest.raises(ValueError, match="Labels count.*!=.*embeddings count"):
            ClusterAnalyzer(
                labels=synthetic_data['labels'][:50],  # Wrong size
                embeddings=synthetic_data['embeddings'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

        # Test wrong dtype for labels
        with pytest.raises(ValueError, match="Labels must have dtype int32"):
            ClusterAnalyzer(
                labels=synthetic_data['labels'].astype(np.float32),
                embeddings=synthetic_data['embeddings'],
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

        # Test wrong embedding dimensions
        with pytest.raises(ValueError, match="Embeddings must have 768 dimensions"):
            ClusterAnalyzer(
                labels=synthetic_data['labels'],
                embeddings=np.random.randn(100, 512).astype(np.float32),
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

        # Test NaN in embeddings
        bad_embeddings = synthetic_data['embeddings'].copy()
        bad_embeddings[0, 0] = np.nan
        with pytest.raises(ValueError, match="Embeddings contain NaN values"):
            ClusterAnalyzer(
                labels=synthetic_data['labels'],
                embeddings=bad_embeddings,
                centroids=synthetic_data['centroids'],
                ground_truth=synthetic_data['ground_truth']
            )

    def test_map_clusters_to_categories(self, synthetic_data):
        """Test cluster-to-category mapping using majority voting (AC-1)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        mapping = analyzer.map_clusters_to_categories()

        # Verify mapping exists for all clusters
        assert len(mapping) == 4
        assert all(cluster_id in mapping for cluster_id in range(4))

        # Verify mapping values are category names
        assert all(
            category in ["World", "Sports", "Business", "Sci/Tech"]
            for category in mapping.values()
        )

        # Cluster 1 should map to Sports (100% category 1)
        assert mapping[1] == "Sports"

    def test_calculate_cluster_purity(self, synthetic_data):
        """Test cluster purity calculation (AC-3)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        purity = analyzer.calculate_cluster_purity()

        # Verify structure
        assert 'per_cluster' in purity
        assert 'average' in purity
        assert len(purity['per_cluster']) == 4

        # Verify purity values are in valid range
        assert all(0 <= p <= 1 for p in purity['per_cluster'].values())
        assert 0 <= purity['average'] <= 1

        # Cluster 1 should have 100% purity (all category 1)
        assert purity['per_cluster'][1] == 1.0

    def test_extract_representative_documents(self, synthetic_data):
        """Test representative document extraction (AC-2, AC-7)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth'],
            titles=synthetic_data['titles']
        )

        representatives = analyzer.extract_representative_documents(cluster_id=0, k=10)

        # Verify correct number of representatives
        assert len(representatives) == 10

        # Verify all representatives have required fields
        for doc in representatives:
            assert 'document_id' in doc
            assert 'category' in doc
            assert 'distance' in doc
            assert 'title' in doc

        # Verify sorted by distance (AC-7)
        distances = [doc['distance'] for doc in representatives]
        assert all(
            distances[i] <= distances[i+1]
            for i in range(len(distances)-1)
        ), "Representatives not sorted by distance"

        # Verify closest document has smallest distance
        assert distances[0] == min(distances)

    def test_extract_representative_documents_invalid_cluster(self, synthetic_data):
        """Test error handling for invalid cluster_id (AC-9)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        # Test invalid cluster ID
        with pytest.raises(ValueError, match="Invalid cluster_id"):
            analyzer.extract_representative_documents(cluster_id=10, k=10)

        with pytest.raises(ValueError, match="Invalid cluster_id"):
            analyzer.extract_representative_documents(cluster_id=-1, k=10)

    def test_get_category_distribution(self, synthetic_data):
        """Test category distribution analysis (AC-6)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        distribution = analyzer.get_category_distribution(cluster_id=1)

        # Verify all categories present
        assert len(distribution) == 4
        assert set(distribution.keys()) == {"World", "Sports", "Business", "Sci/Tech"}

        # Verify percentages sum to ~1.0 (AC-6)
        total = sum(distribution.values())
        assert abs(total - 1.0) < 1e-6, f"Distribution sum {total} != 1.0"

        # Verify all percentages in valid range
        assert all(0 <= p <= 1 for p in distribution.values())

        # Cluster 1 should be 100% Sports
        assert distribution["Sports"] == 1.0

    def test_get_category_distribution_invalid_cluster(self, synthetic_data):
        """Test error handling for invalid cluster_id in distribution (AC-9)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        with pytest.raises(ValueError, match="Invalid cluster_id"):
            analyzer.get_category_distribution(cluster_id=10)

    def test_generate_analysis_report(self, synthetic_data):
        """Test cluster analysis report generation (AC-4)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth'],
            titles=synthetic_data['titles']
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "cluster_analysis.txt"

            result_path = analyzer.generate_analysis_report(
                output_path=report_path,
                dataset_name="Test Dataset",
                n_documents=100,
                clustering_params={"K": 4, "random_state": 42}
            )

            # Verify report was created
            assert result_path.exists()
            assert result_path.stat().st_size > 0

            # Verify report content
            content = result_path.read_text()
            assert "Cluster Analysis Report" in content
            assert "Test Dataset" in content
            assert "K=4" in content
            assert "CLUSTER 0" in content
            assert "Category Distribution:" in content
            assert "Top 10 Representative Documents:" in content
            assert "OVERALL STATISTICS" in content

    def test_export_cluster_labels_json(self, synthetic_data):
        """Test cluster labels JSON export (AC-5)."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "cluster_labels.json"

            result_path = analyzer.export_cluster_labels_json(
                output_path=json_path,
                n_documents=100
            )

            # Verify JSON was created
            assert result_path.exists()

            # Verify JSON schema (AC-5)
            with open(result_path) as f:
                data = json.load(f)

            assert data['n_clusters'] == 4
            assert data['n_documents'] == 100
            assert 'average_purity' in data
            assert 'timestamp' in data
            assert len(data['clusters']) == 4

            # Verify cluster data structure
            for cluster_id in range(4):
                cluster_data = data['clusters'][str(cluster_id)]
                assert 'label' in cluster_data
                assert 'purity' in cluster_data
                assert 'size' in cluster_data
                assert 'dominant_category' in cluster_data
                assert 'distribution' in cluster_data

                # Verify purity in valid range
                assert 0 <= cluster_data['purity'] <= 1

    def test_caching_behavior(self, synthetic_data):
        """Test that mapping and purity are cached after first calculation."""
        analyzer = ClusterAnalyzer(
            labels=synthetic_data['labels'],
            embeddings=synthetic_data['embeddings'],
            centroids=synthetic_data['centroids'],
            ground_truth=synthetic_data['ground_truth']
        )

        # First call should compute
        mapping1 = analyzer.map_clusters_to_categories()
        # Second call should return cached result
        mapping2 = analyzer.map_clusters_to_categories()

        assert mapping1 is mapping2  # Same object (cached)

        # Same for purity
        purity1 = analyzer.calculate_cluster_purity()
        purity2 = analyzer.calculate_cluster_purity()

        assert purity1 is purity2  # Same object (cached)
