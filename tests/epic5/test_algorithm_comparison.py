"""
Unit tests for algorithm comparison module.

Tests the AlgorithmComparison class for cross-algorithm analysis.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

# Import module under test
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.evaluation.algorithm_comparison import AlgorithmComparison


class TestAlgorithmComparison:
    """Test suite for AlgorithmComparison class."""

    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics for testing."""
        return {
            "silhouette_score": 0.001,
            "davies_bouldin_index": 25.0,
            "cluster_purity": {"overall": 0.25}
        }

    @pytest.fixture
    def mock_labels(self):
        """Create mock cluster labels."""
        return np.array([0, 1, 2, 3, 0, 1, 2, 3] * 100, dtype=np.int32)

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        return np.random.rand(800, 768).astype(np.float32)

    @pytest.fixture
    def mock_ground_truth(self):
        """Create mock ground truth labels."""
        return np.array([0, 1, 2, 3, 0, 1, 2, 3] * 100, dtype=np.int32)

    def test_initialization(self):
        """Test AlgorithmComparison initialization."""
        comparison = AlgorithmComparison()
        assert comparison.algorithms == {}
        assert comparison.embeddings is None
        assert comparison.ground_truth is None

    def test_add_algorithm(self, mock_metrics, mock_labels):
        """Test adding an algorithm to comparison."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("K-Means", mock_metrics, mock_labels, runtime=45.0)

        assert "K-Means" in comparison.algorithms
        assert comparison.algorithms["K-Means"]["runtime"] == 45.0
        assert np.array_equal(comparison.algorithms["K-Means"]["labels"], mock_labels)

    def test_add_algorithm_missing_metrics(self, mock_labels):
        """Test that adding algorithm with missing metrics raises error."""
        comparison = AlgorithmComparison()
        incomplete_metrics = {"silhouette_score": 0.001}  # Missing other required metrics

        with pytest.raises(ValueError, match="Missing required metric"):
            comparison.add_algorithm("Test", incomplete_metrics, mock_labels)

    def test_set_embeddings(self, mock_embeddings):
        """Test setting embeddings."""
        comparison = AlgorithmComparison()
        comparison.set_embeddings(mock_embeddings)

        assert comparison.embeddings is not None
        assert comparison.embeddings.shape == (800, 768)

    def test_set_embeddings_invalid_shape(self):
        """Test that invalid embeddings shape raises error."""
        comparison = AlgorithmComparison()
        invalid_embeddings = np.random.rand(800)  # 1D instead of 2D

        with pytest.raises(ValueError, match="Embeddings must be 2D"):
            comparison.set_embeddings(invalid_embeddings)

    def test_set_ground_truth(self, mock_ground_truth):
        """Test setting ground truth labels."""
        comparison = AlgorithmComparison()
        comparison.set_ground_truth(mock_ground_truth)

        assert comparison.ground_truth is not None
        assert len(comparison.ground_truth) == 800

    def test_create_comparison_matrix(self, mock_metrics, mock_labels):
        """Test creating comparison matrix."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("K-Means", mock_metrics, mock_labels, runtime=45.0)
        comparison.add_algorithm("DBSCAN", mock_metrics, mock_labels, runtime=120.0)

        matrix = comparison.create_comparison_matrix()

        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 2  # Two algorithms
        assert "algorithm" in matrix.columns
        assert "silhouette_score" in matrix.columns
        assert "runtime_seconds" in matrix.columns

    def test_generate_confusion_matrices(self, mock_metrics, mock_labels, mock_ground_truth):
        """Test generating confusion matrices."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("K-Means", mock_metrics, mock_labels)
        comparison.set_ground_truth(mock_ground_truth)

        confusion_matrices = comparison.generate_confusion_matrices()

        assert "K-Means" in confusion_matrices
        assert confusion_matrices["K-Means"].shape == (4, 4)

    def test_generate_confusion_matrices_without_ground_truth(self, mock_metrics, mock_labels):
        """Test that generating confusion matrices without ground truth raises error."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("K-Means", mock_metrics, mock_labels)

        with pytest.raises(ValueError, match="Ground truth must be set"):
            comparison.generate_confusion_matrices()

    def test_identify_best_algorithms(self, mock_metrics, mock_labels):
        """Test identifying best algorithms."""
        comparison = AlgorithmComparison()

        # Add algorithms with different metrics
        metrics1 = {
            "silhouette_score": 0.002,  # Best
            "davies_bouldin_index": 30.0,
            "cluster_purity": {"overall": 0.30}
        }
        metrics2 = {
            "silhouette_score": 0.001,
            "davies_bouldin_index": 20.0,  # Best (lower is better)
            "cluster_purity": {"overall": 0.25}
        }

        comparison.add_algorithm("Algo1", metrics1, mock_labels, runtime=45.0)  # Best speed
        comparison.add_algorithm("Algo2", metrics2, mock_labels, runtime=120.0)

        best = comparison.identify_best_algorithms()

        assert "best_silhouette" in best
        assert "best_davies_bouldin" in best
        assert "best_speed" in best
        assert best["best_silhouette"] == "Algo1"
        assert best["best_davies_bouldin"] == "Algo2"
        assert best["best_speed"] == "Algo1"

    def test_export_to_csv(self, mock_metrics, mock_labels):
        """Test exporting comparison matrix to CSV."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("K-Means", mock_metrics, mock_labels, runtime=45.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.csv"
            comparison.export_to_csv(output_path)

            assert output_path.exists()

            # Verify CSV content
            df = pd.read_csv(output_path)
            assert len(df) == 1
            assert "algorithm" in df.columns

    def test_export_to_json(self, mock_metrics, mock_labels, mock_ground_truth):
        """Test exporting comprehensive results to JSON."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("K-Means", mock_metrics, mock_labels, runtime=45.0)
        comparison.set_ground_truth(mock_ground_truth)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.json"
            comparison.export_to_json(output_path)

            assert output_path.exists()

            # Verify JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert "metadata" in data
            assert "comparison_matrix" in data
            assert "best_algorithms" in data
            assert "confusion_matrices" in data

    def test_get_summary_statistics(self, mock_metrics, mock_labels):
        """Test calculating summary statistics."""
        comparison = AlgorithmComparison()
        comparison.add_algorithm("Algo1", mock_metrics, mock_labels, runtime=45.0)
        comparison.add_algorithm("Algo2", mock_metrics, mock_labels, runtime=120.0)

        summary = comparison.get_summary_statistics()

        assert "silhouette_score" in summary
        assert "runtime_seconds" in summary
        assert "min" in summary["runtime_seconds"]
        assert "max" in summary["runtime_seconds"]
        assert "mean" in summary["runtime_seconds"]

    def test_dbscan_noise_handling(self):
        """Test handling of DBSCAN noise points (-1 labels)."""
        comparison = AlgorithmComparison()

        # Create labels with noise points
        labels_with_noise = np.array([0, 1, 2, -1, 0, 1, -1, 2] * 100, dtype=np.int32)
        metrics = {
            "silhouette_score": 0.001,
            "davies_bouldin_index": 25.0,
            "cluster_purity": {"overall": 0.25}
        }

        comparison.add_algorithm("DBSCAN", metrics, labels_with_noise)

        matrix = comparison.create_comparison_matrix()

        assert matrix.loc[0, "n_noise_points"] > 0
        assert matrix.loc[0, "n_clusters_discovered"] == 3  # 0, 1, 2 (excluding -1)


class TestAlgorithmComparisonIntegration:
    """Integration tests using mock data that simulates real scenario."""

    def test_full_comparison_pipeline(self):
        """Test complete comparison pipeline with multiple algorithms."""
        # Create mock data
        n_samples = 1000
        embeddings = np.random.rand(n_samples, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, n_samples, dtype=np.int32)

        # Initialize comparison
        comparison = AlgorithmComparison()
        comparison.set_embeddings(embeddings)
        comparison.set_ground_truth(ground_truth)

        # Add multiple algorithms
        algorithms = {
            "K-Means": (np.random.randint(0, 4, n_samples, dtype=np.int32), 45.0),
            "DBSCAN": (np.random.randint(-1, 5, n_samples, dtype=np.int32), 120.0),
            "Hierarchical": (np.random.randint(0, 4, n_samples, dtype=np.int32), 200.0),
            "GMM": (np.random.randint(0, 4, n_samples, dtype=np.int32), 80.0),
        }

        for name, (labels, runtime) in algorithms.items():
            metrics = {
                "silhouette_score": np.random.uniform(0.0, 0.002),
                "davies_bouldin_index": np.random.uniform(20.0, 30.0),
                "cluster_purity": {"overall": np.random.uniform(0.2, 0.3)}
            }
            comparison.add_algorithm(name, metrics, labels, runtime)

        # Create comparison matrix
        matrix = comparison.create_comparison_matrix()
        assert len(matrix) == 4

        # Generate confusion matrices
        confusion_matrices = comparison.generate_confusion_matrices()
        assert len(confusion_matrices) == 4

        # Identify best algorithms
        best = comparison.identify_best_algorithms()
        assert "best_silhouette" in best
        assert "best_speed" in best

        # Get summary statistics
        summary = comparison.get_summary_statistics()
        assert "silhouette_score" in summary
        assert summary["runtime_seconds"]["min"] < summary["runtime_seconds"]["max"]

    def test_export_all_formats(self):
        """Test exporting to all supported formats."""
        # Create minimal comparison
        comparison = AlgorithmComparison()
        labels = np.array([0, 1, 2, 3] * 100, dtype=np.int32)
        ground_truth = np.array([0, 1, 2, 3] * 100, dtype=np.int32)
        metrics = {
            "silhouette_score": 0.001,
            "davies_bouldin_index": 25.0,
            "cluster_purity": {"overall": 0.25}
        }

        comparison.add_algorithm("K-Means", metrics, labels, runtime=45.0)
        comparison.set_ground_truth(ground_truth)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Export CSV
            csv_path = tmpdir / "comparison.csv"
            comparison.export_to_csv(csv_path)
            assert csv_path.exists()
            assert csv_path.stat().st_size > 0

            # Export JSON
            json_path = tmpdir / "comparison.json"
            comparison.export_to_json(json_path)
            assert json_path.exists()
            assert json_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
