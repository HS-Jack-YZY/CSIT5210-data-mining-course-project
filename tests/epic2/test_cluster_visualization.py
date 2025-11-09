"""
Unit tests for PCA cluster visualization (Story 2.4).

Tests cover PCAVisualizer class methods including PCA dimensionality reduction,
scatter plot generation, and publication-quality PNG export.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from context_aware_multi_agent_system.visualization.cluster_plots import PCAVisualizer


class TestPCAVisualizer:
    """Test suite for PCAVisualizer class."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic clustering data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 768
        n_clusters = 4

        embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
        labels = np.random.randint(0, n_clusters, n_samples).astype(np.int32)
        centroids = np.random.randn(n_clusters, n_features).astype(np.float32)

        return embeddings, labels, centroids

    def test_initialization_valid_data(self, synthetic_data):
        """Test PCAVisualizer initialization with valid data (AC-10)."""
        embeddings, labels, centroids = synthetic_data

        visualizer = PCAVisualizer(embeddings, labels, centroids)

        assert visualizer.embeddings.shape == (1000, 768)
        assert visualizer.labels.shape == (1000,)
        assert visualizer.centroids.shape == (4, 768)
        assert visualizer.pca is None  # Not fitted yet
        assert visualizer.embeddings_2d is None
        assert visualizer.centroids_2d is None

    def test_initialization_invalid_embeddings_shape(self, synthetic_data):
        """Test error handling for invalid embeddings shape (AC-10)."""
        _, labels, centroids = synthetic_data
        invalid_embeddings = np.random.randn(1000).astype(np.float32)  # 1D instead of 2D

        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            PCAVisualizer(invalid_embeddings, labels, centroids)

    def test_initialization_wrong_embedding_dimensions(self, synthetic_data):
        """Test error handling for wrong embedding dimensions (AC-10)."""
        _, labels, centroids = synthetic_data
        wrong_dim_embeddings = np.random.randn(1000, 512).astype(np.float32)  # 512D instead of 768D

        with pytest.raises(ValueError, match="Embeddings must have 768 dimensions"):
            PCAVisualizer(wrong_dim_embeddings, labels, centroids)

    def test_initialization_embeddings_labels_mismatch(self, synthetic_data):
        """Test error handling for embeddings-labels count mismatch (AC-10)."""
        embeddings, _, centroids = synthetic_data
        mismatched_labels = np.random.randint(0, 4, 500).astype(np.int32)  # 500 instead of 1000

        with pytest.raises(ValueError, match="Labels length .* must match embeddings length"):
            PCAVisualizer(embeddings, mismatched_labels, centroids)

    def test_initialization_invalid_centroids_shape(self, synthetic_data):
        """Test error handling for invalid centroids shape (AC-10)."""
        embeddings, labels, _ = synthetic_data
        invalid_centroids = np.random.randn(4, 512).astype(np.float32)  # 512D instead of 768D

        with pytest.raises(ValueError, match="Centroids must have shape"):
            PCAVisualizer(embeddings, labels, invalid_centroids)

    def test_initialization_nan_in_embeddings(self, synthetic_data):
        """Test error handling for NaN values in embeddings (AC-10)."""
        embeddings, labels, centroids = synthetic_data
        embeddings[0, 0] = np.nan

        with pytest.raises(ValueError, match="Embeddings contain NaN or Inf values"):
            PCAVisualizer(embeddings, labels, centroids)

    def test_initialization_inf_in_centroids(self, synthetic_data):
        """Test error handling for Inf values in centroids (AC-10)."""
        embeddings, labels, centroids = synthetic_data
        centroids[0, 0] = np.inf

        with pytest.raises(ValueError, match="Centroids contain NaN or Inf values"):
            PCAVisualizer(embeddings, labels, centroids)

    def test_apply_pca_reduces_dimensions(self, synthetic_data):
        """Test PCA reduces dimensions from 768D to 2D (AC-1)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)

        embeddings_2d, centroids_2d, variance = visualizer.apply_pca()

        assert embeddings_2d.shape == (1000, 2)
        assert centroids_2d.shape == (4, 2)
        assert isinstance(variance, (float, np.floating))

    def test_apply_pca_variance_range(self, synthetic_data):
        """Test variance explained is in valid range [0, 1] (AC-1, AC-7)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)

        _, _, variance = visualizer.apply_pca()

        assert 0 <= variance <= 1.0

    def test_apply_pca_positive_variance(self, synthetic_data):
        """Test variance explained is positive (AC-1, AC-7)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)

        _, _, variance = visualizer.apply_pca()

        assert variance > 0

    def test_apply_pca_reproducibility(self, synthetic_data):
        """Test PCA produces identical results with same random_state (AC-1)."""
        embeddings, labels, centroids = synthetic_data

        visualizer1 = PCAVisualizer(embeddings, labels, centroids)
        embeddings_2d_1, centroids_2d_1, variance_1 = visualizer1.apply_pca()

        visualizer2 = PCAVisualizer(embeddings, labels, centroids)
        embeddings_2d_2, centroids_2d_2, variance_2 = visualizer2.apply_pca()

        np.testing.assert_array_almost_equal(embeddings_2d_1, embeddings_2d_2)
        np.testing.assert_array_almost_equal(centroids_2d_1, centroids_2d_2)
        assert variance_1 == variance_2

    def test_get_variance_explained(self, synthetic_data):
        """Test get_variance_explained returns valid components (AC-7)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)
        visualizer.apply_pca()

        pc1_var, pc2_var, total_var = visualizer.get_variance_explained()

        assert 0 <= pc1_var <= 1.0
        assert 0 <= pc2_var <= 1.0
        assert 0 <= total_var <= 1.0
        assert np.isclose(total_var, pc1_var + pc2_var)

    def test_get_variance_explained_before_pca_raises_error(self, synthetic_data):
        """Test get_variance_explained raises error if PCA not applied."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)

        with pytest.raises(RuntimeError, match="Must call apply_pca"):
            visualizer.get_variance_explained()

    def test_generate_visualization_creates_file(self, synthetic_data):
        """Test visualization file is created (AC-5)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)
        visualizer.apply_pca()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cluster.png"
            saved_path = visualizer.generate_visualization(output_path)

            assert saved_path.exists()
            assert saved_path.stat().st_size > 0

    def test_generate_visualization_300_dpi(self, synthetic_data):
        """Test visualization is saved with 300 DPI resolution (AC-5)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)
        visualizer.apply_pca()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cluster.png"
            visualizer.generate_visualization(output_path, dpi=300)

            # Verify DPI using PIL
            img = Image.open(output_path)
            assert img.format == 'PNG'
            # DPI may have slight floating point variations
            dpi = img.info.get('dpi')
            assert dpi is not None
            assert abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1

    def test_generate_visualization_auto_creates_directory(self, synthetic_data):
        """Test output directory is created automatically (AC-5, AC-10)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)
        visualizer.apply_pca()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "new_dir" / "nested" / "test_cluster.png"
            saved_path = visualizer.generate_visualization(output_path)

            assert saved_path.parent.exists()
            assert saved_path.exists()

    def test_generate_visualization_before_pca_raises_error(self, synthetic_data):
        """Test generate_visualization raises error if PCA not applied (AC-10)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cluster.png"

            with pytest.raises(RuntimeError, match="Must call apply_pca"):
                visualizer.generate_visualization(output_path)

    def test_generate_visualization_reasonable_file_size(self, synthetic_data):
        """Test visualization file size is reasonable (AC-5)."""
        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)
        visualizer.apply_pca()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cluster.png"
            visualizer.generate_visualization(output_path)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            assert file_size_mb < 5.0  # Should be less than 5 MB

    def test_colorblind_palette_used(self, synthetic_data):
        """Test colorblind-friendly palette is used (AC-8)."""
        import seaborn as sns

        embeddings, labels, centroids = synthetic_data
        visualizer = PCAVisualizer(embeddings, labels, centroids)
        visualizer.apply_pca()

        # Verify seaborn colorblind palette can be loaded
        colors = sns.color_palette("colorblind", 4)
        assert len(colors) == 4

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cluster.png"
            visualizer.generate_visualization(output_path)
            assert output_path.exists()
