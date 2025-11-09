"""
Integration tests for PCA visualization pipeline (Story 2.4).

Tests cover full visualization workflow from data loading to PNG export,
using actual cluster results from Story 2.2.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from context_aware_multi_agent_system.config import Paths


class TestVisualizationPipeline:
    """Integration tests for full PCA visualization pipeline."""

    @pytest.fixture(scope="class")
    def paths(self):
        """Get paths configuration."""
        return Paths()

    def test_required_input_files_exist(self, paths):
        """Test all required input files exist (AC-10)."""
        embeddings_path = paths.data_embeddings / "train_embeddings.npy"
        assignments_path = paths.data_processed / "cluster_assignments.csv"
        centroids_path = paths.data_processed / "centroids.npy"

        assert embeddings_path.exists(), f"Embeddings not found: {embeddings_path}"
        assert assignments_path.exists(), f"Cluster assignments not found: {assignments_path}"
        assert centroids_path.exists(), f"Centroids not found: {centroids_path}"

    def test_embeddings_valid_shape(self, paths):
        """Test embeddings have valid shape (120000, 768) (AC-10)."""
        embeddings_path = paths.data_embeddings / "train_embeddings.npy"
        embeddings = np.load(embeddings_path)

        assert embeddings.shape[0] == 120000, f"Expected 120000 samples, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == 768, f"Expected 768 dimensions, got {embeddings.shape[1]}"
        assert embeddings.dtype == np.float32

    def test_cluster_assignments_valid(self, paths):
        """Test cluster assignments are valid (AC-10)."""
        assignments_path = paths.data_processed / "cluster_assignments.csv"
        df = pd.read_csv(assignments_path)

        assert 'cluster_id' in df.columns
        assert len(df) == 120000

        labels = df['cluster_id'].values
        unique_labels = np.unique(labels)

        assert np.all((unique_labels >= 0) & (unique_labels <= 3)), \
            f"Expected labels in [0, 3], got {unique_labels}"

    def test_centroids_valid_shape(self, paths):
        """Test centroids have valid shape (4, 768) (AC-10)."""
        centroids_path = paths.data_processed / "centroids.npy"
        centroids = np.load(centroids_path)

        assert centroids.shape == (4, 768), f"Expected shape (4, 768), got {centroids.shape}"
        assert centroids.dtype == np.float32

    def test_full_visualization_script_execution(self, paths):
        """Test full visualization script executes successfully (AC-1 to AC-10)."""
        script_path = paths.project_root / "scripts" / "04_visualize_clusters.py"
        assert script_path.exists(), f"Script not found: {script_path}"

        # Run visualization script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        # Check script succeeded
        assert result.returncode == 0, \
            f"Script failed with stderr: {result.stderr}\nstdout: {result.stdout}"

        # Verify output file was created
        output_path = paths.project_root / "visualizations" / "cluster_pca.png"
        assert output_path.exists(), f"Visualization not created: {output_path}"

    def test_visualization_output_valid_png(self, paths):
        """Test visualization output is a valid PNG file (AC-5)."""
        output_path = paths.project_root / "visualizations" / "cluster_pca.png"

        # Skip if file doesn't exist (script hasn't been run yet)
        if not output_path.exists():
            pytest.skip("Visualization not generated yet - run 04_visualize_clusters.py first")

        # Verify PNG format
        img = Image.open(output_path)
        assert img.format == 'PNG'

        # Verify dimensions are reasonable
        width, height = img.size
        assert width > 0 and height > 0
        assert width <= 10000 and height <= 10000  # Sanity check

    def test_visualization_300_dpi_resolution(self, paths):
        """Test visualization has 300 DPI resolution (AC-5)."""
        output_path = paths.project_root / "visualizations" / "cluster_pca.png"

        if not output_path.exists():
            pytest.skip("Visualization not generated yet")

        img = Image.open(output_path)
        dpi = img.info.get('dpi')

        assert dpi is not None, "DPI information not found in PNG"
        # DPI may have slight floating point variations
        assert abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1, f"Expected ~300 DPI, got {dpi}"

    def test_visualization_file_size_reasonable(self, paths):
        """Test visualization file size is reasonable (AC-5)."""
        output_path = paths.project_root / "visualizations" / "cluster_pca.png"

        if not output_path.exists():
            pytest.skip("Visualization not generated yet")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        assert file_size_mb > 0, "File is empty"
        assert file_size_mb < 5.0, f"File too large: {file_size_mb:.2f} MB (expected < 5 MB)"

    def test_optional_plotly_html_generated(self, paths):
        """Test optional Plotly HTML visualization is generated (AC-6)."""
        html_path = paths.project_root / "visualizations" / "cluster_pca.html"

        # This is optional, so we only check if it exists
        if html_path.exists():
            file_size_mb = html_path.stat().st_size / (1024 * 1024)
            assert file_size_mb > 0, "HTML file is empty"
            assert file_size_mb < 20.0, f"HTML file too large: {file_size_mb:.2f} MB"

            # Verify it's HTML
            with open(html_path, 'r') as f:
                content = f.read(100)
                assert '<html>' in content.lower() or '<!DOCTYPE' in content

    def test_execution_time_under_5_minutes(self, paths):
        """Test visualization completes within 5 minutes (Performance constraint)."""
        import time

        script_path = paths.project_root / "scripts" / "04_visualize_clusters.py"

        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.time() - start_time

        assert result.returncode == 0
        assert elapsed < 300, f"Execution took {elapsed:.1f}s (expected < 300s)"

    def test_variance_explained_logged(self, paths):
        """Test variance explained is logged correctly (AC-7)."""
        script_path = paths.project_root / "scripts" / "04_visualize_clusters.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0

        # Check logs contain variance information
        output = result.stdout + result.stderr
        assert "PC1 variance:" in output
        assert "PC2 variance:" in output
        assert "Total variance explained:" in output

    def test_emoji_prefixed_logging(self, paths):
        """Test emoji-prefixed logging is used (AC-9)."""
        script_path = paths.project_root / "scripts" / "04_visualize_clusters.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0

        output = result.stdout + result.stderr

        # Check for emoji prefixes
        assert "📊" in output  # INFO logs
        assert "✅" in output  # SUCCESS logs
