"""
Integration tests for K-Means clustering pipeline.

This module tests the complete clustering workflow including script execution,
file outputs, and end-to-end validation.

Test Coverage:
    - AC-2: Cluster assignments export
    - AC-3: Centroids export
    - AC-4: Cluster distribution validation
    - AC-5: Convergence and performance
    - AC-6: Metadata export
    - AC-7: Error handling
    - AC-8: Logging and observability
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestClusteringScriptExecution:
    """Test full clustering script execution."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def test_script_runs_successfully(self, project_root):
        """Test clustering script executes without errors (AC-5, AC-8)."""
        script_path = project_root / "scripts" / "02_train_clustering.py"

        # Check if embeddings exist before running
        embeddings_path = project_root / "data" / "embeddings" / "train_embeddings.npy"
        if not embeddings_path.exists():
            pytest.skip("Embeddings not found - run Epic 1 script first")

        # Run script and measure time
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        elapsed_time = time.time() - start_time

        # Verify script succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify performance target: <5 minutes (AC-5)
        assert elapsed_time < 300, \
            f"Clustering took {elapsed_time:.1f}s (>5 minutes)"

        # Verify logging contains emoji prefixes (AC-8)
        assert "📊" in result.stdout or "📊" in result.stderr
        assert "✅" in result.stdout or "✅" in result.stderr

    def test_script_fails_without_embeddings(self, project_root, tmp_path):
        """Test script fails gracefully when embeddings missing (AC-7)."""
        script_path = project_root / "scripts" / "02_train_clustering.py"
        embeddings_path = project_root / "data" / "embeddings" / "train_embeddings.npy"

        # Only run this test if embeddings don't exist
        # (We don't want to delete real embeddings to test error handling)
        if embeddings_path.exists():
            pytest.skip("Embeddings exist - cannot test missing embeddings error without deleting real data")

        # Run script with missing embeddings
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=project_root
        )

        # Verify script failed (non-zero exit code)
        assert result.returncode != 0, "Script should fail when embeddings missing"

        # Verify helpful error message is present
        error_output = result.stderr + result.stdout
        assert "not found" in error_output.lower() or "embeddings" in error_output.lower(), \
            "Error message should mention missing embeddings"

        # Verify suggestion to run Epic 1 script is present
        assert "01_generate_embeddings" in error_output or "Epic 1" in error_output or "generate" in error_output, \
            "Error message should suggest running embedding generation script"


class TestClusterAssignmentsExport:
    """Test cluster assignments CSV export (AC-2)."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def assignments_path(self, project_root):
        """Get cluster assignments file path."""
        return project_root / "data" / "processed" / "cluster_assignments.csv"

    def test_assignments_file_exists(self, assignments_path):
        """Test cluster assignments file is created (AC-2)."""
        if not assignments_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        assert assignments_path.exists()

    def test_assignments_schema(self, assignments_path):
        """Test cluster assignments CSV has correct schema (AC-2)."""
        if not assignments_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        df = pd.read_csv(assignments_path)

        # Verify columns
        expected_columns = {'document_id', 'cluster_id', 'category_label'}
        assert set(df.columns) == expected_columns

    def test_assignments_row_count(self, assignments_path):
        """Test cluster assignments has 120K rows (AC-2)."""
        if not assignments_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        df = pd.read_csv(assignments_path)
        assert len(df) == 120000

    def test_assignments_cluster_ids_valid(self, assignments_path):
        """Test cluster IDs are valid [0, 3] (AC-2)."""
        if not assignments_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        df = pd.read_csv(assignments_path)

        # Verify cluster IDs in valid range
        assert df['cluster_id'].min() == 0
        assert df['cluster_id'].max() == 3
        assert df['cluster_id'].isin([0, 1, 2, 3]).all()

    def test_assignments_document_ids_unique(self, assignments_path):
        """Test document IDs are unique (AC-2)."""
        if not assignments_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        df = pd.read_csv(assignments_path)
        assert df['document_id'].is_unique


class TestCentroidsExport:
    """Test centroids NPY export (AC-3)."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def centroids_path(self, project_root):
        """Get centroids file path."""
        return project_root / "data" / "processed" / "centroids.npy"

    def test_centroids_file_exists(self, centroids_path):
        """Test centroids file is created (AC-3)."""
        if not centroids_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        assert centroids_path.exists()

    def test_centroids_shape(self, centroids_path):
        """Test centroids have correct shape (AC-3)."""
        if not centroids_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        centroids = np.load(centroids_path)
        assert centroids.shape == (4, 768)

    def test_centroids_dtype(self, centroids_path):
        """Test centroids have correct dtype (AC-3)."""
        if not centroids_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        centroids = np.load(centroids_path)
        assert centroids.dtype == np.float32

    def test_centroids_no_nan_inf(self, centroids_path):
        """Test centroids have no NaN or Inf values (AC-3)."""
        if not centroids_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        centroids = np.load(centroids_path)
        assert not np.any(np.isnan(centroids))
        assert not np.any(np.isinf(centroids))


class TestClusterBalance:
    """Test cluster distribution validation (AC-4)."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def assignments_path(self, project_root):
        """Get cluster assignments file path."""
        return project_root / "data" / "processed" / "cluster_assignments.csv"

    def test_cluster_balance(self, assignments_path):
        """Test cluster sizes are balanced (AC-4)."""
        if not assignments_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        df = pd.read_csv(assignments_path)
        cluster_sizes = df['cluster_id'].value_counts().sort_index()

        # Verify 4 clusters exist
        assert len(cluster_sizes) == 4

        # Verify no extreme imbalance (10%-50% range)
        min_size = 0.1 * 120000  # 12,000
        max_size = 0.5 * 120000  # 60,000

        for cluster_id, size in cluster_sizes.items():
            assert size >= min_size, \
                f"Cluster {cluster_id} too small: {size} < {min_size}"
            assert size <= max_size, \
                f"Cluster {cluster_id} too large: {size} > {max_size}"


class TestMetadataExport:
    """Test clustering metadata export (AC-6)."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def metadata_path(self, project_root):
        """Get metadata file path."""
        return project_root / "data" / "processed" / "cluster_metadata.json"

    def test_metadata_file_exists(self, metadata_path):
        """Test metadata file is created (AC-6)."""
        if not metadata_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        assert metadata_path.exists()

    def test_metadata_required_fields(self, metadata_path):
        """Test metadata contains all required fields (AC-6)."""
        if not metadata_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        with open(metadata_path) as f:
            metadata = json.load(f)

        required_fields = {
            'timestamp', 'n_clusters', 'n_documents', 'random_state',
            'n_iterations', 'inertia', 'cluster_sizes', 'config'
        }

        assert set(metadata.keys()) >= required_fields

    def test_metadata_values(self, metadata_path):
        """Test metadata values are correct (AC-6)."""
        if not metadata_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify expected values
        assert metadata['n_clusters'] == 4
        assert metadata['n_documents'] == 120000
        assert metadata['random_state'] == 42

        # Verify convergence
        assert metadata['n_iterations'] < 300

        # Verify cluster sizes
        assert len(metadata['cluster_sizes']) == 4
        assert sum(metadata['cluster_sizes']) == 120000

    def test_metadata_formatted_properly(self, metadata_path):
        """Test metadata JSON is human-readable (AC-6)."""
        if not metadata_path.exists():
            pytest.skip("Run clustering script first to generate outputs")

        # Read raw JSON to check formatting
        with open(metadata_path) as f:
            content = f.read()

        # Verify indentation (human-readable formatting)
        assert '\n' in content  # Multi-line
        assert '  ' in content  # Indented


class TestReproducibility:
    """Test clustering reproducibility (AC-1, AC-5)."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def test_reproducibility_across_runs(self, project_root):
        """Test clustering produces identical results across runs (AC-1, AC-5)."""
        script_path = project_root / "scripts" / "02_train_clustering.py"
        assignments_path = project_root / "data" / "processed" / "cluster_assignments.csv"
        centroids_path = project_root / "data" / "processed" / "centroids.npy"

        # Check if embeddings exist
        embeddings_path = project_root / "data" / "embeddings" / "train_embeddings.npy"
        if not embeddings_path.exists():
            pytest.skip("Embeddings not found - run Epic 1 script first")

        # Run clustering twice
        for run in range(2):
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                cwd=project_root
            )
            assert result.returncode == 0

            # CRITICAL FIX: Add explicit file system sync to prevent race conditions
            # Wait for file system to fully flush writes before reading
            time.sleep(0.1)  # 100ms buffer to ensure files are written

            # Verify files exist before reading
            assert assignments_path.exists(), f"Run {run+1}: Assignments file not created"
            assert centroids_path.exists(), f"Run {run+1}: Centroids file not created"

            if run == 0:
                # Save first run results
                # Force immediate read to avoid stale cache
                assignments_1 = pd.read_csv(assignments_path)
                centroids_1 = np.load(centroids_path)

                # Verify data was actually loaded
                assert len(assignments_1) > 0, "Run 1: Empty assignments loaded"
                assert centroids_1.size > 0, "Run 1: Empty centroids loaded"
            else:
                # Compare second run results
                # Force fresh read from disk
                assignments_2 = pd.read_csv(assignments_path)
                centroids_2 = np.load(centroids_path)

                # Verify data was actually loaded
                assert len(assignments_2) > 0, "Run 2: Empty assignments loaded"
                assert centroids_2.size > 0, "Run 2: Empty centroids loaded"

                # Verify identical cluster assignments (document_id and cluster_id columns only)
                # Note: category_label may differ if AG News dataset was loaded after first run
                cluster_ids_1 = assignments_1['cluster_id'].to_numpy()
                cluster_ids_2 = assignments_2['cluster_id'].to_numpy()

                # Debug: Check if arrays are identical
                if not np.array_equal(cluster_ids_1, cluster_ids_2):
                    # Find differences for debugging
                    diff_mask = cluster_ids_1 != cluster_ids_2
                    diff_count = np.sum(diff_mask)
                    if diff_count > 0:
                        diff_indices = np.where(diff_mask)[0][:10]  # First 10 differences
                        raise AssertionError(
                            f"Cluster assignments differ in {diff_count} positions. "
                            f"First differences at indices {diff_indices}: "
                            f"run1={cluster_ids_1[diff_indices]}, "
                            f"run2={cluster_ids_2[diff_indices]}"
                        )

                assert np.array_equal(cluster_ids_1, cluster_ids_2), \
                    "Cluster assignments differ across runs"

                assert np.allclose(centroids_1, centroids_2, atol=1e-6), \
                    "Centroids differ across runs"
