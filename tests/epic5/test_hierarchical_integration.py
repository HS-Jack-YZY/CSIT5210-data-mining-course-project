"""
Integration tests for hierarchical clustering pipeline.

Tests cover:
- AC-1: Full clustering pipeline execution
- AC-3: Dendrogram generation
- AC-5: Cluster assignments export
- AC-6: Memory and performance monitoring
- AC-7: Logging and observability
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

import json
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Paths
from context_aware_multi_agent_system.models.hierarchical_clustering import HierarchicalClustering
from context_aware_multi_agent_system.visualization.dendrogram_plot import generate_dendrogram


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for integration testing."""
    np.random.seed(42)
    return np.random.randn(1000, 768).astype(np.float32)


@pytest.fixture
def sample_ground_truth():
    """Create sample ground truth labels."""
    np.random.seed(42)
    return np.random.randint(0, 4, size=1000).astype(np.int32)


@pytest.fixture
def temp_paths(tmp_path):
    """Create temporary paths for integration testing."""
    return {
        'assignments': tmp_path / 'hierarchical_assignments.csv',
        'metrics': tmp_path / 'hierarchical_metrics.json',
        'dendrogram': tmp_path / 'dendrogram.png',
        'linkage_comparison': tmp_path / 'linkage_comparison.csv'
    }


class TestFullHierarchicalPipeline:
    """Test full hierarchical clustering pipeline (AC-1, AC-2)."""

    def test_full_pipeline_execution(self, sample_embeddings, sample_ground_truth):
        """Test full hierarchical clustering pipeline executes without errors."""
        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')

        # Step 1: Fit clustering
        labels, dendrogram_data = clustering.fit_predict(sample_embeddings)

        # Validate labels
        assert labels.shape == (1000,)
        assert labels.dtype == np.int32
        assert set(labels) == {0, 1, 2, 3}

        # Validate dendrogram data
        assert 'linkage_matrix' in dendrogram_data
        assert dendrogram_data['linkage_method'] == 'ward'

        # Step 2: Calculate metrics
        metrics = clustering.calculate_metrics(labels, sample_embeddings, sample_ground_truth)

        # Validate metrics
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'cluster_purity' in metrics
        assert 'cluster_sizes' in metrics
        assert sum(metrics['cluster_sizes']) == 1000

    def test_linkage_comparison_pipeline(self, sample_embeddings, sample_ground_truth):
        """Test linkage method comparison pipeline (AC-2)."""
        clustering = HierarchicalClustering(n_clusters=4)

        # Compare linkage methods
        start_time = time.time()
        comparison_df = clustering.compare_linkage_methods(
            sample_embeddings,
            sample_ground_truth,
            methods=['ward', 'complete']
        )
        end_time = time.time()

        # Validate comparison results
        assert len(comparison_df) == 2
        assert 'linkage_method' in comparison_df.columns
        assert 'silhouette_score' in comparison_df.columns
        assert 'runtime_seconds' in comparison_df.columns

        # Validate runtime tracking
        total_runtime = end_time - start_time
        assert total_runtime > 0
        assert all(comparison_df['runtime_seconds'] > 0)


class TestDendrogramGeneration:
    """Test dendrogram generation (AC-3, AC-10)."""

    def test_dendrogram_generation_creates_file(self, sample_embeddings, temp_paths):
        """Test dendrogram generation creates PNG file."""
        output_path = generate_dendrogram(
            sample_embeddings,
            linkage_method='ward',
            output_path=temp_paths['dendrogram'],
            n_clusters=4,
            truncate_mode='lastp',
            p=30,
            dpi=300
        )

        # Validate file exists
        assert output_path.exists()
        assert output_path.suffix == '.png'

        # Validate file has content
        assert output_path.stat().st_size > 0

    def test_dendrogram_generation_with_truncation(self, sample_embeddings, temp_paths):
        """Test dendrogram generation with truncation mode."""
        output_path = generate_dendrogram(
            sample_embeddings,
            linkage_method='ward',
            output_path=temp_paths['dendrogram'],
            truncate_mode='lastp',
            p=20
        )

        # Should succeed with truncation
        assert output_path.exists()

    def test_dendrogram_validates_embeddings_shape(self):
        """Test dendrogram generation validates embeddings shape."""
        bad_embeddings = np.random.randn(100, 512).astype(np.float32)

        with pytest.raises(ValueError, match="Embeddings must have 768 dimensions"):
            generate_dendrogram(bad_embeddings)

    def test_dendrogram_validates_embeddings_dtype(self):
        """Test dendrogram generation validates embeddings dtype."""
        bad_embeddings = np.random.randn(100, 768).astype(np.float64)

        with pytest.raises(ValueError, match="Embeddings must have dtype float32"):
            generate_dendrogram(bad_embeddings)


class TestClusterAssignmentsExport:
    """Test cluster assignments export (AC-5)."""

    def test_assignments_export_format(self, sample_embeddings, sample_ground_truth, temp_paths):
        """Test cluster assignments export to CSV."""
        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels, _ = clustering.fit_predict(sample_embeddings)

        # Map ground truth to category names
        category_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        ground_truth_str = np.array([category_map[label] for label in sample_ground_truth])

        # Create assignments DataFrame
        assignments_df = pd.DataFrame({
            'document_id': np.arange(len(labels)),
            'cluster_id': labels,
            'ground_truth_category': ground_truth_str,
            'linkage_method': 'ward'
        })

        # Save to CSV
        assignments_df.to_csv(temp_paths['assignments'], index=False)

        # Validate saved file
        loaded_df = pd.read_csv(temp_paths['assignments'])

        # Check schema
        assert list(loaded_df.columns) == [
            'document_id', 'cluster_id', 'ground_truth_category', 'linkage_method'
        ]

        # Check data integrity
        assert len(loaded_df) == 1000
        assert set(loaded_df['cluster_id']) == {0, 1, 2, 3}
        assert set(loaded_df['ground_truth_category']) == {'World', 'Sports', 'Business', 'Sci/Tech'}
        assert all(loaded_df['linkage_method'] == 'ward')


class TestMetricsExport:
    """Test metrics export (AC-4)."""

    def test_metrics_export_format(self, sample_embeddings, sample_ground_truth, temp_paths):
        """Test metrics export to JSON."""
        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels, _ = clustering.fit_predict(sample_embeddings)
        metrics = clustering.calculate_metrics(labels, sample_embeddings, sample_ground_truth)

        # Create full metrics structure
        metrics_full = {
            "timestamp": "2025-11-09T12:00:00",
            "algorithm": "hierarchical",
            "linkage_method": "ward",
            "n_clusters": 4,
            "n_documents": 1000,
            "silhouette_score": metrics['silhouette_score'],
            "davies_bouldin_index": metrics['davies_bouldin_index'],
            "cluster_purity": metrics['cluster_purity'],
            "cluster_sizes": metrics['cluster_sizes'],
            "runtime_seconds": 120.5
        }

        # Save to JSON
        with open(temp_paths['metrics'], 'w') as f:
            json.dump(metrics_full, f, indent=2)

        # Validate saved file
        with open(temp_paths['metrics'], 'r') as f:
            loaded_metrics = json.load(f)

        # Check required fields
        assert loaded_metrics['algorithm'] == 'hierarchical'
        assert loaded_metrics['linkage_method'] == 'ward'
        assert loaded_metrics['n_clusters'] == 4
        assert 'silhouette_score' in loaded_metrics
        assert 'davies_bouldin_index' in loaded_metrics
        assert 'cluster_purity' in loaded_metrics
        assert 'cluster_sizes' in loaded_metrics
        assert sum(loaded_metrics['cluster_sizes']) == 1000


class TestPerformanceMonitoring:
    """Test performance monitoring (AC-6)."""

    def test_runtime_tracking(self, sample_embeddings):
        """Test runtime tracking for clustering operations."""
        clustering = HierarchicalClustering(n_clusters=4, linkage='ward')

        start_time = time.time()
        labels, _ = clustering.fit_predict(sample_embeddings)
        end_time = time.time()

        runtime = end_time - start_time

        # Validate runtime is positive and reasonable
        assert runtime > 0
        assert runtime < 60  # Should complete in less than 60 seconds for 1K samples

    def test_linkage_comparison_runtime_tracking(self, sample_embeddings, sample_ground_truth):
        """Test runtime tracking in linkage comparison."""
        clustering = HierarchicalClustering(n_clusters=4)

        comparison_df = clustering.compare_linkage_methods(
            sample_embeddings,
            sample_ground_truth,
            methods=['ward', 'complete']
        )

        # Validate runtime column exists and has positive values
        assert 'runtime_seconds' in comparison_df.columns
        assert all(comparison_df['runtime_seconds'] > 0)
        assert all(comparison_df['runtime_seconds'] < 60)


class TestReproducibility:
    """Test reproducibility (AC-9)."""

    def test_hierarchical_clustering_deterministic(self, sample_embeddings):
        """Test hierarchical clustering produces identical results across runs."""
        clustering1 = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels1, _ = clustering1.fit_predict(sample_embeddings)

        clustering2 = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels2, _ = clustering2.fit_predict(sample_embeddings)

        # Results should be identical
        assert np.array_equal(labels1, labels2)

    def test_metrics_deterministic(self, sample_embeddings, sample_ground_truth):
        """Test metrics calculation is deterministic."""
        clustering1 = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels1, _ = clustering1.fit_predict(sample_embeddings)
        metrics1 = clustering1.calculate_metrics(labels1, sample_embeddings, sample_ground_truth)

        clustering2 = HierarchicalClustering(n_clusters=4, linkage='ward')
        labels2, _ = clustering2.fit_predict(sample_embeddings)
        metrics2 = clustering2.calculate_metrics(labels2, sample_embeddings, sample_ground_truth)

        # Metrics should be identical
        assert metrics1['silhouette_score'] == metrics2['silhouette_score']
        assert metrics1['davies_bouldin_index'] == metrics2['davies_bouldin_index']
        assert metrics1['cluster_purity'] == metrics2['cluster_purity']
        assert metrics1['cluster_sizes'] == metrics2['cluster_sizes']
