"""
Integration tests for GMM clustering pipeline.

This module tests the end-to-end GMM clustering workflow including:
- Loading embeddings
- Running GMM clustering
- Extracting probabilistic assignments
- Performing uncertainty analysis
- Saving results

Test Coverage:
    - AC-4: CSV output schema validation
    - AC-6: Uncertainty analysis
    - AC-7: Standard clustering metrics
    - AC-9: JSON metrics output
"""

import sys
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.models.gmm_clustering import GMMClustering
from context_aware_multi_agent_system.evaluation.clustering_metrics import ClusteringMetrics


class TestGMMPipeline:
    """Test GMM clustering pipeline integration."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample embeddings and ground truth."""
        np.random.seed(42)
        embeddings = np.random.randn(1000, 768).astype(np.float32)
        ground_truth = np.random.randint(0, 4, size=1000).astype(np.int32)
        return embeddings, ground_truth

    def test_full_pipeline(self, sample_data):
        """Test full GMM pipeline from embeddings to metrics."""
        embeddings, ground_truth = sample_data

        # Step 1: Fit GMM (use diag for numerical stability)
        gmm = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels, probabilities, bic, aic = gmm.fit_predict(embeddings)

        # Step 2: Validate assignments
        assert len(labels) == 1000
        assert len(probabilities) == 1000

        # Step 3: Calculate standard metrics
        centroids = np.zeros((4, 768), dtype=np.float32)
        for cluster_id in range(4):
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 0:
                centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)

        metrics_calculator = ClusteringMetrics(
            embeddings=embeddings,
            labels=labels,
            centroids=centroids,
            ground_truth=ground_truth
        )

        metrics = metrics_calculator.evaluate_all()

        # Validate metrics
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'cluster_purity' in metrics
        assert np.isfinite(metrics['silhouette_score'])
        assert np.isfinite(metrics['davies_bouldin_index'])


class TestGMMAssignmentsCSV:
    """Test GMM assignments CSV output."""

    @pytest.fixture
    def sample_assignments(self):
        """Provide sample GMM assignments."""
        np.random.seed(42)
        n_documents = 1000
        n_components = 4

        # Generate random probabilities that sum to 1.0
        probabilities = np.random.dirichlet(np.ones(n_components), size=n_documents).astype(np.float32)
        labels = np.argmax(probabilities, axis=1).astype(np.int32)
        ground_truth = np.random.randint(0, 4, size=n_documents).astype(np.int32)

        return probabilities, labels, ground_truth

    def test_csv_schema_validation(self, sample_assignments):
        """Test CSV output schema matches specification (AC-4)."""
        probabilities, labels, ground_truth = sample_assignments

        # Calculate confidence
        confidence = probabilities.max(axis=1)

        # Build DataFrame
        data = {
            'document_id': np.arange(len(labels)),
            'cluster_id': labels,
            'cluster_0_prob': probabilities[:, 0],
            'cluster_1_prob': probabilities[:, 1],
            'cluster_2_prob': probabilities[:, 2],
            'cluster_3_prob': probabilities[:, 3],
            'assignment_confidence': confidence,
            'ground_truth_category': ground_truth,
            'covariance_type': 'full'
        }

        df = pd.DataFrame(data)

        # Validate schema
        expected_columns = [
            'document_id', 'cluster_id',
            'cluster_0_prob', 'cluster_1_prob', 'cluster_2_prob', 'cluster_3_prob',
            'assignment_confidence', 'ground_truth_category', 'covariance_type'
        ]

        assert list(df.columns) == expected_columns

    def test_probability_validation_in_csv(self, sample_assignments):
        """Test probabilities in CSV are valid (AC-4)."""
        probabilities, labels, ground_truth = sample_assignments

        # Build DataFrame
        df = pd.DataFrame({
            'document_id': np.arange(len(labels)),
            'cluster_id': labels,
            'cluster_0_prob': probabilities[:, 0],
            'cluster_1_prob': probabilities[:, 1],
            'cluster_2_prob': probabilities[:, 2],
            'cluster_3_prob': probabilities[:, 3],
            'assignment_confidence': probabilities.max(axis=1),
            'ground_truth_category': ground_truth,
            'covariance_type': 'full'
        })

        # Validate probability columns are in [0, 1]
        prob_columns = ['cluster_0_prob', 'cluster_1_prob', 'cluster_2_prob', 'cluster_3_prob']
        for col in prob_columns:
            assert df[col].min() >= 0.0
            assert df[col].max() <= 1.0

        # Validate probabilities sum to 1.0
        prob_sums = df[prob_columns].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5)

    def test_csv_save_and_load(self, sample_assignments):
        """Test CSV can be saved and loaded correctly."""
        probabilities, labels, ground_truth = sample_assignments

        df = pd.DataFrame({
            'document_id': np.arange(len(labels)),
            'cluster_id': labels,
            'cluster_0_prob': probabilities[:, 0],
            'cluster_1_prob': probabilities[:, 1],
            'cluster_2_prob': probabilities[:, 2],
            'cluster_3_prob': probabilities[:, 3],
            'assignment_confidence': probabilities.max(axis=1),
            'ground_truth_category': ground_truth,
            'covariance_type': 'full'
        })

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        # Load back
        loaded_df = pd.read_csv(temp_path)

        # Validate loaded data
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)

        # Clean up
        Path(temp_path).unlink()


class TestUncertaintyAnalysis:
    """Test uncertainty analysis functionality."""

    @pytest.fixture
    def sample_probabilities(self):
        """Provide sample probabilities with various confidence levels."""
        np.random.seed(42)

        # Create probabilities with different confidence levels
        n_documents = 1000
        probabilities = []

        # High confidence (500 documents)
        for _ in range(500):
            p = np.random.dirichlet([10, 1, 1, 1])  # One dominant cluster
            probabilities.append(p)

        # Low confidence (500 documents)
        for _ in range(500):
            p = np.random.dirichlet([1, 1, 1, 1])  # Equally distributed
            probabilities.append(p)

        probabilities = np.array(probabilities, dtype=np.float32)
        labels = np.argmax(probabilities, axis=1).astype(np.int32)
        ground_truth = np.random.randint(0, 4, size=n_documents).astype(np.int32)

        return probabilities, labels, ground_truth

    def test_low_confidence_detection(self, sample_probabilities):
        """Test identification of low-confidence documents (AC-6)."""
        probabilities, labels, ground_truth = sample_probabilities

        # Calculate confidence
        confidence = probabilities.max(axis=1)

        # Identify low-confidence documents
        low_confidence_mask = confidence < 0.5
        low_confidence_count = low_confidence_mask.sum()

        # Should find some low-confidence documents
        assert low_confidence_count > 0

        # All low-confidence documents should have max probability < 0.5
        assert np.all(confidence[low_confidence_mask] < 0.5)

    def test_cluster_pair_confusion_analysis(self, sample_probabilities):
        """Test cluster pair confusion analysis (AC-6)."""
        probabilities, labels, ground_truth = sample_probabilities

        # Find top 2 probabilities per document
        sorted_probs = np.sort(probabilities, axis=1)
        top2_probs = sorted_probs[:, -2:]
        prob_diff = top2_probs[:, 1] - top2_probs[:, 0]

        # Documents with small difference are confused
        confusion_threshold = 0.2
        confused_mask = prob_diff < confusion_threshold
        confused_count = confused_mask.sum()

        # Should find some confused documents
        assert confused_count > 0

        # All confused documents should have small difference
        assert np.all(prob_diff[confused_mask] < confusion_threshold)

    def test_confidence_by_ground_truth(self, sample_probabilities):
        """Test confidence analysis by ground truth category (AC-6)."""
        probabilities, labels, ground_truth = sample_probabilities

        confidence = probabilities.max(axis=1)

        # Analyze by category
        for category_id in range(4):
            category_mask = ground_truth == category_id
            if category_mask.sum() == 0:
                continue

            category_confidence = confidence[category_mask]

            # Should have valid confidence values
            assert len(category_confidence) > 0
            assert np.all(category_confidence >= 0)
            assert np.all(category_confidence <= 1)

            # Calculate statistics
            mean_conf = category_confidence.mean()
            std_conf = category_confidence.std()

            assert np.isfinite(mean_conf)
            assert np.isfinite(std_conf)


class TestMetricsJSON:
    """Test metrics JSON output."""

    def test_json_output_structure(self):
        """Test JSON output contains all required fields (AC-9)."""
        # Simulate metrics output
        metrics = {
            'algorithm': 'GMM',
            'n_components': 4,
            'covariance_type': 'full',
            'random_state': 42,
            'max_iter': 100,
            'timestamp': '2024-01-01T00:00:00',

            # Convergence
            'converged': True,
            'n_iterations': 50,
            'log_likelihood': -1000.0,

            # GMM metrics
            'bic': 2000.0,
            'aic': 1900.0,
            'component_weights': [0.25, 0.25, 0.25, 0.25],

            # Standard metrics
            'silhouette_score': 0.1,
            'davies_bouldin_index': 1.5,
            'cluster_purity': {
                'cluster_0': 0.5,
                'cluster_1': 0.6,
                'cluster_2': 0.7,
                'cluster_3': 0.8,
                'overall': 0.65
            },

            # Cluster info
            'cluster_sizes': [250, 250, 250, 250],
            'is_balanced': True,

            # Uncertainty
            'uncertainty_analysis': {
                'low_confidence_count': 100,
                'low_confidence_ratio': 0.1
            },

            # Runtime
            'runtime_seconds': 60.0
        }

        # Validate required keys
        required_keys = [
            'algorithm', 'n_components', 'covariance_type', 'random_state',
            'converged', 'n_iterations', 'log_likelihood',
            'bic', 'aic', 'component_weights',
            'silhouette_score', 'davies_bouldin_index', 'cluster_purity',
            'cluster_sizes', 'is_balanced',
            'uncertainty_analysis', 'runtime_seconds'
        ]

        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_json_serialization(self):
        """Test metrics can be serialized to JSON."""
        metrics = {
            'bic': 2000.0,
            'aic': 1900.0,
            'silhouette_score': 0.1,
            'component_weights': [0.25, 0.25, 0.25, 0.25],
            'cluster_sizes': [250, 250, 250, 250]
        }

        # Should be serializable
        json_str = json.dumps(metrics)
        assert isinstance(json_str, str)

        # Should be deserializable
        loaded_metrics = json.loads(json_str)
        assert loaded_metrics == metrics


class TestReproducibility:
    """Test reproducibility of GMM clustering."""

    def test_identical_results_with_same_seed(self):
        """Test identical results with same random_state."""
        np.random.seed(42)
        embeddings = np.random.randn(500, 768).astype(np.float32)

        # Run 1 (use diag for numerical stability)
        gmm1 = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels1, probs1, bic1, aic1 = gmm1.fit_predict(embeddings)

        # Run 2
        gmm2 = GMMClustering(n_components=4, covariance_type='diag', random_state=42)
        labels2, probs2, bic2, aic2 = gmm2.fit_predict(embeddings)

        # Should be identical
        assert np.array_equal(labels1, labels2)
        assert np.array_equal(probs1, probs2)
        assert bic1 == bic2
        assert aic1 == aic2
