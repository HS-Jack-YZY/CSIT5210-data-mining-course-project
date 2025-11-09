"""
Clustering quality metrics for evaluating K-Means clustering results.
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


class ClusteringMetrics:
    """Comprehensive cluster quality evaluation metrics."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        ground_truth: np.ndarray
    ):
        """Initialize cluster quality evaluation."""
        # Validate embeddings
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got {len(embeddings.shape)}D"
            )

        if embeddings.shape[1] != 768:
            raise ValueError(
                f"Embeddings must have 768 dimensions"
            )

        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Embeddings must have dtype float32, got {embeddings.dtype}"
            )

        if np.any(np.isnan(embeddings)):
            raise ValueError("Embeddings contain NaN values")

        if np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain Inf values")

        # Validate labels
        if len(labels.shape) != 1:
            raise ValueError(
                f"Labels must be 1D array"
            )

        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Labels count mismatch"
            )

        if labels.dtype != np.int32:
            raise ValueError(
                f"Labels must have dtype int32"
            )

        # Validate centroids
        if len(centroids.shape) != 2:
            raise ValueError(
                f"Centroids must be 2D array"
            )

        if centroids.shape[1] != 768:
            raise ValueError(
                f"Centroids must have 768 dimensions"
            )

        if centroids.dtype != np.float32:
            raise ValueError(
                f"Centroids must have dtype float32"
            )

        if np.any(np.isnan(centroids)):
            raise ValueError("Centroids contain NaN values")

        if np.any(np.isinf(centroids)):
            raise ValueError("Centroids contain Inf values")

        # Validate ground truth
        if len(ground_truth.shape) != 1:
            raise ValueError(
                f"Ground truth must be 1D array"
            )

        if ground_truth.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Ground truth count mismatch"
            )

        if ground_truth.dtype != np.int32:
            raise ValueError(
                f"Ground truth must have dtype int32"
            )

        # Validate label ranges
        n_clusters = centroids.shape[0]

        if labels.min() < 0 or labels.max() >= n_clusters:
            raise ValueError(
                f"Invalid cluster labels"
            )

        if ground_truth.min() < 0 or ground_truth.max() > 3:
            raise ValueError(
                f"Invalid ground truth labels"
            )

        # Store validated inputs
        self.embeddings = embeddings
        self.labels = labels
        self.centroids = centroids
        self.ground_truth = ground_truth
        self.n_clusters = n_clusters

        logger.info(
            f"✅ Initialized ClusteringMetrics: {embeddings.shape[0]} documents, {n_clusters} clusters"
        )

    def calculate_silhouette_score(self) -> float:
        """Calculate Silhouette Score for cluster quality."""
        logger.info("📊 Calculating Silhouette Score...")

        score = silhouette_score(self.embeddings, self.labels, metric='euclidean')

        assert -1.0 <= score <= 1.0

        logger.info(f"✅ Silhouette Score: {score:.4f}")

        return float(score)

    def calculate_davies_bouldin_index(self) -> float:
        """Calculate Davies-Bouldin Index for cluster quality."""
        logger.info("📊 Computing Davies-Bouldin Index...")

        index = davies_bouldin_score(self.embeddings, self.labels)

        assert index > 0

        logger.info(f"✅ Davies-Bouldin Index: {index:.4f} (lower is better)")

        return float(index)

    def calculate_intra_cluster_distance(self) -> Dict[str, float]:
        """Calculate intra-cluster distance for each cluster."""
        logger.info("📊 Calculating intra-cluster distances...")

        intra_distances = {}
        weighted_sum = 0.0
        total_documents = 0

        for cluster_id in range(self.n_clusters):
            cluster_mask = (self.labels == cluster_id)
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_size = cluster_embeddings.shape[0]

            if cluster_size == 0:
                logger.warning(f"⚠️ Cluster {cluster_id} is empty")
                intra_distances[f'cluster_{cluster_id}'] = 0.0
                continue

            centroid = self.centroids[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            mean_distance = distances.mean()
            intra_distances[f'cluster_{cluster_id}'] = float(mean_distance)

            weighted_sum += mean_distance * cluster_size
            total_documents += cluster_size

        overall_distance = weighted_sum / total_documents if total_documents > 0 else 0.0
        intra_distances['overall'] = float(overall_distance)

        logger.info(f"✅ Intra-cluster distances computed")

        return intra_distances

    def calculate_inter_cluster_distance(self) -> Dict[str, Any]:
        """Calculate inter-cluster distance between cluster centroids."""
        logger.info("📊 Calculating inter-cluster distances...")

        centroid_distances = euclidean_distances(self.centroids)

        upper_triangle_indices = np.triu_indices(self.n_clusters, k=1)
        pairwise_distances = centroid_distances[upper_triangle_indices]

        min_distance = float(pairwise_distances.min())
        max_distance = float(pairwise_distances.max())
        mean_distance = float(pairwise_distances.mean())

        result = {
            'min': min_distance,
            'max': max_distance,
            'mean': mean_distance,
            'pairwise': pairwise_distances.tolist()
        }

        logger.info(f"✅ Inter-cluster distances computed")

        return result

    def calculate_cluster_purity(self) -> Dict[str, float]:
        """Calculate cluster purity against ground truth AG News labels."""
        logger.info("📊 Evaluating cluster purity...")

        purity_scores = {}
        weighted_sum = 0.0
        total_documents = 0

        for cluster_id in range(self.n_clusters):
            cluster_mask = (self.labels == cluster_id)
            cluster_ground_truth = self.ground_truth[cluster_mask]
            cluster_size = len(cluster_ground_truth)

            if cluster_size == 0:
                logger.warning(f"⚠️ Cluster {cluster_id} is empty")
                purity_scores[f'cluster_{cluster_id}'] = 0.0
                continue

            category_counts = np.bincount(cluster_ground_truth, minlength=4)
            dominant_category = category_counts.argmax()
            dominant_count = category_counts[dominant_category]

            purity = dominant_count / cluster_size
            purity_scores[f'cluster_{cluster_id}'] = float(purity)

            weighted_sum += purity * cluster_size
            total_documents += cluster_size

        overall_purity = weighted_sum / total_documents if total_documents > 0 else 0.0
        purity_scores['overall'] = float(overall_purity)

        logger.info(f"✅ Cluster purity: {overall_purity:.1%}")

        return purity_scores

    def generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix comparing cluster assignments with ground truth."""
        logger.info("📊 Generating confusion matrix...")

        cm = confusion_matrix(self.ground_truth, self.labels)

        assert cm.shape == (4, 4)
        assert cm.sum() == len(self.embeddings)

        logger.info("✅ Confusion matrix generated")

        return cm

    def validate_cluster_balance(self) -> Tuple[bool, Dict[str, int]]:
        """Validate cluster size distribution and check for imbalance."""
        logger.info("📊 Validating cluster balance...")

        cluster_sizes_array = np.bincount(self.labels, minlength=self.n_clusters)
        cluster_sizes = {i: int(cluster_sizes_array[i]) for i in range(self.n_clusters)}

        total_documents = len(self.embeddings)
        min_threshold = 0.1 * total_documents
        max_threshold = 0.5 * total_documents

        is_balanced = True

        for cluster_id, size in cluster_sizes.items():
            if size < min_threshold or size > max_threshold:
                is_balanced = False
                logger.warning(f"⚠️ Cluster {cluster_id} imbalanced: {size} documents ({size/total_documents:.1%})")

        if is_balanced:
            logger.info("✅ Cluster balance validated")
        else:
            logger.warning("⚠️ Cluster imbalance detected")

        return is_balanced, cluster_sizes

    def evaluate_all(self) -> Dict[str, Any]:
        """Compute all cluster quality metrics."""
        logger.info("📊 Running comprehensive cluster quality evaluation...")

        silhouette = self.calculate_silhouette_score()
        davies_bouldin = self.calculate_davies_bouldin_index()
        intra_distance = self.calculate_intra_cluster_distance()
        inter_distance = self.calculate_inter_cluster_distance()
        purity = self.calculate_cluster_purity()
        is_balanced, cluster_sizes = self.validate_cluster_balance()

        results = {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'intra_cluster_distance': intra_distance,
            'inter_cluster_distance': inter_distance,
            'cluster_purity': purity,
            'cluster_sizes': list(cluster_sizes.values()),
            'is_balanced': is_balanced
        }

        logger.info("✅ Cluster quality evaluation complete")

        return results
