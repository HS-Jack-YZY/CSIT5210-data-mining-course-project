"""
Cluster analysis and semantic labeling for K-Means clustering results.

This module provides functionality to analyze cluster composition, map clusters to
semantic categories, calculate purity metrics, and extract representative documents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Analyze and semantically label clusters based on ground truth categories.

    This class provides comprehensive cluster analysis including:
    - Mapping clusters to dominant semantic categories via majority voting
    - Calculating cluster purity metrics
    - Extracting representative documents closest to centroids
    - Computing category distribution within each cluster
    - Generating human-readable analysis reports

    Example:
        >>> analyzer = ClusterAnalyzer(labels, embeddings, centroids, ground_truth)
        >>> mapping = analyzer.map_clusters_to_categories()
        >>> purity = analyzer.calculate_cluster_purity()
        >>> representatives = analyzer.extract_representative_documents(cluster_id=0, k=10)
    """

    # AG News category mapping
    CATEGORY_NAMES = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }

    def __init__(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        ground_truth: np.ndarray,
        titles: np.ndarray = None,
        descriptions: np.ndarray = None
    ):
        """Initialize cluster analyzer.

        Args:
            labels: Cluster assignments from K-Means (120000,) int32
            embeddings: Document embeddings (120000, 768) float32
            centroids: Cluster centroids from K-Means (4, 768) float32
            ground_truth: AG News ground truth category labels (120000,) int32
            titles: Optional document titles for representative documents
            descriptions: Optional document descriptions for representative documents

        Raises:
            ValueError: If input validation fails (shape mismatch, invalid dtypes, etc.)
        """
        # Validate labels
        if len(labels.shape) != 1:
            raise ValueError(f"Labels must be 1D array, got {len(labels.shape)}D")

        if labels.dtype != np.int32:
            raise ValueError(f"Labels must have dtype int32, got {labels.dtype}")

        # Validate embeddings
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got {len(embeddings.shape)}D")

        if embeddings.shape[1] != 768:
            raise ValueError(f"Embeddings must have 768 dimensions, got {embeddings.shape[1]}")

        if embeddings.dtype != np.float32:
            raise ValueError(f"Embeddings must have dtype float32, got {embeddings.dtype}")

        if np.any(np.isnan(embeddings)):
            raise ValueError("Embeddings contain NaN values")

        if np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain Inf values")

        # Validate centroids
        if len(centroids.shape) != 2:
            raise ValueError(f"Centroids must be 2D array, got {len(centroids.shape)}D")

        if centroids.shape[1] != 768:
            raise ValueError(f"Centroids must have 768 dimensions, got {centroids.shape[1]}")

        if centroids.dtype != np.float32:
            raise ValueError(f"Centroids must have dtype float32, got {centroids.dtype}")

        if np.any(np.isnan(centroids)):
            raise ValueError("Centroids contain NaN values")

        if np.any(np.isinf(centroids)):
            raise ValueError("Centroids contain Inf values")

        # Validate ground truth
        if len(ground_truth.shape) != 1:
            raise ValueError(f"Ground truth must be 1D array, got {len(ground_truth.shape)}D")

        if ground_truth.dtype != np.int32:
            raise ValueError(f"Ground truth must have dtype int32, got {ground_truth.dtype}")

        # Validate shape consistency
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Labels count {len(labels)} != embeddings count {len(embeddings)}"
            )

        if ground_truth.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Ground truth count {len(ground_truth)} != embeddings count {len(embeddings)}"
            )

        # Validate label ranges
        n_clusters = centroids.shape[0]
        if labels.min() < 0 or labels.max() >= n_clusters:
            raise ValueError(
                f"Invalid cluster labels: min={labels.min()}, max={labels.max()}, expected [0, {n_clusters-1}]"
            )

        if ground_truth.min() < 0 or ground_truth.max() > 3:
            raise ValueError(
                f"Invalid ground truth labels: min={ground_truth.min()}, max={ground_truth.max()}, expected [0, 3]"
            )

        # Store validated inputs
        self.labels = labels
        self.embeddings = embeddings
        self.centroids = centroids
        self.ground_truth = ground_truth
        self.n_clusters = n_clusters
        self.titles = titles
        self.descriptions = descriptions

        # Cache for computed results
        self._cluster_mapping = None
        self._purity_scores = None

        logger.info(
            f"✅ Initialized ClusterAnalyzer: {embeddings.shape[0]} documents, {n_clusters} clusters"
        )

    def map_clusters_to_categories(self) -> Dict[int, str]:
        """Map each cluster to its dominant AG News category using majority voting.

        For each cluster, this method:
        1. Extracts all documents in the cluster
        2. Counts documents per AG News category (0-3)
        3. Assigns cluster to category with maximum count

        Returns:
            Mapping of cluster_id → category label
            Example: {0: "Sports", 1: "World", 2: "Business", 3: "Sci/Tech"}

        Example:
            >>> analyzer = ClusterAnalyzer(labels, embeddings, centroids, ground_truth)
            >>> mapping = analyzer.map_clusters_to_categories()
            >>> print(mapping[0])  # e.g., "Sports"
        """
        if self._cluster_mapping is not None:
            return self._cluster_mapping

        logger.info("📊 Mapping clusters to dominant categories...")

        mapping = {}

        for cluster_id in range(self.n_clusters):
            # Get documents in cluster
            cluster_mask = (self.labels == cluster_id)
            cluster_ground_truth = self.ground_truth[cluster_mask]

            # Count documents per category
            category_counts = np.bincount(cluster_ground_truth, minlength=4)

            # Find dominant category (majority voting)
            dominant_category = int(category_counts.argmax())
            dominant_count = category_counts[dominant_category]

            # Map to category name
            category_name = self.CATEGORY_NAMES[dominant_category]
            mapping[cluster_id] = category_name

            purity = dominant_count / len(cluster_ground_truth)
            logger.info(
                f"✅ Cluster {cluster_id}: {category_name} "
                f"({dominant_count}/{len(cluster_ground_truth)} = {purity:.1%})"
            )

        self._cluster_mapping = mapping
        return mapping

    def calculate_cluster_purity(self) -> Dict[str, float]:
        """Calculate cluster purity for each cluster and overall average.

        Cluster purity is the percentage of documents in a cluster that belong to
        the dominant category. Higher purity indicates better cluster-category alignment.

        - Purity >70% indicates good clustering
        - Purity 50-70% indicates fair clustering
        - Purity <50% indicates poor clustering

        Returns:
            Dict with per-cluster purity and average:
            {
                "per_cluster": {0: 0.85, 1: 0.82, 2: 0.84, 3: 0.83},
                "average": 0.835
            }

        Example:
            >>> purity = analyzer.calculate_cluster_purity()
            >>> print(f"Average purity: {purity['average']:.1%}")
            >>> print(f"Cluster 0 purity: {purity['per_cluster'][0]:.1%}")
        """
        if self._purity_scores is not None:
            return self._purity_scores

        logger.info("📊 Calculating cluster purity...")

        purity_scores = {}
        weighted_sum = 0.0
        total_documents = 0

        for cluster_id in range(self.n_clusters):
            # Get documents in cluster
            cluster_mask = (self.labels == cluster_id)
            cluster_ground_truth = self.ground_truth[cluster_mask]
            cluster_size = len(cluster_ground_truth)

            if cluster_size == 0:
                logger.warning(f"⚠️ Cluster {cluster_id} is empty")
                purity_scores[cluster_id] = 0.0
                continue

            # Count documents per category
            category_counts = np.bincount(cluster_ground_truth, minlength=4)
            dominant_count = category_counts.max()

            # Calculate purity
            purity = dominant_count / cluster_size
            purity_scores[cluster_id] = float(purity)

            # Accumulate for average
            weighted_sum += purity * cluster_size
            total_documents += cluster_size

            # Log warning if below threshold
            if purity < 0.70:
                logger.warning(
                    f"⚠️ Cluster {cluster_id} purity {purity:.1%} below target 70%"
                )
            else:
                logger.info(f"✅ Cluster {cluster_id} purity: {purity:.1%}")

        # Calculate overall average purity
        overall_purity = weighted_sum / total_documents if total_documents > 0 else 0.0

        result = {
            "per_cluster": purity_scores,
            "average": float(overall_purity)
        }

        logger.info(f"✅ Average cluster purity: {overall_purity:.1%}")

        self._purity_scores = result
        return result

    def extract_representative_documents(
        self,
        cluster_id: int,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract k most representative documents for a cluster.

        Representative documents are those closest to the cluster centroid,
        measured by Euclidean distance. The closest document represents the
        "most typical" example of the cluster's semantic theme.

        Args:
            cluster_id: Cluster to analyze (0-3)
            k: Number of representatives to extract (default: 10)

        Returns:
            List of dicts with document metadata, sorted by distance (closest first):
            [
                {
                    "document_id": int,
                    "title": str (if available),
                    "description": str (if available),
                    "category": str,
                    "distance": float
                },
                ...
            ]

        Raises:
            ValueError: If cluster_id is invalid

        Example:
            >>> representatives = analyzer.extract_representative_documents(cluster_id=0, k=10)
            >>> closest = representatives[0]
            >>> print(f"Most typical document: {closest['title']}")
            >>> print(f"Distance to centroid: {closest['distance']:.4f}")
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(
                f"Invalid cluster_id {cluster_id}, expected [0, {self.n_clusters-1}]"
            )

        logger.info(f"📊 Extracting {k} representative documents for cluster {cluster_id}...")

        # Get documents in cluster
        cluster_mask = (self.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = self.embeddings[cluster_mask]
        cluster_ground_truth = self.ground_truth[cluster_mask]

        if len(cluster_indices) == 0:
            logger.warning(f"⚠️ Cluster {cluster_id} is empty")
            return []

        # Calculate distances to centroid
        centroid = self.centroids[cluster_id]
        distances = euclidean_distances([centroid], cluster_embeddings)[0]

        # Get top k closest documents
        top_k = min(k, len(distances))
        top_k_indices = distances.argsort()[:top_k]

        # Build representative document list
        representatives = []
        for i in top_k_indices:
            global_idx = cluster_indices[i]
            category_id = int(cluster_ground_truth[i])

            doc_info = {
                "document_id": int(global_idx),
                "category": self.CATEGORY_NAMES[category_id],
                "distance": float(distances[i])
            }

            # Add title if available
            if self.titles is not None:
                doc_info["title"] = str(self.titles[global_idx])

            # Add description if available
            if self.descriptions is not None:
                doc_info["description"] = str(self.descriptions[global_idx])

            representatives.append(doc_info)

        # Verify sorted order
        distances_list = [doc["distance"] for doc in representatives]
        assert all(
            distances_list[i] <= distances_list[i+1]
            for i in range(len(distances_list)-1)
        ), "Representatives not sorted by distance"

        logger.info(f"✅ Extracted {len(representatives)} representative documents")

        return representatives

    def get_category_distribution(self, cluster_id: int) -> Dict[str, float]:
        """Get category distribution for a cluster.

        Computes the percentage of documents from each AG News category within
        the specified cluster.

        Args:
            cluster_id: Cluster to analyze (0-3)

        Returns:
            Percentage distribution across categories (sums to ~1.0):
            {"Sports": 0.85, "World": 0.10, "Business": 0.03, "Sci/Tech": 0.02}

        Raises:
            ValueError: If cluster_id is invalid

        Example:
            >>> distribution = analyzer.get_category_distribution(cluster_id=0)
            >>> for category, percentage in distribution.items():
            >>>     print(f"{category}: {percentage:.1%}")
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(
                f"Invalid cluster_id {cluster_id}, expected [0, {self.n_clusters-1}]"
            )

        # Get documents in cluster
        cluster_mask = (self.labels == cluster_id)
        cluster_ground_truth = self.ground_truth[cluster_mask]
        cluster_size = len(cluster_ground_truth)

        if cluster_size == 0:
            logger.warning(f"⚠️ Cluster {cluster_id} is empty")
            return {name: 0.0 for name in self.CATEGORY_NAMES.values()}

        # Count documents per category
        category_counts = np.bincount(cluster_ground_truth, minlength=4)

        # Convert to percentages
        distribution = {}
        for category_id, category_name in self.CATEGORY_NAMES.items():
            percentage = float(category_counts[category_id]) / cluster_size
            distribution[category_name] = percentage

        # Validate sum ~= 1.0
        total = sum(distribution.values())
        assert abs(total - 1.0) < 1e-6, f"Distribution sum {total} != 1.0"

        return distribution

    def generate_analysis_report(
        self,
        output_path: Path,
        dataset_name: str = "AG News",
        n_documents: int = 120000,
        clustering_params: Dict[str, Any] = None
    ) -> Path:
        """Generate comprehensive cluster analysis text report.

        The report includes:
        - Cluster metadata (ID, dominant category, purity, size)
        - Category distribution breakdown for each cluster
        - Top 10 representative document titles (if available)
        - Overall statistics (average purity, total documents, clustering quality)

        Args:
            output_path: Path to save report
            dataset_name: Name of dataset (default: "AG News")
            n_documents: Total number of documents (default: 120000)
            clustering_params: Optional clustering parameters to include in report

        Returns:
            Path to saved report file

        Raises:
            ValueError: If analysis not yet run (call map_clusters_to_categories first)

        Example:
            >>> report_path = analyzer.generate_analysis_report(
            >>>     output_path=Path("results/cluster_analysis.txt"),
            >>>     clustering_params={"K": 4, "random_state": 42}
            >>> )
        """
        # Ensure analysis has been run
        if self._cluster_mapping is None:
            self.map_clusters_to_categories()

        if self._purity_scores is None:
            self.calculate_cluster_purity()

        logger.info("📊 Generating cluster analysis report...")

        # Build report content
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("Cluster Analysis Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Dataset: {dataset_name} ({n_documents:,} training documents)")

        if clustering_params:
            params_str = ", ".join(f"{k}={v}" for k, v in clustering_params.items())
            lines.append(f"Clustering: K-Means ({params_str})")

        lines.append("=" * 80)
        lines.append("")

        # Per-cluster analysis
        for cluster_id in range(self.n_clusters):
            # Get cluster metadata
            category_label = self._cluster_mapping[cluster_id]
            purity = self._purity_scores['per_cluster'][cluster_id]

            cluster_mask = (self.labels == cluster_id)
            cluster_size = cluster_mask.sum()

            # Cluster header
            lines.append(f"CLUSTER {cluster_id}: {category_label.upper()} (Purity: {purity:.1%})")
            lines.append("-" * 80)
            lines.append(f"Size: {cluster_size:,} documents")
            lines.append("")

            # Category distribution
            distribution = self.get_category_distribution(cluster_id)
            lines.append("Category Distribution:")
            for category, percentage in sorted(distribution.items(), key=lambda x: -x[1]):
                count = int(percentage * cluster_size)
                lines.append(f"  - {category}: {percentage:.1%} ({count:,} documents)")
            lines.append("")

            # Representative documents
            representatives = self.extract_representative_documents(cluster_id, k=10)
            if representatives:
                lines.append("Top 10 Representative Documents:")
                for i, doc in enumerate(representatives, 1):
                    distance = doc['distance']
                    if 'title' in doc:
                        title = doc['title'][:70] + "..." if len(doc['title']) > 70 else doc['title']
                        lines.append(f"  {i}. [Distance: {distance:.4f}] {title}")
                    else:
                        lines.append(f"  {i}. [Distance: {distance:.4f}] Document {doc['document_id']}")
            lines.append("")
            lines.append("")

        # Overall statistics
        avg_purity = self._purity_scores['average']

        # Determine quality assessment
        if avg_purity >= 0.70:
            quality = "GOOD"
        elif avg_purity >= 0.50:
            quality = "FAIR"
        else:
            quality = "POOR"

        lines.append("OVERALL STATISTICS:")
        lines.append("-" * 80)
        lines.append(f"- Average Purity: {avg_purity:.1%}")
        lines.append(f"- Total Documents: {n_documents:,}")
        lines.append(f"- Number of Clusters: {self.n_clusters}")
        lines.append(f"- Clustering Quality: {quality} (purity {'>' if avg_purity >= 0.70 else '<'}70%)")
        lines.append("=" * 80)

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_content = "\n".join(lines)
        output_path.write_text(report_content)

        # Validate file exists and has content
        assert output_path.exists(), f"Report not saved: {output_path}"
        assert output_path.stat().st_size > 0, f"Report is empty: {output_path}"

        logger.info(f"✅ Report saved: {output_path}")

        return output_path

    def export_cluster_labels_json(
        self,
        output_path: Path,
        n_documents: int = 120000
    ) -> Path:
        """Export cluster labels and metadata to JSON.

        Args:
            output_path: Path to save JSON file
            n_documents: Total number of documents (default: 120000)

        Returns:
            Path to saved JSON file

        Example:
            >>> json_path = analyzer.export_cluster_labels_json(
            >>>     output_path=Path("results/cluster_labels.json")
            >>> )
        """
        # Ensure analysis has been run
        if self._cluster_mapping is None:
            self.map_clusters_to_categories()

        if self._purity_scores is None:
            self.calculate_cluster_purity()

        logger.info("📊 Exporting cluster labels to JSON...")

        # Build JSON structure
        clusters_data = {}

        for cluster_id in range(self.n_clusters):
            category_label = self._cluster_mapping[cluster_id]
            purity = self._purity_scores['per_cluster'][cluster_id]

            cluster_mask = (self.labels == cluster_id)
            cluster_size = int(cluster_mask.sum())

            distribution = self.get_category_distribution(cluster_id)

            clusters_data[str(cluster_id)] = {
                "label": category_label,
                "purity": float(purity),
                "size": cluster_size,
                "dominant_category": category_label,
                "distribution": {k: float(v) for k, v in distribution.items()}
            }

        json_data = {
            "timestamp": datetime.now().isoformat(),
            "n_clusters": self.n_clusters,
            "n_documents": n_documents,
            "average_purity": float(self._purity_scores['average']),
            "clusters": clusters_data
        }

        # Save JSON
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Validate file exists
        assert output_path.exists(), f"JSON not saved: {output_path}"

        logger.info(f"✅ Cluster labels saved: {output_path}")

        return output_path
