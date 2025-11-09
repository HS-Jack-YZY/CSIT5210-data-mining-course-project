"""
Comprehensive algorithm comparison module for clustering evaluation.

This module provides the AlgorithmComparison class for comparing multiple clustering
algorithms (K-Means, DBSCAN, Hierarchical, GMM) across standardized metrics.

Classes:
    AlgorithmComparison: Cross-algorithm analysis and comparison
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class AlgorithmComparison:
    """
    Comprehensive comparison across multiple clustering algorithms.

    Aggregates results from different clustering algorithms, normalizes metrics,
    generates comparison matrices, and performs ground truth alignment analysis.

    Attributes:
        algorithms: Dictionary of algorithm results {algorithm_name: metrics_dict}
        embeddings: Document embeddings for visualization
        ground_truth: Ground truth labels for evaluation

    Usage:
        >>> comparison = AlgorithmComparison()
        >>> comparison.add_algorithm("K-Means", kmeans_results)
        >>> comparison.add_algorithm("DBSCAN", dbscan_results)
        >>> matrix = comparison.create_comparison_matrix()
        >>> comparison.export_to_json("results/comparison.json")
    """

    def __init__(self):
        """Initialize AlgorithmComparison with empty results."""
        self.algorithms: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.ground_truth: Optional[np.ndarray] = None
        self.comparison_matrix: Optional[pd.DataFrame] = None

        logger.info("✅ Initialized AlgorithmComparison")

    def add_algorithm(
        self,
        name: str,
        metrics: Dict[str, Any],
        labels: Optional[np.ndarray] = None,
        runtime: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add algorithm results to comparison.

        Args:
            name: Algorithm name (e.g., "K-Means", "DBSCAN")
            metrics: Dictionary of evaluation metrics
            labels: Cluster labels (n_samples,)
            runtime: Algorithm runtime in seconds
            parameters: Algorithm parameters used

        Raises:
            ValueError: If required metrics are missing
        """
        # Validate required metrics
        required_metrics = ["silhouette_score", "davies_bouldin_index", "cluster_purity"]
        for metric in required_metrics:
            if metric not in metrics:
                # Handle nested purity structure
                if metric == "cluster_purity":
                    if "cluster_purity" in metrics and "overall" in metrics["cluster_purity"]:
                        continue
                    raise ValueError(
                        f"Missing required metric '{metric}' for algorithm '{name}'"
                    )
                else:
                    raise ValueError(
                        f"Missing required metric '{metric}' for algorithm '{name}'"
                    )

        # Store algorithm results
        self.algorithms[name] = {
            "metrics": metrics,
            "labels": labels,
            "runtime": runtime,
            "parameters": parameters or {}
        }

        logger.info(f"✅ Added algorithm: {name}")

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Set document embeddings for visualization.

        Args:
            embeddings: Document embeddings (n_samples, n_features)

        Raises:
            ValueError: If embeddings shape is invalid
        """
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got {len(embeddings.shape)}D"
            )

        self.embeddings = embeddings
        logger.info(f"✅ Set embeddings: {embeddings.shape}")

    def set_ground_truth(self, ground_truth: np.ndarray) -> None:
        """
        Set ground truth labels for evaluation.

        Args:
            ground_truth: Ground truth category labels (n_samples,)

        Raises:
            ValueError: If ground truth shape is invalid
        """
        if len(ground_truth.shape) != 1:
            raise ValueError(
                f"Ground truth must be 1D array, got {len(ground_truth.shape)}D"
            )

        self.ground_truth = ground_truth
        logger.info(f"✅ Set ground truth: {len(ground_truth)} samples")

    def create_comparison_matrix(self) -> pd.DataFrame:
        """
        Create unified comparison matrix with all algorithms and metrics.

        Returns:
            DataFrame with algorithms as rows, metrics as columns

        Example output:
            | algorithm    | silhouette | davies_bouldin | purity | n_clusters | runtime |
            |--------------|-----------|----------------|--------|------------|---------|
            | K-Means      | 0.0008    | 26.21          | 0.253  | 4          | 45.2    |
            | DBSCAN       | 0.0012    | 24.15          | 0.261  | 5          | 320.5   |
            | Hierarchical | 0.0010    | 25.43          | 0.255  | 4          | 420.1   |
            | GMM          | 0.0009    | 25.89          | 0.257  | 4          | 180.3   |
        """
        if not self.algorithms:
            raise ValueError("No algorithms added to comparison")

        logger.info("📊 Creating comparison matrix...")

        comparison_data = []

        for name, data in self.algorithms.items():
            metrics = data["metrics"]
            labels = data["labels"]
            runtime = data["runtime"]
            parameters = data["parameters"]

            # Extract common metrics
            silhouette = metrics.get("silhouette_score", np.nan)
            davies_bouldin = metrics.get("davies_bouldin_index", np.nan)

            # Handle nested cluster_purity structure
            if isinstance(metrics.get("cluster_purity"), dict):
                purity = metrics["cluster_purity"].get("overall", np.nan)
            else:
                purity = metrics.get("cluster_purity", np.nan)

            # Count clusters
            n_clusters = len(np.unique(labels[labels >= 0])) if labels is not None else np.nan

            # Count noise points (DBSCAN specific)
            n_noise = np.sum(labels == -1) if labels is not None else 0

            # Extract convergence iterations (K-Means, GMM specific)
            convergence_iterations = metrics.get("n_iter") or parameters.get("convergence_iterations")

            row = {
                "algorithm": name,
                "silhouette_score": silhouette,
                "davies_bouldin_index": davies_bouldin,
                "cluster_purity": purity,
                "n_clusters_discovered": n_clusters,
                "n_noise_points": n_noise,
                "runtime_seconds": runtime or np.nan,
                "convergence_iterations": convergence_iterations,
                "parameters": str(parameters) if parameters else ""
            }

            comparison_data.append(row)

        # Create DataFrame
        self.comparison_matrix = pd.DataFrame(comparison_data)

        # Round numeric columns for readability
        numeric_columns = [
            "silhouette_score",
            "davies_bouldin_index",
            "cluster_purity",
            "runtime_seconds"
        ]

        for col in numeric_columns:
            if col in self.comparison_matrix.columns:
                self.comparison_matrix[col] = self.comparison_matrix[col].round(6)

        logger.info(f"✅ Comparison matrix created: {len(self.comparison_matrix)} algorithms")

        return self.comparison_matrix

    def generate_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """
        Generate confusion matrix for each algorithm vs ground truth.

        Returns:
            Dictionary mapping {algorithm_name: confusion_matrix}

        Raises:
            ValueError: If ground truth not set or no algorithm labels available
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth must be set before generating confusion matrices")

        logger.info("📊 Generating confusion matrices...")

        confusion_matrices = {}

        for name, data in self.algorithms.items():
            labels = data["labels"]

            if labels is None:
                logger.warning(f"⚠️ Skipping {name}: no labels available")
                continue

            # Filter out noise points for DBSCAN
            valid_mask = labels >= 0
            valid_labels = labels[valid_mask]
            valid_ground_truth = self.ground_truth[valid_mask]

            # Generate confusion matrix
            cm = confusion_matrix(valid_ground_truth, valid_labels)

            confusion_matrices[name] = cm

            logger.info(f"✅ Generated confusion matrix for {name}: {cm.shape}")

        return confusion_matrices

    def identify_best_algorithms(self) -> Dict[str, str]:
        """
        Identify best algorithm for each criterion.

        Returns:
            Dictionary mapping {criterion: best_algorithm_name}

        Example:
            {
                "best_overall_quality": "GMM",
                "best_speed": "K-Means",
                "best_silhouette": "DBSCAN",
                "best_purity": "Hierarchical",
                "best_noise_handling": "DBSCAN"
            }
        """
        if self.comparison_matrix is None:
            self.create_comparison_matrix()

        logger.info("📊 Identifying best algorithms...")

        best_algorithms = {}

        # Best Silhouette Score (higher is better)
        best_silhouette_idx = self.comparison_matrix["silhouette_score"].idxmax()
        best_algorithms["best_silhouette"] = self.comparison_matrix.loc[best_silhouette_idx, "algorithm"]

        # Best Davies-Bouldin Index (lower is better)
        best_db_idx = self.comparison_matrix["davies_bouldin_index"].idxmin()
        best_algorithms["best_davies_bouldin"] = self.comparison_matrix.loc[best_db_idx, "algorithm"]

        # Best Cluster Purity (higher is better)
        best_purity_idx = self.comparison_matrix["cluster_purity"].idxmax()
        best_algorithms["best_purity"] = self.comparison_matrix.loc[best_purity_idx, "algorithm"]

        # Best Speed (lower runtime is better)
        best_speed_idx = self.comparison_matrix["runtime_seconds"].idxmin()
        best_algorithms["best_speed"] = self.comparison_matrix.loc[best_speed_idx, "algorithm"]

        # Noise handling (algorithm with most noise points - usually DBSCAN)
        if "n_noise_points" in self.comparison_matrix.columns:
            noise_handling_idx = self.comparison_matrix["n_noise_points"].idxmax()
            best_algorithms["best_noise_handling"] = self.comparison_matrix.loc[noise_handling_idx, "algorithm"]

        logger.info("✅ Identified best algorithms per criterion")

        return best_algorithms

    def export_to_csv(self, output_path: Path) -> None:
        """
        Export comparison matrix to CSV file.

        Args:
            output_path: Path to save CSV file
        """
        if self.comparison_matrix is None:
            self.create_comparison_matrix()

        self.comparison_matrix.to_csv(output_path, index=False)
        logger.info(f"✅ Exported comparison matrix to {output_path}")

    def export_to_json(self, output_path: Path) -> None:
        """
        Export comprehensive comparison results to JSON.

        Args:
            output_path: Path to save JSON file

        JSON structure:
            {
                "metadata": {...},
                "comparison_matrix": [...],
                "best_algorithms": {...},
                "confusion_matrices": {...}
            }
        """
        if self.comparison_matrix is None:
            self.create_comparison_matrix()

        # Generate confusion matrices
        confusion_matrices = self.generate_confusion_matrices()

        # Identify best algorithms
        best_algorithms = self.identify_best_algorithms()

        # Create comprehensive output
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "n_algorithms": len(self.algorithms),
                "algorithms": list(self.algorithms.keys())
            },
            "comparison_matrix": self.comparison_matrix.to_dict(orient="records"),
            "best_algorithms": best_algorithms,
            "confusion_matrices": {
                name: cm.tolist() for name, cm in confusion_matrices.items()
            },
            "per_algorithm_details": {
                name: {
                    "metrics": data["metrics"],
                    "runtime": data["runtime"],
                    "parameters": data["parameters"]
                }
                for name, data in self.algorithms.items()
            }
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✅ Exported comprehensive results to {output_path}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all algorithms.

        Returns:
            Dictionary with min/max/mean/std for each metric
        """
        if self.comparison_matrix is None:
            self.create_comparison_matrix()

        logger.info("📊 Calculating summary statistics...")

        numeric_columns = [
            "silhouette_score",
            "davies_bouldin_index",
            "cluster_purity",
            "runtime_seconds"
        ]

        summary = {}

        for col in numeric_columns:
            if col in self.comparison_matrix.columns:
                summary[col] = {
                    "min": float(self.comparison_matrix[col].min()),
                    "max": float(self.comparison_matrix[col].max()),
                    "mean": float(self.comparison_matrix[col].mean()),
                    "std": float(self.comparison_matrix[col].std())
                }

        logger.info("✅ Summary statistics calculated")

        return summary
