"""
PCA cluster visualization for K-Means clustering results.

This module provides visualization tools for projecting high-dimensional
embeddings to 2D space using PCA and generating publication-quality plots.

Classes:
    PCAVisualizer: PCA-based cluster visualization with matplotlib/seaborn
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCAVisualizer:
    """
    PCA-based cluster visualization for high-dimensional embeddings.

    This class applies PCA dimensionality reduction (768D → 2D) and generates
    scatter plots showing cluster assignments with centroids marked.

    Attributes:
        embeddings: Document embeddings (n_documents, 768) float32
        labels: Cluster assignments (n_documents,) int32
        centroids: Cluster centroids (n_clusters, 768) float32
        pca: Fitted PCA model (None until apply_pca is called)
        embeddings_2d: 2D projected embeddings (None until apply_pca is called)
        centroids_2d: 2D projected centroids (None until apply_pca is called)

    Example:
        >>> from context_aware_multi_agent_system.config import Paths
        >>> import numpy as np
        >>> paths = Paths()
        >>> embeddings = np.load(paths.data_embeddings / "train_embeddings.npy")
        >>> labels = pd.read_csv(paths.data_processed / "cluster_assignments.csv")['cluster_id'].values
        >>> centroids = np.load(paths.data_processed / "centroids.npy")
        >>> visualizer = PCAVisualizer(embeddings, labels, centroids)
        >>> embeddings_2d, centroids_2d, variance = visualizer.apply_pca()
        >>> output_path = visualizer.generate_visualization(Path("visualizations/cluster_pca.png"))
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ):
        """
        Initialize PCA visualizer with clustering results.

        Args:
            embeddings: Document embeddings (n_documents, 768) float32
            labels: Cluster assignments (n_documents,) int32, values in [0, n_clusters-1]
            centroids: Cluster centroids (n_clusters, 768) float32

        Raises:
            ValueError: If input shapes are invalid or data types are incorrect
            ValueError: If embeddings contain NaN or Inf values
        """
        # Validate embeddings shape
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != 768:
            raise ValueError(
                f"Embeddings must have 768 dimensions, got {embeddings.shape[1]}"
            )

        # Validate labels shape and range
        if len(labels.shape) != 1:
            raise ValueError(
                f"Labels must be 1D array, got shape {labels.shape}"
            )
        if len(labels) != len(embeddings):
            raise ValueError(
                f"Labels length ({len(labels)}) must match embeddings length ({len(embeddings)})"
            )

        # Validate centroids shape
        n_clusters = len(np.unique(labels))
        if centroids.shape != (n_clusters, 768):
            raise ValueError(
                f"Centroids must have shape ({n_clusters}, 768), got {centroids.shape}"
            )

        # Validate no NaN/Inf
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain NaN or Inf values")
        if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
            raise ValueError("Centroids contain NaN or Inf values")

        self.embeddings = embeddings
        self.labels = labels
        self.centroids = centroids
        self.pca: PCA = None
        self.embeddings_2d: np.ndarray = None
        self.centroids_2d: np.ndarray = None

        logger.info(
            f"📊 Initialized PCAVisualizer: {len(embeddings)} documents, "
            f"{n_clusters} clusters, 768D embeddings"
        )

    def apply_pca(self, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply PCA dimensionality reduction to embeddings and centroids.

        Args:
            n_components: Number of principal components (default: 2)

        Returns:
            Tuple containing:
                - embeddings_2d: Projected embeddings (n_documents, 2) float64
                - centroids_2d: Projected centroids (n_clusters, 2) float64
                - variance_explained: Combined variance of PC1 + PC2 (float in [0, 1])

        Example:
            >>> embeddings_2d, centroids_2d, variance = visualizer.apply_pca()
            >>> assert embeddings_2d.shape == (120000, 2)
            >>> assert centroids_2d.shape == (4, 2)
            >>> assert 0 <= variance <= 1.0
        """
        logger.info(f"📊 Applying PCA dimensionality reduction (768D → {n_components}D)...")

        # Initialize PCA with fixed random_state for reproducibility
        self.pca = PCA(n_components=n_components, random_state=42)

        # Fit PCA on embeddings and transform both embeddings and centroids
        self.embeddings_2d = self.pca.fit_transform(self.embeddings)
        self.centroids_2d = self.pca.transform(self.centroids)

        # Calculate total variance explained
        variance_explained = self.pca.explained_variance_ratio_.sum()

        logger.info(
            f"✅ PCA complete. Variance explained: {variance_explained*100:.1f}%"
        )

        return self.embeddings_2d, self.centroids_2d, variance_explained

    def get_variance_explained(self) -> Tuple[float, float, float]:
        """
        Get variance explained by PC1, PC2, and total.

        Returns:
            Tuple of (pc1_variance, pc2_variance, total_variance) as floats in [0, 1]

        Raises:
            RuntimeError: If apply_pca has not been called yet
        """
        if self.pca is None:
            raise RuntimeError("Must call apply_pca() before get_variance_explained()")

        pc1_var = self.pca.explained_variance_ratio_[0]
        pc2_var = self.pca.explained_variance_ratio_[1]
        total_var = pc1_var + pc2_var

        return pc1_var, pc2_var, total_var

    def generate_visualization(
        self,
        output_path: Path,
        dpi: int = 300,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Path:
        """
        Generate and save cluster visualization with PCA projection.

        Args:
            output_path: Path to save PNG file
            dpi: Resolution in dots per inch (default: 300 for publication quality)
            figsize: Figure size in inches (width, height), default: (10, 8)

        Returns:
            Path to saved visualization file

        Raises:
            RuntimeError: If apply_pca has not been called yet
            FileNotFoundError: If output directory cannot be created

        Example:
            >>> output_path = Path("visualizations/cluster_pca.png")
            >>> saved_path = visualizer.generate_visualization(output_path)
            >>> assert saved_path.exists()
        """
        if self.embeddings_2d is None or self.centroids_2d is None:
            raise RuntimeError("Must call apply_pca() before generate_visualization()")

        logger.info("📊 Generating cluster scatter plot...")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get colorblind-friendly palette
        n_clusters = len(np.unique(self.labels))
        colors = sns.color_palette("colorblind", n_clusters)

        # Plot each cluster
        for cluster_id in range(n_clusters):
            mask = (self.labels == cluster_id)
            ax.scatter(
                self.embeddings_2d[mask, 0],
                self.embeddings_2d[mask, 1],
                c=[colors[cluster_id]],
                label=f'Cluster {cluster_id}',
                s=5,
                alpha=0.6
            )

        # Plot centroids
        ax.scatter(
            self.centroids_2d[:, 0],
            self.centroids_2d[:, 1],
            marker='*',
            s=300,
            c=colors[:n_clusters],
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=10
        )

        # Get variance explained for axis labels
        pc1_var, pc2_var, _ = self.get_variance_explained()

        # Format plot
        ax.set_xlabel(f'PC1 ({pc1_var*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pc2_var*100:.1f}% variance)')
        ax.set_title('K-Means Clustering of AG News (K=4, PCA Projection)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        logger.info("✅ Scatter plot rendered with 4 clusters")

        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        logger.info(f"📊 Saving visualization to {output_path}...")
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        # Validate file was created
        if not output_path.exists():
            raise FileNotFoundError(f"Failed to save visualization to {output_path}")

        if output_path.stat().st_size == 0:
            raise ValueError(f"Saved visualization file is empty: {output_path}")

        logger.info(f"✅ Visualization saved ({dpi} DPI PNG)")

        return output_path
