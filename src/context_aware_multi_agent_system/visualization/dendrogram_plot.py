"""
Dendrogram visualization for hierarchical clustering results.

This module provides visualization tools for generating dendrograms from
hierarchical clustering linkage matrices with support for truncation and
cluster boundary markers.

Functions:
    generate_dendrogram: Create dendrogram visualization with matplotlib
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

logger = logging.getLogger(__name__)


def generate_dendrogram(
    embeddings: np.ndarray,
    linkage_method: str = 'ward',
    output_path: Optional[Path] = None,
    n_clusters: int = 4,
    truncate_mode: Optional[str] = 'lastp',
    p: int = 30,
    title: Optional[str] = None,
    figsize: tuple = (12, 8),
    dpi: int = 300
) -> Path:
    """
    Generate dendrogram visualization for hierarchical clustering.

    Args:
        embeddings: Document embeddings (n_samples, 768) float32 or sampled subset
        linkage_method: Linkage method used (ward/complete/average/single)
        output_path: Where to save dendrogram PNG (default: reports/figures/dendrogram.png)
        n_clusters: Number of clusters for boundary marker (default: 4)
        truncate_mode: Truncation strategy ('lastp' or None for full dendrogram)
        p: Number of last merged clusters to show if truncate_mode='lastp'
        title: Custom title for dendrogram (auto-generated if None)
        figsize: Figure size in inches (width, height)
        dpi: Image resolution (default: 300 DPI)

    Returns:
        Path to saved dendrogram PNG file

    Raises:
        ValueError: If embeddings are invalid or output directory cannot be created

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> output_path = generate_dendrogram(
        ...     embeddings,
        ...     linkage_method='ward',
        ...     output_path=Path('reports/figures/dendrogram.png')
        ... )
        >>> print(f"Dendrogram saved: {output_path}")
    """
    # Validate embeddings
    if len(embeddings.shape) != 2:
        raise ValueError(
            f"Embeddings must be 2D array, got {len(embeddings.shape)}D"
        )

    if embeddings.shape[1] != 768:
        raise ValueError(
            f"Embeddings must have 768 dimensions, got {embeddings.shape[1]}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Embeddings must have dtype float32, got {embeddings.dtype}"
        )

    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError("Embeddings contain NaN or Inf values")

    logger.info(
        f"📊 Generating dendrogram: {embeddings.shape[0]} samples, "
        f"linkage={linkage_method}, truncate={truncate_mode}, p={p}"
    )

    # Set default output path
    if output_path is None:
        from ..config import Paths
        paths = Paths()
        output_path = paths.reports_figures / "dendrogram.png"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create linkage matrix
    logger.info("📊 Computing linkage matrix...")
    Z = scipy_linkage(embeddings, method=linkage_method)
    logger.info("✅ Linkage matrix computed")

    # Create figure
    plt.figure(figsize=figsize)

    # Generate dendrogram
    logger.info("📊 Generating dendrogram plot...")

    dendrogram_kwargs = {
        'color_threshold': None,  # Will be set based on n_clusters
        'above_threshold_color': 'gray',
        'leaf_font_size': 10
    }

    # Add truncation if specified
    if truncate_mode == 'lastp':
        dendrogram_kwargs['truncate_mode'] = truncate_mode
        dendrogram_kwargs['p'] = p
        dendrogram_kwargs['show_leaf_counts'] = True

    # Generate dendrogram
    dendro = dendrogram(Z, **dendrogram_kwargs)

    # Add cluster boundary line (horizontal line at n_clusters cut)
    # Calculate the height at which to cut for n_clusters
    if n_clusters > 1 and len(Z) > 0:
        # The (n_clusters-1)th merge from the end
        cut_height = Z[-(n_clusters-1), 2] if len(Z) >= (n_clusters-1) else Z[-1, 2]
        plt.axhline(y=cut_height, c='red', linestyle='--', linewidth=2,
                   label=f'n_clusters={n_clusters} cut')
        plt.legend(loc='upper right')

    # Add labels and title
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance (Linkage Height)', fontsize=12)

    # Generate title
    if title is None:
        truncation_note = f" (Truncated: Last {p} Merges)" if truncate_mode == 'lastp' else ""
        title = f"Hierarchical Clustering Dendrogram (AG News, {linkage_method.capitalize()} Linkage){truncation_note}"

    plt.title(title, fontsize=14, fontweight='bold')

    # Add interpretation notes as text
    notes = [
        f"• Linkage Method: {linkage_method.capitalize()}",
        f"• Number of Clusters: {n_clusters}",
        f"• Samples: {embeddings.shape[0]:,}"
    ]

    if truncate_mode == 'lastp':
        notes.append(f"• Truncation: Last {p} merges shown")

    notes_text = "\n".join(notes)
    plt.text(
        0.02, 0.98, notes_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    logger.info(f"📊 Saving dendrogram to {output_path}...")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Validate file exists
    if not output_path.exists():
        raise ValueError(f"Dendrogram not saved: {output_path}")

    if output_path.stat().st_size == 0:
        raise ValueError(f"Dendrogram file is empty: {output_path}")

    logger.info(f"✅ Dendrogram saved: {output_path} ({dpi} DPI)")

    return output_path


def generate_interactive_dendrogram(
    embeddings: np.ndarray,
    linkage_method: str = 'ward',
    output_path: Optional[Path] = None,
    n_clusters: int = 4,
    title: Optional[str] = None
) -> Path:
    """
    Generate interactive dendrogram visualization using Plotly.

    Args:
        embeddings: Document embeddings (n_samples, 768) float32
        linkage_method: Linkage method used (ward/complete/average/single)
        output_path: Where to save dendrogram HTML (default: reports/figures/dendrogram.html)
        n_clusters: Number of clusters for boundary marker (default: 4)
        title: Custom title for dendrogram (auto-generated if None)

    Returns:
        Path to saved dendrogram HTML file

    Raises:
        ImportError: If plotly is not installed
        ValueError: If embeddings are invalid

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> output_path = generate_interactive_dendrogram(
        ...     embeddings,
        ...     linkage_method='ward',
        ...     output_path=Path('reports/figures/dendrogram.html')
        ... )
    """
    try:
        import plotly.figure_factory as ff
    except ImportError:
        raise ImportError(
            "plotly is required for interactive dendrograms. "
            "Install it with: pip install plotly"
        )

    # Validate embeddings
    if len(embeddings.shape) != 2:
        raise ValueError(
            f"Embeddings must be 2D array, got {len(embeddings.shape)}D"
        )

    if embeddings.shape[1] != 768:
        raise ValueError(
            f"Embeddings must have 768 dimensions, got {embeddings.shape[1]}"
        )

    logger.info(
        f"📊 Generating interactive dendrogram: {embeddings.shape[0]} samples, "
        f"linkage={linkage_method}"
    )

    # Set default output path
    if output_path is None:
        from ..config import Paths
        paths = Paths()
        output_path = paths.reports_figures / "dendrogram.html"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create linkage matrix
    logger.info("📊 Computing linkage matrix...")
    Z = scipy_linkage(embeddings, method=linkage_method)
    logger.info("✅ Linkage matrix computed")

    # Generate title
    if title is None:
        title = f"Hierarchical Clustering Dendrogram (AG News, {linkage_method.capitalize()} Linkage)"

    # Create interactive dendrogram
    logger.info("📊 Creating interactive plotly dendrogram...")
    fig = ff.create_dendrogram(
        embeddings,
        linkagefun=lambda x: scipy_linkage(x, method=linkage_method),
        color_threshold=None
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Distance (Linkage Height)",
        width=1200,
        height=800
    )

    # Save HTML
    logger.info(f"📊 Saving interactive dendrogram to {output_path}...")
    fig.write_html(str(output_path))

    # Validate file exists
    if not output_path.exists():
        raise ValueError(f"Interactive dendrogram not saved: {output_path}")

    logger.info(f"✅ Interactive dendrogram saved: {output_path}")

    return output_path
