"""
Evaluation module for clustering quality metrics.

This module provides tools for evaluating cluster quality using standard metrics
like Silhouette Score, Davies-Bouldin Index, cluster purity, and confusion matrices.

Classes:
    ClusteringMetrics: Comprehensive cluster quality evaluation
"""

from .clustering_metrics import ClusteringMetrics

__all__ = ['ClusteringMetrics']
