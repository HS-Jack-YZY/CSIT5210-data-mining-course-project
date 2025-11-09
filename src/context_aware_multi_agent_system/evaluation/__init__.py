"""
Evaluation module for clustering quality metrics.

This module provides tools for evaluating cluster quality using standard metrics
like Silhouette Score, Davies-Bouldin Index, cluster purity, and confusion matrices.
It also provides cluster analysis and semantic labeling functionality.

Classes:
    ClusteringMetrics: Comprehensive cluster quality evaluation
    ClusterAnalyzer: Cluster analysis and semantic labeling
"""

from .clustering_metrics import ClusteringMetrics
from .cluster_analysis import ClusterAnalyzer

__all__ = ['ClusteringMetrics', 'ClusterAnalyzer']
