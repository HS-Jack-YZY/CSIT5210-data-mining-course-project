#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster Analysis Script

This script performs comprehensive cluster analysis including:
- Mapping clusters to dominant AG News categories
- Calculating cluster purity metrics
- Extracting representative documents
- Generating human-readable analysis report
- Exporting cluster labels to JSON

Usage:
    python scripts/05_analyze_clusters.py

Outputs:
    - results/cluster_analysis.txt: Human-readable analysis report
    - results/cluster_labels.json: Structured cluster labels and metadata
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.evaluation import ClusterAnalyzer
from context_aware_multi_agent_system.data.load_dataset import DatasetLoader
from context_aware_multi_agent_system.utils.reproducibility import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main orchestration function for cluster analysis."""
    start_time = time.time()

    logger.info("📊 Starting cluster analysis and labeling...")

    # Set seed for reproducibility
    set_seed(42)
    logger.info("✅ Set random seed to 42")

    # Load configuration
    config = Config()
    paths = Paths()

    # ========== Load Cluster Assignments ==========
    logger.info("📊 Loading cluster assignments and ground truth labels...")

    assignments_path = paths.data_processed / "cluster_assignments.csv"
    if not assignments_path.exists():
        logger.error(
            f"❌ Cluster assignments not found: {assignments_path}\n"
            "Run 'python scripts/02_train_clustering.py' first"
        )
        sys.exit(1)

    df = pd.read_csv(assignments_path)
    labels = df['cluster_id'].values.astype(np.int32)
    logger.info(f"✅ Loaded {len(labels)} cluster assignments")

    # ========== Load Embeddings ==========
    embeddings_path = paths.data_embeddings / "train_embeddings.npy"
    if not embeddings_path.exists():
        logger.error(
            f"❌ Embeddings not found: {embeddings_path}\n"
            "Run 'python scripts/01_generate_embeddings.py' first"
        )
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded {embeddings.shape[0]} embeddings")

    # ========== Load Centroids ==========
    centroids_path = paths.data_processed / "centroids.npy"
    if not centroids_path.exists():
        logger.error(
            f"❌ Centroids not found: {centroids_path}\n"
            "Run 'python scripts/02_train_clustering.py' first"
        )
        sys.exit(1)

    centroids = np.load(centroids_path)
    logger.info(f"✅ Loaded {centroids.shape[0]} centroids")

    # ========== Load Ground Truth Labels ==========
    dataset_loader = DatasetLoader(config)
    train_dataset, _ = dataset_loader.load_ag_news()
    ground_truth = np.array(train_dataset['label'], dtype=np.int32)
    logger.info(f"✅ Loaded {len(ground_truth)} ground truth labels")

    # Extract text for representative documents (AG News only has 'text' field)
    texts = np.array(train_dataset['text'])
    # Truncate texts for display (first 100 chars as "title")
    titles = np.array([text[:100] for text in texts])

    # ========== Validate Input Shapes ==========
    logger.info("📊 Validating input data consistency...")

    if len(labels) != len(embeddings):
        logger.error(
            f"❌ Shape mismatch: Labels count {len(labels)} != embeddings count {len(embeddings)}"
        )
        sys.exit(1)

    if len(ground_truth) != len(embeddings):
        logger.error(
            f"❌ Shape mismatch: Ground truth count {len(ground_truth)} != embeddings count {len(embeddings)}"
        )
        sys.exit(1)

    if labels.min() < 0 or labels.max() >= centroids.shape[0]:
        logger.error(
            f"❌ Invalid cluster labels: min={labels.min()}, max={labels.max()}, expected [0, {centroids.shape[0]-1}]"
        )
        sys.exit(1)

    logger.info("✅ Input validation successful")

    # ========== Initialize ClusterAnalyzer ==========
    logger.info("📊 Initializing cluster analyzer...")

    analyzer = ClusterAnalyzer(
        labels=labels,
        embeddings=embeddings,
        centroids=centroids,
        ground_truth=ground_truth,
        titles=titles
    )

    # ========== Map Clusters to Categories ==========
    logger.info("📊 Mapping clusters to dominant categories...")

    cluster_mapping = analyzer.map_clusters_to_categories()

    logger.info("✅ Cluster-to-category mapping complete:")
    for cluster_id, category in cluster_mapping.items():
        logger.info(f"   - Cluster {cluster_id}: {category}")

    # ========== Calculate Cluster Purity ==========
    logger.info("📊 Calculating cluster purity...")

    purity_results = analyzer.calculate_cluster_purity()

    avg_purity = purity_results['average']
    logger.info(f"✅ Average cluster purity: {avg_purity:.1%}")

    # Check purity threshold
    if avg_purity < 0.70:
        logger.warning(
            f"⚠️ Average purity {avg_purity:.1%} below target 70% "
            "(acceptable for MVP, may indicate clustering could be improved)"
        )
    else:
        logger.info("✅ Cluster purity meets target threshold (>70%)")

    # ========== Extract Representative Documents ==========
    logger.info("📊 Extracting representative documents...")

    total_representatives = 0
    for cluster_id in range(centroids.shape[0]):
        representatives = analyzer.extract_representative_documents(cluster_id, k=10)
        total_representatives += len(representatives)

    logger.info(f"✅ Extracted {total_representatives} representative documents (10 per cluster)")

    # ========== Generate Cluster Analysis Report ==========
    logger.info("📊 Generating cluster analysis report...")

    # Create results directory
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate text report
    report_path = results_dir / "cluster_analysis.txt"
    analyzer.generate_analysis_report(
        output_path=report_path,
        dataset_name="AG News",
        n_documents=len(embeddings),
        clustering_params={"K": centroids.shape[0], "random_state": 42}
    )

    logger.info(f"✅ Report saved: {report_path}")

    # ========== Export Cluster Labels JSON ==========
    logger.info("📊 Exporting cluster labels to JSON...")

    json_path = results_dir / "cluster_labels.json"
    analyzer.export_cluster_labels_json(
        output_path=json_path,
        n_documents=len(embeddings)
    )

    logger.info(f"✅ Labels saved: {json_path}")

    # ========== Display Summary ==========
    elapsed = time.time() - start_time

    logger.info("=" * 80)
    logger.info("✅ Cluster Analysis Complete")
    logger.info(f"   - Clusters analyzed: {centroids.shape[0]}")
    logger.info(f"   - Average purity: {avg_purity:.1%}")
    logger.info(f"   - Total documents: {len(embeddings):,}")
    logger.info(f"   - Representative docs: {total_representatives} (10 per cluster)")
    logger.info(f"   - Report: {report_path}")
    logger.info(f"   - Labels: {json_path}")
    logger.info(f"   - Execution time: {elapsed:.1f}s")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Cluster analysis failed: {e}", exc_info=True)
        sys.exit(1)
