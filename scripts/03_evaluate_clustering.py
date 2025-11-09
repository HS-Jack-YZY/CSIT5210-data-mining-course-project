#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.evaluation import ClusteringMetrics
from context_aware_multi_agent_system.data.load_dataset import DatasetLoader
from context_aware_multi_agent_system.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    
    logger.info("Starting cluster quality evaluation...")
    
    set_seed(42)
    logger.info("Set random seed to 42")
    
    config = Config()
    paths = Paths()
    
    # Load cluster assignments
    assignments_path = paths.data_processed / "cluster_assignments.csv"
    if not assignments_path.exists():
        logger.error(f"Cluster assignments not found: {assignments_path}")
        sys.exit(1)
    
    df = pd.read_csv(assignments_path)
    labels = df['cluster_id'].values.astype(np.int32)
    logger.info(f"Loaded {len(labels)} cluster assignments")
    
    # Load embeddings
    embeddings_path = paths.data_embeddings / "train_embeddings.npy"
    if not embeddings_path.exists():
        logger.error(f"Embeddings not found: {embeddings_path}")
        sys.exit(1)
    
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded {embeddings.shape[0]} embeddings")
    
    # Load centroids
    centroids_path = paths.data_processed / "centroids.npy"
    if not centroids_path.exists():
        logger.error(f"Centroids not found: {centroids_path}")
        sys.exit(1)
    
    centroids = np.load(centroids_path)
    logger.info(f"Loaded {centroids.shape[0]} centroids")
    
    # Load ground truth
    dataset_loader = DatasetLoader(config)
    train_dataset, _ = dataset_loader.load_ag_news()
    ground_truth = np.array(train_dataset['label'], dtype=np.int32)
    logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Initialize metrics calculator
    metrics_calculator = ClusteringMetrics(
        embeddings=embeddings,
        labels=labels,
        centroids=centroids,
        ground_truth=ground_truth
    )
    
    # Calculate all metrics
    logger.info("Computing cluster quality metrics...")
    metrics = metrics_calculator.evaluate_all()
    
    # Generate confusion matrix
    confusion_mat = metrics_calculator.generate_confusion_matrix()
    
    # Save results
    quality_path = paths.data_processed / "cluster_quality.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(quality_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved cluster quality metrics: {quality_path}")
    
    # Save confusion matrix
    confusion_path = paths.data_processed / "confusion_matrix.npy"
    np.save(confusion_path, confusion_mat)
    logger.info(f"Saved confusion matrix: {confusion_path}")
    
    # Update cluster metadata
    metadata_path = paths.data_processed / "cluster_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata['quality_metrics'] = {
        'timestamp': datetime.now().isoformat(),
        'silhouette_score': metrics['silhouette_score'],
        'davies_bouldin_index': metrics['davies_bouldin_index'],
        'cluster_purity': metrics['cluster_purity'],
        'is_balanced': metrics['is_balanced']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Updated cluster metadata: {metadata_path}")
    
    # Display summary
    silhouette = metrics['silhouette_score']
    purity = metrics['cluster_purity']['overall']
    
    logger.info("=" * 60)
    logger.info("Cluster Quality Evaluation Complete")
    logger.info("=" * 60)
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    logger.info(f"Cluster Purity: {purity:.4f} ({purity*100:.1f}%)")
    logger.info(f"Cluster Balance: {'Balanced' if metrics['is_balanced'] else 'Imbalanced'}")
    logger.info("=" * 60)
    
    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Cluster quality evaluation failed: {e}", exc_info=True)
        sys.exit(1)
