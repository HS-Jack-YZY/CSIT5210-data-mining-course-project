#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Initialization Script

This script initializes specialized agents for the multi-agent classification system:
- Creates 4 SpecializedAgent instances (one per cluster)
- Assigns cluster-specific documents to each agent
- Calculates context size reduction metrics
- Exports agent metadata to JSON

Usage:
    python scripts/06_initialize_agents.py

Outputs:
    - results/agent_metadata.json: Agent initialization metadata with context reduction stats
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.data.load_dataset import DatasetLoader
from context_aware_multi_agent_system.models.agent import SpecializedAgent, create_agent_registry
from context_aware_multi_agent_system.utils.reproducibility import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main orchestration function for agent initialization."""
    start_time = time.time()

    logger.info("📊 Starting agent initialization...")

    # Set seed for reproducibility
    set_seed(42)
    logger.info("✅ Set random seed to 42")

    # Load configuration
    config = Config()
    paths = Paths()

    # ========== Load Cluster Assignments ==========
    logger.info("📊 Loading cluster assignments...")

    assignments_path = paths.data_processed / "cluster_assignments.csv"
    if not assignments_path.exists():
        logger.error(
            f"❌ Cluster assignments not found: {assignments_path}\n"
            "Run 'python scripts/02_train_clustering.py' first"
        )
        sys.exit(1)

    assignments_df = pd.read_csv(assignments_path)
    logger.info(f"✅ Loaded {len(assignments_df)} cluster assignments")

    # Validate cluster assignments
    unique_clusters = sorted(assignments_df['cluster_id'].unique())
    if unique_clusters != [0, 1, 2, 3]:
        logger.error(
            f"❌ Invalid cluster IDs: expected [0, 1, 2, 3], got {unique_clusters}\n"
            "Clustering must produce exactly 4 clusters"
        )
        sys.exit(1)

    # ========== Load Cluster Labels ==========
    logger.info("📊 Loading cluster labels...")

    labels_path = paths.results / "cluster_labels.json"
    if not labels_path.exists():
        logger.error(
            f"❌ Cluster labels not found: {labels_path}\n"
            "Run 'python scripts/05_analyze_clusters.py' first"
        )
        sys.exit(1)

    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    # Extract cluster labels mapping
    cluster_labels = {
        int(cluster_id): cluster_info['label']
        for cluster_id, cluster_info in labels_data['clusters'].items()
    }
    logger.info(f"✅ Loaded cluster labels: {cluster_labels}")

    # ========== Load AG News Training Documents ==========
    logger.info("📊 Loading AG News training documents...")

    dataset_loader = DatasetLoader(config)
    train_dataset, _ = dataset_loader.load_ag_news()

    # Convert to list of dicts format expected by agents
    documents = [
        {
            'id': idx,
            'text': train_dataset['text'][idx],
            'label': train_dataset['label'][idx]
        }
        for idx in range(len(train_dataset))
    ]
    logger.info(f"✅ Loaded {len(documents)} training documents")

    # Validate document count matches assignments
    if len(documents) != len(assignments_df):
        logger.error(
            f"❌ Document count mismatch: {len(documents)} documents, "
            f"{len(assignments_df)} assignments\n"
            "Dataset and cluster assignments must have same size"
        )
        sys.exit(1)

    # ========== Create Agent Registry ==========
    logger.info("📊 Creating specialized agents...")

    try:
        registry = create_agent_registry(
            cluster_assignments_df=assignments_df,
            documents=documents,
            cluster_labels=cluster_labels
        )
    except ValueError as e:
        logger.error(f"❌ Agent registry creation failed: {e}")
        sys.exit(1)

    logger.info(f"✅ Created {len(registry)} specialized agents")

    # ========== Calculate Context Size Reduction ==========
    logger.info("📊 Calculating context size reduction metrics...")

    # Calculate baseline context size (all documents)
    baseline_context_size = sum(len(doc['text']) for doc in documents)
    logger.info(f"📊 Baseline context size: {baseline_context_size:,} characters")

    # Calculate per-agent metrics
    agent_metrics = {}
    total_agent_context = 0

    for cluster_id, agent in registry.items():
        agent_context_size = agent.get_context_size()
        total_agent_context += agent_context_size
        reduction_pct = 1 - (agent_context_size / baseline_context_size)

        # Estimate token count (rough estimate: 1 token ≈ 4 chars)
        estimated_tokens = agent_context_size // 4

        agent_metrics[cluster_id] = {
            'cluster_id': cluster_id,
            'cluster_label': agent.cluster_label,
            'num_documents': len(agent.get_documents()),
            'context_size_chars': agent_context_size,
            'context_size_tokens': estimated_tokens,
            'reduction_percentage': round(reduction_pct, 4)
        }

        logger.info(
            f"✅ Context Reduction: Agent {cluster_id} ({agent.cluster_label}) uses "
            f"{(1 - reduction_pct) * 100:.1f}% of baseline ({reduction_pct * 100:.1f}% reduction)"
        )

    # Calculate average reduction
    average_reduction = sum(m['reduction_percentage'] for m in agent_metrics.values()) / len(agent_metrics)
    logger.info(f"📊 Average context reduction: {average_reduction * 100:.1f}%")

    # Validate total documents conservation
    total_docs = sum(len(agent.get_documents()) for agent in registry.values())
    if total_docs != len(documents):
        logger.warning(
            f"⚠️ Document count mismatch: {total_docs} total across agents, "
            f"{len(documents)} original documents"
        )

    # ========== Save Agent Metadata ==========
    logger.info("📊 Saving agent metadata...")

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_agents': len(registry),
        'baseline_context_size': baseline_context_size,
        'baseline_context_tokens': baseline_context_size // 4,  # Rough estimate
        'agents': {
            str(cluster_id): metrics
            for cluster_id, metrics in agent_metrics.items()
        },
        'average_reduction': round(average_reduction, 4),
        'total_documents': len(documents),
        'documents_per_agent': {
            str(cid): len(agent.get_documents())
            for cid, agent in registry.items()
        }
    }

    metadata_path = paths.results / "agent_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Saved agent metadata to: {metadata_path}")

    # ========== Completion Summary ==========
    elapsed_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("✅ Agent Initialization Complete!")
    logger.info(f"📊 Total agents created: {len(registry)}")
    logger.info(f"📊 Average context reduction: {average_reduction * 100:.1f}%")
    logger.info(f"📊 Baseline context: {baseline_context_size:,} chars (~{baseline_context_size // 4:,} tokens)")
    logger.info(f"📊 Metadata saved: {metadata_path}")
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
