"""
Embedding generation script for AG News dataset.

This script generates embeddings for all AG News documents using Google Gemini API
with efficient caching, checkpoint system, and cost tracking.

Usage:
    python scripts/01_generate_embeddings.py

Features:
    - Batch embedding generation with Gemini API
    - Automatic caching to avoid repeated API calls
    - Checkpoint system for resumable generation
    - Cost tracking and performance metrics
    - Error handling with retry logic
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from tenacity import RetryError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.data.load_dataset import DatasetLoader
from context_aware_multi_agent_system.features.embedding_service import EmbeddingService
from context_aware_multi_agent_system.features.embedding_cache import EmbeddingCache
from context_aware_multi_agent_system.evaluation.cost_calculator import (
    estimate_embedding_cost,
    estimate_tokens
)
from context_aware_multi_agent_system.utils.reproducibility import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_path: Path,
    split: str,
    last_index: int,
    metadata: Dict[str, Any]
) -> None:
    """
    Save checkpoint for resumable embedding generation.

    Args:
        checkpoint_path: Path to checkpoint file
        split: Dataset split name
        last_index: Last processed document index
        metadata: Additional metadata (batch_size, total_batches, etc.)
    """
    checkpoint_data = {
        "split": split,
        "last_processed_index": last_index,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metadata
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"💾 Saved checkpoint: processed {last_index} documents")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load checkpoint if exists.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint data dict or empty dict if no checkpoint
    """
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        logger.info(
            f"⏯️ Resuming from checkpoint: last processed index = {checkpoint['last_processed_index']}"
        )
        return checkpoint
    return {}


def delete_checkpoint(checkpoint_path: Path) -> None:
    """
    Delete checkpoint file after successful completion.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("✅ Checkpoint deleted (generation complete)")


def generate_embeddings_for_split(
    documents: list,
    split: str,
    config: Config,
    paths: Paths,
    service: EmbeddingService,
    cache: EmbeddingCache
) -> Dict[str, Any]:
    """
    Generate embeddings for a dataset split with checkpoint support.

    Args:
        documents: List of text documents
        split: Dataset split name ("train" or "test")
        config: Configuration object
        paths: Paths object
        service: EmbeddingService instance
        cache: EmbeddingCache instance

    Returns:
        Statistics dict with counts, cost, and time metrics
    """
    # Check if cached embeddings exist
    if config.get("embedding.cache_enabled", True) and cache.exists(split):
        logger.warning(f"⚠️ Using cached embeddings for {split} split from {paths.data_embeddings}")
        embeddings, metadata = cache.load(split)
        return {
            "from_cache": True,
            "num_documents": len(documents),
            "api_calls": 0,
            "tokens": 0,
            "cost": 0.0,
            "time": 0.0
        }

    # Checkpoint setup
    checkpoint_path = paths.data_embeddings / f".checkpoint_{split}.json"
    checkpoint = load_checkpoint(checkpoint_path) if config.get("embedding.checkpoint_enabled", True) else {}

    start_index = checkpoint.get("last_processed_index", 0)
    batch_size = config.get("embedding.batch_size", 100)
    use_batch_api = config.get("embedding.use_batch_api", True)

    # Initialize tracking
    embeddings_list = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    successful_batches = 0
    failed_batches = 0
    total_tokens = 0
    start_time = time.time()

    logger.info(f"📊 Starting embedding generation for {split} split")
    logger.info(f"📊 Total documents: {len(documents)}, Batch size: {batch_size}, Total batches: {total_batches}")

    if start_index > 0:
        logger.info(f"⏯️ Resuming from document {start_index}")

    # Process documents in batches
    for i in range(start_index, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1

        # Log progress every 1000 documents
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            docs_per_sec = i / elapsed if elapsed > 0 else 0
            logger.info(
                f"📊 Progress: {i}/{len(documents)} documents processed "
                f"({docs_per_sec:.1f} docs/sec)"
            )

        logger.info(f"🔄 Processing batch {batch_num}/{total_batches} (documents {i}-{i+len(batch)-1})")

        try:
            # Generate embeddings for batch
            batch_embeddings = service.generate_batch(batch, batch_size=batch_size)
            embeddings_list.append(batch_embeddings)
            successful_batches += 1

            # Estimate tokens for this batch
            batch_text = " ".join(batch)
            batch_tokens = estimate_tokens(batch_text)
            total_tokens += batch_tokens

            # Save checkpoint after each successful batch
            if config.get("embedding.checkpoint_enabled", True):
                save_checkpoint(
                    checkpoint_path,
                    split,
                    i + len(batch),
                    {
                        "batch_size": batch_size,
                        "total_batches": total_batches,
                        "successful_batches": successful_batches,
                        "failed_batches": failed_batches
                    }
                )

        except RetryError as e:
            logger.error(
                f"❌ Batch {batch_num} failed after 3 retries, skipping: {e}"
            )
            failed_batches += 1
            # Continue processing remaining batches
            continue

        except Exception as e:
            logger.error(
                f"❌ Unexpected error in batch {batch_num}, skipping: {e}"
            )
            failed_batches += 1
            continue

    # Concatenate all embeddings
    if not embeddings_list:
        raise RuntimeError("No embeddings generated - all batches failed")

    embeddings = np.concatenate(embeddings_list, axis=0)

    # Validate final shape
    expected_shape = (len(documents), 768)
    if embeddings.shape != expected_shape:
        logger.warning(
            f"⚠️ Shape mismatch: got {embeddings.shape}, expected {expected_shape}. "
            f"Some batches may have failed."
        )

    # Calculate metrics
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / successful_batches if successful_batches > 0 else 0
    estimated_cost = estimate_embedding_cost(total_tokens, use_batch_api=use_batch_api)

    # Prepare metadata
    metadata = {
        "model": config.get("embedding.model", "gemini-embedding-001"),
        "dimensions": 768,
        "num_documents": len(documents),
        "dataset": "ag_news",
        "split": split,
        "api_calls": successful_batches,
        "estimated_cost": estimated_cost,
        "total_tokens": total_tokens,
        "batch_size": batch_size,
        "use_batch_api": use_batch_api,
        "failed_batches": failed_batches,
        "total_time": total_time
    }

    # Save embeddings to cache
    if config.get("embedding.cache_enabled", True):
        cache.save(embeddings, split, metadata)
        logger.info(f"💾 Saved {split} embeddings to cache")

    # Delete checkpoint on successful completion
    if config.get("embedding.checkpoint_enabled", True):
        delete_checkpoint(checkpoint_path)

    return {
        "from_cache": False,
        "num_documents": len(documents),
        "api_calls": successful_batches,
        "tokens": total_tokens,
        "cost": estimated_cost,
        "time": total_time,
        "avg_time_per_batch": avg_time_per_batch,
        "failed_batches": failed_batches,
        "successful_batches": successful_batches
    }


def main():
    """Main function for embedding generation."""
    # Set seed for reproducibility
    set_seed(42)
    logger.info("🎲 Set random seed to 42 for reproducibility")

    # Load configuration
    config = Config()
    paths = Paths()
    logger.info("✅ Configuration loaded")

    # Initialize services
    service = EmbeddingService(
        api_key=config.gemini_api_key,
        model=config.get("embedding.model", "gemini-embedding-001")
    )
    cache = EmbeddingCache(paths.data_embeddings)
    logger.info("✅ Services initialized")

    # Load dataset
    loader = DatasetLoader(config)
    train_dataset, test_dataset = loader.load_ag_news()
    logger.info(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")

    # Combine title and text for embedding
    train_docs = [f"{item['text']}" for item in train_dataset]
    test_docs = [f"{item['text']}" for item in test_dataset]

    # Generate embeddings for train split
    logger.info("=" * 80)
    logger.info("📊 TRAIN SPLIT")
    logger.info("=" * 80)
    train_stats = generate_embeddings_for_split(
        train_docs, "train", config, paths, service, cache
    )

    # Generate embeddings for test split
    logger.info("=" * 80)
    logger.info("📊 TEST SPLIT")
    logger.info("=" * 80)
    test_stats = generate_embeddings_for_split(
        test_docs, "test", config, paths, service, cache
    )

    # Display final summary
    logger.info("=" * 80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 80)

    total_docs = train_stats["num_documents"] + test_stats["num_documents"]
    total_api_calls = train_stats["api_calls"] + test_stats["api_calls"]
    total_tokens = train_stats["tokens"] + test_stats["tokens"]
    total_cost = train_stats["cost"] + test_stats["cost"]
    total_time = train_stats["time"] + test_stats["time"]

    logger.info(f"📄 Total documents processed: {total_docs:,}")
    logger.info(f"📡 Total API calls made: {total_api_calls:,}")
    logger.info(f"🔤 Total tokens consumed: {total_tokens:,}")
    logger.info(f"💰 Estimated total cost: ${total_cost:.4f} USD")
    logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if not train_stats["from_cache"]:
        logger.info(f"📊 Train - Successful batches: {train_stats['successful_batches']}")
        logger.info(f"⚠️ Train - Failed batches: {train_stats['failed_batches']}")
        logger.info(f"⏱️ Train - Avg time per batch: {train_stats['avg_time_per_batch']:.2f} seconds")

    if not test_stats["from_cache"]:
        logger.info(f"📊 Test - Successful batches: {test_stats['successful_batches']}")
        logger.info(f"⚠️ Test - Failed batches: {test_stats['failed_batches']}")
        logger.info(f"⏱️ Test - Avg time per batch: {test_stats['avg_time_per_batch']:.2f} seconds")

    # Cost validation
    if total_cost > 5.0:
        logger.warning(
            f"⚠️ WARNING: Total cost ${total_cost:.4f} exceeds target of $5.00"
        )
    else:
        logger.info(f"✅ Cost is below $5 target (${total_cost:.4f})")

    logger.info("=" * 80)
    logger.info("✅ Embedding generation complete!")


if __name__ == "__main__":
    main()
