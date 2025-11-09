"""
Embedding cache management for storing and loading embeddings.

This module provides efficient caching of embeddings to disk using numpy binary
format (.npy) for arrays and JSON for metadata, preventing redundant API calls.

Classes:
    EmbeddingCache: Save and load embeddings with metadata tracking
    CacheNotFoundError: Custom exception for missing cache files
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class CacheNotFoundError(FileNotFoundError):
    """
    Exception raised when embedding cache files are not found.

    This exception is raised when attempting to load embeddings from cache
    but the .npy or .json files don't exist.

    Usage:
        raise CacheNotFoundError("train", cache_dir)
    """

    def __init__(self, split: str, cache_dir: Path):
        """
        Initialize CacheNotFoundError with split name and cache directory.

        Args:
            split: Dataset split name (e.g., "train", "test")
            cache_dir: Cache directory path
        """
        message = (
            f"Cache not found for split '{split}' in {cache_dir}\n"
            f"Expected files:\n"
            f"  - {cache_dir}/{split}_embeddings.npy\n"
            f"  - {cache_dir}/{split}_metadata.json\n"
            f"Run embedding generation first to create cache."
        )
        super().__init__(message)


class EmbeddingCache:
    """
    Cache manager for embedding storage and retrieval.

    Provides methods to save embeddings and metadata to disk in an efficient
    binary format (.npy for arrays, .json for metadata) and load them back.

    Features:
        - Embeddings saved as .npy files (compact, fast I/O)
        - Metadata saved as .json files (human-readable, versioned)
        - Automatic timestamp tracking
        - Dtype validation (float32)
        - Cache existence checking
        - Cache clearing for cleanup

    Usage:
        >>> from pathlib import Path
        >>> cache = EmbeddingCache(Path("data/embeddings"))
        >>>
        >>> # Save embeddings
        >>> embeddings = np.random.rand(100, 768).astype(np.float32)
        >>> metadata = {"model": "gemini-embedding-001", "num_documents": 100}
        >>> cache.save(embeddings, "train", metadata)
        >>>
        >>> # Load embeddings
        >>> loaded_embeddings, loaded_metadata = cache.load("train")
        >>> loaded_embeddings.shape
        (100, 768)
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize EmbeddingCache with cache directory.

        Args:
            cache_dir: Directory for storing embedding cache files

        Example:
            >>> from context_aware_multi_agent_system.config import Paths
            >>> paths = Paths()
            >>> cache = EmbeddingCache(paths.data_embeddings)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Initialized embedding cache at: {self.cache_dir}")

    def save(
        self,
        embeddings: np.ndarray,
        split: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save embeddings and metadata to cache.

        Saves embeddings as {split}_embeddings.npy and metadata as
        {split}_metadata.json. Automatically adds timestamp to metadata
        if not present.

        Args:
            embeddings: Embedding array to save (must be float32)
            split: Dataset split name (e.g., "train", "test")
            metadata: Metadata dictionary (model, dimensions, num_documents, etc.)

        Returns:
            Path to saved embeddings file

        Raises:
            ValueError: If embeddings dtype is not float32

        Example:
            >>> embeddings = np.random.rand(100, 768).astype(np.float32)
            >>> metadata = {
            ...     "model": "gemini-embedding-001",
            ...     "dimensions": 768,
            ...     "num_documents": 100
            ... }
            >>> path = cache.save(embeddings, "train", metadata)
        """
        # Validate embeddings dtype
        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Embeddings must be float32, got {embeddings.dtype}. "
                f"Convert with: embeddings.astype(np.float32)"
            )

        # Construct file paths
        embeddings_path = self.cache_dir / f"{split}_embeddings.npy"
        metadata_path = self.cache_dir / f"{split}_metadata.json"

        # Add timestamp to metadata if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        # Save embeddings as .npy file
        np.save(embeddings_path, embeddings)
        logger.info(f"💾 Saved embeddings to {embeddings_path}")

        # Save metadata as .json file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata to {metadata_path}")

        return embeddings_path

    def load(self, split: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings and metadata from cache.

        Loads embeddings from {split}_embeddings.npy and metadata from
        {split}_metadata.json. Validates dtype is float32.

        Args:
            split: Dataset split name (e.g., "train", "test")

        Returns:
            Tuple of (embeddings array, metadata dict)

        Raises:
            CacheNotFoundError: If cache files don't exist
            ValueError: If loaded embeddings dtype is not float32

        Example:
            >>> embeddings, metadata = cache.load("train")
            >>> embeddings.shape
            (100, 768)
            >>> metadata["model"]
            "gemini-embedding-001"
        """
        # Construct file paths
        embeddings_path = self.cache_dir / f"{split}_embeddings.npy"
        metadata_path = self.cache_dir / f"{split}_metadata.json"

        # Check if files exist
        if not embeddings_path.exists() or not metadata_path.exists():
            raise CacheNotFoundError(split, self.cache_dir)

        # Load embeddings from .npy file
        embeddings = np.load(embeddings_path)
        logger.info(f"✅ Loaded embeddings from {embeddings_path}")

        # Load metadata from .json file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✅ Loaded metadata from {metadata_path}")

        # Validate embeddings dtype
        if embeddings.dtype != np.float32:
            raise ValueError(
                f"Loaded embeddings have invalid dtype: {embeddings.dtype}. "
                f"Expected float32. Cache may be corrupted."
            )

        return embeddings, metadata

    def exists(self, split: str) -> bool:
        """
        Check if cache files exist for a given split.

        Args:
            split: Dataset split name (e.g., "train", "test")

        Returns:
            True if both .npy and .json files exist, False otherwise

        Example:
            >>> if cache.exists("train"):
            ...     embeddings, metadata = cache.load("train")
            ... else:
            ...     # Generate embeddings
            ...     pass
        """
        embeddings_path = self.cache_dir / f"{split}_embeddings.npy"
        metadata_path = self.cache_dir / f"{split}_metadata.json"

        exists = embeddings_path.exists() and metadata_path.exists()

        if exists:
            logger.info(f"✅ Cache exists for split: {split}")
        else:
            logger.info(f"⚠️ Cache not found for split: {split}")

        return exists

    def clear(self, split: str) -> None:
        """
        Delete cache files for a given split.

        Removes both .npy and .json files for the specified split.
        Handles FileNotFoundError gracefully (no error if files don't exist).

        Args:
            split: Dataset split name (e.g., "train", "test")

        Example:
            >>> cache.clear("train")  # Delete train embeddings and metadata
        """
        embeddings_path = self.cache_dir / f"{split}_embeddings.npy"
        metadata_path = self.cache_dir / f"{split}_metadata.json"

        # Delete embeddings file
        try:
            embeddings_path.unlink()
            logger.info(f"🗑️ Deleted {embeddings_path}")
        except FileNotFoundError:
            logger.info(f"⚠️ Embeddings file not found: {embeddings_path}")

        # Delete metadata file
        try:
            metadata_path.unlink()
            logger.info(f"🗑️ Deleted {metadata_path}")
        except FileNotFoundError:
            logger.info(f"⚠️ Metadata file not found: {metadata_path}")

        logger.info(f"✅ Cleared cache for split: {split}")
