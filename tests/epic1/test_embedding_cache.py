"""
Tests for EmbeddingCache class (Story 1.4).

This module tests embedding cache save/load functionality, metadata tracking,
and cache management operations.

Test Coverage:
    - AC-4: Embedding Cache Functional
"""

import json
import pytest
import numpy as np
from pathlib import Path

from src.context_aware_multi_agent_system.features import (
    EmbeddingCache,
    CacheNotFoundError
)


class TestEmbeddingCacheInitialization:
    """Test EmbeddingCache initialization and setup."""

    def test_initialization_with_valid_path(self, tmp_path):
        """Test successful initialization with valid cache directory."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_initialization_creates_directory(self, tmp_path):
        """Test cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"

        # Directory doesn't exist yet
        assert not cache_dir.exists()

        # Initialize cache
        cache = EmbeddingCache(cache_dir)

        # Directory now exists
        assert cache_dir.exists()

    def test_initialization_with_existing_directory(self, tmp_path):
        """Test initialization with already existing directory."""
        cache_dir = tmp_path / "existing_cache"
        cache_dir.mkdir()

        # Initialize cache
        cache = EmbeddingCache(cache_dir)

        # Directory still exists
        assert cache_dir.exists()


class TestCacheSaveOperation:
    """Test embedding cache save functionality (AC-4)."""

    def test_save_embeddings_and_metadata(self, tmp_path):
        """
        AC-4: Test saving embeddings and metadata to cache.

        Given: Embeddings generated
        When: I call EmbeddingCache().save(embeddings, "train", metadata)
        Then: Embeddings saved to data/embeddings/train_embeddings.npy
              Metadata saved to data/embeddings/train_metadata.json
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create test embeddings
        embeddings = np.random.rand(100, 768).astype(np.float32)
        metadata = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 100,
            "dataset": "ag_news",
            "split": "train"
        }

        # Save embeddings
        saved_path = cache.save(embeddings, "train", metadata)

        # Verify files exist
        assert saved_path.exists()
        assert (cache_dir / "train_embeddings.npy").exists()
        assert (cache_dir / "train_metadata.json").exists()

    def test_save_validates_dtype_float32(self, tmp_path):
        """
        AC-4: Test saved embeddings are float32 dtype.

        Given: Embeddings generated
        When: I call save()
        Then: Saved embeddings are float32 dtype
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create embeddings with wrong dtype
        embeddings_float64 = np.random.rand(10, 768).astype(np.float64)

        with pytest.raises(ValueError, match="float32"):
            cache.save(embeddings_float64, "test", {})

    def test_save_adds_timestamp_to_metadata(self, tmp_path):
        """
        AC-4: Test timestamp automatically added to metadata.

        Given: Metadata without timestamp
        When: I call save()
        Then: Timestamp added to metadata
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create embeddings and metadata without timestamp
        embeddings = np.random.rand(10, 768).astype(np.float32)
        metadata = {
            "model": "gemini-embedding-001",
            "num_documents": 10
        }

        # Save embeddings
        cache.save(embeddings, "test", metadata)

        # Load metadata
        metadata_path = cache_dir / "test_metadata.json"
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)

        # Verify timestamp added
        assert "timestamp" in saved_metadata

    def test_save_metadata_includes_all_fields(self, tmp_path):
        """
        AC-4: Test metadata includes all required fields.

        Given: Embeddings generated
        When: I call save()
        Then: Metadata includes model, dimensions, num_documents, timestamp, etc.
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create embeddings
        embeddings = np.random.rand(50, 768).astype(np.float32)
        metadata = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 50,
            "dataset": "ag_news",
            "split": "train",
            "api_calls": 1,
            "estimated_cost": 0.0001
        }

        # Save embeddings
        cache.save(embeddings, "train", metadata)

        # Load metadata
        metadata_path = cache_dir / "train_metadata.json"
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)

        # Verify all fields present
        assert saved_metadata["model"] == "gemini-embedding-001"
        assert saved_metadata["dimensions"] == 768
        assert saved_metadata["num_documents"] == 50
        assert saved_metadata["dataset"] == "ag_news"
        assert saved_metadata["split"] == "train"
        assert saved_metadata["api_calls"] == 1
        assert saved_metadata["estimated_cost"] == 0.0001
        assert "timestamp" in saved_metadata


class TestCacheLoadOperation:
    """Test embedding cache load functionality (AC-4)."""

    def test_load_embeddings_and_metadata(self, tmp_path):
        """
        AC-4: Test loading embeddings and metadata from cache.

        Given: Embeddings saved to cache
        When: I call EmbeddingCache().load("train")
        Then: Returns tuple of (embeddings, metadata)
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create and save embeddings
        embeddings_original = np.random.rand(100, 768).astype(np.float32)
        metadata_original = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 100
        }

        cache.save(embeddings_original, "train", metadata_original)

        # Load embeddings
        embeddings_loaded, metadata_loaded = cache.load("train")

        # Verify return types
        assert isinstance(embeddings_loaded, np.ndarray)
        assert isinstance(metadata_loaded, dict)

    def test_load_embeddings_match_original(self, tmp_path):
        """
        AC-4: Test loaded embeddings match original.

        Given: Embeddings saved to cache
        When: I call load()
        Then: Loaded embeddings match original (np.allclose check)
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create and save embeddings
        embeddings_original = np.random.rand(100, 768).astype(np.float32)
        metadata = {"model": "gemini-embedding-001"}

        cache.save(embeddings_original, "test", metadata)

        # Load embeddings
        embeddings_loaded, _ = cache.load("test")

        # Verify embeddings match
        assert np.allclose(embeddings_original, embeddings_loaded)

    def test_load_metadata_matches_saved(self, tmp_path):
        """
        AC-4: Test loaded metadata matches saved metadata.

        Given: Embeddings saved with metadata
        When: I call load()
        Then: Loaded metadata matches saved metadata
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create and save embeddings
        embeddings = np.random.rand(50, 768).astype(np.float32)
        metadata_original = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 50,
            "dataset": "ag_news"
        }

        cache.save(embeddings, "train", metadata_original)

        # Load metadata
        _, metadata_loaded = cache.load("train")

        # Verify metadata matches
        assert metadata_loaded["model"] == "gemini-embedding-001"
        assert metadata_loaded["dimensions"] == 768
        assert metadata_loaded["num_documents"] == 50
        assert metadata_loaded["dataset"] == "ag_news"

    def test_load_raises_cache_not_found_error(self, tmp_path):
        """
        AC-4: Test CacheNotFoundError raised when cache missing.

        Given: Cache files don't exist
        When: I call load()
        Then: Raises CacheNotFoundError with helpful message
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Try to load non-existent cache
        with pytest.raises(CacheNotFoundError) as exc_info:
            cache.load("nonexistent")

        # Verify error message includes split name and cache directory
        error_message = str(exc_info.value)
        assert "nonexistent" in error_message
        assert str(cache_dir) in error_message

    def test_load_validates_dtype_float32(self, tmp_path):
        """Test loaded embeddings dtype validation."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Manually create cache with wrong dtype
        embeddings_wrong = np.random.rand(10, 768).astype(np.float64)
        np.save(cache_dir / "wrong_embeddings.npy", embeddings_wrong)

        # Create metadata file
        with open(cache_dir / "wrong_metadata.json", 'w') as f:
            json.dump({"model": "test"}, f)

        # Try to load
        with pytest.raises(ValueError, match="float32"):
            cache.load("wrong")


class TestCacheRoundtrip:
    """Test save-then-load roundtrip consistency (AC-4)."""

    def test_roundtrip_preserves_embeddings(self, tmp_path):
        """
        AC-4: Test roundtrip (save then load) preserves embeddings.

        Given: Embeddings generated
        When: I save then load embeddings
        Then: Loaded embeddings exactly match original
        """
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create embeddings
        embeddings_original = np.random.rand(200, 768).astype(np.float32)
        metadata = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 200
        }

        # Save and load
        cache.save(embeddings_original, "roundtrip", metadata)
        embeddings_loaded, metadata_loaded = cache.load("roundtrip")

        # Verify exact match
        assert np.allclose(embeddings_original, embeddings_loaded)
        assert embeddings_loaded.shape == embeddings_original.shape
        assert embeddings_loaded.dtype == embeddings_original.dtype

    def test_roundtrip_preserves_metadata(self, tmp_path):
        """Test roundtrip preserves all metadata fields."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create embeddings with detailed metadata
        embeddings = np.random.rand(100, 768).astype(np.float32)
        metadata_original = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 100,
            "dataset": "ag_news",
            "split": "train",
            "api_calls": 5,
            "estimated_cost": 0.005,
            "batch_size": 20
        }

        # Save and load
        cache.save(embeddings, "metadata_test", metadata_original)
        _, metadata_loaded = cache.load("metadata_test")

        # Verify all metadata fields preserved
        for key in metadata_original:
            assert key in metadata_loaded
            assert metadata_loaded[key] == metadata_original[key]


class TestCacheExistsMethod:
    """Test cache existence checking functionality."""

    def test_exists_returns_true_when_cache_exists(self, tmp_path):
        """Test exists() returns True when cache files exist."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create cache
        embeddings = np.random.rand(10, 768).astype(np.float32)
        cache.save(embeddings, "test", {"model": "test"})

        # Check existence
        assert cache.exists("test") is True

    def test_exists_returns_false_when_cache_missing(self, tmp_path):
        """Test exists() returns False when cache files don't exist."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Check non-existent cache
        assert cache.exists("nonexistent") is False

    def test_exists_returns_false_when_only_embeddings_file_exists(self, tmp_path):
        """Test exists() returns False when only .npy file exists."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create only embeddings file (no metadata)
        embeddings = np.random.rand(10, 768).astype(np.float32)
        np.save(cache_dir / "incomplete_embeddings.npy", embeddings)

        # Check existence (should be False without metadata)
        assert cache.exists("incomplete") is False


class TestCacheClearMethod:
    """Test cache clearing functionality."""

    def test_clear_deletes_cache_files(self, tmp_path):
        """Test clear() deletes both .npy and .json files."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create cache
        embeddings = np.random.rand(10, 768).astype(np.float32)
        cache.save(embeddings, "to_delete", {"model": "test"})

        # Verify files exist
        assert (cache_dir / "to_delete_embeddings.npy").exists()
        assert (cache_dir / "to_delete_metadata.json").exists()

        # Clear cache
        cache.clear("to_delete")

        # Verify files deleted
        assert not (cache_dir / "to_delete_embeddings.npy").exists()
        assert not (cache_dir / "to_delete_metadata.json").exists()

    def test_clear_handles_nonexistent_cache_gracefully(self, tmp_path):
        """Test clear() doesn't error on non-existent cache."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Clear non-existent cache (should not raise error)
        cache.clear("nonexistent")

    def test_clear_only_affects_specified_split(self, tmp_path):
        """Test clear() only deletes specified split, not others."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Create multiple caches
        embeddings = np.random.rand(10, 768).astype(np.float32)
        cache.save(embeddings, "train", {"model": "test"})
        cache.save(embeddings, "test", {"model": "test"})

        # Clear only train cache
        cache.clear("train")

        # Verify train deleted, test remains
        assert not cache.exists("train")
        assert cache.exists("test")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_save_and_load_very_small_embeddings(self, tmp_path):
        """Test cache handles very small embedding arrays."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Single embedding
        embeddings = np.random.rand(1, 768).astype(np.float32)
        cache.save(embeddings, "single", {"model": "test"})

        # Load and verify
        loaded, _ = cache.load("single")
        assert loaded.shape == (1, 768)

    def test_save_and_load_very_large_embeddings(self, tmp_path):
        """Test cache handles large embedding arrays."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Large embedding array
        embeddings = np.random.rand(10000, 768).astype(np.float32)
        cache.save(embeddings, "large", {"model": "test"})

        # Load and verify
        loaded, _ = cache.load("large")
        assert loaded.shape == (10000, 768)

    def test_metadata_with_nested_structures(self, tmp_path):
        """Test cache handles complex nested metadata."""
        cache_dir = tmp_path / "embeddings"
        cache = EmbeddingCache(cache_dir)

        # Complex metadata with nested structures
        embeddings = np.random.rand(10, 768).astype(np.float32)
        metadata = {
            "model": "gemini-embedding-001",
            "config": {
                "batch_size": 100,
                "retry_attempts": 3
            },
            "stats": {
                "mean": 0.5,
                "std": 0.1
            }
        }

        # Save and load
        cache.save(embeddings, "nested", metadata)
        _, loaded_metadata = cache.load("nested")

        # Verify nested structure preserved
        assert loaded_metadata["config"]["batch_size"] == 100
        assert loaded_metadata["stats"]["mean"] == 0.5
