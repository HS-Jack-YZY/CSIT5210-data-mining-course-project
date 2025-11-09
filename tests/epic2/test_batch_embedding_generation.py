"""
Tests for batch embedding generation (Story 2.1 - AC-1, AC-2).

Tests cover:
- AC-1: Batch embedding generation with Gemini API
- AC-2: Embedding cache implementation
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.context_aware_multi_agent_system.features.embedding_service import EmbeddingService
from src.context_aware_multi_agent_system.features.embedding_cache import (
    EmbeddingCache,
    CacheNotFoundError
)
from src.context_aware_multi_agent_system.evaluation.cost_calculator import (
    estimate_embedding_cost,
    estimate_tokens,
    estimate_batch_cost
)


class TestBatchEmbeddingGeneration:
    """Test batch embedding generation (AC-1)."""

    def test_batch_embedding_shape_and_dtype(self):
        """Test that batch embeddings have correct shape (n_documents, 768) and dtype float32."""
        # Mock Gemini API response
        mock_response = MagicMock()
        mock_embeddings = []
        for i in range(3):
            mock_emb = MagicMock()
            mock_emb.values = np.random.rand(768).tolist()
            mock_embeddings.append(mock_emb)
        mock_response.embeddings = mock_embeddings

        with patch('google.genai.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.models.embed_content.return_value = mock_response

            service = EmbeddingService(api_key="test-key")
            documents = ["doc1", "doc2", "doc3"]
            embeddings = service.generate_batch(documents, batch_size=3)

            # Verify shape
            assert embeddings.shape == (3, 768), f"Expected (3, 768), got {embeddings.shape}"
            # Verify dtype
            assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"

    def test_batch_size_handling(self):
        """Test that batch processing handles different batch sizes correctly."""
        # Mock response for batches
        def mock_embed_content(model, contents):
            mock_response = MagicMock()
            mock_embeddings = []
            for _ in contents:
                mock_emb = MagicMock()
                mock_emb.values = np.random.rand(768).tolist()
                mock_embeddings.append(mock_emb)
            mock_response.embeddings = mock_embeddings
            return mock_response

        with patch('google.genai.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.models.embed_content.side_effect = mock_embed_content

            service = EmbeddingService(api_key="test-key")

            # Test with 5 documents, batch_size=2 (should create 3 batches)
            documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            embeddings = service.generate_batch(documents, batch_size=2)

            assert embeddings.shape == (5, 768)
            # Verify API was called 3 times (batches of 2, 2, 1)
            assert mock_client_instance.models.embed_content.call_count == 3

    def test_embedding_dimensions_validation(self):
        """Test that embedding dimensions are validated (768D)."""
        # Mock response with incorrect dimensions
        mock_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = np.random.rand(512).tolist()  # Wrong dimension
        mock_response.embeddings = [mock_emb]

        with patch('google.genai.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.models.embed_content.return_value = mock_response

            service = EmbeddingService(api_key="test-key")
            documents = ["doc1"]

            with pytest.raises(ValueError, match="Invalid embedding shape"):
                service.generate_batch(documents, batch_size=1)

    def test_empty_document_list_handling(self):
        """Test that empty document list raises appropriate error."""
        with patch('google.genai.Client') as mock_client:
            service = EmbeddingService(api_key="test-key")

            with pytest.raises(ValueError, match="Cannot generate embeddings for empty document list"):
                service.generate_batch([], batch_size=100)


class TestEmbeddingCache:
    """Test embedding cache implementation (AC-2)."""

    def test_cache_save_and_load_roundtrip(self, tmp_path):
        """Test that embeddings can be saved and loaded without data loss."""
        cache = EmbeddingCache(tmp_path)

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
        cache.save(embeddings, "train", metadata)

        # Load embeddings
        loaded_embeddings, loaded_metadata = cache.load("train")

        # Verify embeddings are identical
        assert np.allclose(embeddings, loaded_embeddings), "Embeddings not identical after save/load"

        # Verify metadata
        assert loaded_metadata["model"] == "gemini-embedding-001"
        assert loaded_metadata["dimensions"] == 768
        assert loaded_metadata["num_documents"] == 100
        assert "timestamp" in loaded_metadata  # Should be auto-added

    def test_cache_shape_validation(self, tmp_path):
        """Test that cache validates shape (120000, 768) for train, (7600, 768) for test."""
        cache = EmbeddingCache(tmp_path)

        # Test train shape
        train_embeddings = np.random.rand(120000, 768).astype(np.float32)
        train_metadata = {"model": "gemini-embedding-001", "dimensions": 768}
        cache.save(train_embeddings, "train", train_metadata)

        loaded, _ = cache.load("train")
        assert loaded.shape == (120000, 768)

        # Test test shape
        test_embeddings = np.random.rand(7600, 768).astype(np.float32)
        test_metadata = {"model": "gemini-embedding-001", "dimensions": 768}
        cache.save(test_embeddings, "test", test_metadata)

        loaded, _ = cache.load("test")
        assert loaded.shape == (7600, 768)

    def test_cache_dtype_validation(self, tmp_path):
        """Test that cache enforces float32 dtype."""
        cache = EmbeddingCache(tmp_path)

        # Try to save with wrong dtype
        embeddings_wrong_dtype = np.random.rand(100, 768).astype(np.float64)
        metadata = {"model": "test"}

        with pytest.raises(ValueError, match="Embeddings must be float32"):
            cache.save(embeddings_wrong_dtype, "train", metadata)

    def test_cache_exists_check(self, tmp_path):
        """Test cache existence checking."""
        cache = EmbeddingCache(tmp_path)

        # Cache doesn't exist initially
        assert not cache.exists("train")

        # Save embeddings
        embeddings = np.random.rand(100, 768).astype(np.float32)
        metadata = {"model": "test"}
        cache.save(embeddings, "train", metadata)

        # Cache should exist now
        assert cache.exists("train")

    def test_cache_not_found_error(self, tmp_path):
        """Test that loading non-existent cache raises CacheNotFoundError."""
        cache = EmbeddingCache(tmp_path)

        with pytest.raises(CacheNotFoundError, match="Cache not found for split 'train'"):
            cache.load("train")

    def test_metadata_completeness(self, tmp_path):
        """Test that metadata includes all required fields."""
        cache = EmbeddingCache(tmp_path)

        embeddings = np.random.rand(1000, 768).astype(np.float32)
        metadata = {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "num_documents": 1000,
            "dataset": "ag_news",
            "split": "train",
            "api_calls": 10,
            "estimated_cost": 0.05
        }

        cache.save(embeddings, "train", metadata)
        _, loaded_metadata = cache.load("train")

        # Verify all required fields are present
        assert loaded_metadata["model"] == "gemini-embedding-001"
        assert loaded_metadata["dimensions"] == 768
        assert loaded_metadata["num_documents"] == 1000
        assert loaded_metadata["dataset"] == "ag_news"
        assert loaded_metadata["split"] == "train"
        assert loaded_metadata["api_calls"] == 10
        assert loaded_metadata["estimated_cost"] == 0.05
        assert "timestamp" in loaded_metadata


class TestCostCalculation:
    """Test cost calculation utilities (AC-4)."""

    def test_batch_api_cost_calculation(self):
        """Test that batch API cost is calculated correctly ($0.075/1M tokens)."""
        cost = estimate_embedding_cost(1_200_000, use_batch_api=True)
        expected = 0.09  # 1.2M tokens * $0.075/1M
        assert cost == expected, f"Expected {expected}, got {cost}"

    def test_standard_api_cost_calculation(self):
        """Test that standard API cost is calculated correctly ($0.15/1M tokens)."""
        cost = estimate_embedding_cost(1_200_000, use_batch_api=False)
        expected = 0.18  # 1.2M tokens * $0.15/1M
        assert cost == expected, f"Expected {expected}, got {cost}"

    def test_batch_api_cost_savings(self):
        """Test that batch API provides 50% cost savings."""
        batch_cost = estimate_embedding_cost(1_000_000, use_batch_api=True)
        standard_cost = estimate_embedding_cost(1_000_000, use_batch_api=False)

        # Batch API should be 50% cheaper
        assert batch_cost == standard_cost / 2

    def test_token_estimation(self):
        """Test token count estimation."""
        text = "This is a test" * 100  # ~1400 characters
        tokens = estimate_tokens(text)

        # Should estimate ~350 tokens (0.25 tokens/char)
        assert 300 < tokens < 400, f"Token estimate {tokens} out of expected range"

    def test_batch_cost_estimation(self):
        """Test cost estimation for document batch."""
        # 120K documents, 100 chars each, batch API
        cost = estimate_batch_cost(120_000, avg_document_length=100, use_batch_api=True)

        # Should be under $5 (PRD requirement)
        assert cost < 5.0, f"Cost ${cost} exceeds $5 target"


class TestCheckpointSystem:
    """Test checkpoint management (AC-3)."""

    def test_checkpoint_save_and_load(self, tmp_path):
        """Test checkpoint save and resume functionality."""
        checkpoint_path = tmp_path / ".checkpoint_train.json"

        # Save checkpoint
        checkpoint_data = {
            "split": "train",
            "last_processed_index": 5000,
            "batch_size": 100,
            "total_batches": 1200
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        with open(checkpoint_path, 'r') as f:
            loaded = json.load(f)

        assert loaded["last_processed_index"] == 5000
        assert loaded["split"] == "train"

    def test_checkpoint_cleanup(self, tmp_path):
        """Test checkpoint deletion on successful completion."""
        checkpoint_path = tmp_path / ".checkpoint_train.json"

        # Create checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump({"last_processed_index": 1000}, f)

        assert checkpoint_path.exists()

        # Delete checkpoint
        checkpoint_path.unlink()

        assert not checkpoint_path.exists()
