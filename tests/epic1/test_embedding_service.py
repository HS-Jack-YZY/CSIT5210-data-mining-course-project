"""
Tests for EmbeddingService class (Story 1.4).

This module tests Gemini API integration, authentication, retry logic,
and embedding generation functionality.

Test Coverage:
    - AC-1: Gemini API Authentication Successful
    - AC-2: Gemini API Error Handling
    - AC-3: Retry Logic Functional
"""

import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tenacity import RetryError

from src.context_aware_multi_agent_system.features import (
    EmbeddingService,
    AuthenticationError
)
from src.context_aware_multi_agent_system.config import Config


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization and setup."""

    def test_initialization_with_valid_api_key(self):
        """Test successful initialization with valid API key."""
        service = EmbeddingService(api_key="test-api-key-12345")

        assert service.client is not None
        assert service.model == "gemini-embedding-001"

    def test_initialization_with_custom_model(self):
        """Test initialization with custom model name."""
        service = EmbeddingService(
            api_key="test-api-key-12345",
            model="custom-model-001"
        )

        assert service.model == "custom-model-001"

    def test_initialization_with_empty_api_key(self):
        """Test initialization fails with empty API key."""
        with pytest.raises(ValueError, match="API key cannot be None or empty"):
            EmbeddingService(api_key="")

    def test_initialization_with_none_api_key(self):
        """Test initialization fails with None API key."""
        with pytest.raises(ValueError, match="API key cannot be None or empty"):
            EmbeddingService(api_key=None)


class TestAuthentication:
    """Test API authentication functionality (AC-1, AC-2)."""

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_successful_authentication(self, mock_client_class):
        """
        AC-1: Test successful API authentication.

        Given: Valid API key in .env
        When: I call EmbeddingService(api_key).test_connection()
        Then: Returns True, test embedding generated, shape (768,), dtype float32
        """
        # Mock the API response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock successful embedding response
        mock_response = MagicMock()
        mock_response.embedding = [0.1] * 768  # 768-dimensional embedding
        mock_client.models.embed_content.return_value = mock_response

        # Test authentication
        service = EmbeddingService(api_key="valid-api-key")
        result = service.test_connection()

        # Verify authentication successful
        assert result is True

        # Verify API was called with "Hello world"
        mock_client.models.embed_content.assert_called_once()
        call_args = mock_client.models.embed_content.call_args
        assert call_args[1]['content'] == "Hello world"

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_authentication_failure_invalid_key(self, mock_client_class):
        """
        AC-2: Test authentication failure with invalid API key.

        Given: Invalid API key
        When: I call EmbeddingService(api_key).test_connection()
        Then: Raises AuthenticationError with helpful message
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock authentication failure
        mock_client.models.embed_content.side_effect = Exception("Invalid API key")

        # Test authentication failure
        service = EmbeddingService(api_key="invalid-api-key")

        with pytest.raises(AuthenticationError) as exc_info:
            service.test_connection()

        # Verify error message includes helpful guidance
        error_message = str(exc_info.value)
        assert "Invalid API key" in error_message
        assert ".env" in error_message

    def test_config_gemini_api_key_missing(self, tmp_path, monkeypatch):
        """
        AC-2: Test Config raises ValueError when GEMINI_API_KEY not found.

        Given: Missing GEMINI_API_KEY in environment
        When: I access config.gemini_api_key
        Then: Raises ValueError with helpful message
        """
        # Remove GEMINI_API_KEY from environment
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Create temporary config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
dataset:
  name: ag_news
  categories: 4
clustering:
  algorithm: kmeans
  n_clusters: 4
  random_state: 42
  max_iter: 300
  init: k-means++
embedding:
  model: gemini-embedding-001
  batch_size: 100
  cache_dir: data/embeddings
  output_dimensionality: 768
classification:
  method: cosine_similarity
  threshold: 0.7
metrics:
  cost_per_1M_tokens_under_200k: 0.075
  cost_per_1M_tokens_over_200k: 0.15
  target_cost_reduction: 0.90
""")

        # Try to access API key
        config = Config(config_path=str(config_file))

        with pytest.raises(ValueError, match="GEMINI_API_KEY not found"):
            _ = config.gemini_api_key

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_api_key_never_exposed_in_logs(self, mock_client_class, caplog):
        """
        AC-2: Test API key value never exposed in logs or error messages.

        Given: API authentication fails
        When: Error is logged
        Then: API key value is masked (not shown in logs)
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock authentication failure
        mock_client.models.embed_content.side_effect = Exception("Unauthorized")

        # Test with actual API key that should NOT appear in logs
        secret_key = "sk-secret-api-key-do-not-expose"
        service = EmbeddingService(api_key=secret_key)

        try:
            service.test_connection()
        except AuthenticationError:
            pass

        # Verify secret key never appears in logs
        for record in caplog.records:
            assert secret_key not in record.message


class TestEmbeddingGeneration:
    """Test single and batch embedding generation (AC-1)."""

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_single_embedding_generation(self, mock_client_class):
        """
        AC-1: Test single embedding generation.

        Given: Valid API credentials
        When: I generate embedding for single document
        Then: Embedding shape is (768,) and dtype is float32
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock embedding response
        mock_response = MagicMock()
        mock_response.embedding = [0.5] * 768
        mock_client.models.embed_content.return_value = mock_response

        # Generate embedding
        service = EmbeddingService(api_key="test-api-key")
        embedding = service.generate_embedding("Hello world")

        # Verify shape and dtype
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_batch_embedding_generation(self, mock_client_class):
        """
        AC-1: Test batch embedding generation.

        Given: Valid API credentials
        When: I generate embeddings for 100 documents
        Then: Embeddings shape is (100, 768) and dtype is float32
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock batch embedding response (returns list of embeddings)
        def mock_embed_content(model, contents):
            mock_response = MagicMock()
            mock_embeddings = []
            for _ in contents:
                mock_emb = MagicMock()
                mock_emb.values = [0.5] * 768
                mock_embeddings.append(mock_emb)
            mock_response.embeddings = mock_embeddings
            return mock_response

        mock_client.models.embed_content.side_effect = mock_embed_content

        # Generate batch embeddings
        service = EmbeddingService(api_key="test-api-key")
        documents = [f"Document {i}" for i in range(100)]
        embeddings = service.generate_batch(documents, batch_size=50)

        # Verify shape and dtype
        assert embeddings.shape == (100, 768)
        assert embeddings.dtype == np.float32

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_embedding_shape_validation(self, mock_client_class):
        """Test embedding shape validation catches invalid dimensions."""
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock invalid embedding response (wrong dimension)
        mock_response = MagicMock()
        mock_response.embedding = [0.5] * 512  # Wrong: should be 768
        mock_client.models.embed_content.return_value = mock_response

        # Generate embedding
        service = EmbeddingService(api_key="test-api-key")

        with pytest.raises(ValueError, match="Invalid embedding shape"):
            service.generate_embedding("Test")


class TestRetryLogic:
    """Test retry logic with exponential backoff (AC-3)."""

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_retry_on_network_failure(self, mock_client_class):
        """
        AC-3: Test retry logic activates on network failure.

        Given: Network error occurs during API call
        When: Retry logic activates
        Then: Up to 3 retry attempts made with exponential backoff
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock network failures then success
        mock_response = MagicMock()
        mock_response.embedding = [0.5] * 768

        mock_client.models.embed_content.side_effect = [
            Exception("Connection timeout"),  # Attempt 1: fail
            Exception("Connection timeout"),  # Attempt 2: fail
            mock_response  # Attempt 3: success
        ]

        # Generate embedding with retries
        service = EmbeddingService(api_key="test-api-key")
        embedding = service.generate_embedding("Test")

        # Verify retry attempts
        assert mock_client.models.embed_content.call_count == 3
        assert embedding.shape == (768,)

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_retry_logging(self, mock_client_class, caplog):
        """
        AC-3: Test retry attempts are logged.

        Given: Network errors occur
        When: Retry logic activates
        Then: Each retry attempt logged with attempt number
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock network failures then success
        mock_response = MagicMock()
        mock_response.embedding = [0.5] * 768

        mock_client.models.embed_content.side_effect = [
            Exception("Connection timeout"),
            mock_response
        ]

        # Generate embedding with retries
        service = EmbeddingService(api_key="test-api-key")

        with caplog.at_level("WARNING"):
            embedding = service.generate_embedding("Test")

        # Verify retry was logged
        assert any("API call failed" in record.message for record in caplog.records)

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_max_retry_attempts_exceeded(self, mock_client_class):
        """
        AC-3: Test failure after 3 retry attempts.

        Given: Network errors persist
        When: All 3 retry attempts fail
        Then: Exception raised with context
        """
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock persistent network failures
        mock_client.models.embed_content.side_effect = Exception("Connection timeout")

        # Attempt to generate embedding
        service = EmbeddingService(api_key="test-api-key")

        with pytest.raises(Exception):
            service.generate_embedding("Test")

        # Verify 3 attempts were made
        assert mock_client.models.embed_content.call_count == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_empty_document_embedding(self, mock_client_class):
        """Test embedding generation for empty string."""
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock embedding response
        mock_response = MagicMock()
        mock_response.embedding = [0.0] * 768
        mock_client.models.embed_content.return_value = mock_response

        # Generate embedding for empty string
        service = EmbeddingService(api_key="test-api-key")
        embedding = service.generate_embedding("")

        # Verify embedding generated
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_batch_with_single_document(self, mock_client_class):
        """Test batch processing with single document."""
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock batch embedding response
        def mock_embed_content(model, contents):
            mock_response = MagicMock()
            mock_embeddings = []
            for _ in contents:
                mock_emb = MagicMock()
                mock_emb.values = [0.5] * 768
                mock_embeddings.append(mock_emb)
            mock_response.embeddings = mock_embeddings
            return mock_response

        mock_client.models.embed_content.side_effect = mock_embed_content

        # Generate batch with single document
        service = EmbeddingService(api_key="test-api-key")
        embeddings = service.generate_batch(["Single document"])

        # Verify shape
        assert embeddings.shape == (1, 768)
        assert embeddings.dtype == np.float32

    @patch('src.context_aware_multi_agent_system.features.embedding_service.genai.Client')
    def test_batch_with_custom_batch_size(self, mock_client_class):
        """Test batch processing with custom batch size."""
        # Mock the API client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock batch embedding response
        def mock_embed_content(model, contents):
            mock_response = MagicMock()
            mock_embeddings = []
            for _ in contents:
                mock_emb = MagicMock()
                mock_emb.values = [0.5] * 768
                mock_embeddings.append(mock_emb)
            mock_response.embeddings = mock_embeddings
            return mock_response

        mock_client.models.embed_content.side_effect = mock_embed_content

        # Generate batch with custom batch size
        service = EmbeddingService(api_key="test-api-key")
        documents = [f"Doc {i}" for i in range(10)]
        embeddings = service.generate_batch(documents, batch_size=3)

        # Verify shape (should process in 4 batches: 3+3+3+1)
        assert embeddings.shape == (10, 768)
        assert embeddings.dtype == np.float32

        # Verify number of API calls (4 batches using Batch API)
        assert mock_client.models.embed_content.call_count == 4
