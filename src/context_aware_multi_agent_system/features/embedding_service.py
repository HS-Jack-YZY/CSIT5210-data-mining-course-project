"""
Embedding service for generating semantic embeddings using Google Gemini API.

This module provides a robust interface to the Gemini Embedding API with
authentication, retry logic, and batch processing capabilities.

Classes:
    EmbeddingService: Generate embeddings via Gemini API with retry logic
    AuthenticationError: Custom exception for API authentication failures
"""

import logging
from typing import List
import numpy as np
from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_not_exception_type,
    before_sleep_log,
    after_log,
    RetryError
)

# Configure logging
logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """
    Exception raised when Gemini API authentication fails.

    This exception is raised when the API key is invalid, missing, or
    authentication fails for any reason.

    Usage:
        raise AuthenticationError("Invalid API key. Copy .env.example to .env and add your API key.")
    """

    def __init__(self, message: str):
        """
        Initialize AuthenticationError with helpful message.

        Args:
            message: Error description with troubleshooting guidance
        """
        super().__init__(message)


class EmbeddingService:
    """
    Service for generating embeddings using Google Gemini API.

    This service provides methods for authentication testing, single embedding
    generation, and batch processing with automatic retry logic for resilience.

    Features:
        - API authentication verification
        - Single embedding generation (768 dimensions)
        - Batch embedding generation with configurable batch size
        - Automatic retry with exponential backoff (3 attempts, 4-16s delays)
        - Shape and dtype validation (768, float32)
        - Comprehensive logging with emoji prefixes

    Usage:
        >>> service = EmbeddingService(api_key="your-api-key")
        >>> service.test_connection()  # Verify authentication
        True
        >>> embedding = service.generate_embedding("Hello world")
        >>> embedding.shape
        (768,)
        >>> embeddings = service.generate_batch(["doc1", "doc2"], batch_size=100)
        >>> embeddings.shape
        (2, 768)
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        """
        Initialize EmbeddingService with Gemini API credentials.

        Args:
            api_key: Gemini API key (from .env GEMINI_API_KEY)
            model: Gemini embedding model name (default: gemini-embedding-001)

        Raises:
            ValueError: If api_key is None or empty string

        Example:
            >>> from context_aware_multi_agent_system.config import Config
            >>> config = Config()
            >>> service = EmbeddingService(config.gemini_api_key)
        """
        if not api_key:
            raise ValueError("API key cannot be None or empty")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        logger.info(f"📊 Initialized Gemini API client with model: {model}")

    def test_connection(self) -> bool:
        """
        Test API authentication by generating a test embedding.

        Generates embedding for "Hello world" to verify API key validity
        and connection. Validates embedding shape (768,) and dtype (float32).

        Returns:
            True if authentication successful

        Raises:
            AuthenticationError: If API key is invalid or authentication fails

        Example:
            >>> service = EmbeddingService(api_key)
            >>> service.test_connection()
            True
        """
        try:
            # Generate test embedding
            logger.info("📡 Testing API authentication...")
            embedding = self.generate_embedding("Hello world")

            # Validate embedding shape and dtype
            if embedding.shape != (768,):
                raise AuthenticationError(
                    f"Unexpected embedding shape: {embedding.shape}. "
                    f"Expected (768,). API may have changed."
                )

            if embedding.dtype != np.float32:
                raise AuthenticationError(
                    f"Unexpected embedding dtype: {embedding.dtype}. "
                    f"Expected float32. API may have changed."
                )

            logger.info("✅ API authentication successful")
            return True

        except RetryError as e:
            # Extract the original exception from RetryError
            if e.last_attempt.exception():
                original_error = e.last_attempt.exception()
                error_msg = str(original_error)
            else:
                error_msg = str(e)

            if "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise AuthenticationError(
                    f"Invalid API key. {error_msg}\n"
                    "Copy .env.example to .env and add your API key."
                )
            else:
                raise AuthenticationError(
                    f"API authentication failed: {error_msg}\n"
                    "Copy .env.example to .env and add your API key."
                )

        except Exception as e:
            # Handle authentication failures
            error_msg = str(e)

            if "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise AuthenticationError(
                    f"Invalid API key. {error_msg}\n"
                    "Copy .env.example to .env and add your API key."
                )
            else:
                raise AuthenticationError(
                    f"API authentication failed: {error_msg}\n"
                    "Copy .env.example to .env and add your API key."
                )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=16),
        retry=retry_if_not_exception_type(ValueError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text document.

        Uses Gemini API to generate 768-dimensional semantic embedding.
        Automatically retries on failure with exponential backoff (3 attempts,
        4-16s delays).

        Args:
            text: Input text to embed

        Returns:
            Embedding array with shape (768,) and dtype float32

        Raises:
            Exception: If all retry attempts fail

        Example:
            >>> embedding = service.generate_embedding("Hello world")
            >>> embedding.shape
            (768,)
            >>> embedding.dtype
            dtype('float32')
        """
        try:
            # Call Gemini API
            response = self.client.models.embed_content(
                model=self.model,
                content=text
            )

            # Extract embedding from response
            if hasattr(response, 'embedding'):
                embedding = np.array(response.embedding, dtype=np.float32)
            elif isinstance(response, dict) and 'embedding' in response:
                embedding = np.array(response['embedding'], dtype=np.float32)
            else:
                # Handle unexpected response format
                raise ValueError(f"Unexpected API response format: {type(response)}")

            # Validate embedding shape
            if embedding.shape != (768,):
                raise ValueError(
                    f"Invalid embedding shape: {embedding.shape}. Expected (768,)"
                )

            return embedding

        except Exception as e:
            logger.warning(f"⚠️ API call failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=16),
        retry=retry_if_not_exception_type(ValueError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )
    def generate_batch(self, documents: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple documents using Gemini Batch API.

        Uses Gemini's batch embedding API for cost efficiency ($0.075/1M tokens
        vs $0.15/1M for standard API). Processes documents in batches and
        automatically retries on failure with exponential backoff.

        Args:
            documents: List of text documents to embed
            batch_size: Number of documents per API call (default: 100)

        Returns:
            Embedding array with shape (n_documents, 768) and dtype float32

        Raises:
            ValueError: If embedding validation fails
            Exception: If all retry attempts fail

        Example:
            >>> docs = ["doc1", "doc2", "doc3"]
            >>> embeddings = service.generate_batch(docs, batch_size=100)
            >>> embeddings.shape
            (3, 768)
            >>> embeddings.dtype
            dtype('float32')
        """
        if not documents:
            raise ValueError("Cannot generate embeddings for empty document list")

        embeddings_list = []
        n_batches = (len(documents) + batch_size - 1) // batch_size

        logger.info(f"📊 Processing {len(documents)} documents in {n_batches} batches")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"🔄 Processing batch {batch_num}/{n_batches} (documents {i}-{i+len(batch)-1})")

            try:
                # Call Gemini Batch API with list of contents
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch
                )

                # Extract embeddings from batch response
                # Response should have embeddings list matching input batch size
                if hasattr(response, 'embeddings'):
                    batch_embeddings = response.embeddings
                elif isinstance(response, dict) and 'embeddings' in response:
                    batch_embeddings = response['embeddings']
                else:
                    raise ValueError(f"Unexpected batch API response format: {type(response)}")

                # Convert each embedding to numpy array
                for idx, emb_response in enumerate(batch_embeddings):
                    if hasattr(emb_response, 'values'):
                        embedding = np.array(emb_response.values, dtype=np.float32)
                    elif isinstance(emb_response, (list, np.ndarray)):
                        embedding = np.array(emb_response, dtype=np.float32)
                    else:
                        raise ValueError(f"Unexpected embedding format at index {idx}: {type(emb_response)}")

                    # Validate embedding shape
                    if embedding.shape != (768,):
                        raise ValueError(
                            f"Invalid embedding shape at index {idx}: {embedding.shape}. Expected (768,)"
                        )

                    embeddings_list.append(embedding)

            except Exception as e:
                logger.warning(f"⚠️ Batch API call failed: {e}")
                raise

        # Concatenate all embeddings
        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Validate final shape
        expected_shape = (len(documents), 768)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Invalid batch embeddings shape: {embeddings.shape}. "
                f"Expected {expected_shape}"
            )

        logger.info(f"✅ Successfully generated {len(documents)} embeddings")
        return embeddings
