"""Features module for embedding generation and caching."""

from .embedding_service import EmbeddingService, AuthenticationError
from .embedding_cache import EmbeddingCache, CacheNotFoundError

__all__ = [
    "EmbeddingService",
    "AuthenticationError",
    "EmbeddingCache",
    "CacheNotFoundError",
]
