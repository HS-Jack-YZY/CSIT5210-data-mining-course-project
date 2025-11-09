"""
Cost calculation utilities for Gemini API usage.

This module provides cost estimation functions for embedding generation
and other API operations using Google Gemini API pricing.

Functions:
    estimate_embedding_cost: Calculate embedding generation cost
    estimate_tokens: Estimate token count for text
"""

from typing import Union


def estimate_embedding_cost(
    num_tokens: int,
    use_batch_api: bool = True
) -> float:
    """
    Estimate cost for embedding generation using Gemini API.

    Uses official Gemini API pricing for embedding generation.
    Batch API provides 50% cost savings over standard API.

    Args:
        num_tokens: Total number of tokens to embed
        use_batch_api: Whether to use Batch API (default: True)

    Returns:
        Estimated cost in USD with 4 decimal places

    Example:
        >>> # Batch API cost for 1.2M tokens
        >>> cost = estimate_embedding_cost(1_200_000, use_batch_api=True)
        >>> cost
        0.09
        >>> # Standard API cost for 1.2M tokens
        >>> cost = estimate_embedding_cost(1_200_000, use_batch_api=False)
        >>> cost
        0.18
    """
    # Gemini API pricing (USD per 1M tokens)
    BATCH_API_COST = 0.075  # $0.075 per 1M tokens
    STANDARD_API_COST = 0.15  # $0.15 per 1M tokens

    # Select pricing based on API type
    cost_per_million = BATCH_API_COST if use_batch_api else STANDARD_API_COST

    # Calculate total cost
    cost = (num_tokens / 1_000_000) * cost_per_million

    # Round to 4 decimal places
    return round(cost, 4)


def estimate_tokens(text: str, avg_tokens_per_char: float = 0.25) -> int:
    """
    Estimate token count for text.

    Uses average ratio of tokens to characters. Gemini tokenization
    typically produces ~4 characters per token (0.25 tokens/char).

    Args:
        text: Input text to estimate tokens for
        avg_tokens_per_char: Average tokens per character (default: 0.25)

    Returns:
        Estimated token count

    Example:
        >>> text = "This is a test document with some text."
        >>> tokens = estimate_tokens(text)
        >>> tokens
        10
    """
    return int(len(text) * avg_tokens_per_char)


def estimate_batch_cost(
    num_documents: int,
    avg_document_length: int = 100,
    use_batch_api: bool = True
) -> float:
    """
    Estimate total cost for embedding a batch of documents.

    Combines token estimation with cost calculation for convenience.

    Args:
        num_documents: Number of documents to embed
        avg_document_length: Average characters per document (default: 100)
        use_batch_api: Whether to use Batch API (default: True)

    Returns:
        Estimated cost in USD with 4 decimal places

    Example:
        >>> # Cost for 120K documents, 100 chars each
        >>> cost = estimate_batch_cost(120_000, avg_document_length=100)
        >>> cost
        2.25
    """
    # Estimate total tokens
    total_chars = num_documents * avg_document_length
    total_tokens = estimate_tokens("x" * total_chars)

    # Calculate cost
    return estimate_embedding_cost(total_tokens, use_batch_api)
