"""
Embedding module for RAG pipeline.

Handles embedding generation with support for multiple providers
and caching for efficiency.
"""

from src.embedding.base_embedder import BaseEmbedder, embed_chunks, batch_embed
from src.embedding.openai_embedder import OpenAIEmbedder
from src.embedding.embedding_cache import EmbeddingCache

__all__ = [
    "BaseEmbedder",
    "embed_chunks",
    "batch_embed",
    "OpenAIEmbedder",
    "EmbeddingCache",
]
