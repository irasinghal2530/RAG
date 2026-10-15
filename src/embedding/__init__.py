"""
Embedding module for RAG pipeline.

Handles embedding generation using sentence-transformers.
"""

from src.embedding.base_embedder import BaseEmbedder, embed_chunks, batch_embed

__all__ = [
    "BaseEmbedder",
    "embed_chunks",
    "batch_embed",
]
