"""
Reranking module for RAG pipeline.

Provides reranking capabilities to improve retrieval quality
using cross-encoders or external APIs like Cohere.
"""

from src.reranking.base_reranker import BaseReranker, rerank
from src.reranking.cohere_reranker import CohereReranker

__all__ = [
    "BaseReranker",
    "rerank",
    "CohereReranker",
]
