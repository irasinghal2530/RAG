"""
Reranking module for RAG pipeline.

Rescores retrieved chunks with a cross-encoder.
"""

from src.reranking.base_reranker import CrossEncoderReranker

__all__ = [
    "CrossEncoderReranker",
]
