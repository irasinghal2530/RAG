"""
Retrieval module for RAG pipeline.

Handles vector similarity search using ChromaDB.
"""

from src.retrieval.retriever import Retriever, VectorStore

__all__ = [
    "Retriever",
    "VectorStore",
]
