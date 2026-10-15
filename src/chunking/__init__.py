"""
Chunking module for RAG pipeline.

Handles document chunking using Docling's HybridChunker.
"""

from src.chunking.chunker import (
    DocumentChunker,
    chunk_document,
    batch_chunk,
    load_docling_json,
)

__all__ = [
    "DocumentChunker",
    "chunk_document",
    "batch_chunk",
    "load_docling_json",
]
