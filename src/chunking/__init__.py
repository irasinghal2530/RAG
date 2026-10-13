"""
Chunking module for RAG pipeline.

Handles document chunking strategies with configurable overlap and metadata tagging.
"""

from src.chunking.chunker import (
    DocumentChunker,
    chunk_document,
    batch_chunk,
)
from src.chunking.overlap_manager import OverlapManager
from src.chunking.metadata_tagger import MetadataTagger, tag_chunks

__all__ = [
    "DocumentChunker",
    "chunk_document",
    "batch_chunk",
    "OverlapManager",
    "MetadataTagger",
    "tag_chunks",
]
