"""
Ingestion module for RAG pipeline.

Handles loading documents from various sources (PDFs, URLs, text files)
and splitting them into processable units.
"""

from src.ingestion.loader import DocumentLoader, load_pdfs, load_documents
from src.ingestion.splitter import DocumentSplitter, split_document, batch_split

__all__ = [
    "DocumentLoader",
    "load_pdfs",
    "load_documents",
    "DocumentSplitter",
    "split_document",
    "batch_split",
]
