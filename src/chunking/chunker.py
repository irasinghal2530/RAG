"""
Document chunker for RAG pipeline.

Chunks DoclingDocument objects into smaller, retrieval-ready pieces.

Usage:
    from src.chunking.chunker import DocumentChunker
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(document, document_id="DOC-001")
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from docling.chunking import HybridChunker
from docling_core.types import DoclingDocument

from src.config import load_document_titles, load_retriever_config


class DocumentChunker:
    """Chunks documents into retrieval-ready pieces."""

    def __init__(
        self,
        merge_peers: Optional[bool] = None,
    ):
        """
        Initialize the document chunker.

        Chunk size is set by HybridChunker's tokenizer (all-MiniLM-L6-v2 by
        default). Changing it renumbers chunk ids, which invalidates the
        relevant_chunk_ids labels in the eval sets.

        Args:
            merge_peers: Merge undersized peer chunks (default: retriever_config.yaml).
        """
        processing_cfg = load_retriever_config().get("processing", {})
        self.merge_peers = (
            processing_cfg.get("merge_peers", True) if merge_peers is None else merge_peers
        )
        self._chunker = None
    
    @property
    def chunker(self) -> HybridChunker:
        """Lazy-load the hybrid chunker."""
        if self._chunker is None:
            self._chunker = HybridChunker(merge_peers=self.merge_peers)
        return self._chunker
    
    def _extract_page_info(self, chunk) -> tuple:
        """Extract page numbers and content type from chunk metadata."""
        page_start = None
        page_end = None
        content_type = "text"
        
        if chunk.meta.doc_items:
            pages = []
            content_types = set()
            
            for item in chunk.meta.doc_items:
                # Get page numbers
                if hasattr(item, "prov") and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, "page_no") and prov.page_no is not None:
                            pages.append(prov.page_no)
                
                # Get content type (DocItemLabel values: "table", "picture",
                # "list_item", "section_header", "title", ...)
                if hasattr(item, "label"):
                    content_types.add(str(getattr(item.label, "value", item.label)).lower())

            if pages:
                page_start = min(pages)
                page_end = max(pages)

            # Determine content type
            if content_types:
                if "table" in content_types:
                    content_type = "table"
                elif "picture" in content_types:
                    content_type = "image"
                elif "list_item" in content_types:
                    content_type = "list"
                elif content_types <= {"section_header", "title"}:
                    content_type = "heading"
        
        return page_start, page_end, content_type

    def _document_title(self, document: DoclingDocument, document_id: str) -> str:
        """Title from config/document_titles.yaml, else the first heading."""
        title = load_document_titles().get(document_id)
        if title:
            return title
        for item in document.texts:
            if str(getattr(item, "label", "")).lower().endswith(("title", "section_header")):
                return item.text
        return document_id

    def chunk_document(
        self,
        document: DoclingDocument,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into retrieval-ready pieces.
        
        Args:
            document: A DoclingDocument object.
            document_id: Identifier for the document.
            
        Returns:
            list: List of chunk dictionaries.
        """
        chunks = []
        title = self._document_title(document, document_id)

        for index, chunk in enumerate(self.chunker.chunk(document)):
            page_start, page_end, content_type = self._extract_page_info(chunk)

            chunks.append({
                "chunk_id": f"{document_id}_chunk_{index + 1:03d}",
                "document_id": document_id,
                "document_title": title,
                "text": chunk.text,
                # Headings alone don't say which product a chunk is about
                # (several brochures share sections like "Enclosure").
                "contextualized_text": f"{title}\n{self.chunker.contextualize(chunk)}",
                "headings": chunk.meta.headings,
                "page_start": page_start,
                "page_end": page_end,
                "content_type": content_type,
            })
        
        return chunks
    
    def save_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        output_dir: Path,
    ) -> Path:
        """
        Save chunks to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries.
            document_id: Document identifier.
            output_dir: Directory to save chunks.
            
        Returns:
            Path: Path to saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{document_id}.json"
        
        output_path.write_text(
            json.dumps({
                "document_id": document_id,
                "total_chunks": len(chunks),
                "chunks": chunks,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        return output_path


def chunk_document(
    document: DoclingDocument,
    document_id: str,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk a single document.
    
    Args:
        document: DoclingDocument to chunk.
        document_id: Document identifier.
        **kwargs: Additional arguments for DocumentChunker.
        
    Returns:
        list: List of chunks.
    """
    chunker = DocumentChunker(**kwargs)
    return chunker.chunk_document(document, document_id)


def batch_chunk(
    documents: List[tuple],
    output_dir: Path,
    **kwargs,
) -> int:
    """
    Chunk multiple documents and save to disk.
    
    Args:
        documents: List of (pdf_path, DoclingDocument) tuples.
        output_dir: Directory to save chunk files.
        **kwargs: Additional arguments for DocumentChunker.
        
    Returns:
        int: Total number of chunks created.
    """
    chunker = DocumentChunker(**kwargs)
    output_dir = Path(output_dir)
    total_chunks = 0
    
    for pdf_path, document in documents:
        document_id = pdf_path.stem
        chunks = chunker.chunk_document(document, document_id)
        chunker.save_chunks(chunks, document_id, output_dir)
        
        print(f"Chunked: {pdf_path.name} -> {len(chunks)} chunks")
        total_chunks += len(chunks)
    
    return total_chunks


def load_docling_json(json_path: Path) -> DoclingDocument:
    """Load a DoclingDocument from a saved JSON file."""
    return DoclingDocument.model_validate_json(
        json_path.read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    # Demo: chunk from saved JSON files
    JSON_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "docling"
    CHUNKS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks"
    
    if JSON_DIR.exists():
        chunker = DocumentChunker()
        json_files = sorted(JSON_DIR.glob("*.json"))
        
        total = 0
        for json_path in json_files:
            document = load_docling_json(json_path)
            chunks = chunker.chunk_document(document, json_path.stem)
            chunker.save_chunks(chunks, json_path.stem, CHUNKS_DIR)
            print(f"Chunked: {json_path.name} -> {len(chunks)} chunks")
            total += len(chunks)
        
        print(f"\nTotal chunks: {total}")
    else:
        print(f"Directory not found: {JSON_DIR}")
