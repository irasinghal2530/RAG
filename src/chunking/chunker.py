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


class DocumentChunker:
    """Chunks documents into retrieval-ready pieces."""
    
    def __init__(
        self,
        merge_peers: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the document chunker.
        
        Args:
            merge_peers: Merge peer elements in chunking.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks.
        """
        self.merge_peers = merge_peers
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
                
                # Get content type
                if hasattr(item, "label"):
                    content_types.add(item.label)
            
            if pages:
                page_start = min(pages)
                page_end = max(pages)
            
            # Determine content type
            if content_types:
                if "table" in content_types or "Table" in content_types:
                    content_type = "table"
                elif "picture" in content_types or "Picture" in content_types:
                    content_type = "image"
                elif "list" in content_types or "List" in content_types:
                    content_type = "list"
                elif any("heading" in t.lower() for t in content_types if isinstance(t, str)):
                    content_type = "heading"
        
        return page_start, page_end, content_type
    
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
        
        for index, chunk in enumerate(self.chunker.chunk(document)):
            page_start, page_end, content_type = self._extract_page_info(chunk)
            
            chunks.append({
                "chunk_id": f"{document_id}_chunk_{index + 1:03d}",
                "document_id": document_id,
                "text": chunk.text,
                "contextualized_text": self.chunker.contextualize(chunk),
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
