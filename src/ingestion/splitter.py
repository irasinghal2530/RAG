"""
Document splitter for RAG pipeline.

Splits documents into logical sections and saves parsed artifacts.

Usage:
    from src.ingestion.splitter import DocumentSplitter
    
    splitter = DocumentSplitter(output_dir="data/processed/docling")
    splitter.split_and_save(documents)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class DocumentSplitter:
    """Splits and saves parsed documents."""
    
    def __init__(self, output_dir: str | Path):
        """
        Initialize the document splitter.
        
        Args:
            output_dir: Directory to save split documents.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_artifacts(
        self,
        document,
        document_id: str,
    ) -> Tuple[Path, Path]:
        """
        Save document as JSON and Markdown artifacts.
        
        Args:
            document: The DoclingDocument object.
            document_id: Unique identifier for the document.
            
        Returns:
            tuple: (json_path, markdown_path)
        """
        json_path = self.output_dir / f"{document_id}.json"
        markdown_path = self.output_dir / f"{document_id}.md"
        
        # Save as JSON
        document.save_as_json(json_path, ensure_ascii=False)
        
        # Save as Markdown
        markdown_path.write_text(
            document.export_to_markdown(),
            encoding="utf-8",
        )
        
        print(f"Saved: {json_path.name} | {markdown_path.name}")
        
        return json_path, markdown_path
    
    def split_document(
        self,
        doc_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Split a single document and save artifacts.
        
        Args:
            doc_data: Document dictionary from loader.
            
        Returns:
            dict: Updated document data with artifact paths.
        """
        document = doc_data["document"]
        document_id = doc_data["document_id"]
        
        json_path, markdown_path = self.save_artifacts(document, document_id)
        
        return {
            **doc_data,
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }
    
    def batch_split(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Split multiple documents and save artifacts.
        
        Args:
            documents: List of document dictionaries from loader.
            
        Returns:
            list: Updated documents with artifact paths.
        """
        print(f"Splitting {len(documents)} documents...")
        
        results = []
        for index, doc_data in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] Splitting: {doc_data['document_id']}")
            result = self.split_document(doc_data)
            results.append(result)
        
        print("=" * 60)
        print(f"Split complete. {len(results)} documents processed.")
        print(f"Artifacts saved to: {self.output_dir}")
        print("=" * 60)
        
        return results


def split_document(
    doc_data: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Convenience function to split a single document.
    
    Args:
        doc_data: Document dictionary from loader.
        output_dir: Directory to save artifacts.
        
    Returns:
        dict: Updated document data.
    """
    splitter = DocumentSplitter(output_dir)
    return splitter.split_document(doc_data)


def batch_split(
    documents: List[Dict[str, Any]],
    output_dir: str | Path,
) -> List[Dict[str, Any]]:
    """
    Convenience function to split multiple documents.
    
    Args:
        documents: List of document dictionaries.
        output_dir: Directory to save artifacts.
        
    Returns:
        list: Updated documents.
    """
    splitter = DocumentSplitter(output_dir)
    return splitter.batch_split(documents)


if __name__ == "__main__":
    from src.ingestion.loader import load_pdfs
    
    RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "docling"
    
    if RAW_DIR.exists():
        documents = load_pdfs(RAW_DIR)
        
        if documents:
            results = batch_split(documents, OUTPUT_DIR)
            print(f"\nProcessed {len(results)} documents")
    else:
        print(f"Directory not found: {RAW_DIR}")
