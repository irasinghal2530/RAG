"""
Document loader for RAG pipeline.

Loads PDFs, URLs, and other document formats into a unified structure.

Usage:
    from src.ingestion.loader import DocumentLoader
    
    loader = DocumentLoader()
    documents = loader.load_directory("data/raw")
"""

import os
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


class DocumentLoader:
    """Unified document loader supporting multiple formats."""
    
    def __init__(
        self,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        generate_picture_images: bool = True,
        images_scale: float = 2.0,
    ):
        """
        Initialize the document loader.
        
        Args:
            do_ocr: Enable OCR for scanned documents.
            do_table_structure: Extract table structure.
            generate_picture_images: Extract images from documents.
            images_scale: Scale factor for extracted images.
        """
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.generate_picture_images = generate_picture_images
        self.images_scale = images_scale
        self._converter = None
    
    @property
    def converter(self) -> DocumentConverter:
        """Lazy-load the document converter."""
        if self._converter is None:
            options = PdfPipelineOptions(
                do_ocr=self.do_ocr,
                do_table_structure=self.do_table_structure,
                generate_picture_images=self.generate_picture_images,
                images_scale=self.images_scale,
                accelerator_options=AcceleratorOptions(
                    num_threads=os.cpu_count() or 4,
                ),
            )
            
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=options,
                    ),
                },
            )
        return self._converter
    
    def load_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Load a single PDF document.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            dict: Document data with id, source, and parsed content.
        """
        pdf_path = Path(pdf_path)
        print(f"Loading: {pdf_path.name}")
        
        result = self.converter.convert(pdf_path)
        
        return {
            "document_id": pdf_path.stem,
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "document": result.document,
        }
    
    def load_directory(
        self,
        input_dir: str | Path,
        pattern: str = "*.pdf",
    ) -> List[Dict[str, Any]]:
        """
        Load all documents from a directory.
        
        Args:
            input_dir: Directory containing documents.
            pattern: Glob pattern for file matching.
            
        Returns:
            list: List of loaded document dictionaries.
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        files = sorted(input_dir.glob(pattern))
        
        if not files:
            print(f"No files matching '{pattern}' found in: {input_dir}")
            return []
        
        print(f"Found {len(files)} files to load.")
        
        documents = []
        failed = 0
        
        for index, file_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] Loading: {file_path.name}")
            
            try:
                doc = self.load_pdf(file_path)
                documents.append(doc)
            except Exception:
                failed += 1
                print(f"Failed to load: {file_path.name}")
                traceback.print_exc()
        
        print("=" * 60)
        print(f"Loading complete. Success: {len(documents)}, Failed: {failed}")
        print("=" * 60)
        
        return documents


def load_pdfs(input_dir: str | Path, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to load all PDFs from a directory.
    
    Args:
        input_dir: Directory containing PDF files.
        **kwargs: Additional arguments for DocumentLoader.
        
    Returns:
        list: List of loaded documents.
    """
    loader = DocumentLoader(**kwargs)
    return loader.load_directory(input_dir, pattern="*.pdf")


def load_documents(
    input_dir: str | Path,
    pattern: str = "*.*",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Load documents matching a pattern from a directory.
    
    Args:
        input_dir: Directory containing documents.
        pattern: Glob pattern for file matching.
        **kwargs: Additional arguments for DocumentLoader.
        
    Returns:
        list: List of loaded documents.
    """
    loader = DocumentLoader(**kwargs)
    return loader.load_directory(input_dir, pattern=pattern)


if __name__ == "__main__":
    # Demo usage
    RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
    
    if RAW_DIR.exists():
        documents = load_pdfs(RAW_DIR)
        print(f"\nLoaded {len(documents)} documents")
        for doc in documents:
            print(f"  - {doc['document_id']}: {doc['source_file']}")
    else:
        print(f"Directory not found: {RAW_DIR}")
