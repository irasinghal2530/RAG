"""
PDF parsing module for RAG pipeline.

This module uses Docling to parse PDFs into structured DoclingDocuments
and saves artifacts (JSON/Markdown).

Usage:
    Standalone:
        python parser.py
        → Parses all PDFs in input_dir, saves JSON/MD to output_dir

    As library:
        from parser import batch_parse
        parsed_docs = batch_parse(input_dir, output_dir)
        # Returns list of (pdf_path, DoclingDocument) tuples
        # Pass to chunker.batch_chunk() for in-memory chunking
"""

import logging
import os
import traceback
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


logger = logging.getLogger(__name__)


def build_converter() -> DocumentConverter:
    """Build and return the Docling PDF converter."""

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        images_scale=2.0,
        accelerator_options=AcceleratorOptions(
            num_threads=os.cpu_count() or 4,
        ),
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options,
            ),
        },
    )


def parse_single_pdf(
    converter: DocumentConverter,
    pdf_path: Path,
):
    """
    Parse a single PDF using Docling.

    Args:
        converter: The Docling DocumentConverter.
        pdf_path: Path to the PDF file.

    Returns:
        DoclingDocument: Parsed document.
    """

    logger.info("Parsing: %s", pdf_path.name)

    result = converter.convert(pdf_path)

    return result.document


def save_artifacts(
    document,
    pdf_path: Path,
    output_dir: Path,
):
    """
    Save the parsed document as JSON and Markdown.

    Args:
        document: The DoclingDocument.
        pdf_path: Original PDF path (used for naming).
        output_dir: Directory to save artifacts.

    Returns:
        tuple: (json_path, markdown_path)
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_dir / f"{pdf_path.stem}.json"
    markdown_path = output_dir / f"{pdf_path.stem}.md"

    # Save structured Docling JSON
    document.save_as_json(
        json_path,
        ensure_ascii=False,
    )

    # Save Markdown representation
    markdown_path.write_text(
        document.export_to_markdown(),
        encoding="utf-8",
    )

    logger.info(
        "Saved: %s | %s",
        json_path.name,
        markdown_path.name,
    )

    return json_path, markdown_path


def batch_parse(
    input_dir,
    output_dir,
):
    """
    Parse all PDFs in input_dir and save artifacts.

    Returns a list of (pdf_path, document) tuples for further processing.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Directory to save JSON/Markdown artifacts.

    Returns:
        list[tuple]: List of (pdf_path, DoclingDocument) for each successful parse.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    pdf_files = sorted(
        input_dir.glob("*.pdf")
    )

    if not pdf_files:
        logger.warning(
            "No PDF files found in: %s",
            input_dir,
        )
        return []

    logger.info(
        "Found %d PDF files.",
        len(pdf_files),
    )

    # Build converter once.
    converter = build_converter()

    parsed_documents = []
    failed = 0

    for index, pdf_path in enumerate(pdf_files, start=1):

        logger.info(
            "[%d/%d] Parsing %s",
            index,
            len(pdf_files),
            pdf_path.name,
        )

        try:
            # Parse PDF → DoclingDocument
            document = parse_single_pdf(
                converter,
                pdf_path,
            )

            # Save artifacts (JSON/Markdown)
            save_artifacts(
                document,
                pdf_path,
                output_dir,
            )

            # Add to results for downstream processing
            parsed_documents.append((pdf_path, document))

        except Exception:
            failed += 1

            logger.error(
                "Failed to parse: %s",
                pdf_path.name,
            )

            logger.error(
                traceback.format_exc()
            )

    logger.info("=" * 60)
    logger.info("Parsing complete.")
    logger.info("Successful: %d", len(parsed_documents))
    logger.info("Failed:    %d", failed)
    logger.info("Total:     %d", len(pdf_files))
    logger.info("=" * 60)

    return parsed_documents


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
    )

    INPUT_DIR = (
        r"C:\personal-projects\RAG\study_rag"
        r"\data\raw_pdfs\corpus_pdfs"
    )

    OUTPUT_DIR = (
        r"C:\personal-projects\RAG\study_rag"
        r"\data\processed\docling"
    )

    # Parse PDFs and save artifacts (JSON/Markdown)
    parsed = batch_parse(
        INPUT_DIR,
        OUTPUT_DIR,
    )

    print(f"\nParsed {len(parsed)} documents.")
    print(f"Artifacts saved to: {OUTPUT_DIR}")
    print("\nRun chunker.py next to generate chunks.")
