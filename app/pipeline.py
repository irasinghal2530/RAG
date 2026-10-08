"""
Main RAG pipeline: Parse PDFs → Chunk documents.

This script orchestrates the full flow:
1. parser.py: PDF → DoclingDocument → save JSON/Markdown
2. chunker.py: DoclingDocument → chunks → save chunks

Usage:
    uv run python -m app.pipeline
"""

import logging
from pathlib import Path

from app.processing.parser import batch_parse
from app.chunking.chunker import batch_chunk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_pipeline(
    input_dir: str,
    output_dir: str,
):
    """
    Run the full parse → chunk pipeline.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Base output directory for all artifacts.
    """

    output_dir = Path(output_dir)

    # Step 1: Parse all PDFs
    logger.info("=" * 60)
    logger.info("STEP 1: Parsing PDFs")
    logger.info("=" * 60)

    parsed_documents = batch_parse(
        input_dir=input_dir,
        output_dir=output_dir,  # JSON/Markdown saved here
    )

    if not parsed_documents:
        logger.warning("No documents parsed. Exiting.")
        return

    # Step 2: Chunk all parsed documents
    logger.info("=" * 60)
    logger.info("STEP 2: Chunking documents")
    logger.info("=" * 60)

    chunks_dir = output_dir / "chunks"

    total_chunks = batch_chunk(
        parsed_documents=parsed_documents,
        output_dir=chunks_dir,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Documents parsed: %d", len(parsed_documents))
    logger.info("Total chunks: %d", total_chunks)
    logger.info("Artifacts saved to: %s", output_dir)
    logger.info("Chunks saved to: %s", chunks_dir)
    logger.info("=" * 60)


if __name__ == "__main__":

    INPUT_DIR = (
        r"C:\personal-projects\RAG\study_rag"
        r"\data\raw_pdfs\corpus_pdfs"
    )

    OUTPUT_DIR = (
        r"C:\personal-projects\RAG\study_rag"
        r"\data\processed\docling"
    )

    run_pipeline(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )
