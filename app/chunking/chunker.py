"""
Chunking module for RAG pipeline.

This module chunks DoclingDocument objects into smaller pieces
suitable for embedding and retrieval.

Usage:
    Standalone (loads from saved JSON files):
        python chunker.py
        → Loads Docling JSONs from json_dir, saves chunks to output_dir

    As library (in-memory DoclingDocuments):
        from chunker import batch_chunk
        from parser import batch_parse

        parsed_docs = batch_parse(input_dir, output_dir)
        batch_chunk(parsed_docs, chunks_dir)
        # Chunks in-memory DoclingDocument objects directly
"""

import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.types import DoclingDocument


def chunk_document(document):
    """
    Chunk a DoclingDocument into retrieval-ready pieces.

    Args:
        document: A DoclingDocument object (from parser.py).

    Returns:
        list[dict]: List of chunk dictionaries.
    """

    chunker = HybridChunker(
        merge_peers=True,
    )

    chunks = []

    for index, chunk in enumerate(chunker.chunk(document)):
        chunks.append(
            {
                "chunk_id": f"chunk_{index}",
                "text": chunk.text,
                "contextualized_text": chunker.contextualize(chunk),
                "headings": chunk.meta.headings,
            }
        )

    return chunks


def save_chunks(
    chunks: list[dict],
    document_id: str,
    output_dir: Path,
):
    """
    Save generated chunks as JSON.

    Args:
        chunks: List of chunk dictionaries.
        document_id: Identifier for the document.
        output_dir: Directory to save the chunks.

    Returns:
        Path: Path to the saved JSON file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{document_id}.json"

    output_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "chunks": chunks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


def batch_chunk(
    parsed_documents: list[tuple],
    output_dir: Path,
):
    """
    Chunk multiple parsed documents.

    Args:
        parsed_documents: List of (pdf_path, DoclingDocument) tuples
                          (as returned by parser.batch_parse).
        output_dir: Directory to save chunk JSON files.

    Returns:
        int: Total number of chunks created.
    """

    output_dir = Path(output_dir)
    total_chunks = 0

    for pdf_path, document in parsed_documents:
        document_id = pdf_path.stem

        chunks = chunk_document(document)
        save_chunks(chunks, document_id, output_dir)

        print(f"Chunked: {pdf_path.name} → {len(chunks)} chunks")
        total_chunks += len(chunks)

    return total_chunks


def load_docling_json(json_path: Path) -> DoclingDocument:
    """
    Load a DoclingDocument from a saved JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        DoclingDocument: The loaded document.
    """
    return DoclingDocument.model_validate_json(
        json_path.read_text(encoding="utf-8")
    )


def batch_chunk_from_json(
    json_dir: Path,
    output_dir: Path,
):
    """
    Load DoclingDocument JSONs from a directory and chunk them.

    Args:
        json_dir: Directory containing Docling JSON files.
        output_dir: Directory to save chunk JSON files.

    Returns:
        int: Total number of chunks created.
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return 0

    print(f"Found {len(json_files)} JSON files to chunk.")

    total_chunks = 0

    for index, json_path in enumerate(json_files, start=1):
        print(f"[{index}/{len(json_files)}] Loading: {json_path.name}")

        document = load_docling_json(json_path)
        document_id = json_path.stem

        chunks = chunk_document(document)
        save_chunks(chunks, document_id, output_dir)

        print(f"  → {len(chunks)} chunks")
        total_chunks += len(chunks)

    print("=" * 60)
    print(f"Chunking complete. Total chunks: {total_chunks}")
    print("=" * 60)

    return total_chunks


if __name__ == "__main__":

    JSON_DIR = (
        r"C:\personal-projects\RAG\study_rag"
        r"\data\processed\docling"
    )

    CHUNKS_DIR = (
        r"C:\personal-projects\RAG\study_rag"
        r"\data\processed\chunks"
    )

    batch_chunk_from_json(JSON_DIR, CHUNKS_DIR)
