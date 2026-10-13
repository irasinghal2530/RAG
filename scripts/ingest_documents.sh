#!/bin/bash
# Ingest Documents Script
# Runs the full ingestion pipeline: Load → Parse → Chunk → Embed → Index
# Retrieval is a separate command: python main.py retrieve "your query"

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "RAG Document Ingestion Pipeline"
echo "=========================================="

RAW_DIR="${RAW_DIR:-data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
VECTORDB_DIR="${VECTORDB_DIR:-data/vectordb}"

if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Raw documents directory not found: $RAW_DIR"
    exit 1
fi

PDF_COUNT=$(find "$RAW_DIR" -name "*.pdf" | wc -l)
echo "Found $PDF_COUNT PDF files in $RAW_DIR"

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "No PDF files to process. Exiting."
    exit 0
fi

EXTRA_ARGS=()
if [ "${RESET:-0}" = "1" ]; then
    EXTRA_ARGS+=(--reset)
fi

python main.py ingest \
    --raw-dir "$RAW_DIR" \
    --processed-dir "$PROCESSED_DIR" \
    --vectordb-dir "$VECTORDB_DIR" \
    "${EXTRA_ARGS[@]}"
