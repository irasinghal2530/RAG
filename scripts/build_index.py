"""
Build Vector Index Script

Indexes all embeddings into the vector store.

Usage:
    python scripts/build_index.py [--reset]
"""

import argparse
from pathlib import Path

from src.retrieval.retriever import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Build vector index from embeddings"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/processed/embeddings",
        help="Directory containing embedding JSON files",
    )
    parser.add_argument(
        "--vectordb-dir",
        type=str,
        default="data/vectordb",
        help="Directory to store vector database",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="Collection name",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset (delete) existing collection",
    )
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    vectordb_dir = Path(args.vectordb_dir)
    
    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        return 1
    
    print("=" * 60)
    print("Building Vector Index")
    print("=" * 60)
    print(f"Embeddings: {embeddings_dir}")
    print(f"Vector DB:  {vectordb_dir}")
    print(f"Collection: {args.collection}")
    print(f"Reset:      {args.reset}")
    print("=" * 60)
    
    # Initialize vector store
    store = VectorStore(
        persist_dir=vectordb_dir,
        collection_name=args.collection,
    )
    
    # Index all embeddings
    total = store.index_all(
        embeddings_dir=embeddings_dir,
        reset=args.reset,
    )
    
    # Print stats
    stats = store.get_stats()
    
    print("\nVector Store Statistics:")
    print(f"  Collection:      {stats['collection_name']}")
    print(f"  Total chunks:    {stats['total_chunks']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Document IDs:    {stats['document_ids']}")
    print(f"  Content types:   {stats['content_types']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
