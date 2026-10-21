"""
RAG Production System - Main Entry Point

Commands:

    # Offline pipeline: load → parse → chunk → embed → store in vector DB
    python main.py ingest
    python main.py ingest --reset

    # Online retrieval against an existing vector DB
    python main.py retrieve "What is the maintenance schedule?"

    # Full RAG: retrieve, then answer with the LLM and cite sources
    python main.py ask "What is the maintenance schedule?"

    # Basic evaluation (retrieval + generation metrics)
    python main.py eval --eval-file data/eval_set.json

    # RAGAS evaluation (comprehensive RAG assessment)
    python main.py ragas --eval-file data/eval_set.json
    python main.py ragas --eval-file data/eval_set.json --metrics faithfulness context_precision
"""

import argparse
import sys
from pathlib import Path


def cmd_ingest(args):
    """Load, parse, chunk, embed, and store documents in the vector DB."""
    from src.ingestion import load_pdfs, batch_split
    from src.chunking.chunker import DocumentChunker, load_docling_json
    from src.embedding import batch_embed
    from src.retrieval.retriever import VectorStore

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    vectordb_dir = Path(args.vectordb_dir)

    print("=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)

    print("\nStep 1: Loading and parsing documents...")
    docs = load_pdfs(raw_dir)

    if not docs:
        print("No documents to process.")
        return

    docling_dir = processed_dir / "docling"
    batch_split(docs, docling_dir)

    print("\nStep 2: Chunking documents...")
    chunks_dir = processed_dir / "chunks"
    chunker = DocumentChunker()

    total_chunks = 0
    for json_path in sorted(docling_dir.glob("*.json")):
        doc = load_docling_json(json_path)
        chunks = chunker.chunk_document(doc, json_path.stem)
        chunker.save_chunks(chunks, json_path.stem, chunks_dir)
        print(f"  Chunked: {json_path.name} -> {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"Total chunks: {total_chunks}")

    print("\nStep 3: Generating embeddings...")
    embeddings_dir = processed_dir / "embeddings"
    batch_embed(chunks_dir, embeddings_dir)

    print("\nStep 4: Building vector index...")
    store = VectorStore(persist_dir=vectordb_dir)
    store.index_all(embeddings_dir, reset=args.reset)

    stats = store.get_stats()
    print("\nVector Store Statistics:")
    print(f"  Total chunks:    {stats['total_chunks']}")
    print(f"  Total documents: {stats['total_documents']}")

    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)


def cmd_retrieve(args):
    """Retrieve relevant chunks for a query from the vector DB."""
    from src.retrieval import Retriever

    print("=" * 60)
    print(f"Query: {args.query}")
    print("=" * 60)

    retriever = Retriever(args.vectordb_dir)
    results = retriever.search(args.query, top_k=args.top_k)

    print(retriever.format_results(results))


def cmd_ask(args):
    """Retrieve chunks for a question and answer it with the LLM."""
    from src.retrieval import Retriever
    from src.generation import LLMClient

    retriever = Retriever(args.vectordb_dir)
    llm_client = LLMClient()

    results = retriever.search(args.question, top_k=args.top_k)
    if not results:
        print("No chunks retrieved; run `python main.py ingest` first.")
        return

    response = llm_client.generate_with_context(args.question, results)

    print("\n" + "=" * 60)
    print(f"Question: {args.question}")
    print("=" * 60)
    print(response["content"])
    print("\nSources:")
    for i, source in enumerate(response["sources"], start=1):
        pages = ""
        if source.get("page_start"):
            pages = f", pages {source['page_start']}-{source.get('page_end') or source['page_start']}"
        print(f"  [Source {i}] {source['chunk_id']} ({source.get('document_title') or source['document_id']}{pages})")
    print(f"\nModel: {response['model']} | tokens: {response['usage'].get('total_tokens', 'N/A')}")


def cmd_eval(args):
    """Run evaluation suite."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/run_eval.py", "--eval-file", args.eval_file]

    if args.vectordb_dir:
        cmd.extend(["--vectordb-dir", args.vectordb_dir])
    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.skip_generation:
        cmd.append("--skip-generation")

    sys.exit(subprocess.run(cmd).returncode)


def cmd_ragas(args):
    """Run RAGAS evaluation suite."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/run_ragas_eval.py", "--eval-file", args.eval_file]

    if args.vectordb_dir:
        cmd.extend(["--vectordb-dir", args.vectordb_dir])
    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.metrics:
        cmd.extend(["--metrics"] + args.metrics)
    if args.output:
        cmd.extend(["--output", args.output])
    if args.skip_generation:
        cmd.append("--skip-generation")
    if args.no_hybrid:
        cmd.append("--no-hybrid")
    if args.no_rerank:
        cmd.append("--no-rerank")

    sys.exit(subprocess.run(cmd).returncode)


def main():
    # LLM answers contain characters such as U+202F (narrow space in "125 psig")
    # that the Windows cp1252 console cannot encode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="RAG Production System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest --reset\n"
            '  python main.py retrieve "What is the maintenance schedule?"\n'
            '  python main.py ask "What is the maintenance schedule?"\n'
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load, parse, chunk, embed, and store documents",
    )
    ingest_parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing raw documents",
    )
    ingest_parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory for processed data",
    )
    ingest_parser.add_argument(
        "--vectordb-dir",
        default="data/vectordb",
        help="Directory for vector database",
    )
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before indexing",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve chunks for a query from the vector DB",
    )
    retrieve_parser.add_argument("query", help="Search query")
    retrieve_parser.add_argument(
        "--vectordb-dir",
        default="data/vectordb",
        help="Vector database directory",
    )
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results (default: retriever_config.yaml)",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Answer a question from the vector DB with the LLM (full RAG)",
    )
    ask_parser.add_argument("question", help="Question to answer")
    ask_parser.add_argument(
        "--vectordb-dir",
        default="data/vectordb",
        help="Vector database directory",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks given to the LLM (default: retriever_config.yaml)",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--eval-file",
        required=True,
        help="Evaluation set JSON file",
    )
    eval_parser.add_argument(
        "--vectordb-dir",
        default="data/vectordb",
        help="Vector database directory",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: retriever_config.yaml)",
    )
    eval_parser.add_argument(
        "--output",
        help="Output file for results (JSON)",
    )
    eval_parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation evaluation",
    )

    # RAGAS evaluation parser
    ragas_parser = subparsers.add_parser(
        "ragas",
        help="Run RAGAS evaluation (comprehensive RAG assessment)",
    )
    ragas_parser.add_argument(
        "--eval-file",
        required=True,
        help="Evaluation set JSON file",
    )
    ragas_parser.add_argument(
        "--vectordb-dir",
        default="data/vectordb",
        help="Vector database directory",
    )
    ragas_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: retriever_config.yaml)",
    )
    ragas_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Dense search only (skip BM25 fusion)",
    )
    ragas_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking",
    )
    ragas_parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_precision",
            "context_recall",
        ],
        help="Specific metrics to compute",
    )
    ragas_parser.add_argument(
        "--output",
        help="Output file for results (JSON)",
    )
    ragas_parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip answer generation",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "ragas":
        cmd_ragas(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
