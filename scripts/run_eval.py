"""
Run Evaluation Script

Executes the evaluation suite on the RAG pipeline.

Usage:
    python scripts/run_eval.py --eval-file data/eval_set.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.retrieval import Retriever
from src.generation import PromptBuilder, LLMClient
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.answer_eval import AnswerEvaluator


def load_eval_set(eval_file: Path) -> List[Dict[str, Any]]:
    """Load evaluation set from JSON file."""
    return json.loads(eval_file.read_text(encoding="utf-8"))


def evaluate_retrieval(
    retriever: Retriever,
    eval_set: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, float]:
    """Evaluate retrieval quality."""
    print("\nEvaluating Retrieval...")
    
    queries = []
    for item in eval_set:
        query = item["question"]
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        
        # Run retrieval
        results = retriever.search(query, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]
        
        queries.append({
            "query": query,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
        })
    
    evaluator = RetrievalEvaluator()
    return evaluator.evaluate(queries)


def evaluate_generation(
    retriever: Retriever,
    llm_client: LLMClient,
    eval_set: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, float]:
    """Evaluate answer generation quality."""
    print("\nEvaluating Generation...")
    
    if llm_client is None:
        print("  Skipping (no LLM client)")
        return {}
    
    prompt_builder = PromptBuilder()
    answer_evaluator = AnswerEvaluator()
    
    samples = []
    for item in eval_set:
        query = item["question"]
        
        # Retrieve context
        results = retriever.search(query, top_k=top_k)
        
        # Build prompt and generate
        messages = prompt_builder.build_messages(query, results)
        response = llm_client.generate(messages)
        
        # Build context string
        context = "\n\n".join([r["document"] for r in results])
        
        samples.append({
            "question": query,
            "answer": response["content"],
            "context": context,
        })
    
    return answer_evaluator.evaluate_batch(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation suite"
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to evaluation set JSON file",
    )
    parser.add_argument(
        "--vectordb-dir",
        type=str,
        default="data/vectordb",
        help="Vector database directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation evaluation (retrieval only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    
    args = parser.parse_args()
    
    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"Error: Evaluation file not found: {eval_file}")
        return 1
    
    print("=" * 60)
    print("RAG Evaluation Suite")
    print("=" * 60)
    print(f"Eval file:   {eval_file}")
    print(f"Vector DB:   {args.vectordb_dir}")
    print(f"Top-K:       {args.top_k}")
    print("=" * 60)
    
    # Load evaluation set
    eval_set = load_eval_set(eval_file)
    print(f"Loaded {len(eval_set)} evaluation samples")
    
    # Initialize retriever
    retriever = Retriever(args.vectordb_dir)
    
    # Retrieval evaluation
    retrieval_metrics = evaluate_retrieval(
        retriever, eval_set, top_k=args.top_k
    )
    
    print("\nRetrieval Metrics:")
    for key, value in sorted(retrieval_metrics.items()):
        print(f"  {key}: {value:.4f}")
    
    # Generation evaluation
    generation_metrics = {}
    if not args.skip_generation:
        try:
            llm_client = LLMClient()
            generation_metrics = evaluate_generation(
                retriever, llm_client, eval_set, top_k=args.top_k
            )
            
            print("\nGeneration Metrics:")
            for key, value in sorted(generation_metrics.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        except (ImportError, ValueError) as e:
            print(f"\nSkipping generation eval: {e}")
    
    # Save results
    if args.output:
        results = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "config": {
                "eval_file": str(eval_file),
                "top_k": args.top_k,
                "num_samples": len(eval_set),
            },
        }
        
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
