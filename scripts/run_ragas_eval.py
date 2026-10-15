"""
Run RAGAS Evaluation Script

Executes comprehensive RAGAS evaluation on the RAG pipeline.

Usage:
    python scripts/run_ragas_eval.py --eval-file data/eval_set.json
    python scripts/run_ragas_eval.py --eval-file data/eval_set.json --metrics faithfulness context_precision
    python scripts/run_ragas_eval.py --eval-file data/eval_set.json --output results/ragas_results.json

Metrics computed:
    - faithfulness: Is the answer grounded in the context?
    - answer_relevancy: Does the answer address the question?
    - context_relevancy: Are the retrieved contexts relevant?
    - context_precision: Are relevant contexts ranked higher?
    - context_recall: Does context cover the ground truth? (requires ground_truth)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

from src.retrieval import Retriever
from src.generation import PromptBuilder, LLMClient
from evaluation.ragas_eval import RagasEvaluator, RagasInput


def load_eval_set(eval_file: Path) -> List[Dict[str, Any]]:
    """Load evaluation set from JSON file."""
    return json.loads(eval_file.read_text(encoding="utf-8"))


def run_rag_pipeline(
    retriever: Retriever,
    llm_client: Optional[LLMClient],
    question: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run the RAG pipeline for a single question.
    
    Returns:
        dict: Contains answer, contexts, and metadata.
    """
    # Retrieve relevant chunks
    results = retriever.search(question, top_k=top_k)
    
    # Extract context strings
    contexts = [r["document"] for r in results]
    
    # Generate answer if LLM is available
    answer = ""
    if llm_client:
        prompt_builder = PromptBuilder()
        messages = prompt_builder.build_messages(question, results)
        response = llm_client.generate(messages)
        answer = response["content"]
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "retrieved_chunks": results,
    }


def evaluate_with_ragas(
    retriever: Retriever,
    llm_client: Optional[LLMClient],
    eval_set: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on the evaluation set.
    
    Args:
        retriever: Retriever instance.
        llm_client: LLM client for answer generation.
        eval_set: Evaluation dataset.
        metrics: List of metrics to compute.
        top_k: Number of chunks to retrieve.
        verbose: Print progress.
        
    Returns:
        dict: RAGAS evaluation results.
    """
    evaluator = RagasEvaluator()
    
    if not evaluator.client:
        print("Error: RAGAS evaluation requires a Google Gemini API key.")
        print("Set GOOGLE_API_KEY in .env (https://aistudio.google.com/apikey).")
        return {}
    
    samples = []
    rag_responses = []
    
    print("\n" + "=" * 60)
    print("Step 1: Running RAG Pipeline")
    print("=" * 60)
    
    for i, item in enumerate(eval_set):
        question = item["question"]
        ground_truth = item.get("ground_truth", item.get("expected_answer"))
        
        if verbose:
            print(f"[{i+1}/{len(eval_set)}] Processing: {question[:50]}...")
        
        # Run RAG pipeline
        rag_result = run_rag_pipeline(retriever, llm_client, question, top_k)
        rag_responses.append(rag_result)
        
        # Create RAGAS input
        sample = RagasInput(
            question=question,
            answer=rag_result["answer"],
            contexts=rag_result["contexts"],
            ground_truth=ground_truth,
        )
        samples.append(sample)
    
    print("\n" + "=" * 60)
    print("Step 2: Computing RAGAS Metrics")
    print("=" * 60)
    
    results = evaluator.evaluate_batch(samples, metrics=metrics, verbose=verbose)
    
    # Add RAG responses to results
    results["rag_responses"] = rag_responses
    
    return results


def print_results(results: Dict[str, Any]):
    """Print formatted RAGAS results."""
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    
    aggregates = results.get("aggregate_metrics", {})
    
    if not aggregates:
        print("No metrics computed.")
        return
    
    print("\nAggregate Metrics:")
    print("-" * 40)
    
    metric_descriptions = {
        "avg_faithfulness": "Faithfulness (answer grounded in context)",
        "avg_answer_relevancy": "Answer Relevancy (answer addresses question)",
        "avg_context_relevancy": "Context Relevancy (contexts are relevant)",
        "avg_context_precision": "Context Precision (relevant contexts ranked higher)",
        "avg_context_recall": "Context Recall (context covers ground truth)",
    }
    
    for key, value in sorted(aggregates.items()):
        description = metric_descriptions.get(key, key)
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {description}")
        print(f"    {bar} {value:.4f}")
        print()
    
    print(f"Total samples evaluated: {results.get('num_samples', 0)}")


def save_results(results: Dict[str, Any], output_path: Path):
    """Save results to JSON file."""
    # Convert non-serializable objects
    serializable = {
        "aggregate_metrics": results.get("aggregate_metrics", {}),
        "per_sample_metrics": results.get("per_sample_metrics", []),
        "num_samples": results.get("num_samples", 0),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Optionally include detailed responses
    if "rag_responses" in results:
        # Simplify for JSON
        responses = []
        for r in results["rag_responses"]:
            responses.append({
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "num_chunks": len(r.get("retrieved_chunks", [])),
            })
        serializable["rag_responses"] = responses
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ragas_eval.py --eval-file data/eval_set.json
  python scripts/run_ragas_eval.py --eval-file data/eval_set.json --metrics faithfulness context_precision
  python scripts/run_ragas_eval.py --eval-file data/eval_set.json --output results/ragas.json
        """,
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
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=[
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_precision",
            "context_recall",
        ],
        help="Specific metrics to compute (default: all)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip answer generation (only evaluate retrieval metrics)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"Error: Evaluation file not found: {eval_file}")
        return 1
    
    print("=" * 60)
    print("RAGAS Evaluation Suite")
    print("=" * 60)
    print(f"Eval file:   {eval_file}")
    print(f"Vector DB:   {args.vectordb_dir}")
    print(f"Top-K:       {args.top_k}")
    if args.metrics:
        print(f"Metrics:     {', '.join(args.metrics)}")
    else:
        print("Metrics:     All available")
    print("=" * 60)
    
    # Load evaluation set
    eval_set = load_eval_set(eval_file)
    print(f"Loaded {len(eval_set)} evaluation samples")
    
    # Initialize retriever
    retriever = Retriever(args.vectordb_dir)
    
    # Initialize LLM client
    llm_client = None
    if not args.skip_generation:
        try:
            llm_client = LLMClient()
            print(f"LLM client initialized ({llm_client.model_name})")
        except (ImportError, ValueError) as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Evaluation will skip answer generation metrics")
    
    # Determine metrics to compute
    metrics = args.metrics
    if args.skip_generation:
        # Only compute retrieval-based metrics
        metrics = [m for m in (metrics or ["context_relevancy", "context_precision"]) 
                   if m in ["context_relevancy", "context_precision", "context_recall"]]
    
    # Run evaluation
    results = evaluate_with_ragas(
        retriever=retriever,
        llm_client=llm_client,
        eval_set=eval_set,
        metrics=metrics,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        save_results(results, Path(args.output))
    
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
