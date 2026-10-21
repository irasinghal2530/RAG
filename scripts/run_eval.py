"""
Run Evaluation Script

Executes the evaluation suite on the RAG pipeline.

Usage:
    python scripts/run_eval.py --eval-file data/eval_set.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Run as a script, sys.path[0] is scripts/, so `src` would not be importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate retrieval quality."""
    print("\nEvaluating Retrieval...")
    
    queries = []
    skipped = 0
    for item in eval_set:
        query = item["question"]
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        if not relevant_ids:
            skipped += 1
            continue

        results = retriever.search(query, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]
        queries.append({
            "query": query,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
        })

    if skipped:
        print(f"  Skipping {skipped} queries with no relevant_chunk_ids")
    if not queries:
        print("  No labeled queries; retrieval metrics not computed.")
        return {}

    # Metrics@k for k past the number retrieved would just repeat @top_k.
    top_k = top_k or retriever.default_top_k
    evaluator = RetrievalEvaluator(k_values=[k for k in (1, 3, 5, 10) if k <= top_k])
    return evaluator.evaluate(queries)


def evaluate_generation(
    retriever: Retriever,
    llm_client: LLMClient,
    eval_set: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate answer generation quality. Returns (averages, per-sample results)."""
    print("\nEvaluating Generation...")

    if llm_client is None:
        print("  Skipping (no LLM client)")
        return {}, []
    
    prompt_builder = PromptBuilder()
    answer_evaluator = AnswerEvaluator()
    
    samples = []
    for item in eval_set:
        query = item["question"]
        
        # Retrieve context
        results = retriever.search(query, top_k=top_k)
        
        # Build prompt and generate
        prompt = prompt_builder.build(query, results)
        response = llm_client.generate([
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ])

        # Judge against what the generator saw: chunks past
        # max_context_length were dropped from the prompt.
        context = "\n\n".join([r["document"] for r in prompt["used_chunks"]])
        
        sample = {
            "question": query,
            "answer": response["content"],
            "context": context,
        }
        sample["scores"] = answer_evaluator.evaluate(
            sample["question"], sample["answer"], sample["context"]
        )
        sample["retrieved_chunk_ids"] = [r["chunk_id"] for r in prompt["used_chunks"]]
        samples.append(sample)
        print_sample(len(samples), len(eval_set), sample, item)

    metrics: Dict[str, Any] = {}
    if samples:
        for key in samples[0]["scores"]:
            metrics[f"avg_{key}"] = sum(s["scores"][key] for s in samples) / len(samples)
    metrics["num_samples"] = len(samples)
    return metrics, samples


def print_sample(
    index: int,
    total: int,
    sample: Dict[str, Any],
    item: Dict[str, Any],
) -> None:
    """Print one question with its generated answer and scores."""
    print(f"\n{'-' * 60}")
    print(f"[{index}/{total}] Q: {sample['question']}")
    print(f"A: {sample['answer']}")
    expected = item.get("ground_truth", item.get("expected_answer"))
    if expected:
        print(f"Expected: {expected}")
    scores = ", ".join(f"{k}={v:.2f}" for k, v in sample["scores"].items())
    print(f"Scores: {scores}")


def main():
    # Answers contain characters (e.g. U+202F) the Windows cp1252 console cannot encode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

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
        default=None,
        help="Number of results to retrieve (default: retriever_config.yaml)",
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
    print(f"Top-K:       {args.top_k or 'from config'}")
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
    generation_samples = []
    if not args.skip_generation:
        try:
            llm_client = LLMClient()
            generation_metrics, generation_samples = evaluate_generation(
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
            "samples": [
                {k: v for k, v in s.items() if k != "context"}
                for s in generation_samples
            ],
            "config": {
                "eval_file": str(eval_file),
                "top_k": args.top_k,
                "num_samples": len(eval_set),
            },
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
